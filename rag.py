"""Lightweight, dependency-free RAG over SEC filings and uploaded PDFs.

Why TF-IDF instead of dense embeddings?
- Zero model downloads, zero GPU, zero API cost (cheapest option).
- Works on Python 3.14 without ONNX runtime hassles.
- Plenty good for retrieval inside a single ~10-K-sized document.

The two main entry points are :func:`extract_text_from_pdf_bytes` (used by the
Streamlit upload flow) and :class:`DocumentIndex` (the in-memory TF-IDF
retriever the agent's tools call into).
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Text extraction                                                             #
# --------------------------------------------------------------------------- #


def extract_text_from_pdf_bytes(data: bytes) -> str:
    """Extract a clean string of text from raw PDF bytes."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pypdf is required for PDF parsing. Install with `pip install pypdf`."
        ) from exc

    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - depends on PDF
            logger.warning("Failed to read a PDF page: %s", exc)
            txt = ""
        if txt:
            pages.append(txt)
    return _normalize_whitespace("\n\n".join(pages))


def extract_text_from_html(html: str) -> str:
    """Strip HTML tags and noisy whitespace from an SEC filing's HTML body."""
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover
        raise ImportError("beautifulsoup4 is required for HTML parsing.") from exc

    soup = BeautifulSoup(html, "lxml") if _has_lxml() else BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return _normalize_whitespace(text)


def _has_lxml() -> bool:
    try:
        import lxml  # noqa: F401

        return True
    except ImportError:  # pragma: no cover
        return False


_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANKLINE_RE = re.compile(r"\n{3,}")


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANKLINE_RE.sub("\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Chunking                                                                    #
# --------------------------------------------------------------------------- #


def chunk_text(
    text: str,
    *,
    target_chars: int = 1200,
    overlap_chars: int = 150,
) -> list[str]:
    """Split ``text`` into overlapping paragraph-ish chunks.

    Optimized for SEC filings: prefer paragraph breaks, then fall back to
    fixed-width windows so we never produce an empty chunk.
    """
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) <= target_chars or not current:
            current.append(para)
            current_len += len(para) + 2
        else:
            chunks.append("\n\n".join(current).strip())
            tail = current[-1] if current else ""
            current = [tail[-overlap_chars:], para] if overlap_chars else [para]
            current_len = sum(len(p) for p in current) + 2 * (len(current) - 1)

    if current:
        chunks.append("\n\n".join(current).strip())

    # Hard cap: if a single paragraph is huge (common in 10-K risk sections),
    # window it.
    final: list[str] = []
    for c in chunks:
        if len(c) <= target_chars * 1.5:
            final.append(c)
            continue
        step = max(1, target_chars - overlap_chars)
        for i in range(0, len(c), step):
            final.append(c[i : i + target_chars])

    return [c for c in final if c.strip()]


# --------------------------------------------------------------------------- #
# Index                                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class RetrievedChunk:
    """A chunk returned from :meth:`DocumentIndex.search`."""

    text: str
    score: float
    rank: int

    def render(self, max_chars: int = 600) -> str:
        body = self.text if len(self.text) <= max_chars else self.text[:max_chars] + "..."
        return f"[chunk #{self.rank} | score={self.score:.3f}]\n{body}"


class DocumentIndex:
    """In-memory TF-IDF index over a single document's chunks."""

    def __init__(self, chunks: Iterable[str], *, label: str = "document") -> None:
        self.label = label
        self.chunks: list[str] = [c for c in chunks if c and c.strip()]
        if not self.chunks:
            raise ValueError("DocumentIndex requires at least one non-empty chunk.")

        self._vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            stop_words="english",
            lowercase=True,
        )
        self._matrix = self._vectorizer.fit_transform(self.chunks)

    def search(self, query: str, *, top_k: int = 4) -> list[RetrievedChunk]:
        """Return the top-``k`` chunks most relevant to ``query``."""
        if not query.strip():
            return []
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix).ravel()
        order = np.argsort(-sims)[:top_k]
        return [
            RetrievedChunk(text=self.chunks[i], score=float(sims[i]), rank=rank + 1)
            for rank, i in enumerate(order)
            if sims[i] > 0
        ]

    @classmethod
    def from_text(cls, text: str, *, label: str = "document") -> "DocumentIndex":
        return cls(chunk_text(text), label=label)


# --------------------------------------------------------------------------- #
# Module-level singleton store for the Streamlit-uploaded document            #
# --------------------------------------------------------------------------- #


class _DocumentStore:
    """Holds the most recently uploaded document so a stateless tool can read it."""

    def __init__(self) -> None:
        self._index: DocumentIndex | None = None
        self._source_label: str = ""

    def set_from_text(self, text: str, *, label: str) -> int:
        if not text.strip():
            self._index = None
            self._source_label = ""
            return 0
        self._index = DocumentIndex.from_text(text, label=label)
        self._source_label = label
        return len(self._index.chunks)

    def clear(self) -> None:
        self._index = None
        self._source_label = ""

    @property
    def index(self) -> DocumentIndex | None:
        return self._index

    @property
    def source_label(self) -> str:
        return self._source_label

    @property
    def is_ready(self) -> bool:
        return self._index is not None


UPLOADED_DOC = _DocumentStore()
