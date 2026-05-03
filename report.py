"""Executive Brief export for Market Sentinel.

Produces two artifacts the Streamlit UI can offer for download:

- Markdown (instant, plain text, easiest to read in any editor)
- PDF       (via :mod:`fpdf2` - pure Python, no system fonts needed)

The brief format intentionally mirrors what an analyst would write up after a
research session: a one-sentence headline, key findings, the model's reasoning
trail, and a sources list.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fpdf import FPDF


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _coerce_step(step: Any) -> tuple[str, str, str]:
    """Normalize a LangChain (action, observation) tuple OR a (label, (a, o)) tuple.

    Returns ``(actor, tool_call_repr, observation_repr)``.
    """
    actor = ""
    payload = step

    if isinstance(step, tuple) and len(step) == 2 and isinstance(step[0], str):
        actor, payload = step

    try:
        action, observation = payload  # type: ignore[misc]
        tool_name = getattr(action, "tool", "tool")
        tool_input = getattr(action, "tool_input", "")
        return (
            actor,
            f"{tool_name}({tool_input})",
            str(observation),
        )
    except Exception:
        return (actor, "(unparsed)", str(payload))


def build_markdown_brief(
    *,
    question: str,
    answer: str,
    mode: str,
    sentiment: dict[str, Any] | None = None,
    technical_brief: str | None = None,
    fundamental_brief: str | None = None,
    intermediate_steps: list[Any] | None = None,
) -> str:
    """Compose a Markdown executive brief."""
    lines: list[str] = []
    lines.append("# Market Sentinel - Executive Brief")
    lines.append("")
    lines.append(f"*Generated: {_now()}*  ")
    lines.append(f"*Mode: {mode}*")
    lines.append("")
    lines.append("## Question")
    lines.append(f"> {question.strip()}")
    lines.append("")

    if sentiment:
        label = sentiment.get("label", "Neutral")
        compound = sentiment.get("compound", 0.0)
        lines.append("## Aggregate Sentiment")
        lines.append(f"- **Label:** {label}")
        lines.append(f"- **Compound score (-1 to 1):** {compound}")
        lines.append("")

    lines.append("## Final Answer")
    lines.append(answer.strip() or "_(no answer)_")
    lines.append("")

    if technical_brief:
        lines.append("## Technical Analyst")
        lines.append(technical_brief.strip())
        lines.append("")
    if fundamental_brief:
        lines.append("## Fundamental Researcher")
        lines.append(fundamental_brief.strip())
        lines.append("")

    if intermediate_steps:
        lines.append("## Reasoning Trail")
        for i, step in enumerate(intermediate_steps, 1):
            actor, call, obs = _coerce_step(step)
            actor_label = f" _{actor}_" if actor else ""
            lines.append(f"**{i}.{actor_label} `{call}`**")
            obs_short = obs if len(obs) <= 600 else obs[:600] + "..."
            lines.append("")
            lines.append("```")
            lines.append(obs_short)
            lines.append("```")
            lines.append("")

    sources = _extract_sources(answer, intermediate_steps)
    if sources:
        lines.append("## Sources")
        for url in sources:
            lines.append(f"- {url}")
        lines.append("")

    lines.append("---")
    lines.append("_Disclaimer: Market Sentinel is a research prototype. "
                 "Not investment advice._")
    return "\n".join(lines)


_URL_RE = re.compile(r"https?://[^\s\)\]\>'\"]+")


def _extract_sources(answer: str, steps: list[Any] | None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for blob in [answer, *(str(s) for s in (steps or []))]:
        for match in _URL_RE.findall(blob):
            url = match.rstrip(".,;:")
            if url not in seen:
                seen.add(url)
                out.append(url)
    return out


# --------------------------------------------------------------------------- #
# PDF                                                                         #
# --------------------------------------------------------------------------- #


_PDF_REPLACEMENTS = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2026": "...",
    "\u00a0": " ",
    "\u2022": "*",
}


def _safe_for_latin1(text: str) -> str:
    for src, dst in _PDF_REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", "replace").decode("latin-1")


def _markdown_to_html_fragment(md: str) -> str:
    """Turn GitHub-flavoured-ish Markdown into HTML for :meth:`FPDF.write_html`.

    The LLM answers (and specialist briefs) are usually Markdown: ``##`` headings,
    ``**bold**``, lists, links, ``---`` rules, and fenced `` ```json`` blocks.
    Feeding that HTML into fpdf2 renders it as real typography instead of raw
    ``##`` / ``**`` characters in the PDF.
    """
    import markdown

    text = (md or "").strip() or "_Empty._"
    # ``nl2br`` keeps single newlines inside paragraphs (common in chat output).
    # ``fenced_code`` + ``tables`` match typical model / tool JSON formatting.
    return markdown.markdown(
        text,
        extensions=["nl2br", "fenced_code", "tables"],
    )


def _pdf_write_markdown_block(pdf: "FPDF", md: str) -> None:
    """Render a Markdown prose block using fpdf2's HTML engine."""
    from fpdf import FPDF

    assert isinstance(pdf, FPDF)
    html = _markdown_to_html_fragment(md)
    with pdf.local_context():
        pdf.write_html(
            html,
            warn_on_tags_not_matching=False,
        )
    pdf.ln(2)


def _pdf_write_sources_html(pdf: "FPDF", urls: list[str]) -> None:
    """Render extracted URLs as a clickable bullet list."""
    from fpdf import FPDF

    assert isinstance(pdf, FPDF)
    if not urls:
        return
    items = "".join(
        f'<li><a href="{_html_escape_attr(u)}">{_html_escape_text(u)}</a></li>'
        for u in urls
    )
    html = f"<ul>{items}</ul>"
    with pdf.local_context():
        pdf.write_html(html, warn_on_tags_not_matching=False)
    pdf.ln(2)


def _html_escape_attr(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _html_escape_text(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def build_pdf_brief(
    *,
    question: str,
    answer: str,
    mode: str,
    sentiment: dict[str, Any] | None = None,
    technical_brief: str | None = None,
    fundamental_brief: str | None = None,
    intermediate_steps: list[Any] | None = None,
) -> bytes:
    """Render the same brief as a downloadable PDF byte string."""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ``new_x="LMARGIN", new_y="NEXT"`` is required in fpdf2 >= 2.8 to reset
    # the X cursor to the left margin after a ``multi_cell`` - otherwise the
    # next call sees ~0 mm of usable horizontal space and raises
    # ``FPDFException: Not enough horizontal space to render a single character``.
    def h1(text: str) -> None:
        pdf.set_font("Helvetica", "B", 16)
        pdf.multi_cell(0, 8, _safe_for_latin1(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    def h2(text: str) -> None:
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 7, _safe_for_latin1(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    def body(text: str) -> None:
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(
            0, 5, _safe_for_latin1(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT
        )
        pdf.ln(1)

    def mono(text: str) -> None:
        pdf.set_font("Courier", "", 8)
        # ``wrapmode="CHAR"`` allows breaking in the middle of long unbroken
        # strings (URLs, JSON blobs, base64).
        pdf.multi_cell(
            0,
            4,
            _safe_for_latin1(text),
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
            wrapmode="CHAR",
        )
        pdf.ln(1)

    h1("Market Sentinel - Executive Brief")
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(
        0,
        5,
        _safe_for_latin1(f"Generated: {_now()}  |  Mode: {mode}"),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.ln(3)

    h2("Question")
    _pdf_write_markdown_block(pdf, question.strip())

    if sentiment:
        h2("Aggregate Sentiment")
        body(
            f"Label: {sentiment.get('label', 'Neutral')}    "
            f"Compound: {sentiment.get('compound', 0.0)}"
        )

    h2("Final Answer")
    _pdf_write_markdown_block(pdf, answer.strip() or "(no answer)")

    if technical_brief:
        h2("Technical Analyst")
        _pdf_write_markdown_block(pdf, technical_brief.strip())

    if fundamental_brief:
        h2("Fundamental Researcher")
        _pdf_write_markdown_block(pdf, fundamental_brief.strip())

    if intermediate_steps:
        h2("Reasoning Trail")
        for i, step in enumerate(intermediate_steps, 1):
            actor, call, obs = _coerce_step(step)
            tag = f"{i}.{(' ' + actor) if actor else ''}  {call}"
            pdf.set_font("Helvetica", "B", 9)
            pdf.multi_cell(
                0, 5, _safe_for_latin1(tag), new_x=XPos.LMARGIN, new_y=YPos.NEXT
            )
            mono(obs[:1000] + ("..." if len(obs) > 1000 else ""))

    sources = _extract_sources(answer, intermediate_steps)
    if sources:
        h2("Sources")
        _pdf_write_sources_html(pdf, sources)

    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(
        0,
        4,
        _safe_for_latin1(
            "Disclaimer: Market Sentinel is a research prototype. Not investment advice."
        ),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )

    raw = pdf.output()  # fpdf2 >=2.2 returns a ``bytearray``.
    if isinstance(raw, str):  # very old fpdf compat
        return raw.encode("latin-1")
    return bytes(raw)
