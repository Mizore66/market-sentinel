"""Shared pytest setup.

Loads ``.env`` at collection time so that ``pytest.mark.skipif`` decorators
on live-LLM tests see ``GROQ_API_KEY`` and friends without needing to
import ``agent.py`` first.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
