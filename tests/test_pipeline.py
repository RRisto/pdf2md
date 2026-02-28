"""Tests for pipeline module."""

from pathlib import Path

from pdf2md.models import Section
from pdf2md.pipeline import _extract_title


# --- _extract_title ---


def test_extract_title_level1_wins():
    sections = [
        Section(title="Preamble", level=0, content="x"),
        Section(title="Main Title", level=1, content="y"),
        Section(title="Subtitle", level=2, content="z"),
    ]
    assert _extract_title(sections, Path("f.pdf")) == "Main Title"


def test_extract_title_skips_preamble():
    sections = [
        Section(title="Preamble", level=0, content="x"),
        Section(title="Real Title", level=1, content="y"),
    ]
    assert _extract_title(sections, Path("f.pdf")) == "Real Title"


def test_extract_title_fallback_to_filename():
    sections = [Section(title="Preamble", level=0, content="x")]
    assert _extract_title(sections, Path("my_doc.pdf")) == "my_doc"


def test_extract_title_empty_sections():
    assert _extract_title([], Path("fallback.pdf")) == "fallback"


def test_extract_title_non_preamble_first():
    """First section that isn't 'Preamble' is used even if level > 1."""
    sections = [Section(title="Overview", level=2, content="x")]
    assert _extract_title(sections, Path("f.pdf")) == "Overview"
