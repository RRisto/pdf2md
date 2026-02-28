"""Tests for output_writer module."""

import json

import yaml

from pdf2md.models import ConversionConfig, DocumentResult, OutputFormat, Section
from pdf2md.output_writer import (
    _build_front_matter,
    _slugify,
    write_json,
    write_markdown,
    write_sections_dir,
)


def _make_config(tmp_path, fmt=OutputFormat.MD, filename="out.md"):
    return ConversionConfig(
        input_path=tmp_path / "input.pdf",
        output_path=tmp_path / filename,
        output_format=fmt,
    )


def _make_doc(sections=None):
    return DocumentResult(
        title="Test Doc",
        source_pdf="input.pdf",
        total_pages=3,
        sections=sections or [],
    )


# --- _slugify ---


def test_slugify_normal():
    assert _slugify("Hello World") == "hello-world"


def test_slugify_special_chars():
    assert _slugify("What's Up?!") == "whats-up"


def test_slugify_empty():
    assert _slugify("") == "untitled"


def test_slugify_long_title():
    long = "a" * 100
    result = _slugify(long)
    assert len(result) <= 60


def test_slugify_spaces_and_underscores():
    assert _slugify("foo_bar  baz") == "foo-bar-baz"


# --- write_markdown ---


def test_write_markdown_front_matter(tmp_path):
    section = Section(title="Intro", level=1, content="Hello.")
    config = _make_config(tmp_path)
    doc = _make_doc(sections=[section])

    write_markdown(doc, config)

    text = config.output_path.read_text()
    assert text.startswith("---\n")
    assert "title: Test Doc" in text
    assert "source: input.pdf" in text
    assert "pages: 3" in text


def test_write_markdown_heading_reconstruction(tmp_path):
    section = Section(title="Chapter", level=2, content="Body text.")
    config = _make_config(tmp_path)
    doc = _make_doc(sections=[section])

    write_markdown(doc, config)

    text = config.output_path.read_text()
    assert "## Chapter" in text
    assert "Body text." in text


def test_write_markdown_preamble_no_heading(tmp_path):
    section = Section(title="Preamble", level=0, content="Intro text.")
    config = _make_config(tmp_path)
    doc = _make_doc(sections=[section])

    write_markdown(doc, config)

    text = config.output_path.read_text()
    assert "# Preamble" not in text
    assert "Intro text." in text


# --- write_sections_dir ---


def test_write_sections_dir_creates_files(tmp_path):
    sections = [
        Section(title="Preamble", level=0, content="Intro."),
        Section(title="Intro", level=1, content="Body."),
    ]
    out_dir = tmp_path / "sections"
    config = _make_config(tmp_path, fmt=OutputFormat.SECTIONS_DIR, filename="sections")
    doc = _make_doc(sections=sections)

    write_sections_dir(doc, config)

    assert (out_dir / "00_preamble.md").exists()
    assert (out_dir / "01_intro.md").exists()
    assert (out_dir / "index.json").exists()


def test_write_sections_dir_manifest_structure(tmp_path):
    sections = [Section(title="Part One", level=1, content="Content.", word_count=1)]
    out_dir = tmp_path / "out"
    config = _make_config(tmp_path, fmt=OutputFormat.SECTIONS_DIR, filename="out")
    doc = _make_doc(sections=sections)

    write_sections_dir(doc, config)

    manifest = json.loads((out_dir / "index.json").read_text())
    assert manifest["title"] == "Test Doc"
    assert manifest["section_count"] == 1
    assert manifest["sections"][0]["title"] == "Part One"
    assert manifest["sections"][0]["filename"] == "00_part-one.md"


def test_write_sections_dir_manifest_token_count(tmp_path):
    sections = [Section(title="A", level=1, content="x.", token_count=42)]
    out_dir = tmp_path / "out"
    config = _make_config(tmp_path, fmt=OutputFormat.SECTIONS_DIR, filename="out")
    doc = _make_doc(sections=sections)

    write_sections_dir(doc, config)

    manifest = json.loads((out_dir / "index.json").read_text())
    assert manifest["sections"][0]["token_count"] == 42


def test_write_sections_dir_manifest_no_token_count_when_none(tmp_path):
    sections = [Section(title="A", level=1, content="x.")]
    out_dir = tmp_path / "out"
    config = _make_config(tmp_path, fmt=OutputFormat.SECTIONS_DIR, filename="out")
    doc = _make_doc(sections=sections)

    write_sections_dir(doc, config)

    manifest = json.loads((out_dir / "index.json").read_text())
    assert "token_count" not in manifest["sections"][0]


def test_write_sections_dir_numbered_filenames(tmp_path):
    sections = [
        Section(title="A", level=1, content="x."),
        Section(title="B", level=1, content="y."),
        Section(title="C", level=1, content="z."),
    ]
    out_dir = tmp_path / "out"
    config = _make_config(tmp_path, fmt=OutputFormat.SECTIONS_DIR, filename="out")
    doc = _make_doc(sections=sections)

    write_sections_dir(doc, config)

    files = sorted(f.name for f in out_dir.iterdir() if f.suffix == ".md")
    assert files == ["00_a.md", "01_b.md", "02_c.md"]


# --- write_json ---


def test_write_json_valid(tmp_path):
    sections = [Section(title="S", level=1, content="Text.")]
    config = _make_config(tmp_path, fmt=OutputFormat.JSON, filename="out.json")
    doc = _make_doc(sections=sections)

    write_json(doc, config)

    data = json.loads(config.output_path.read_text())
    assert data["title"] == "Test Doc"
    assert len(data["sections"]) == 1


def test_write_json_roundtrips(tmp_path):
    sections = [Section(title="S", level=1, content="Content.", word_count=1)]
    config = _make_config(tmp_path, fmt=OutputFormat.JSON, filename="out.json")
    doc = _make_doc(sections=sections)

    write_json(doc, config)

    loaded = DocumentResult.model_validate_json(config.output_path.read_text())
    assert loaded.title == doc.title
    assert loaded.sections[0].title == "S"


# --- _build_front_matter ---


def _parse_front_matter(text: str) -> dict:
    """Extract and parse YAML front-matter from a string."""
    assert text.startswith("---\n")
    end = text.index("---", 4)
    yaml_str = text[4:end]
    return yaml.safe_load(yaml_str)


def test_build_front_matter_basic():
    doc = _make_doc(sections=[Section(title="Intro", level=1, content="Hi.")])
    fm = _build_front_matter(doc)

    assert fm.startswith("---\n")
    assert fm.endswith("---\n\n")

    parsed = _parse_front_matter(fm)
    assert parsed["title"] == "Test Doc"
    assert parsed["source"] == "input.pdf"
    assert parsed["pages"] == 3
    assert parsed["sections"] == 1


def test_build_front_matter_quotes():
    doc = DocumentResult(
        title='He said "hello" and left',
        source_pdf="input.pdf",
        total_pages=1,
        sections=[],
    )
    fm = _build_front_matter(doc)
    parsed = _parse_front_matter(fm)
    assert parsed["title"] == 'He said "hello" and left'


def test_build_front_matter_colons():
    doc = DocumentResult(
        title="Section: Part 1: Overview",
        source_pdf="input.pdf",
        total_pages=1,
        sections=[],
    )
    fm = _build_front_matter(doc)
    parsed = _parse_front_matter(fm)
    assert parsed["title"] == "Section: Part 1: Overview"


def test_build_front_matter_backslashes():
    doc = DocumentResult(
        title="path\\to\\file",
        source_pdf="input.pdf",
        total_pages=1,
        sections=[],
    )
    fm = _build_front_matter(doc)
    parsed = _parse_front_matter(fm)
    assert parsed["title"] == "path\\to\\file"


def test_build_front_matter_unicode():
    doc = DocumentResult(
        title="Ubersicht der Ergebnisse",
        source_pdf="input.pdf",
        total_pages=1,
        sections=[],
    )
    fm = _build_front_matter(doc)
    parsed = _parse_front_matter(fm)
    assert parsed["title"] == "Ubersicht der Ergebnisse"
