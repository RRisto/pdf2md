"""Tests for data models."""

from pathlib import Path

from pdf2md.models import (
    ConversionConfig,
    DocumentMetadata,
    DocumentResult,
    ImageMode,
    OutputFormat,
    PageBlockMetadata,
    PageStat,
    Section,
    TableFormat,
    TocEntry,
    TokenCounting,
)


# --- Enums ---


def test_image_mode_from_string():
    assert ImageMode("files") == ImageMode.FILES
    assert ImageMode("base64") == ImageMode.BASE64
    assert ImageMode("none") == ImageMode.NONE


def test_output_format_from_string():
    assert OutputFormat("md") == OutputFormat.MD
    assert OutputFormat("sections-dir") == OutputFormat.SECTIONS_DIR
    assert OutputFormat("json") == OutputFormat.JSON


def test_table_format_from_string():
    assert TableFormat("markdown") == TableFormat.MARKDOWN
    assert TableFormat("html") == TableFormat.HTML
    assert TableFormat("csv") == TableFormat.CSV


def test_token_counting_from_string():
    assert TokenCounting("off") == TokenCounting.OFF
    assert TokenCounting("estimate") == TokenCounting.ESTIMATE
    assert TokenCounting("tiktoken") == TokenCounting.TIKTOKEN


# --- Section ---


def test_section_defaults():
    s = Section(title="T", level=1, content="text")
    assert s.word_count == 0
    assert s.images == []
    assert s.section_hierarchy == []
    assert s.page_start is None
    assert s.page_end is None
    assert s.token_count is None


# --- DocumentMetadata ---


def test_document_metadata_defaults():
    meta = DocumentMetadata()
    assert meta.table_of_contents == []
    assert meta.page_stats == []
    assert meta.debug_data_path is None
    assert meta.total_pages == 0


def test_document_metadata_total_pages():
    meta = DocumentMetadata(page_stats=[
        PageStat(page_id=0),
        PageStat(page_id=1),
        PageStat(page_id=2),
    ])
    assert meta.total_pages == 3


def test_document_metadata_from_marker_dict():
    """Construct DocumentMetadata from a Marker-shaped dict."""
    raw = {
        "table_of_contents": [
            {"title": "Intro", "heading_level": 1, "page_id": 0, "polygon": [[0, 0]]},
        ],
        "page_stats": [
            {
                "page_id": 0,
                "text_extraction_method": "surya",
                "block_counts": {"Text": 3},
                "block_metadata": {
                    "llm_request_count": 1,
                    "llm_error_count": 0,
                    "llm_tokens_used": 100,
                },
            },
        ],
    }
    meta = DocumentMetadata.model_validate(raw)
    assert len(meta.table_of_contents) == 1
    assert meta.table_of_contents[0].title == "Intro"
    assert meta.total_pages == 1
    assert meta.page_stats[0].block_metadata.llm_tokens_used == 100


def test_document_metadata_extra_fields_ignored():
    """Extra keys from future Marker versions are silently ignored."""
    raw = {"unknown_field": "value", "page_stats": []}
    meta = DocumentMetadata.model_validate(raw)
    assert meta.page_stats == []


def test_document_metadata_serialization_roundtrip():
    meta = DocumentMetadata(
        table_of_contents=[TocEntry(title="Ch1", heading_level=1)],
        page_stats=[PageStat(page_id=0, text_extraction_method="surya")],
    )
    data = meta.model_dump()
    restored = DocumentMetadata.model_validate(data)
    assert restored.total_pages == 1
    assert restored.table_of_contents[0].title == "Ch1"


# --- DocumentResult ---


def test_document_result_construction():
    doc = DocumentResult(title="Doc", source_pdf="f.pdf", total_pages=5)
    assert doc.title == "Doc"
    assert doc.sections == []
    assert isinstance(doc.metadata, DocumentMetadata)


def test_document_result_serialization():
    section = Section(title="S", level=1, content="body", word_count=1)
    doc = DocumentResult(title="D", source_pdf="x.pdf", sections=[section])
    data = doc.model_dump()
    assert data["title"] == "D"
    assert len(data["sections"]) == 1
    assert data["sections"][0]["title"] == "S"


# --- ConversionConfig ---


def test_conversion_config_defaults():
    cfg = ConversionConfig(input_path=Path("in.pdf"), output_path=Path("out.md"))
    assert cfg.output_format == OutputFormat.MD
    assert cfg.image_mode == ImageMode.FILES
    assert cfg.split_level == 2
    assert cfg.paginate is False
    assert cfg.device == "auto"
    assert cfg.quiet is False
    assert cfg.dry_run is False
    assert cfg.token_counting == TokenCounting.OFF
    assert cfg.model_dir is None
    assert cfg.image_dir is None
    assert cfg.page_range is None


def test_conversion_config_table_format_default():
    cfg = ConversionConfig(input_path=Path("in.pdf"), output_path=Path("out.md"))
    assert cfg.table_format == TableFormat.MARKDOWN


def test_conversion_config_dry_run():
    cfg = ConversionConfig(
        input_path=Path("in.pdf"),
        output_path=Path("out.md"),
        dry_run=True,
    )
    assert cfg.dry_run is True


def test_conversion_config_model_dir():
    cfg = ConversionConfig(
        input_path=Path("in.pdf"),
        output_path=Path("out.md"),
        model_dir=Path("/tmp/models"),
    )
    assert cfg.model_dir == Path("/tmp/models")
