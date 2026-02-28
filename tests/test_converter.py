"""Tests for pdf2md.converter — page-range validation."""

import pytest

from pdf2md.converter import validate_page_range
from pdf2md.errors import InvalidPageRangeError


# --- Valid page ranges (should not raise) ---


@pytest.mark.parametrize(
    "page_range",
    [
        "5",              # single page
        "1,5,10",         # multiple pages
        "3-7",            # range
        "1,3-5,10",       # mixed pages and ranges
        "1, 3-5 , 10",   # with whitespace
    ],
    ids=[
        "single_page",
        "multiple_pages",
        "range",
        "mixed",
        "with_whitespace",
    ],
)
def test_valid_page_ranges(page_range: str) -> None:
    """Valid page-range strings must be accepted without error."""
    validate_page_range(page_range)  # should not raise


# --- Invalid page ranges (should raise InvalidPageRangeError) ---


def test_empty_string() -> None:
    with pytest.raises(InvalidPageRangeError, match="must not be empty"):
        validate_page_range("")


def test_non_numeric() -> None:
    with pytest.raises(InvalidPageRangeError, match="Invalid segment"):
        validate_page_range("abc")


def test_trailing_comma() -> None:
    with pytest.raises(InvalidPageRangeError, match="Empty segment"):
        validate_page_range("1,2,")


def test_double_comma() -> None:
    with pytest.raises(InvalidPageRangeError, match="Empty segment"):
        validate_page_range("1,,3")


def test_reversed_range() -> None:
    with pytest.raises(InvalidPageRangeError, match="start.*greater than end"):
        validate_page_range("7-3")


def test_float_value() -> None:
    with pytest.raises(InvalidPageRangeError, match="Invalid segment"):
        validate_page_range("1.5")


def test_triple_segment() -> None:
    with pytest.raises(InvalidPageRangeError, match="Invalid segment"):
        validate_page_range("1-2-3")


# ---------------------------------------------------------------------------
# Mock-based tests for MarkerBridge
# ---------------------------------------------------------------------------

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from pdf2md.converter import (
    ConversionResult,
    MarkerBridge,
    download_models,
    get_default_model_dir,
)
from pdf2md.errors import ConversionError, EmptyOutputError
from pdf2md.models import ConversionConfig, DocumentMetadata, TableFormat


# --- Helpers / fixtures ---


@pytest.fixture()
def dummy_pdf(tmp_path: Path) -> Path:
    """Create a minimal dummy PDF file so path-exists checks pass."""
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    return pdf


def _make_config(pdf_path: Path, **overrides) -> ConversionConfig:
    """Build a ConversionConfig pointing at *pdf_path* with sensible defaults."""
    defaults = dict(
        input_path=pdf_path,
        output_path=pdf_path.parent / "out.md",
    )
    defaults.update(overrides)
    return ConversionConfig(**defaults)


@pytest.fixture()
def marker_mocks():
    """Patch all late-imported Marker modules and return the mock objects.

    Yields a dict with keys:
        create_model_dict, ConfigParser, PdfConverter, text_from_rendered
    """
    fake_artifact_dict = {"model_a": MagicMock(), "model_b": MagicMock()}

    # ConfigParser mock
    mock_config_parser_cls = MagicMock()
    mock_config_parser_inst = MagicMock()
    mock_config_parser_cls.return_value = mock_config_parser_inst
    mock_config_parser_inst.generate_config_dict.return_value = {"some": "config"}
    mock_config_parser_inst.get_processors.return_value = ["proc"]
    mock_config_parser_inst.get_renderer.return_value = "renderer"

    # PdfConverter mock — calling the instance returns a rendered object
    mock_converter_cls = MagicMock()
    mock_converter_inst = MagicMock()
    mock_converter_cls.return_value = mock_converter_inst
    rendered = MagicMock()
    rendered.metadata = {"table_of_contents": [], "page_stats": []}
    mock_converter_inst.return_value = rendered

    # text_from_rendered mock
    mock_text_from_rendered = MagicMock(
        return_value=("# Hello\n\nConverted content.", {}, {"img.png": MagicMock()})
    )

    with (
        patch("marker.models.create_model_dict", return_value=fake_artifact_dict) as m_cmd,
        patch("marker.config.parser.ConfigParser", mock_config_parser_cls) as m_cp,
        patch("marker.converters.pdf.PdfConverter", mock_converter_cls) as m_pdfconv,
        patch("marker.output.text_from_rendered", mock_text_from_rendered) as m_tfr,
    ):
        yield {
            "create_model_dict": m_cmd,
            "ConfigParser": m_cp,
            "config_parser_instance": mock_config_parser_inst,
            "PdfConverter": m_pdfconv,
            "converter_instance": mock_converter_inst,
            "text_from_rendered": m_tfr,
            "rendered": rendered,
            "artifact_dict": fake_artifact_dict,
        }


# ---- TestConfigWiring ----


class TestConfigWiring:
    """MarkerBridge.convert() passes the right config to Marker."""

    def test_default_config(self, dummy_pdf, marker_mocks):
        """Default config: no page_range, device=auto, paginate=False."""
        cfg = _make_config(dummy_pdf)
        bridge = MarkerBridge()
        bridge.convert(cfg)

        # ConfigParser should receive output_format and paginate_output only
        marker_mocks["ConfigParser"].assert_called_once()
        passed_config = marker_mocks["ConfigParser"].call_args[0][0]
        assert passed_config["output_format"] == "markdown"
        assert passed_config["paginate_output"] is False
        assert "page_range" not in passed_config
        assert "device" not in passed_config

    def test_with_page_range(self, dummy_pdf, marker_mocks):
        """page_range is forwarded to marker config."""
        cfg = _make_config(dummy_pdf, page_range="1-3")
        bridge = MarkerBridge()
        bridge.convert(cfg)

        passed_config = marker_mocks["ConfigParser"].call_args[0][0]
        assert passed_config["page_range"] == "1-3"

    def test_device_cpu(self, dummy_pdf, marker_mocks):
        """Explicit device='cpu' is forwarded."""
        cfg = _make_config(dummy_pdf, device="cpu")
        bridge = MarkerBridge()
        bridge.convert(cfg)

        passed_config = marker_mocks["ConfigParser"].call_args[0][0]
        assert passed_config["device"] == "cpu"

    def test_device_auto_not_forwarded(self, dummy_pdf, marker_mocks):
        """device='auto' (default) should NOT appear in marker config."""
        cfg = _make_config(dummy_pdf, device="auto")
        bridge = MarkerBridge()
        bridge.convert(cfg)

        passed_config = marker_mocks["ConfigParser"].call_args[0][0]
        assert "device" not in passed_config

    def test_paginate_true(self, dummy_pdf, marker_mocks):
        """paginate=True is forwarded."""
        cfg = _make_config(dummy_pdf, paginate=True)
        bridge = MarkerBridge()
        bridge.convert(cfg)

        passed_config = marker_mocks["ConfigParser"].call_args[0][0]
        assert passed_config["paginate_output"] is True

    def test_html_table_format(self, dummy_pdf, marker_mocks):
        """HTML table format sets html_tables_in_markdown=True."""
        cfg = _make_config(dummy_pdf, table_format=TableFormat.HTML)
        bridge = MarkerBridge()
        bridge.convert(cfg)

        passed_config = marker_mocks["ConfigParser"].call_args[0][0]
        assert passed_config["html_tables_in_markdown"] is True

    def test_csv_table_format_sets_html_tables_flag(self, dummy_pdf, marker_mocks):
        """CSV table format also sets html_tables_in_markdown=True (HTML is extracted first,
        then post-processed to CSV)."""
        # Make text_from_rendered return an HTML table so CSV conversion can work
        marker_mocks["text_from_rendered"].return_value = (
            "<table><tr><td>a</td><td>b</td></tr></table>",
            {},
            {},
        )
        cfg = _make_config(dummy_pdf, table_format=TableFormat.CSV)
        bridge = MarkerBridge()
        result = bridge.convert(cfg)

        passed_config = marker_mocks["ConfigParser"].call_args[0][0]
        assert passed_config["html_tables_in_markdown"] is True
        # Output should be CSV, not HTML
        assert "```csv" in result.markdown

    def test_markdown_table_format_no_html_flag(self, dummy_pdf, marker_mocks):
        """Default markdown table format should NOT set html_tables_in_markdown."""
        cfg = _make_config(dummy_pdf, table_format=TableFormat.MARKDOWN)
        bridge = MarkerBridge()
        bridge.convert(cfg)

        passed_config = marker_mocks["ConfigParser"].call_args[0][0]
        assert "html_tables_in_markdown" not in passed_config

    def test_converter_receives_config_parser_outputs(self, dummy_pdf, marker_mocks):
        """PdfConverter is constructed with outputs from ConfigParser."""
        cfg = _make_config(dummy_pdf)
        bridge = MarkerBridge()
        bridge.convert(cfg)

        marker_mocks["PdfConverter"].assert_called_once_with(
            config=marker_mocks["config_parser_instance"].generate_config_dict(),
            artifact_dict=marker_mocks["artifact_dict"],
            processor_list=marker_mocks["config_parser_instance"].get_processors(),
            renderer=marker_mocks["config_parser_instance"].get_renderer(),
        )


# ---- TestErrorHandling ----


class TestErrorHandling:
    """Error paths in MarkerBridge.convert()."""

    def test_file_not_found_raises_conversion_error(self, tmp_path, marker_mocks):
        """Non-existent input file raises ConversionError."""
        cfg = _make_config(tmp_path / "no_such_file.pdf")
        bridge = MarkerBridge()
        with pytest.raises(ConversionError, match="Input file not found"):
            bridge.convert(cfg)

    def test_marker_exception_raises_conversion_error(self, dummy_pdf, marker_mocks):
        """Exception during Marker conversion is wrapped in ConversionError."""
        marker_mocks["converter_instance"].side_effect = RuntimeError("boom")
        cfg = _make_config(dummy_pdf)
        bridge = MarkerBridge()
        with pytest.raises(ConversionError, match="Marker conversion failed.*boom"):
            bridge.convert(cfg)

    def test_empty_output_raises_empty_output_error(self, dummy_pdf, marker_mocks):
        """Empty string from Marker raises EmptyOutputError."""
        marker_mocks["text_from_rendered"].return_value = ("", {}, {})
        cfg = _make_config(dummy_pdf)
        bridge = MarkerBridge()
        with pytest.raises(EmptyOutputError, match="no content"):
            bridge.convert(cfg)

    def test_whitespace_only_output_raises_empty_output_error(
        self, dummy_pdf, marker_mocks
    ):
        """Whitespace-only text from Marker raises EmptyOutputError."""
        marker_mocks["text_from_rendered"].return_value = ("   \n\t  ", {}, {})
        cfg = _make_config(dummy_pdf)
        bridge = MarkerBridge()
        with pytest.raises(EmptyOutputError, match="no content"):
            bridge.convert(cfg)

    def test_metadata_failure_issues_warning(self, dummy_pdf, marker_mocks):
        """Bad metadata triggers a warning but does not raise."""
        # Make dict() on rendered.metadata raise
        rendered = marker_mocks["rendered"]
        rendered.metadata = MagicMock()
        # dict(rendered.metadata) calls __iter__ then builds pairs;
        # make it return something that model_validate will reject
        type(rendered).metadata = property(lambda self: _BadMeta())

        cfg = _make_config(dummy_pdf)
        bridge = MarkerBridge()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = bridge.convert(cfg)
            metadata_warnings = [
                x for x in w if "metadata" in str(x.message).lower()
            ]
            assert len(metadata_warnings) == 1
        # Should still return a result with default metadata
        assert result.metadata == DocumentMetadata()

    def test_invalid_page_range_raises_before_model_load(self, dummy_pdf, marker_mocks):
        """Invalid page_range raises InvalidPageRangeError *before* _ensure_models."""
        cfg = _make_config(dummy_pdf, page_range="bad-range!!")
        bridge = MarkerBridge()
        with pytest.raises(InvalidPageRangeError):
            bridge.convert(cfg)
        # Models should never have been loaded
        marker_mocks["create_model_dict"].assert_not_called()


class _BadMeta:
    """Helper that makes dict() succeed but produces data that
    DocumentMetadata.model_validate() will reject."""

    def __iter__(self):
        # Return key-value pairs with wrong types to trigger validation failure
        return iter([("table_of_contents", "NOT_A_LIST_OF_TOC_ENTRIES!!!")])

    def __bool__(self):
        return True


# ---- TestModelCaching ----


class TestModelCaching:
    """Lazy model loading and caching behaviour."""

    def test_models_loaded_once_across_multiple_converts(
        self, dummy_pdf, marker_mocks
    ):
        """create_model_dict is called only once even after multiple convert() calls."""
        bridge = MarkerBridge()
        cfg = _make_config(dummy_pdf)
        bridge.convert(cfg)
        bridge.convert(cfg)
        bridge.convert(cfg)
        marker_mocks["create_model_dict"].assert_called_once()

    def test_different_instances_load_independently(self, dummy_pdf, marker_mocks):
        """Each MarkerBridge instance loads models separately."""
        cfg = _make_config(dummy_pdf)
        MarkerBridge().convert(cfg)
        MarkerBridge().convert(cfg)
        assert marker_mocks["create_model_dict"].call_count == 2


# ---- TestReturnValue ----


class TestReturnValue:
    """ConversionResult returned by convert()."""

    def test_return_value_fields(self, dummy_pdf, marker_mocks):
        """ConversionResult has correct markdown, images, and metadata."""
        cfg = _make_config(dummy_pdf)
        bridge = MarkerBridge()
        result = bridge.convert(cfg)

        assert isinstance(result, ConversionResult)
        assert result.markdown == "# Hello\n\nConverted content."
        assert "img.png" in result.images
        assert isinstance(result.metadata, DocumentMetadata)

    def test_images_default_to_empty_dict(self, dummy_pdf, marker_mocks):
        """When text_from_rendered returns None for images, result.images is {}."""
        marker_mocks["text_from_rendered"].return_value = (
            "some text",
            {},
            None,
        )
        cfg = _make_config(dummy_pdf)
        bridge = MarkerBridge()
        result = bridge.convert(cfg)
        assert result.images == {}


# ---- TestModelDir ----


class TestModelDir:
    """Model directory configuration."""

    def test_custom_model_dir_sets_cache(self, tmp_path, marker_mocks):
        """Custom model_dir is created and used for model caching."""
        model_dir = tmp_path / "custom_models"
        bridge = MarkerBridge(model_dir=model_dir)

        # Need a dummy PDF for convert
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        cfg = _make_config(pdf)

        with patch("pdf2md.converter._set_model_cache_dir") as mock_set_cache:
            # Re-enter the marker mocks context since we need to also patch _set_model_cache_dir
            # But marker_mocks is already active. We just call convert.
            bridge.convert(cfg)

        # The model_dir should have been created
        assert model_dir.exists()

    def test_get_default_model_dir_returns_path(self):
        """get_default_model_dir() returns a Path instance."""
        result = get_default_model_dir()
        assert isinstance(result, Path)
        assert "models" in str(result)


# ---- TestDownloadModels ----


class TestDownloadModels:
    """download_models() function."""

    def test_download_models_calls_create_model_dict(self, marker_mocks):
        """download_models() delegates to create_model_dict."""
        download_models()
        marker_mocks["create_model_dict"].assert_called_once()

    def test_download_models_custom_dir(self, tmp_path, marker_mocks):
        """download_models(custom_dir) creates the dir and sets the cache."""
        custom_dir = tmp_path / "my_models"
        with patch("pdf2md.converter._set_model_cache_dir") as mock_set_cache:
            download_models(custom_dir)
        assert custom_dir.exists()
        mock_set_cache.assert_called_once_with(custom_dir)
        marker_mocks["create_model_dict"].assert_called()
