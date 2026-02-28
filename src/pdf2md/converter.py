"""Bridge to Marker's PDF conversion pipeline."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .errors import ConversionError, EmptyOutputError, InvalidPageRangeError
from .models import ConversionConfig, DocumentMetadata, TableFormat


def _inject_system_certs() -> None:
    """Use OS trust store for SSL so corporate proxies work."""
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass


_inject_system_certs()


_SEGMENT_RE = re.compile(r"^\d+(-\d+)?$")


def validate_page_range(page_range: str) -> None:
    """Validate a page-range string such as ``"1,3-5,10"``.

    Raises :class:`InvalidPageRangeError` for any malformed input.
    """
    if not page_range or not page_range.strip():
        raise InvalidPageRangeError("Page range must not be empty")

    segments = page_range.split(",")
    for i, raw_segment in enumerate(segments):
        segment = raw_segment.strip()
        if not segment:
            raise InvalidPageRangeError(
                f"Empty segment at position {i + 1} in page range: {page_range!r}"
            )
        if not _SEGMENT_RE.match(segment):
            raise InvalidPageRangeError(
                f"Invalid segment {segment!r} in page range: {page_range!r}"
            )
        if "-" in segment:
            start_s, end_s = segment.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start > end:
                raise InvalidPageRangeError(
                    f"Range start ({start}) is greater than end ({end}) "
                    f"in segment {segment!r}"
                )


@dataclass
class ConversionResult:
    """Raw output from Marker conversion."""

    markdown: str
    images: dict[str, Image.Image] = field(default_factory=dict)
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)


def get_default_model_dir() -> Path:
    """Return the platform-appropriate default model cache directory."""
    from platformdirs import user_cache_dir

    return Path(user_cache_dir("datalab")) / "models"


def _set_model_cache_dir(model_dir: Path) -> None:
    """Override Surya's model cache directory."""
    import surya.settings

    surya.settings.settings.MODEL_CACHE_DIR = str(model_dir)


def download_models(model_dir: Path | None = None) -> None:
    """Download Marker's ML models (without converting any PDF).

    Args:
        model_dir: Custom directory for model storage. If *None*, uses the
            default Surya cache location.
    """
    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)
        _set_model_cache_dir(model_dir)

    from marker.models import create_model_dict

    create_model_dict()


class MarkerBridge:
    """Wraps Marker's PdfConverter for single-call PDF -> markdown conversion."""

    def __init__(self, model_dir: Path | None = None) -> None:
        self._artifact_dict: dict[str, Any] | None = None
        self._model_dir = model_dir

    def _ensure_models(self) -> dict[str, Any]:
        """Lazily load Marker's ML models."""
        if self._artifact_dict is None:
            if self._model_dir is not None:
                self._model_dir.mkdir(parents=True, exist_ok=True)
                _set_model_cache_dir(self._model_dir)

            from marker.models import create_model_dict

            self._artifact_dict = create_model_dict()
        return self._artifact_dict

    def convert(self, config: ConversionConfig) -> ConversionResult:
        """Convert a PDF file to markdown using Marker."""
        input_path = config.input_path
        if not input_path.exists():
            raise ConversionError(f"Input file not found: {input_path}")

        if config.page_range is not None:
            validate_page_range(config.page_range)

        artifact_dict = self._ensure_models()

        from marker.config.parser import ConfigParser
        from marker.converters.pdf import PdfConverter
        from marker.output import text_from_rendered

        marker_config: dict[str, Any] = {
            "output_format": "markdown",
            "paginate_output": config.paginate,
        }
        if config.page_range:
            marker_config["page_range"] = config.page_range
        if config.device and config.device != "auto":
            marker_config["device"] = config.device

        if config.table_format in (TableFormat.HTML, TableFormat.CSV):
            marker_config["html_tables_in_markdown"] = True

        config_parser = ConfigParser(marker_config)

        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=artifact_dict,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
        )

        try:
            rendered = converter(str(input_path))
        except Exception as e:
            raise ConversionError(f"Marker conversion failed: {e}") from e

        text, _, images = text_from_rendered(rendered)

        if config.table_format == TableFormat.CSV:
            text = _convert_html_tables_to_csv(text)

        if not text or not text.strip():
            raise EmptyOutputError(f"Conversion produced no content for: {input_path}")

        metadata = DocumentMetadata()
        if hasattr(rendered, "metadata") and rendered.metadata:
            try:
                raw = dict(rendered.metadata)
                metadata = DocumentMetadata.model_validate(raw)
            except Exception:
                warnings.warn(
                    "Could not convert document metadata; metadata will use defaults",
                    stacklevel=2,
                )

        return ConversionResult(
            markdown=text,
            images=images or {},
            metadata=metadata,
        )


def _convert_html_tables_to_csv(text: str) -> str:
    """Replace HTML <table> blocks with fenced CSV code blocks."""
    import re as _re

    pattern = _re.compile(r"<table\b[^>]*>.*?</table>", _re.DOTALL)
    return pattern.sub(lambda m: _html_table_to_csv_block(m.group(0)), text)


def _html_table_to_csv_block(html: str) -> str:
    """Convert a single HTML table to a fenced CSV block."""
    import csv
    import io
    from html.parser import HTMLParser

    class _TableParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows: list[list[str]] = []
            self._current_row: list[str] | None = None
            self._current_cell: list[str] | None = None

        def handle_starttag(self, tag, attrs):
            if tag == "tr":
                self._current_row = []
            elif tag in ("td", "th"):
                self._current_cell = []

        def handle_endtag(self, tag):
            if tag in ("td", "th") and self._current_cell is not None:
                if self._current_row is not None:
                    self._current_row.append("".join(self._current_cell).strip())
                self._current_cell = None
            elif tag == "tr" and self._current_row is not None:
                self.rows.append(self._current_row)
                self._current_row = None

        def handle_data(self, data):
            if self._current_cell is not None:
                self._current_cell.append(data)

    parser = _TableParser()
    parser.feed(html)

    if not parser.rows:
        return html

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for row in parser.rows:
        writer.writerow(row)

    return "```csv\n" + buf.getvalue() + "```"
