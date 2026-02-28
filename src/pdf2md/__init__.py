"""pdf2md - Convert PDF files to structured Markdown for LLM consumption."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import ConversionConfig, DocumentResult, ImageMode, OutputFormat, Section
from .pipeline import run_pipeline

__version__ = "0.1.0"

__all__ = [
    "convert_pdf",
    "run_pipeline",
    "ConversionConfig",
    "DocumentResult",
    "Section",
    "ImageMode",
    "OutputFormat",
]


def convert_pdf(
    input_path: str | Path,
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> DocumentResult:
    """Convert a PDF to structured Markdown.

    This is the main convenience function for programmatic use.

    Args:
        input_path: Path to the input PDF file.
        output_path: Output path. If *None*, defaults to ``<input_stem>.md``
            next to the input file.
        **kwargs: Additional options forwarded to :class:`ConversionConfig`
            (e.g. ``image_mode``, ``split_level``, ``page_range``).

    Returns:
        A :class:`DocumentResult` with title, sections, and metadata.
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.with_suffix(".md")
    else:
        output_path = Path(output_path)

    config = ConversionConfig(
        input_path=input_path,
        output_path=output_path,
        **kwargs,
    )
    return run_pipeline(config)
