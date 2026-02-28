"""Core conversion pipeline — no CLI/Rich dependencies."""

from __future__ import annotations

from pathlib import Path

from .converter import MarkerBridge
from .image_handler import process_images
from .models import ConversionConfig, DocumentResult, Section
from .output_writer import write_output
from .section_splitter import split_markdown


def run_pipeline(config: ConversionConfig) -> DocumentResult:
    """Execute the full conversion pipeline and return the document result.

    This is the main programmatic entry point. It converts the PDF, processes
    images, splits into sections, builds the DocumentResult, and writes output.
    """
    bridge = MarkerBridge(model_dir=config.model_dir)

    # Step 1: Convert PDF
    result = bridge.convert(config)

    # Step 2: Process images
    markdown = process_images(
        markdown=result.markdown,
        images=result.images,
        mode=config.image_mode,
        image_dir=config.image_dir,
        output_path=config.output_path,
    )

    # Step 3: Split into sections
    sections = split_markdown(
        markdown,
        split_level=config.split_level,
        token_counting=config.token_counting,
    )

    # Step 4: Build document result
    title = _extract_title(sections, config.input_path)
    doc = DocumentResult(
        title=title,
        source_pdf=config.input_path.name,
        total_pages=result.metadata.total_pages or None,
        sections=sections,
        metadata=result.metadata,
    )

    # Step 5: Write output (skip in dry-run mode)
    if not config.dry_run:
        write_output(doc, config)

    return doc


def _extract_title(sections: list[Section], input_path: Path) -> str:
    """Extract a document title from sections or fall back to filename."""
    for section in sections:
        if section.level <= 1 and section.title != "Preamble":
            return section.title
    if sections and sections[0].title != "Preamble":
        return sections[0].title
    return input_path.stem
