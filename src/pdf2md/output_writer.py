"""Output writers for different format targets."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from .models import ConversionConfig, DocumentResult, OutputFormat, Section


def _build_front_matter(result: DocumentResult) -> str:
    """Build YAML front-matter with proper escaping."""
    data = {
        "title": result.title,
        "source": result.source_pdf,
        "pages": result.total_pages,
        "sections": len(result.sections),
    }
    yaml_str = yaml.safe_dump(
        data,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
    return "---\n" + yaml_str + "---\n\n"


def write_output(result: DocumentResult, config: ConversionConfig) -> None:
    """Write the converted document in the configured output format."""
    writers = {
        OutputFormat.MD: write_markdown,
        OutputFormat.SECTIONS_DIR: write_sections_dir,
        OutputFormat.JSON: write_json,
    }
    writers[config.output_format](result, config)


def write_markdown(result: DocumentResult, config: ConversionConfig) -> None:
    """Write a single markdown file with YAML front-matter."""
    front_matter = _build_front_matter(result)

    body_parts: list[str] = []
    for section in result.sections:
        if section.level > 0:
            heading = "#" * section.level + " " + section.title
            body_parts.append(heading + "\n\n" + section.content)
        else:
            body_parts.append(section.content)

    content = front_matter + "\n\n".join(body_parts) + "\n"
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(content, encoding="utf-8")


def write_sections_dir(result: DocumentResult, config: ConversionConfig) -> None:
    """Write each section as a separate file in a directory, plus index.json."""
    out_dir = config.output_path
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict] = []

    for i, section in enumerate(result.sections):
        slug = _slugify(section.title)
        filename = f"{i:02d}_{slug}.md"
        filepath = out_dir / filename

        if section.level > 0:
            heading = "#" * section.level + " " + section.title
            content = heading + "\n\n" + section.content + "\n"
        else:
            content = section.content + "\n"

        filepath.write_text(content, encoding="utf-8")

        entry = {
            "index": i,
            "filename": filename,
            "title": section.title,
            "level": section.level,
            "word_count": section.word_count,
            "page_start": section.page_start,
            "page_end": section.page_end,
        }
        if section.token_count is not None:
            entry["token_count"] = section.token_count
        manifest_entries.append(entry)

    index = {
        "title": result.title,
        "source_pdf": result.source_pdf,
        "total_pages": result.total_pages,
        "section_count": len(result.sections),
        "sections": manifest_entries,
    }

    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


def write_json(result: DocumentResult, config: ConversionConfig) -> None:
    """Write the full DocumentResult as JSON."""
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def _slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text[:60].strip("-") or "untitled"
