"""Pydantic data models for pdf2md."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class ImageMode(str, Enum):
    FILES = "files"
    BASE64 = "base64"
    NONE = "none"


class OutputFormat(str, Enum):
    MD = "md"
    SECTIONS_DIR = "sections-dir"
    JSON = "json"


class TableFormat(str, Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    CSV = "csv"


class TokenCounting(str, Enum):
    OFF = "off"
    ESTIMATE = "estimate"
    TIKTOKEN = "tiktoken"


# --- Typed metadata models (matching Marker's output) ---


class TocEntry(BaseModel):
    title: str = ""
    heading_level: int = 0
    page_id: int = 0
    polygon: list[list[float]] = Field(default_factory=list)


class PageBlockMetadata(BaseModel):
    llm_request_count: int = 0
    llm_error_count: int = 0
    llm_tokens_used: int = 0


class PageStat(BaseModel):
    page_id: int = 0
    text_extraction_method: str = ""
    block_counts: dict[str, int] = Field(default_factory=dict)
    block_metadata: PageBlockMetadata = Field(default_factory=PageBlockMetadata)

    model_config = {"extra": "ignore"}


class DocumentMetadata(BaseModel):
    table_of_contents: list[TocEntry] = Field(default_factory=list)
    page_stats: list[PageStat] = Field(default_factory=list)
    debug_data_path: str | None = None

    model_config = {"extra": "ignore"}

    @property
    def total_pages(self) -> int:
        return len(self.page_stats)


# --- Core domain models ---


class Section(BaseModel):
    title: str
    level: int
    content: str
    page_start: int | None = None
    page_end: int | None = None
    section_hierarchy: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    word_count: int = 0
    token_count: int | None = None


class DocumentResult(BaseModel):
    title: str
    source_pdf: str
    total_pages: int | None = None
    sections: list[Section] = Field(default_factory=list)
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)


class ConversionConfig(BaseModel):
    input_path: Path
    output_path: Path
    output_format: OutputFormat = OutputFormat.MD
    image_mode: ImageMode = ImageMode.FILES
    image_dir: Path | None = None
    page_range: str | None = None
    split_level: int = 2
    table_format: TableFormat = TableFormat.MARKDOWN
    paginate: bool = False
    device: str = "auto"
    quiet: bool = False
    dry_run: bool = False
    token_counting: TokenCounting = TokenCounting.OFF
    model_dir: Path | None = None
