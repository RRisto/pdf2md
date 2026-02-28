"""Split markdown content into sections by headings."""

from __future__ import annotations

import re

from .models import Section, TokenCounting

# Matches ATX headings: # Title, ## Title, etc.
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Matches page markers injected by Marker: {page_number}
PAGE_MARKER_RE = re.compile(r"\{(\d+)\}")

# Matches markdown image references
IMAGE_REF_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def parse_page_markers(text: str) -> int | None:
    """Extract the last page number from page markers in text."""
    matches = PAGE_MARKER_RE.findall(text)
    if matches:
        return int(matches[-1])
    return None


def parse_first_page_marker(text: str) -> int | None:
    """Extract the first page number from page markers in text."""
    match = PAGE_MARKER_RE.search(text)
    if match:
        return int(match.group(1))
    return None


def extract_image_refs(text: str) -> list[str]:
    """Extract image paths/names from markdown image references."""
    return [m[1] for m in IMAGE_REF_RE.findall(text)]


def compute_token_count(text: str, mode: TokenCounting) -> int | None:
    """Estimate token count for a text string.

    Args:
        text: The text to count tokens for.
        mode: Counting strategy — ``off`` returns None, ``estimate`` uses a
            word-count heuristic, ``tiktoken`` uses the cl100k_base encoding.
    """
    if mode == TokenCounting.OFF:
        return None
    if mode == TokenCounting.ESTIMATE:
        return int(len(text.split()) * 1.3)
    if mode == TokenCounting.TIKTOKEN:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    return None


def split_markdown(
    markdown: str,
    split_level: int = 2,
    token_counting: TokenCounting = TokenCounting.OFF,
) -> list[Section]:
    """Split markdown into sections based on heading level.

    Headings at or above `split_level` start a new section.
    Content before the first heading becomes a "Preamble" section.
    """
    if not markdown or not markdown.strip():
        return []

    sections: list[Section] = []
    hierarchy: list[str] = []

    # Find all heading positions
    heading_matches = list(HEADING_RE.finditer(markdown))

    if not heading_matches:
        # No headings: single section with all content
        sections.append(_make_section(
            title="Preamble",
            level=0,
            content=markdown.strip(),
            hierarchy=[],
            token_counting=token_counting,
        ))
        return sections

    # Content before first heading
    first_heading_pos = heading_matches[0].start()
    preamble = markdown[:first_heading_pos].strip()
    if preamble:
        sections.append(_make_section(
            title="Preamble",
            level=0,
            content=preamble,
            hierarchy=[],
            token_counting=token_counting,
        ))

    for i, match in enumerate(heading_matches):
        level = len(match.group(1))
        title = match.group(2).strip()

        # Determine content: from end of this heading line to start of next heading
        content_start = match.end()
        if i + 1 < len(heading_matches):
            content_end = heading_matches[i + 1].start()
        else:
            content_end = len(markdown)

        content = markdown[content_start:content_end].strip()

        if level <= split_level:
            # This is a split point — update hierarchy
            # Trim hierarchy to current level
            hierarchy = hierarchy[:level - 1]
            hierarchy.append(title)

            sections.append(_make_section(
                title=title,
                level=level,
                content=content,
                hierarchy=list(hierarchy),
                token_counting=token_counting,
            ))
        else:
            # Sub-heading below split level: append to current section
            if sections:
                heading_line = match.group(0)
                sections[-1].content = (
                    sections[-1].content + "\n\n" + heading_line + "\n\n" + content
                ).strip()
            else:
                # No parent section yet, create one
                heading_line = match.group(0)
                sections.append(_make_section(
                    title=title,
                    level=level,
                    content=heading_line + "\n\n" + content,
                    hierarchy=[title],
                    token_counting=token_counting,
                ))

    # Recalculate computed fields after sub-heading merges
    for section in sections:
        section.word_count = len(section.content.split())
        section.images = extract_image_refs(section.content)
        section.page_start = parse_first_page_marker(section.content)
        section.page_end = parse_page_markers(section.content)
        section.token_count = compute_token_count(section.content, token_counting)

    return sections


def _make_section(
    title: str,
    level: int,
    content: str,
    hierarchy: list[str],
    token_counting: TokenCounting = TokenCounting.OFF,
) -> Section:
    """Create a Section with computed fields."""
    return Section(
        title=title,
        level=level,
        content=content,
        page_start=parse_first_page_marker(content),
        page_end=parse_page_markers(content),
        section_hierarchy=hierarchy,
        images=extract_image_refs(content),
        word_count=len(content.split()),
        token_count=compute_token_count(content, token_counting),
    )
