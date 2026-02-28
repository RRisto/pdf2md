"""Tests for section_splitter module."""

from pdf2md.models import TokenCounting
from pdf2md.section_splitter import (
    compute_token_count,
    extract_image_refs,
    parse_first_page_marker,
    parse_page_markers,
    split_markdown,
)


# --- parse_page_markers ---


def test_parse_page_markers_no_markers():
    assert parse_page_markers("Hello world") is None


def test_parse_page_markers_single():
    assert parse_page_markers("Some text {5} here") == 5


def test_parse_page_markers_multiple_returns_last():
    assert parse_page_markers("{1} text {3} more {7}") == 7


# --- extract_image_refs ---


def test_extract_image_refs_standard():
    text = "Look at ![photo](img.png) and ![chart](chart.jpg)"
    assert extract_image_refs(text) == ["img.png", "chart.jpg"]


def test_extract_image_refs_empty_alt():
    assert extract_image_refs("![](path.png)") == ["path.png"]


def test_extract_image_refs_no_refs():
    assert extract_image_refs("No images here") == []


# --- split_markdown ---


def test_split_empty_string():
    assert split_markdown("") == []


def test_split_whitespace_only():
    assert split_markdown("   \n\n  ") == []


def test_split_no_headings():
    sections = split_markdown("Just a paragraph.\n\nAnother one.")
    assert len(sections) == 1
    assert sections[0].title == "Preamble"
    assert sections[0].level == 0
    assert "Just a paragraph" in sections[0].content


def test_split_single_heading():
    md = "# Introduction\n\nSome content here."
    sections = split_markdown(md, split_level=2)
    assert len(sections) == 1
    assert sections[0].title == "Introduction"
    assert sections[0].level == 1
    assert sections[0].content == "Some content here."


def test_split_preamble_before_heading():
    md = "Preamble text.\n\n# Heading\n\nBody."
    sections = split_markdown(md, split_level=2)
    assert len(sections) == 2
    assert sections[0].title == "Preamble"
    assert sections[1].title == "Heading"


def test_split_multiple_headings():
    md = "# First\n\nContent 1.\n\n## Second\n\nContent 2.\n\n# Third\n\nContent 3."
    sections = split_markdown(md, split_level=2)
    assert len(sections) == 3
    assert sections[0].title == "First"
    assert sections[1].title == "Second"
    assert sections[2].title == "Third"


def test_subheadings_merge_into_parent():
    md = "## Parent\n\nParent content.\n\n### Child\n\nChild content."
    sections = split_markdown(md, split_level=2)
    assert len(sections) == 1
    assert sections[0].title == "Parent"
    assert "### Child" in sections[0].content
    assert "Child content." in sections[0].content


def test_section_hierarchy():
    md = "# Top\n\nA.\n\n## Sub\n\nB."
    sections = split_markdown(md, split_level=2)
    assert sections[0].section_hierarchy == ["Top"]
    assert sections[1].section_hierarchy == ["Top", "Sub"]


def test_computed_fields_word_count():
    md = "# Title\n\none two three four five"
    sections = split_markdown(md)
    assert sections[0].word_count == 5


def test_computed_fields_images():
    md = "# Section\n\n![alt](pic.png)\n\nMore text."
    sections = split_markdown(md)
    assert sections[0].images == ["pic.png"]


def test_computed_fields_page_markers():
    md = "# Section\n\n{3} Some text {5}"
    sections = split_markdown(md)
    assert sections[0].page_start == 3
    assert sections[0].page_end == 5


# --- parse_first_page_marker ---


def test_parse_first_page_marker_no_markers():
    assert parse_first_page_marker("Hello world") is None


def test_parse_first_page_marker_single():
    assert parse_first_page_marker("Some text {5} here") == 5


def test_parse_first_page_marker_multiple_returns_first():
    assert parse_first_page_marker("{1} text {3} more {7}") == 1


def test_split_level_1():
    md = "# A\n\nContent.\n\n## B\n\nSub content."
    sections = split_markdown(md, split_level=1)
    assert len(sections) == 1
    assert sections[0].title == "A"
    assert "## B" in sections[0].content


# --- compute_token_count ---


def test_compute_token_count_off():
    assert compute_token_count("hello world", TokenCounting.OFF) is None


def test_compute_token_count_estimate():
    result = compute_token_count("one two three four five", TokenCounting.ESTIMATE)
    assert result == int(5 * 1.3)


def test_compute_token_count_estimate_empty():
    # Empty string has 0 words (split returns [''] but that's 1 element, however
    # "".split() returns [])
    result = compute_token_count("", TokenCounting.ESTIMATE)
    assert result == 0


# --- split_markdown with token counting ---


def test_split_markdown_default_token_count_none():
    md = "# Title\n\nSome content here."
    sections = split_markdown(md)
    assert sections[0].token_count is None


def test_split_markdown_with_estimate_token_counting():
    md = "# Title\n\none two three four five"
    sections = split_markdown(md, token_counting=TokenCounting.ESTIMATE)
    assert sections[0].token_count is not None
    assert sections[0].token_count == int(5 * 1.3)
