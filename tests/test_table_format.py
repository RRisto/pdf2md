"""Tests for table format conversion helpers."""

from pdf2md.converter import _convert_html_tables_to_csv, _html_table_to_csv_block


def test_html_table_to_csv_block_simple():
    html = "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>"
    result = _html_table_to_csv_block(html)
    assert result == "```csv\nA,B\n1,2\n```"


def test_html_table_to_csv_block_with_header():
    html = (
        "<table>"
        "<tr><th>Name</th><th>Age</th></tr>"
        "<tr><td>Alice</td><td>30</td></tr>"
        "</table>"
    )
    result = _html_table_to_csv_block(html)
    assert result == "```csv\nName,Age\nAlice,30\n```"


def test_convert_html_tables_to_csv_passthrough():
    text = "# Hello\n\nSome paragraph text.\n"
    result = _convert_html_tables_to_csv(text)
    assert result == text


def test_convert_html_tables_to_csv_mixed():
    text = (
        "# Title\n\n"
        "<table><tr><td>X</td><td>Y</td></tr></table>\n\n"
        "More text.\n"
    )
    result = _convert_html_tables_to_csv(text)
    assert "```csv\nX,Y\n```" in result
    assert "# Title" in result
    assert "More text." in result


def test_html_table_to_csv_block_empty_table():
    html = "<table></table>"
    result = _html_table_to_csv_block(html)
    assert result == html


def test_html_table_to_csv_block_cells_with_commas():
    html = (
        "<table>"
        "<tr><td>hello, world</td><td>foo</td></tr>"
        "<tr><td>bar</td><td>a, b, c</td></tr>"
        "</table>"
    )
    result = _html_table_to_csv_block(html)
    assert result == '```csv\n"hello, world",foo\nbar,"a, b, c"\n```'
