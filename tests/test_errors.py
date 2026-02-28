"""Tests for exception hierarchy."""

from pdf2md.errors import (
    ConversionError,
    EmptyOutputError,
    InvalidPageRangeError,
    Pdf2MdError,
)


def test_conversion_error_is_pdf2md_error():
    assert isinstance(ConversionError("fail"), Pdf2MdError)


def test_invalid_page_range_is_pdf2md_error():
    assert isinstance(InvalidPageRangeError("bad range"), Pdf2MdError)


def test_empty_output_is_pdf2md_error():
    assert isinstance(EmptyOutputError("no content"), Pdf2MdError)


def test_error_message_preserved():
    e = ConversionError("something broke")
    assert str(e) == "something broke"


def test_base_is_exception():
    assert isinstance(Pdf2MdError("x"), Exception)
