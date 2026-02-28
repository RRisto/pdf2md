"""Custom exceptions for pdf2md."""


class Pdf2MdError(Exception):
    """Base exception for pdf2md."""


class ConversionError(Pdf2MdError):
    """Raised when PDF conversion fails."""


class InvalidPageRangeError(Pdf2MdError):
    """Raised when page range specification is invalid."""


class EmptyOutputError(Pdf2MdError):
    """Raised when conversion produces no content."""
