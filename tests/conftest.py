"""Shared test fixtures."""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: requires ML models")


@pytest.fixture
def synthetic_pdf(tmp_path):
    """Generate a 2-page PDF with known content using fpdf2."""
    from fpdf import FPDF

    pdf = FPDF()

    # Page 1
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(text="Test Document")
    pdf.ln(15)
    pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(w=0, text="This is the first page of the test document. It contains a paragraph of text for testing the PDF to Markdown conversion pipeline.")

    # Page 2
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(text="Section Two")
    pdf.ln(12)
    pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(w=0, text="This is the second page with another section. It provides additional content for verifying multi-page conversion.")

    path = tmp_path / "test_input.pdf"
    pdf.output(str(path))
    return path
