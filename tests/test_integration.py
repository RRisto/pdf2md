"""Integration tests — require ML models to be available."""

import json

import pytest

from pdf2md.converter import MarkerBridge
from pdf2md.models import ConversionConfig, OutputFormat


@pytest.mark.slow
class TestIntegration:
    def test_convert_to_markdown(self, synthetic_pdf, tmp_path):
        output = tmp_path / "output.md"
        config = ConversionConfig(
            input_path=synthetic_pdf,
            output_path=output,
            output_format=OutputFormat.MD,
        )

        from pdf2md.pipeline import run_pipeline
        doc = run_pipeline(config)

        text = output.read_text()
        assert text.startswith("---\n")
        assert "title:" in text
        assert len(doc.sections) >= 1

    def test_convert_to_json(self, synthetic_pdf, tmp_path):
        output = tmp_path / "output.json"
        config = ConversionConfig(
            input_path=synthetic_pdf,
            output_path=output,
            output_format=OutputFormat.JSON,
        )

        from pdf2md.pipeline import run_pipeline
        doc = run_pipeline(config)

        data = json.loads(output.read_text())
        assert "title" in data
        assert "sections" in data
        assert len(data["sections"]) >= 1

    def test_convert_to_sections_dir(self, synthetic_pdf, tmp_path):
        output = tmp_path / "sections"
        config = ConversionConfig(
            input_path=synthetic_pdf,
            output_path=output,
            output_format=OutputFormat.SECTIONS_DIR,
        )

        from pdf2md.pipeline import run_pipeline
        run_pipeline(config)

        assert output.is_dir()
        index_path = output / "index.json"
        assert index_path.exists()
        index = json.loads(index_path.read_text())
        assert len(index["sections"]) >= 1
        # Verify at least one .md file exists
        md_files = list(output.glob("*.md"))
        assert len(md_files) >= 1

    def test_marker_bridge_returns_conversion_result(self, synthetic_pdf, tmp_path):
        config = ConversionConfig(
            input_path=synthetic_pdf,
            output_path=tmp_path / "output.md",
        )
        bridge = MarkerBridge()
        result = bridge.convert(config)

        assert result.markdown
        assert len(result.markdown.strip()) > 0
