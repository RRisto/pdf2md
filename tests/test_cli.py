"""Tests for CLI helpers and Click group."""

from pathlib import Path

from click.testing import CliRunner

from pdf2md.cli import DefaultGroup, _default_output_path, _extract_title, main
from pdf2md.models import OutputFormat, Section


# --- _default_output_path ---


def test_default_output_path_md():
    result = _default_output_path(Path("/dir/report.pdf"), OutputFormat.MD)
    assert result == Path("/dir/report.md")


def test_default_output_path_json():
    result = _default_output_path(Path("/dir/report.pdf"), OutputFormat.JSON)
    assert result == Path("/dir/report.json")


def test_default_output_path_sections_dir():
    result = _default_output_path(Path("/dir/report.pdf"), OutputFormat.SECTIONS_DIR)
    assert result == Path("/dir/report")


# --- _extract_title ---


def test_extract_title_level1_wins():
    sections = [
        Section(title="Preamble", level=0, content="x"),
        Section(title="Main Title", level=1, content="y"),
        Section(title="Subtitle", level=2, content="z"),
    ]
    assert _extract_title(sections, Path("f.pdf")) == "Main Title"


def test_extract_title_skips_preamble():
    # _extract_title first looks for level<=1 non-Preamble sections;
    # if none found, it checks sections[0] != "Preamble", then falls back to filename.
    sections = [
        Section(title="Preamble", level=0, content="x"),
        Section(title="Real Title", level=1, content="y"),
    ]
    assert _extract_title(sections, Path("f.pdf")) == "Real Title"


def test_extract_title_fallback_to_filename():
    sections = [Section(title="Preamble", level=0, content="x")]
    assert _extract_title(sections, Path("my_doc.pdf")) == "my_doc"


def test_extract_title_empty_sections():
    assert _extract_title([], Path("fallback.pdf")) == "fallback"


# --- DefaultGroup ---


def test_default_group_inserts_default_command():
    import click

    @click.group(cls=DefaultGroup)
    def grp():
        pass

    @grp.command()
    @click.argument("name")
    def convert(name):
        click.echo(f"converting {name}")

    runner = CliRunner()
    result = runner.invoke(grp, ["myfile.pdf"])
    assert result.exit_code == 0
    assert "converting myfile.pdf" in result.output


def test_default_group_known_subcommand_unchanged():
    import click

    @click.group(cls=DefaultGroup)
    def grp():
        pass

    @grp.command()
    def convert():
        click.echo("convert ran")

    @grp.command()
    def other():
        click.echo("other ran")

    runner = CliRunner()
    result = runner.invoke(grp, ["other"])
    assert "other ran" in result.output


def test_default_group_flags_not_treated_as_subcommand():
    """Flags starting with '-' are not mistaken for subcommand names,
    so they stay as group-level args (not routed to convert)."""
    import click

    @click.group(cls=DefaultGroup)
    def grp():
        pass

    @grp.command()
    @click.argument("name")
    def convert(name):
        click.echo(f"converting {name}")

    runner = CliRunner()
    # "--unknown" starts with "-", so DefaultGroup does NOT prepend "convert"
    result = runner.invoke(grp, ["--unknown"])
    assert result.exit_code != 0  # group rejects unknown flag


# --- main CLI help ---


def test_help_shows_commands():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "convert" in result.output
    assert "download-models" in result.output


def test_help_shows_table_format():
    runner = CliRunner()
    result = runner.invoke(main, ["convert", "--help"])
    assert result.exit_code == 0
    assert "--table-format" in result.output


def test_help_shows_dry_run():
    runner = CliRunner()
    result = runner.invoke(main, ["convert", "--help"])
    assert result.exit_code == 0
    assert "--dry-run" in result.output


def test_help_shows_token_counting():
    runner = CliRunner()
    result = runner.invoke(main, ["convert", "--help"])
    assert result.exit_code == 0
    assert "--token-counting" in result.output
