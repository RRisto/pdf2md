"""CLI entry point for pdf2md."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .converter import download_models, get_default_model_dir
from .errors import Pdf2MdError
from .models import ConversionConfig, DocumentResult, ImageMode, OutputFormat, TableFormat, TokenCounting
from .pipeline import run_pipeline, _extract_title

console = Console(stderr=True)


class DefaultGroup(click.Group):
    """Click group that falls back to a default command when no subcommand matches."""

    def __init__(self, *args, default_cmd: str = "convert", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.default_cmd = default_cmd

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        # If the first arg is not a known command, insert the default command name.
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            args = [self.default_cmd] + args
        return super().parse_args(ctx, args)


@click.group(cls=DefaultGroup)
def main() -> None:
    """Convert PDF files to structured Markdown for LLM consumption."""


@main.command()
@click.argument("input_pdf", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o", "--output", "output_path", type=click.Path(path_type=Path), default=None,
    help="Output path (default: <input_stem>.md)",
)
@click.option(
    "--output-format", type=click.Choice(["md", "sections-dir", "json"]), default="md",
    help="Output format.",
)
@click.option(
    "--image-mode", type=click.Choice(["files", "base64", "none"]), default="files",
    help="How to handle images.",
)
@click.option(
    "--image-dir", type=click.Path(path_type=Path), default=None,
    help="Directory for extracted images.",
)
@click.option(
    "--page-range", type=str, default=None,
    help='Pages to process, e.g. "0,5-10,20".',
)
@click.option(
    "--split-level", type=int, default=2,
    help="Heading level for section splits.",
)
@click.option(
    "--table-format", type=click.Choice(["markdown", "html", "csv"]), default="markdown",
    help="Table output format.",
)
@click.option(
    "--paginate/--no-paginate", default=False,
    help="Include page number markers.",
)
@click.option(
    "--device", type=click.Choice(["auto", "cpu", "cuda", "mps"]), default="auto",
    help="Compute device.",
)
@click.option(
    "--model-dir", type=click.Path(path_type=Path), default=None,
    help="Custom directory for ML model storage.",
)
@click.option(
    "--quiet", is_flag=True, default=False,
    help="Suppress progress output.",
)
@click.option(
    "--dry-run", is_flag=True, default=False,
    help="Run pipeline but only print a summary instead of writing output.",
)
@click.option(
    "--token-counting", type=click.Choice(["off", "estimate", "tiktoken"]), default="off",
    help="Token count estimation mode.",
)
def convert(
    input_pdf: Path,
    output_path: Path | None,
    output_format: str,
    image_mode: str,
    image_dir: Path | None,
    page_range: str | None,
    split_level: int,
    table_format: str,
    paginate: bool,
    device: str,
    model_dir: Path | None,
    quiet: bool,
    dry_run: bool,
    token_counting: str,
) -> None:
    """Convert a PDF file to structured Markdown."""
    fmt = OutputFormat(output_format)
    img_mode = ImageMode(image_mode)

    if output_path is None:
        output_path = _default_output_path(input_pdf, fmt)

    config = ConversionConfig(
        input_path=input_pdf,
        output_path=output_path,
        output_format=fmt,
        image_mode=img_mode,
        image_dir=image_dir,
        page_range=page_range,
        split_level=split_level,
        table_format=TableFormat(table_format),
        paginate=paginate,
        device=device,
        quiet=quiet,
        dry_run=dry_run,
        token_counting=TokenCounting(token_counting),
        model_dir=model_dir,
    )

    try:
        _run_pipeline(config)
    except Pdf2MdError as e:
        raise click.ClickException(str(e)) from e


@main.command("download-models")
@click.option(
    "--model-dir", type=click.Path(path_type=Path), default=None,
    help="Custom directory for ML model storage.",
)
@click.option(
    "--quiet", is_flag=True, default=False,
    help="Suppress progress output.",
)
def download_models_cmd(model_dir: Path | None, quiet: bool) -> None:
    """Download ML models without converting any PDF."""
    if not quiet:
        target = model_dir or get_default_model_dir()
        console.print(f"[bold]Downloading models to[/bold] {target} ...")

    try:
        download_models(model_dir)
    except Exception as e:
        raise click.ClickException(f"Model download failed: {e}") from e

    if not quiet:
        console.print("[green]Models downloaded successfully.[/green]")


def _run_pipeline(config: ConversionConfig) -> None:
    """Execute the full conversion pipeline with CLI progress output."""
    name = config.input_path.name
    ctx = (
        console.status(f"[bold]Converting[/bold] {name} ...", spinner="dots")
        if not config.quiet
        else nullcontext()
    )

    with ctx:
        doc = run_pipeline(config)

    if config.dry_run:
        _print_dry_run_summary(doc, config)
        return

    if not config.quiet:
        console.print(f"  Split into {len(doc.sections)} section(s)")
        console.print(f"[green]Written to[/green] {config.output_path}")


def _print_dry_run_summary(doc: DocumentResult, config: ConversionConfig) -> None:
    """Print a summary table of the conversion result without writing files."""
    table = Table(title=f"Dry-run summary: {config.input_path.name}")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Level", justify="center")
    table.add_column("Title")
    table.add_column("Words", justify="right")
    table.add_column("Images", justify="right")
    table.add_column("Pages")

    for i, section in enumerate(doc.sections):
        pages = ""
        if section.page_start is not None:
            pages = str(section.page_start)
            if section.page_end is not None and section.page_end != section.page_start:
                pages += f"-{section.page_end}"

        table.add_row(
            str(i),
            str(section.level),
            section.title,
            str(section.word_count),
            str(len(section.images)),
            pages,
        )

    console.print(table)


def _default_output_path(input_pdf: Path, fmt: OutputFormat) -> Path:
    """Derive default output path from input filename and format."""
    stem = input_pdf.stem
    if fmt == OutputFormat.MD:
        return input_pdf.parent / f"{stem}.md"
    elif fmt == OutputFormat.SECTIONS_DIR:
        return input_pdf.parent / stem
    elif fmt == OutputFormat.JSON:
        return input_pdf.parent / f"{stem}.json"
    return input_pdf.parent / f"{stem}.md"
