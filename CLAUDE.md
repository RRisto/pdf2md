# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
uv sync --native-tls                                    # Install dependencies
uv run pdf2md --help                                     # Run CLI
uv run pdf2md input.pdf -o out.md
uv run python -m pytest tests/ -v -m "not slow"         # Run fast tests
uv run python -m pytest tests/test_integration.py -m slow  # Run integration tests (needs ML models)
```

The `--native-tls` flag is required because the environment uses a corporate proxy with a custom CA certificate. The `truststore` dependency handles this at runtime for Marker's model downloads.

## Architecture

The tool converts PDFs to structured Markdown via a five-stage pipeline orchestrated by `pipeline.py`:

```
CLI args → ConversionConfig (models.py)
         ↓
[1] MarkerBridge.convert()     (converter.py)  — Marker PdfConverter → raw markdown + PIL images + DocumentMetadata
         ↓
[2] process_images()           (image_handler.py) — save/base64/strip images, rewrite markdown refs
         ↓
[3] split_markdown()           (section_splitter.py) — regex split on ATX headings, build Section list, optional token counting
         ↓
[4] DocumentResult assembly    (pipeline.py)  — title extraction, metadata, section list
         ↓
[5] write_output()             (output_writer.py) — write as .md (YAML front-matter), sections-dir, or JSON
```

**Key modules:**
- `pipeline.py` — core pipeline logic (`run_pipeline()`), no CLI/Rich dependencies. Also the programmatic Python API entry point.
- `cli.py` — Click CLI, thin wrapper around `run_pipeline()` with Rich spinner, dry-run summary, and progress output.
- `converter.py` — `MarkerBridge` wrapping Marker's `PdfConverter`, page-range validation, table format handling (HTML/CSV), typed metadata conversion.
- `models.py` — Pydantic models: `ConversionConfig`, `DocumentResult`, `Section`, `DocumentMetadata` (typed, with `TocEntry`, `PageStat`, `PageBlockMetadata`), enums (`ImageMode`, `OutputFormat`, `TableFormat`, `TokenCounting`).
- `section_splitter.py` — ATX heading split, sub-heading merge, `compute_token_count()`.
- `image_handler.py` — image processing with extracted pure functions (`resolve_image_dir`, `resolve_image_rel_path`).
- `output_writer.py` — YAML front-matter via `pyyaml` (`_build_front_matter()`), sections-dir manifest with optional `token_count`.
- `__init__.py` — public API: `convert_pdf()`, re-exports `run_pipeline`, `ConversionConfig`, `DocumentResult`, `Section`.

**Key design decisions:**
- Marker ML models are lazily loaded once per `MarkerBridge` instance (`_ensure_models`)
- `converter.py` injects `truststore` at module load time to handle corporate proxy SSL for model downloads
- Page-range validation runs before model loading (fail fast)
- `DocumentMetadata` is a typed Pydantic model with `extra="ignore"` for forward compatibility with future Marker versions
- Section splitter merges sub-headings below `split_level` into their parent section and recalculates computed fields (word_count, images, page markers, token_count) after merging
- Token counting supports three modes: `off` (default), `estimate` (word_count * 1.3), `tiktoken` (cl100k_base encoding, optional dependency)
- Table format: `markdown` (default), `html` (Marker's HTML tables), `csv` (HTML tables post-processed to fenced CSV blocks using stdlib `html.parser`)
- YAML front-matter uses `yaml.safe_dump()` for proper escaping of special characters
- Output formats: `md` (single file + YAML front-matter), `sections-dir` (numbered files + `index.json` manifest), `json` (full `DocumentResult` serialized)
- Dry-run mode skips `write_output()` and prints a Rich summary table instead

## Dependencies

- `marker-pdf` — the PDF conversion engine (Marker + Surya OCR). Downloads ~1GB of ML models on first run to `~/Library/Caches/datalab/models/`
- `truststore` — uses macOS keychain for SSL verification (corporate proxy workaround)
- `pyyaml` — YAML front-matter generation with proper escaping
- `tiktoken` (optional) — precise token counting with OpenAI's cl100k_base encoding

## Testing

```bash
uv run python -m pytest tests/ -v -m "not slow"            # Fast tests (146 tests, no ML models needed)
uv run python -m pytest tests/test_integration.py -m slow   # Integration tests (need ML models)
```

Test files mirror source modules. `test_converter.py` uses mock-based tests (patches Marker imports). Integration tests use `fpdf2` to generate synthetic PDFs.

## Custom PyPI Index

`pyproject.toml` configures a custom PyPI mirror under `[tool.uv]`. Always use `--native-tls` with `uv` commands.
