# pdf2md

Convert PDF files to structured Markdown for LLM consumption.

Uses [Marker](https://github.com/VikParuchuri/marker) (deep-learning pipeline) for high-quality extraction of headings, tables, equations (LaTeX), and images from PDFs like arxiv papers and books.

## Installation

```bash
uv sync --native-tls
```

## Usage

```bash
# Basic conversion
uv run pdf2md paper.pdf -o paper.md

# Explicit convert subcommand (equivalent)
uv run pdf2md convert paper.pdf -o paper.md

# First 30 pages of a book
uv run pdf2md book.pdf --page-range "0-29" -o chapter.md

# Inline images as base64
uv run pdf2md paper.pdf --image-mode base64 -o paper.md

# Split into a directory of section files
uv run pdf2md book.pdf --output-format sections-dir -o book/

# JSON output for programmatic use
uv run pdf2md paper.pdf --output-format json -o paper.json

# Tables as HTML or CSV instead of markdown
uv run pdf2md paper.pdf --table-format html -o paper.md
uv run pdf2md paper.pdf --table-format csv -o paper.md

# Token counting (estimate or precise with tiktoken)
uv run pdf2md paper.pdf --token-counting estimate -o paper.md

# Dry-run: see section summary without writing output
uv run pdf2md paper.pdf --dry-run

# Use a custom model directory
uv run pdf2md paper.pdf --model-dir ./models -o paper.md

# Suppress progress output
uv run pdf2md paper.pdf --quiet -o paper.md
```

### Python API

```python
from pdf2md import convert_pdf

# Simple conversion
doc = convert_pdf("paper.pdf", "paper.md")

# With options
doc = convert_pdf("paper.pdf", "paper.md", page_range="0-9", image_mode="base64")

# Access results
print(doc.title)
for section in doc.sections:
    print(f"  {section.level}: {section.title} ({section.word_count} words)")
```

### Pre-downloading models

Models are downloaded automatically on first conversion, but you can
pre-download them (useful for CI, Docker builds, or air-gapped setups):

```bash
# Download to the platform default cache directory
uv run pdf2md download-models

# Download to a custom directory
uv run pdf2md download-models --model-dir ./models
```

The default cache location is platform-dependent:

| Platform | Default path |
|----------|-------------|
| macOS | `~/Library/Caches/datalab/models/` |
| Linux | `~/.cache/datalab/models/` |
| Windows | `C:\Users\<user>\AppData\Local\datalab\Cache\models\` |

## CLI Options

```
pdf2md convert INPUT.pdf [OPTIONS]

  -o, --output PATH                         Output path (default: <input_stem>.md)
  --output-format [md|sections-dir|json]    Output format (default: md)
  --image-mode [files|base64|none]          How to handle images (default: files)
  --image-dir PATH                          Directory for extracted images
  --page-range TEXT                         Pages to process, e.g. "0,5-10,20"
  --split-level INT                         Heading level for section splits (default: 2)
  --table-format [markdown|html|csv]        Table output format (default: markdown)
  --paginate / --no-paginate                Include page number markers
  --device [auto|cpu|cuda|mps]              Compute device (default: auto)
  --model-dir PATH                          Custom directory for ML model storage
  --quiet                                   Suppress progress output
  --dry-run                                 Print section summary without writing output
  --token-counting [off|estimate|tiktoken]  Token count estimation mode (default: off)
```

```
pdf2md download-models [OPTIONS]

  --model-dir PATH  Custom directory for ML model storage
  --quiet           Suppress progress output
```

## Output Formats

| Format | Description |
|--------|-------------|
| `md` | Single Markdown file with YAML front-matter (title, source, pages, section count) |
| `sections-dir` | Directory of numbered section files (`00_preamble.md`, `01_introduction.md`, ...) plus `index.json` manifest |
| `json` | Full document structure as JSON with sections array |

## How It Works

```
PDF → Marker (OCR + layout analysis) → raw markdown + images + metadata
    → image processing (save/base64/strip)
    → section splitting (by heading level, optional token counting)
    → output writing (md/sections-dir/json) or dry-run summary
```

## ML Models

Marker uses a five-model pipeline via [Surya](https://github.com/VikParuchuri/surya). All models are downloaded from HuggingFace on first run (~3.3 GB total on disk).

| Model | Purpose | Size |
|-------|---------|------|
| **Text detection** (`line_det`) | Locates text regions and bounding boxes in page images | 81 MB |
| **Text recognition** (`text_recognition`) | OCR — reads text from detected regions (multimodal vision + language model) | 1.3 GB |
| **Layout analysis** (`layout`) | Classifies page regions: headings, paragraphs, tables, figures, footnotes, etc. | 1.4 GB |
| **Table recognition** (`table_recognition`) | Extracts table structure — rows, columns, spans, headers | 209 MB |
| **OCR error detection** (`ocr_error_detection`) | Flags low-confidence OCR output for correction | 277 MB |

### System Requirements

**Disk:** ~3.3 GB for model weights, plus the Python environment (~2 GB for PyTorch and dependencies).

**RAM:** 8 GB minimum. The two largest models (text recognition and layout analysis) are loaded together during conversion. Expect ~4-6 GB peak memory on CPU, less on GPU where models use float16/bfloat16 precision.

**GPU (optional but recommended):**
- **NVIDIA CUDA** — fastest option. Models use float16 with Flash Attention 2 when available.
- **Apple Silicon MPS** — supported, uses float16. Good performance on M-series Macs.
- **CPU** — works but significantly slower, especially for long documents. Uses float32.

Select the device with `--device` (`auto` detects the best available).

**Python:** 3.13+
