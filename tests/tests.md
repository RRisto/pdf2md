# Tests

## test_section_splitter.py

Tests for `split_markdown()`, `parse_page_markers()`, `extract_image_refs()` from `section_splitter.py`.

- `test_parse_page_markers_no_markers` — returns `None` when no `{N}` markers present
- `test_parse_page_markers_single` — extracts page number from a single marker
- `test_parse_page_markers_multiple_returns_last` — returns the last marker when multiple exist
- `test_extract_image_refs_standard` — extracts paths from `![alt](path)` refs
- `test_extract_image_refs_empty_alt` — handles `![](path)` with no alt text
- `test_extract_image_refs_no_refs` — returns empty list when no image refs
- `test_split_empty_string` — empty input returns `[]`
- `test_split_whitespace_only` — whitespace-only input returns `[]`
- `test_split_no_headings` — no headings produces single "Preamble" section
- `test_split_single_heading` — single heading produces one section with correct title/level/content
- `test_split_preamble_before_heading` — text before first heading becomes a Preamble section
- `test_split_multiple_headings` — multiple headings at/above split_level each become sections
- `test_subheadings_merge_into_parent` — headings below split_level merge into parent section content
- `test_section_hierarchy` — `section_hierarchy` tracks nested heading titles
- `test_computed_fields_word_count` — `word_count` is calculated from section content
- `test_computed_fields_images` — `images` list is populated from image refs in content
- `test_computed_fields_page_markers` — `page_start`/`page_end` set from page markers
- `test_split_level_1` — `split_level=1` merges level-2 headings into parent

## test_image_handler.py

Tests for `strip_images()`, `inline_base64()`, `save_image_files()`, `process_images()` from `image_handler.py`.

- `test_strip_images_removes_refs` — removes `![...](...)` from markdown
- `test_strip_images_preserves_other_text` — leaves non-image text unchanged
- `test_strip_images_multiple` — removes all image refs in one pass
- `test_inline_base64_replaces_ref` — replaces image ref with `data:image/png;base64,...` URI
- `test_inline_base64_jpeg_format` — uses `image/jpeg` MIME type for `.jpg` files
- `test_inline_base64_unknown_ref_unchanged` — leaves refs unchanged when image name not in dict
- `test_save_image_files_writes_to_dir` — saves PIL image to derived `_images` dir, rewrites ref
- `test_save_image_files_explicit_dir` — saves to explicitly provided image directory
- `test_save_image_files_unknown_ref_unchanged` — unknown image refs left as-is
- `test_process_images_none_strips` — `ImageMode.NONE` strips all image refs
- `test_process_images_base64_inlines` — `ImageMode.BASE64` inlines as data URIs
- `test_process_images_files_saves` — `ImageMode.FILES` saves to disk
- `test_process_images_empty_images_passthrough` — empty images dict returns markdown unchanged

## test_output_writer.py

Tests for `_slugify()`, `write_markdown()`, `write_sections_dir()`, `write_json()` from `output_writer.py`.

- `test_slugify_normal` — "Hello World" becomes "hello-world"
- `test_slugify_special_chars` — punctuation is stripped
- `test_slugify_empty` — empty string becomes "untitled"
- `test_slugify_long_title` — truncated to 60 chars
- `test_slugify_spaces_and_underscores` — spaces and underscores become hyphens
- `test_write_markdown_front_matter` — output starts with YAML front-matter containing title, source, pages
- `test_write_markdown_heading_reconstruction` — section headings are prefixed with `#` markers
- `test_write_markdown_preamble_no_heading` — level-0 preamble has no heading prefix
- `test_write_sections_dir_creates_files` — creates numbered `.md` files and `index.json`
- `test_write_sections_dir_manifest_structure` — `index.json` has correct title, section_count, sections array
- `test_write_sections_dir_numbered_filenames` — filenames are `00_slug.md`, `01_slug.md`, etc.
- `test_write_json_valid` — output is valid JSON with expected fields
- `test_write_json_roundtrips` — JSON output round-trips through `DocumentResult.model_validate_json()`

## test_models.py

Tests for Pydantic models and enums from `models.py`.

- `test_image_mode_from_string` — `ImageMode("files")` resolves to `ImageMode.FILES`, etc.
- `test_output_format_from_string` — `OutputFormat("md")` resolves to `OutputFormat.MD`, etc.
- `test_section_defaults` — `Section` has `word_count=0`, empty lists, `None` pages by default
- `test_document_result_construction` — `DocumentResult` initializes with empty sections/metadata
- `test_document_result_serialization` — `model_dump()` produces expected dict structure
- `test_conversion_config_defaults` — `ConversionConfig` defaults: MD format, FILES images, split_level=2
- `test_conversion_config_model_dir` — `model_dir` field accepts a `Path`

## test_cli.py

Tests for CLI helpers and Click group from `cli.py`.

- `test_default_output_path_md` — MD format: `report.pdf` → `report.md`
- `test_default_output_path_json` — JSON format: `report.pdf` → `report.json`
- `test_default_output_path_sections_dir` — sections-dir format: `report.pdf` → `report`
- `test_extract_title_level1_wins` — first level-1 non-Preamble heading wins as title
- `test_extract_title_skips_preamble` — "Preamble" title is skipped in favor of real headings
- `test_extract_title_fallback_to_filename` — falls back to input filename stem
- `test_extract_title_empty_sections` — empty sections list falls back to filename
- `test_default_group_inserts_default_command` — bare `file.pdf` arg routed to `convert`
- `test_default_group_known_subcommand_unchanged` — explicit subcommand names pass through
- `test_default_group_flags_not_treated_as_subcommand` — flags starting with `-` are not mistaken for subcommands
- `test_help_shows_commands` — `--help` output lists both `convert` and `download-models`

## test_errors.py

Tests for exception hierarchy from `errors.py`.

- `test_conversion_error_is_pdf2md_error` — `ConversionError` is a subclass of `Pdf2MdError`
- `test_invalid_page_range_is_pdf2md_error` — `InvalidPageRangeError` is a subclass of `Pdf2MdError`
- `test_empty_output_is_pdf2md_error` — `EmptyOutputError` is a subclass of `Pdf2MdError`
- `test_error_message_preserved` — exception message accessible via `str()`
- `test_base_is_exception` — `Pdf2MdError` is a subclass of `Exception`
