"""Tests for image_handler module."""

import base64
from pathlib import Path

from PIL import Image

from pdf2md.image_handler import (
    inline_base64,
    process_images,
    resolve_image_dir,
    resolve_image_rel_path,
    save_image_files,
    strip_images,
)
from pdf2md.models import ImageMode


def _make_image(size=(2, 2), color="red"):
    """Create a small PIL image for testing."""
    return Image.new("RGB", size, color)


# --- strip_images ---


def test_strip_images_removes_refs():
    md = "Text ![alt](img.png) more text."
    assert strip_images(md) == "Text  more text."


def test_strip_images_preserves_other_text():
    md = "No images here."
    assert strip_images(md) == "No images here."


def test_strip_images_multiple():
    md = "![a](1.png) middle ![b](2.jpg)"
    assert strip_images(md) == " middle "


# --- inline_base64 ---


def test_inline_base64_replaces_ref():
    images = {"img.png": _make_image()}
    result = inline_base64("![photo](img.png)", images)
    assert result.startswith("![photo](data:image/png;base64,")
    # Verify it's valid base64
    b64_data = result.split("base64,")[1].rstrip(")")
    decoded = base64.b64decode(b64_data)
    assert len(decoded) > 0


def test_inline_base64_jpeg_format():
    images = {"photo.jpg": _make_image()}
    result = inline_base64("![pic](photo.jpg)", images)
    assert "data:image/jpeg;base64," in result


def test_inline_base64_unknown_ref_unchanged():
    images = {"other.png": _make_image()}
    md = "![alt](missing.png)"
    assert inline_base64(md, images) == md


# --- resolve_image_dir ---


def test_resolve_image_dir_explicit_dir(tmp_path):
    """When an explicit image_dir is given, it is returned as-is."""
    explicit = tmp_path / "my_imgs"
    assert resolve_image_dir(explicit, Path("/some/out.md")) == explicit


def test_resolve_image_dir_file_output():
    """When output_path has a suffix (file), derive <stem>_images next to it."""
    result = resolve_image_dir(None, Path("/docs/report.md"))
    assert result == Path("/docs/report_images")


def test_resolve_image_dir_dir_output():
    """When output_path has no suffix (directory), use <output>/images."""
    result = resolve_image_dir(None, Path("/docs/output_dir"))
    assert result == Path("/docs/output_dir/images")


def test_resolve_image_dir_none_output():
    """When both image_dir and output_path are None, fall back to 'images'."""
    result = resolve_image_dir(None, None)
    assert result == Path("images")


# --- resolve_image_rel_path ---


def test_resolve_image_rel_path_file_output():
    """When output_path has a suffix, relative path uses image_dir.name."""
    result = resolve_image_rel_path(
        Path("/docs/report_images"), Path("/docs/report.md"), "fig.png"
    )
    assert result == Path("report_images/fig.png")


def test_resolve_image_rel_path_dir_output():
    """When output_path has no suffix (directory), relative path uses 'images'."""
    result = resolve_image_rel_path(
        Path("/docs/output_dir/images"), Path("/docs/output_dir"), "fig.png"
    )
    assert result == Path("images/fig.png")


# --- save_image_files ---


def test_save_image_files_writes_to_dir(tmp_path):
    images = {"fig.png": _make_image()}
    output_path = tmp_path / "out.md"
    result = save_image_files("![x](fig.png)", images, image_dir=None, output_path=output_path)

    image_dir = tmp_path / "out_images"
    assert (image_dir / "fig.png").exists()
    assert "out_images/fig.png" in result


def test_save_image_files_explicit_dir(tmp_path):
    images = {"pic.png": _make_image()}
    img_dir = tmp_path / "my_images"
    result = save_image_files("![a](pic.png)", images, image_dir=img_dir, output_path=None)

    assert (img_dir / "pic.png").exists()


def test_save_image_files_unknown_ref_unchanged(tmp_path):
    images = {"known.png": _make_image()}
    md = "![x](unknown.png)"
    result = save_image_files(md, images, image_dir=tmp_path / "imgs", output_path=None)
    assert result == md


# --- process_images routing ---


def test_process_images_none_strips():
    md = "Text ![img](pic.png) end."
    result = process_images(md, {"pic.png": _make_image()}, ImageMode.NONE)
    assert "![" not in result


def test_process_images_base64_inlines():
    images = {"pic.png": _make_image()}
    result = process_images("![alt](pic.png)", images, ImageMode.BASE64)
    assert "data:image/png;base64," in result


def test_process_images_files_saves(tmp_path):
    images = {"pic.png": _make_image()}
    result = process_images(
        "![alt](pic.png)", images, ImageMode.FILES,
        image_dir=tmp_path / "imgs", output_path=None,
    )
    assert (tmp_path / "imgs" / "pic.png").exists()


def test_process_images_empty_images_passthrough():
    md = "![alt](pic.png)"
    result = process_images(md, {}, ImageMode.BASE64)
    assert result == md
