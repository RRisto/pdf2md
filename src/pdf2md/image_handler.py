"""Handle image extraction and embedding from Marker output."""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path

from PIL import Image

from .models import ImageMode

IMAGE_REF_RE = re.compile(r"(!\[[^\]]*\])\(([^)]+)\)")


def process_images(
    markdown: str,
    images: dict[str, Image.Image],
    mode: ImageMode,
    image_dir: Path | None = None,
    output_path: Path | None = None,
) -> str:
    """Process images according to the selected mode and rewrite markdown refs.

    Args:
        markdown: Raw markdown from Marker.
        images: Dict of {image_name: PIL.Image} from Marker.
        mode: How to handle images (files, base64, none).
        image_dir: Explicit directory for saving image files.
        output_path: Output file/dir path, used to derive default image_dir.

    Returns:
        Markdown with rewritten image references.
    """
    if mode == ImageMode.NONE:
        return strip_images(markdown)

    if not images:
        return markdown

    if mode == ImageMode.BASE64:
        return inline_base64(markdown, images)

    if mode == ImageMode.FILES:
        return save_image_files(markdown, images, image_dir, output_path)

    return markdown


def strip_images(markdown: str) -> str:
    """Remove all image references from markdown."""
    return IMAGE_REF_RE.sub("", markdown)


def inline_base64(markdown: str, images: dict[str, Image.Image]) -> str:
    """Replace image references with inline base64 data URIs."""
    def replace_ref(match: re.Match) -> str:
        alt_part = match.group(1)
        image_name = match.group(2)

        if image_name not in images:
            return match.group(0)

        img = images[image_name]
        buf = io.BytesIO()
        fmt = "PNG"
        if image_name.lower().endswith((".jpg", ".jpeg")):
            fmt = "JPEG"
        img.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        mime = "image/png" if fmt == "PNG" else "image/jpeg"
        return f"{alt_part}(data:{mime};base64,{b64})"

    return IMAGE_REF_RE.sub(replace_ref, markdown)


def resolve_image_dir(image_dir: Path | None, output_path: Path | None) -> Path:
    """Determine the directory where images should be saved.

    Args:
        image_dir: Explicit image directory, used as-is when provided.
        output_path: Output file/dir path, used to derive a default when
            *image_dir* is ``None``.

    Returns:
        Resolved image directory path.
    """
    if image_dir is not None:
        return image_dir

    if output_path is None:
        return Path("images")

    output_path = Path(output_path)
    if output_path.suffix:
        return output_path.parent / (output_path.stem + "_images")
    return output_path / "images"


def resolve_image_rel_path(
    image_dir: Path, output_path: Path | None, save_name: str
) -> Path:
    """Compute the relative path from the output location to a saved image.

    Args:
        image_dir: Directory where the image was saved.
        output_path: Output file/dir path (determines relativity strategy).
        save_name: Filename of the saved image.

    Returns:
        Relative ``Path`` suitable for a markdown image reference.
    """
    if output_path and Path(output_path).suffix:
        return Path(image_dir.name) / save_name
    return Path("images") / save_name


def save_image_files(
    markdown: str,
    images: dict[str, Image.Image],
    image_dir: Path | None,
    output_path: Path | None,
) -> str:
    """Save images to files and rewrite markdown refs to relative paths."""
    image_dir = resolve_image_dir(image_dir, output_path)

    image_dir.mkdir(parents=True, exist_ok=True)

    def replace_ref(match: re.Match) -> str:
        alt_part = match.group(1)
        image_name = match.group(2)

        if image_name not in images:
            return match.group(0)

        img = images[image_name]
        # Determine format from name
        ext = Path(image_name).suffix or ".png"
        save_name = Path(image_name).stem + ext
        save_path = image_dir / save_name

        fmt = "JPEG" if ext.lower() in (".jpg", ".jpeg") else "PNG"
        img.save(save_path, format=fmt)

        # Compute relative path from output file to image
        rel = resolve_image_rel_path(image_dir, output_path, save_name)

        return f"{alt_part}({rel})"

    return IMAGE_REF_RE.sub(replace_ref, markdown)
