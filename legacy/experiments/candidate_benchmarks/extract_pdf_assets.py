#!/usr/bin/env python3
"""Extract text pages and embedded images from candidate benchmark PDFs.

This is a lightweight first-pass extractor for newly downloaded olympiad PDFs.
It does not try to infer problem boundaries or rubrics; it only creates stable
page-level artifacts that can be reviewed or fed into a later structuring pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import fitz


def extract_one(pdf_path: Path, out_root: Path, *, render_pages: bool = False, render_dpi: int = 180) -> dict[str, Any]:
    doc = fitz.open(pdf_path)
    out_dir = out_root / pdf_path.stem
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    page_render_dir = out_dir / "page_renders"
    if render_pages:
        page_render_dir.mkdir(parents=True, exist_ok=True)
    pages: list[dict[str, Any]] = []
    for page_index, page in enumerate(doc, start=1):
        page_images = []
        for image_index, image in enumerate(page.get_images(full=True), start=1):
            xref = image[0]
            extracted = doc.extract_image(xref)
            ext = extracted.get("ext", "png")
            image_name = f"page_{page_index:03d}_image_{image_index:02d}.{ext}"
            image_path = image_dir / image_name
            image_path.write_bytes(extracted["image"])
            page_images.append(
                {
                    "file": str(image_path.relative_to(out_dir)),
                    "xref": xref,
                    "width": extracted.get("width"),
                    "height": extracted.get("height"),
                    "colorspace": extracted.get("colorspace"),
                    "ext": ext,
                }
            )
        page_render = None
        if render_pages:
            zoom = render_dpi / 72.0
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            render_name = f"page_{page_index:03d}.png"
            render_path = page_render_dir / render_name
            pixmap.save(render_path)
            page_render = {
                "file": str(render_path.relative_to(out_dir)),
                "dpi": render_dpi,
                "width": pixmap.width,
                "height": pixmap.height,
            }
        pages.append(
            {
                "page": page_index,
                "text": page.get_text("text"),
                "image_count": len(page_images),
                "images": page_images,
                "page_render": page_render,
            }
        )
    payload = {
        "source_pdf": str(pdf_path),
        "page_count": len(pages),
        "pages": pages,
    }
    (out_dir / "pages.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "source_pdf": str(pdf_path),
        "output_dir": str(out_dir),
        "page_count": len(pages),
        "image_count": sum(page["image_count"] for page in pages),
        "rendered_page_count": len(pages) if render_pages else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdfs", nargs="+", type=Path)
    parser.add_argument("--out_root", type=Path, default=Path("data/science_arena/candidate_benchmarks/extracted"))
    parser.add_argument("--summary", type=Path, default=Path(""))
    parser.add_argument("--render_pages", action="store_true", help="Also render every PDF page to PNG for vector diagrams.")
    parser.add_argument("--render_dpi", type=int, default=180)
    args = parser.parse_args()

    results = [
        extract_one(
            pdf.expanduser().resolve(),
            args.out_root.expanduser().resolve(),
            render_pages=args.render_pages,
            render_dpi=args.render_dpi,
        )
        for pdf in args.pdfs
    ]
    summary_path = args.summary or args.out_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
