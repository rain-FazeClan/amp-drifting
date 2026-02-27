"""Convert a PDF (paper.pdf) to Markdown using the markitdown library.

Default behavior:
- Input:  ./paper.pdf
- Output: ./paper.md  (same directory as the input PDF)

You can override paths via CLI args.

Example:
  python convert_paper_pdf_to_md.py
  python convert_paper_pdf_to_md.py --input path\\to\\paper.pdf
  python convert_paper_pdf_to_md.py --input paper.pdf --output paper.md --force
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _extract_markdown(conversion_result: object) -> str:
    """Best-effort extraction of markdown text from markitdown's result object."""

    # Common attributes seen in markitdown examples / return types.
    for attr in ("markdown", "text_content", "content", "text"):
        value = getattr(conversion_result, attr, None)
        if isinstance(value, str) and value.strip():
            return value

    # Sometimes libraries return a plain string.
    if isinstance(conversion_result, str) and conversion_result.strip():
        return conversion_result

    # Last resort: a repr is not useful for markdown; fail loudly.
    raise RuntimeError(
        "Unable to extract markdown text from markitdown conversion result. "
        f"Type={type(conversion_result)!r}. "
        "If this happens, please open an issue with the returned object's attributes."
    )


def convert_pdf_to_markdown(input_pdf: Path, output_md: Path, *, force: bool = False) -> None:
    if not input_pdf.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")
    if input_pdf.suffix.lower() != ".pdf":
        raise ValueError(f"Input file does not look like a PDF: {input_pdf}")

    if output_md.exists() and not force:
        raise FileExistsError(
            f"Output file already exists: {output_md}. "
            "Use --force to overwrite."
        )

    try:
        from markitdown import MarkItDown # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "markitdown is not installed. Install it first, e.g.:\n"
            "  pip install -r requirements.txt\n"
            "or\n"
            "  pip install markitdown\n"
        ) from exc

    converter = MarkItDown()
    result = converter.convert(str(input_pdf))
    markdown = _extract_markdown(result)

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert paper.pdf to Markdown using markitdown",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("paper.pdf"),
        help="Path to input PDF (default: ./paper.pdf)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output Markdown. Default: same dir as input, with .md extension.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )

    args = parser.parse_args()
    input_pdf: Path = args.input

    if args.output is None:
        output_md = input_pdf.with_suffix(".md")
    else:
        output_md = args.output

    convert_pdf_to_markdown(input_pdf, output_md, force=args.force)
    print(f"Converted: {input_pdf} -> {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
