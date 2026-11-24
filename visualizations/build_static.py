"""Render the static fallbacks of loop-widgets.js as loop-<example>-static.png.

The static figures come from the same code as the interactive widgets
(LoopWidgets.staticHTML), so they stay in sync. Needs Quarto (for its bundled
Deno) and macOS Quick Look (qlmanage). Run from anywhere:

    python visualizations/build_static.py
"""

import pathlib
import subprocess
import tempfile

from PIL import Image, ImageChops

HERE = pathlib.Path(__file__).resolve().parent
PAGE_PX = 2000  # 1000 CSS px rendered at 2x
PAD_PX = 32


def crop_to_content(src, dst):
    """Keep the full width (same scale for every figure), drop the empty bottom."""
    im = Image.open(src).convert("RGB")
    box = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
    if box[3] >= im.size[1] - 2:
        raise RuntimeError(f"{dst.name}: content reaches the bottom of the page and may be cut off")
    im.crop((0, 0, im.size[0], box[3] + PAD_PX)).save(dst)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        out = subprocess.run(
            ["quarto", "run", str(HERE / "export-static.js"), tmp],
            check=True, capture_output=True, text=True,
        ).stdout
        names = out.strip().splitlines()[-1].split()
        for name in names:
            html = pathlib.Path(tmp) / f"loop-{name}.html"
            subprocess.run(["qlmanage", "-t", "-s", str(PAGE_PX), "-o", tmp, str(html)],
                           check=True, capture_output=True)
            dst = HERE / f"loop-{name}-static.png"
            crop_to_content(pathlib.Path(tmp) / f"{html.name}.png", dst)
            print("wrote", dst.relative_to(HERE.parent))


if __name__ == "__main__":
    main()
