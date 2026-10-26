#!/usr/bin/env python3
"""Panel icons -> app/frontend/public (Vite copies them into app/backend/static on build).

Motif: three ECU bins (translucent track) filled to different levels (service placement / utilisation), white on the
panel's primary blue. favicon.svg is the same drawing in vector form.
    .venv/bin/python app/frontend/scripts/make_icons.py
"""
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "public"
BLUE, WHITE = (37, 99, 235, 255), (255, 255, 255, 255)
S = 1024                                   # master size, everything below in master pixels
BARS = [(197, 0.55), (427, 0.85), (657, 0.40)]   # (left x, fill level)
BAR_W, TOP, BOT, R_BAR = 170, 222, 802, 34
TRACK = (255, 255, 255, 90)               # empty part of a bin: translucent white


def draw(rounded: bool) -> Image.Image:
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, S - 1, S - 1], radius=224 if rounded else 0, fill=BLUE)
    for x, lv in BARS:
        bar = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        ImageDraw.Draw(bar).rounded_rectangle([x, TOP, x + BAR_W, BOT], radius=R_BAR, fill=TRACK)
        im.alpha_composite(bar)
        d.rounded_rectangle([x, BOT - (BOT - TOP) * lv, x + BAR_W, BOT], radius=R_BAR, fill=WHITE)
    return im


def svg() -> str:
    parts = [f'<rect width="{S}" height="{S}" rx="224" fill="#2563eb"/>']
    for x, lv in BARS:
        parts.append(f'<rect x="{x}" y="{TOP}" width="{BAR_W}" height="{BOT - TOP}" rx="{R_BAR}" fill="#fff" fill-opacity="0.35"/>')
        y = BOT - (BOT - TOP) * lv
        parts.append(f'<rect x="{x}" y="{y:.0f}" width="{BAR_W}" height="{BOT - y:.0f}" rx="{R_BAR}" fill="#fff"/>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {S} {S}">' + "".join(parts) + "</svg>\n"


def main():
    OUT.mkdir(exist_ok=True)
    tab, full = draw(True), draw(False)    # apple-touch: opaque full bleed, iOS rounds the corners itself
    small = lambda im, n: im.resize((n, n), Image.LANCZOS)
    tab.save(OUT / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])
    for n in (16, 32, 48):
        small(tab, n).save(OUT / f"favicon-{n}x{n}.png")
    for n in (120, 152, 167, 180):
        name = "apple-touch-icon.png" if n == 180 else f"apple-touch-icon-{n}x{n}.png"
        small(full, n).convert("RGB").save(OUT / name)
    for n in (192, 512):
        small(tab, n).save(OUT / f"icon-{n}.png")
    (OUT / "favicon.svg").write_text(svg())
    (OUT / "site.webmanifest").write_text(
        '{\n  "name": "my-code Experiments",\n  "short_name": "my-code",\n  "icons": [\n'
        '    {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png"},\n'
        '    {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png"}\n  ],\n'
        '  "theme_color": "#2563eb",\n  "background_color": "#ffffff",\n  "display": "standalone"\n}\n')
    print("icons ->", OUT, sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
