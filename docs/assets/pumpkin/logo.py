"""Pumpkin logo: an open ring with a turned tile in the break at its top, drawn in the FrontierCS mark's language.

The FrontierCS mark is a quarter circle of radius 50 (stroke 11, round ends, near-black #1F1F1F) and a 24 x 24 tile
(corner radius 5.5) one step beyond it at 45 degrees; its ink box is 74.7 units square. Pumpkin keeps the circle of
radius 50 and takes three proportions from that box, measured against Pumpkin's own ink height H:

    tile side   = 24   / 74.7 * H   (32.1 %)
    stroke      = 11   / 74.7 * H   (14.7 %)
    space       = 8.42 / 74.7 * H   (11.3 %; FrontierCS's nearest distance between the tile and the arc)

The tile is turned 45 degrees and centred on the circle at its top; the ring breaks on both sides of it, `space` away
from the tile. H depends on the tile and the stroke, so it is solved first. Both marks drawn at the same height show
the tile, the stroke and the space at the same size.

Writes, next to this file:
  pumpkin_mark.svg        the mark, near-black ring, orange tile
  pumpkin_mark_dark.svg   the mark for dark backgrounds (ring #ECECEC)
  pumpkin_logo.pdf        the lockup: the mark at 24 px, a 9 px gap, "Pumpkin" in Instrument Sans Medium 23 px,
                          letter-spacing -0.01em (the FrontierCS lockup's values); needs Playwright with Chromium
  pumpkin_mark.pdf        the mark alone, as a vector PDF

Usage: python logo.py            (SVGs only)
       python logo.py --pdf      (SVGs and PDFs; the font loads from Google Fonts unless --font points to a .ttf)
"""
import argparse
import math
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
INK, INK_DARK, ACCENT = '#1f1f1f', '#ececec', '#e8710a'

# FrontierCS mark (website lockup, fig/logo.py of the paper)
FCS_BOX, FCS_TILE, FCS_STROKE, FCS_RX = 74.7, 24.0, 11.0, 5.5
FCS_SPACE = math.hypot(45.2, 45.2) - (50 + FCS_STROKE / 2)          # 8.42: tile corner to the arc's outer edge
R = 50.0


def geometry():
    rt, rs, rg, rr = FCS_TILE / FCS_BOX, FCS_STROKE / FCS_BOX, FCS_SPACE / FCS_BOX, FCS_RX / FCS_TILE
    reach = 1 / math.sqrt(2) - rr * (math.sqrt(2) - 1)            # turned rounded square: centre to tip, per side
    H = 2 * R / (1 - rt * reach - rs / 2)                          # ink height: tip of the tile to the ring's bottom edge
    t, s, g = rt * H, rs * H, rg * H
    half, rx = t / 2, rr * t

    def to_tile(x, y):                                              # distance from (x, y) to the tile's edge
        u, v = (x + (y + R)) / math.sqrt(2), (x - (y + R)) / math.sqrt(2)
        a = half - rx
        return math.hypot(max(abs(u) - a, 0), max(abs(v) - a, 0)) - rx

    lo, hi = 0.0, 90.0                                              # angle from the top where the ring ends
    for _ in range(60):
        m = (lo + hi) / 2
        x, y = R * math.sin(math.radians(m)), -R * math.cos(math.radians(m))
        lo, hi = (m, hi) if to_tile(x, y) < g + s / 2 else (lo, m)
    ex, ey = R * math.sin(math.radians(lo)), -R * math.cos(math.radians(lo))
    top = -R - reach * t
    return dict(H=H, t=t, s=s, g=g, rx=rx, ex=ex, ey=ey, box=(-(R + s / 2), top, 2 * (R + s / 2), H))


def mark_svg(ring=INK, tile=ACCENT, height=None):
    G = geometry()
    x0, y0, w, h = G['box']
    size = f' width="{w * height / h:.2f}" height="{height}"' if height else f' width="{w:.2f}" height="{h:.2f}"'
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.2f} {y0:.2f} {w:.2f} {h:.2f}"{size}>'
            f'<path d="M{G["ex"]:.2f} {G["ey"]:.2f} A{R:g} {R:g} 0 1 1 {-G["ex"]:.2f} {G["ey"]:.2f}" fill="none" '
            f'stroke="{ring}" stroke-width="{G["s"]:.2f}" stroke-linecap="round"/>'
            f'<rect x="{-G["t"] / 2:.2f}" y="{-R - G["t"] / 2:.2f}" width="{G["t"]:.2f}" height="{G["t"]:.2f}" '
            f'rx="{G["rx"]:.2f}" transform="rotate(45 0 {-R:g})" fill="{tile}"/></svg>')


def write_pdfs(font):
    from playwright.sync_api import sync_playwright
    face = (f'@font-face{{font-family:IS; src:url("{Path(font).resolve().as_uri()}"); font-weight:500}}' if font else
            '@import url("https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@500&display=block");')
    family = 'IS' if font else "'Instrument Sans'"
    page = f"""<!doctype html><html><head><meta charset="utf-8"><style>{face}
html,body{{margin:0; background:transparent}}
.brand{{position:absolute; left:0; top:0; display:inline-flex; align-items:center; gap:9px; font:500 23px/1 {family};
        letter-spacing:-0.01em; color:{INK}; white-space:nowrap}}
.brand svg{{display:block}}</style></head><body>
<div class="brand" id="lockup">{mark_svg(height=24)}<b style="font-weight:500">Pumpkin</b></div>
<div class="brand" id="mark" style="display:none">{mark_svg()}</div></body></html>"""
    box = """id => { const e = document.getElementById(id), rs = [e.querySelector('svg').getBoundingClientRect()];
      const b = e.querySelector('b'); if (b) { const r = document.createRange(); r.selectNodeContents(b); rs.push(r.getBoundingClientRect()); }
      return [Math.min(...rs.map(r => r.left)), rs[0].top, Math.max(...rs.map(r => r.right)), rs[0].bottom]; }"""
    with tempfile.NamedTemporaryFile('w', suffix='.html', delete=False) as f:
        f.write(page)
    try:
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page()
            pg.goto(Path(f.name).as_uri())
            pg.wait_for_function(f"document.fonts.check('500 23px {family}')", timeout=20000)
            for name, keep in (('pumpkin_logo', 'lockup'), ('pumpkin_mark', 'mark')):
                pg.evaluate(f"""() => {{ for (const e of document.querySelectorAll('.brand'))
                  e.style.display = e.id === '{keep}' ? 'inline-flex' : 'none'; }}""")
                x0, y0, x1, y1 = pg.evaluate(box, keep)
                pg.evaluate(f"() => {{ const e = document.getElementById('{keep}'); e.style.left = '{-x0}px'; e.style.top = '{-y0}px'; }}")
                w, h = x1 - x0, y1 - y0
                pg.pdf(path=str(HERE / f'{name}.pdf'), width=f'{w / 96}in', height=f'{h / 96}in', print_background=False,
                       margin={'top': '0', 'right': '0', 'bottom': '0', 'left': '0'}, page_ranges='1')
                print(f'{name}.pdf {w:.2f} x {h:.2f} px')
            b.close()
    finally:
        Path(f.name).unlink()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--pdf', action='store_true', help='also write the lockup and mark PDFs (needs Playwright)')
    ap.add_argument('--font', help='path to InstrumentSans-Medium.ttf (default: Google Fonts)')
    a = ap.parse_args()
    G = geometry()
    print('H %.2f  tile %.2f  stroke %.2f  space %.2f' % (G['H'], G['t'], G['s'], G['g']))
    (HERE / 'pumpkin_mark.svg').write_text(mark_svg() + '\n')
    (HERE / 'pumpkin_mark_dark.svg').write_text(mark_svg(ring=INK_DARK) + '\n')
    print('pumpkin_mark.svg, pumpkin_mark_dark.svg')
    if a.pdf:
        write_pdfs(a.font)
