"""Render actual replay JSON as labeled software-fixture media; no market claims."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '36a6bc451e600243611ae67f31c842fc3d5a197d'
W, H, FPS, SECONDS = 1280, 720, 12, 15
BG, PANEL, INK, MUTED = '#0b1020', '#141e32', '#eef4ff', '#91a4c4'
TEAL, BLUE, AMBER = '#63efd1', '#7aabff', '#ffd088'


def font(size, bold=False):
    """Use a freely packaged system font, failing clearly if unavailable."""
    suffix = '-Bold' if bold else ''
    path = Path('/usr/share/fonts/truetype/dejavu') / f'DejaVuSans{suffix}.ttf'
    return ImageFont.truetype(str(path), size)


def text(draw, xy, value, size=20, color=INK, bold=False):
    """Draw consistent screen typography."""
    draw.text(xy, str(value), font=font(size, bold), fill=color)


def chart(draw, box, rows, key, title, color, bounds):
    """Plot only observed rows, with fixed axes over the full replay."""
    x, y, width, height = box
    draw.rounded_rectangle((x, y, x + width, y + height), 14, fill=PANEL)
    text(draw, (x + 22, y + 16), title, 18, MUTED)
    left, top, right, bottom = x + 65, y + 54, x + width - 24, y + height - 38
    lo, hi = bounds
    for i in range(4):
        yy = top + (bottom - top) * i / 3
        draw.line((left, yy, right, yy), fill='#27354f')
        text(draw, (x + 15, yy - 8), f'{hi-(hi-lo)*i/3:.1f}', 13, MUTED)
    points = [(left + (right-left)*(r['episode_steps']-1)/35,
               bottom - (bottom-top)*(r[key]-lo)/(hi-lo)) for r in rows]
    if len(points) > 1:
        draw.line(points, fill=color, width=3)
    if points:
        px, py = points[-1]
        draw.ellipse((px-5, py-5, px+5, py+5), fill=color)
    text(draw, (left, bottom+12), 'Event 1', 12, MUTED)
    text(draw, (right-70, bottom+12), 'Event 36', 12, MUTED)


def frame(report, index):
    """Present one exact simulator state; presentation time is slowed down."""
    image = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(image)
    rows = report['rows'][:index+1]
    row = rows[-1]
    text(d, (40, 27), 'DRL4MM', 34, INK, True)
    text(d, (253, 39), 'ORDER EVENT REPLAY', 17, TEAL, True)
    d.rounded_rectangle((847, 29, 1240, 66), 10, fill='#2e2a24')
    text(d, (861, 37), 'SOFTWARE FIXTURE / NO LIVE MARKET', 15, AMBER, True)
    text(d, (40, 86), 'Follow the fills. Reconcile the book.', 29, INK, True)
    text(d, (40, 129), 'CSV events  >  fixed quotes  >  FIFO matching  >  accounting JSON', 18, MUTED)
    for x, name, value, unit in [(40, 'EVENTS CONSUMED', f'{index+3}/38', '2 initial book events'),
                                (346, 'EXECUTED FILLS', row['fill_count'], 'software fixture shares'),
                                (652, 'INVENTORY', row['inventory'], 'shares held'),
                                (958, 'RECONCILED EQUITY', f"{row['equity']:.2f}", 'fixture price units')]:
        d.rounded_rectangle((x, 177, x+282, 290), 12, fill=PANEL)
        text(d, (x+18, 192), name, 13, MUTED, True)
        text(d, (x+18, 215), value, 31, TEAL if name=='INVENTORY' else INK, True)
        text(d, (x+18, 260), unit, 12, MUTED)
    chart(d, (40, 310, 585, 223), rows, 'inventory', 'Inventory after each event', TEAL, (-1, 3))
    chart(d, (645, 310, 595, 223), rows, 'equity', 'Marked equity (fixture units)', BLUE, (-1, 5))
    d.rounded_rectangle((40, 552, 1200+40, 661), 12, fill=PANEL)
    text(d, (58, 565), 'LATEST EXECUTION', 13, MUTED, True)
    fill = report['fills'][index]
    text(d, (58, 590), f"{fill['side'].upper()} {fill['quantity']} @ {fill['price']:.2f}", 23, TEAL, True)
    text(d, (352, 568), 'Cash + marked position = equity', 17, MUTED)
    text(d, (352, 596), f"{row['cash']:.2f} + {row['position_value']:.2f} = {row['equity']:.2f}", 22, INK, True)
    text(d, (886, 568), 'INPUT VERIFIED', 13, MUTED, True)
    text(d, (886, 593), 'SHA-256 '+report['input_sha256'][:14], 16, MUTED)
    text(d, (40, 677), 'No exchange, fees, calibrated latency, or trained-policy performance. Animation slows fixture replay.', 13, MUTED)
    text(d, (1022, 677), 'Source '+SOURCE[:12], 12, MUTED)
    return image


def main():
    """Execute real CLI, reconcile output, then encode GIF and H.264 video."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'demos')
    args = parser.parse_args()
    if not shutil.which('ffmpeg'):
        parser.error('ffmpeg is required')
    manifest = json.loads((ROOT/'demos/source-manifest.json').read_text())
    if manifest['source_commit'] != SOURCE:
        parser.error('source label differs from manifest commit')
    for relative, expected in manifest['git_blob_sha1'].items():
        data = (ROOT/relative).read_bytes()
        actual = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if actual != expected:
            parser.error(f'{relative} differs from recorded source; refresh provenance before rendering')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    result = out/'replay.json'
    subprocess.run([sys.executable, '-m', 'simulator.replay',
                    str(ROOT/'demos/software-fixture.csv'), '--output', str(result)],
                   cwd=ROOT, check=True)
    report = json.loads(result.read_text())
    assert report['complete'] and report['events_consumed'] == 38
    assert len(report['rows']) == len(report['fills']) == 36
    for row in report['rows']:
        assert abs(row['equity']-row['cash']-row['position_value']) < 1e-8
        assert abs(row['equity']-row['realized_pnl']-row['unrealized_pnl']) < 1e-8
    with tempfile.TemporaryDirectory(prefix='drl-demo-') as temp:
        directory = Path(temp)
        for n in range(FPS*SECONDS):
            # Hold first/last state; no interpolated or invented measurements.
            index = max(0, min(35, int((n/FPS-1)*36/(SECONDS-3))))
            frame(report, index).save(directory/f'{n:04d}.png')
        common = ['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                  '-i', str(directory/'%04d.png')]
        subprocess.run(common+['-c:v', 'libx264', '-crf', '22', '-pix_fmt', 'yuv420p',
                              '-movflags', '+faststart', str(out/'replay-demo.mp4')], check=True)
        subprocess.run(common+['-filter_complex',
                              'fps=8,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse',
                              '-loop', '0', str(out/'replay-demo.gif')], check=True)
        frame(report, 35).save(out/'replay-preview.png')
    print('Rendered real simulator output: replay-demo.mp4 and replay-demo.gif')


if __name__ == '__main__':
    main()
