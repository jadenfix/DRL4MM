"""Streaming zip inflation, off-tick snapping and the paired statistics (no network)."""
import io
import zipfile
import zlib

import numpy as np

from mmv2.data import _ZipStream, snap
from mmv2.stats import block_bootstrap, paired, sign_flip_by_cluster


def test_zip_stream_inflates_in_chunks():
    text = ''.join(f'{i},{i * 0.1:.1f},1.0\n' for i in range(20000)).encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('x.csv', text)
    raw = buf.getvalue()
    zs = _ZipStream()
    out = b''.join(zs.feed(raw[i:i + 997]) for i in range(0, len(raw), 997))
    assert out == text and zs.done and zs.crc == zlib.crc32(text)


def test_snap_moves_off_grid_prints_away_from_the_touch():
    p = np.array([29699.9, 29699.92, 29700.0])
    bids, n = snap(p, 0.1, np.floor)
    asks, _ = snap(p, 0.1, np.ceil)
    assert n == 1 and bids.tolist() == [296999, 296999, 297000] and asks.tolist() == [296999, 297000, 297000]


def test_sign_flip_is_exact_and_bootstrap_brackets_the_mean():
    groups = [np.array([1.0, 2.0]), np.array([1.5]), np.array([0.5, 1.0]), np.array([2.0]), np.array([1.0]), np.array([3.0])]
    assert sign_flip_by_cluster(groups) == 2 / 64  # all six day means positive: only the two extreme flips tie
    pt, lo, hi = block_bootstrap(groups, np.mean)
    assert lo <= pt <= hi


def test_paired_matches_episodes_by_key():
    mk = lambda pol, pnl, d, s: {'policy': pol, 'pnl': pnl, 'symbol': 'X', 'day': d, 'start_bin': s}  # noqa: E731
    a = [mk('a', 1.0 + i, f'd{i % 3}', i) for i in range(9)]
    b = [mk('b', 0.5 + i, f'd{i % 3}', i) for i in range(9)]
    t = paired(a, b)
    assert t['n_episodes'] == 9 and t['n_clusters'] == 3 and abs(t['diff_mean'] - 0.5) < 1e-12 and t['episodes_better'] == 1.0
