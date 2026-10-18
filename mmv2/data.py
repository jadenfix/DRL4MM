"""Download, verify and convert Binance USD-M futures public data, one instrument-day at a time.

Source: https://data.binance.vision (daily ``bookTicker`` and ``aggTrades`` archives),
licensed CC BY-NC-SA 4.0 by Binance. Nothing raw is written to disk: each zip is streamed
over HTTPS, hashed with SHA-256 on the fly (compared with the published ``.CHECKSUM``),
inflated in memory and parsed in ~32 MB text blocks. Only a compact per-day NPZ is kept:

* the top of book on a 100 ms grid: the last ``bookTicker`` row (by ``update_id``) whose
  ``transaction_time`` falls in ``[t, t + 100 ms)`` is the state at the end of that bin;
  bins without an update carry the previous state forward;
* every ``aggTrade`` aggregated per (bin, aggressor side, price): summed quantity and count.

Prices are stored as integer ticks relative to ``meta.base_ticks``; quantities in base units.
A day that fails its checksum is discarded.
"""
from __future__ import annotations

import hashlib
import io
import json
import struct
import time
import urllib.request
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

BIN_MS = 100
BINS_PER_DAY = 86_400_000 // BIN_MS
URL = 'https://data.binance.vision/data/futures/um/daily/{kind}/{sym}/{sym}-{kind}-{day}.zip'
LICENSE = 'Binance public market data (data.binance.vision), CC BY-NC-SA 4.0'
BLOCK = 16 * 1024 * 1024
QTY_UNIT = 0.001  # every quantity on these contracts is a multiple of 0.001 base units


def _fetch(url, retries=4, timeout=120):
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(url, timeout=timeout)
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def published_sha256(url):
    with _fetch(url + '.CHECKSUM') as r:
        return r.read().decode().split()[0].strip().lower()


class _ZipStream:
    """Inflate the single member of a zip archive from a forward-only byte stream."""

    def __init__(self):
        self.buf = b''
        self.inflater = None
        self.done = False
        self.crc = 0
        self.expected_crc = None

    def feed(self, data):
        if self.done:
            return b''
        if self.inflater is None:
            self.buf += data
            if len(self.buf) < 30:
                return b''
            sig, _, flags, method, _, _, crc, _, _, nlen, xlen = struct.unpack('<IHHHHHIIIHH', self.buf[:30])
            if sig != 0x04034B50 or method != 8:
                raise ValueError('unexpected zip layout (need one deflated member)')
            if len(self.buf) < 30 + nlen + xlen:
                return b''
            self.expected_crc = None if flags & 0x08 else crc
            data, self.buf = self.buf[30 + nlen + xlen:], b''
            self.inflater = zlib.decompressobj(-15)
        out = self.inflater.decompress(data)
        if self.inflater.eof:
            self.done = True
        self.crc = zlib.crc32(out, self.crc)
        return out


def stream_csv_blocks(url, sha_out):
    """Yield decompressed CSV text blocks that end on line boundaries; record the zip SHA-256."""
    digest = hashlib.sha256()
    zs = _ZipStream()
    pending = b''
    with _fetch(url) as r:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            pending += zs.feed(chunk)
            if len(pending) >= BLOCK:
                cut = pending.rfind(b'\n') + 1
                yield pending[:cut]
                pending = pending[cut:]
    if not zs.done:
        raise ValueError('truncated archive: ' + url)
    if zs.expected_crc is not None and zs.expected_crc != zs.crc:
        raise ValueError('CRC mismatch: ' + url)
    if pending:
        yield pending
    sha_out.append(digest.hexdigest())


def _frame(block, names):
    first = block[:block.find(b'\n')]
    header = not first[:1].isdigit()
    return pd.read_csv(io.BytesIO(block), header=0 if header else None, names=names,
                       dtype={n: 'float64' for n in names})


def infer_tick(prices):
    u = np.unique(prices)
    d = np.diff(u)
    d = d[d > 0]
    raw = float(d.min())
    tick = float(f'{raw:.1g}')
    return tick


def snap(prices, tick, rounding):
    """Integer ticks; the rare off-grid print (e.g. 29699.92 on a 0.1 grid) is moved away from
    the touch (bids down, asks up) and counted."""
    t = np.rint(prices / tick)
    off = np.abs(prices - t * tick) > tick * 1e-6
    if off.any():
        t[off] = rounding(prices[off] / tick)
    return t, int(off.sum())


def convert_book(sym, day, day0_ms):
    names = ['update_id', 'bid', 'bid_qty', 'ask', 'ask_qty', 'ttime', 'etime']
    uid = np.full(BINS_PER_DAY, -1.0)
    bid = np.zeros(BINS_PER_DAY, np.int64)
    ask = np.zeros(BINS_PER_DAY, np.int64)
    bq = np.zeros(BINS_PER_DAY, np.float64)
    aq = np.zeros(BINS_PER_DAY, np.float64)
    upd = np.zeros(BINS_PER_DAY, np.int64)
    tick, rows, dropped, sha, off_tick = None, 0, 0, [], 0
    url = URL.format(kind='bookTicker', sym=sym, day=day)
    for block in stream_csv_blocks(url, sha):
        df = _frame(block, names)
        rows += len(df)
        if tick is None:
            tick = infer_tick(df['bid'].to_numpy())
        b = np.floor((df['ttime'].to_numpy() - day0_ms) / BIN_MS).astype(np.int64)
        ok = (b >= 0) & (b < BINS_PER_DAY)
        dropped += int((~ok).sum())
        b = b[ok]
        u = df['update_id'].to_numpy()[ok]
        bt, nb = snap(df['bid'].to_numpy()[ok], tick, np.floor)
        at, na = snap(df['ask'].to_numpy()[ok], tick, np.ceil)
        off_tick += nb + na
        np.add.at(upd, b, 1)
        order = np.lexsort((u, b))
        b, u = b[order], u[order]
        last = np.r_[b[1:] != b[:-1], True]
        b, u, sel = b[last], u[last], order[last]
        newer = u > uid[b]
        b, sel = b[newer], sel[newer]
        uid[b] = u[newer]
        bid[b] = bt[sel].astype(np.int64)
        ask[b] = at[sel].astype(np.int64)
        bq[b] = df['bid_qty'].to_numpy()[ok][sel]
        aq[b] = df['ask_qty'].to_numpy()[ok][sel]
        del df
    have = uid >= 0
    if not have.any():
        raise ValueError(f'{sym} {day}: no book rows')
    first = int(np.argmax(have))
    idx = np.where(have, np.arange(BINS_PER_DAY), 0)
    np.maximum.accumulate(idx, out=idx)
    idx[:first] = first
    bid, ask, bq, aq = bid[idx], ask[idx], bq[idx], aq[idx]
    crossed = int((ask <= bid).sum())
    if off_tick > 1e-3 * rows:
        raise ValueError(f'{sym} {day}: {off_tick} book prices off the inferred tick {tick}')
    return dict(off_tick=off_tick, tick=tick, bid=bid, ask=ask, bq=bq, aq=aq, upd=upd, rows=rows, dropped=dropped,
                first_bin=first, crossed_bins=crossed, sha=sha[0], url=url)


def convert_trades(sym, day, day0_ms, tick):
    names = ['agg_id', 'price', 'qty', 'first_id', 'last_id', 'ttime', 'buyer_maker']
    parts, rows, dropped, sha, off_tick = [], 0, 0, [], 0
    url = URL.format(kind='aggTrades', sym=sym, day=day)
    for block in stream_csv_blocks(url, sha):
        first = block[:block.find(b'\n')]
        header = not first[:1].isdigit()
        df = pd.read_csv(io.BytesIO(block), header=0 if header else None, names=names,
                         dtype={'price': 'float64', 'qty': 'float64', 'ttime': 'int64', 'buyer_maker': 'str'})
        rows += len(df)
        b = (df['ttime'].to_numpy() - day0_ms) // BIN_MS
        ok = (b >= 0) & (b < BINS_PER_DAY)
        dropped += int((~ok).sum())
        px = df['price'].to_numpy()[ok]
        maker = df['buyer_maker'].to_numpy()[ok]
        side = np.where(np.char.lower(maker.astype(str)) == 'true', -1, 1).astype(np.int64)  # -1: seller aggressor
        t_dn, n_dn = snap(px, tick, np.floor)
        t_up, n_up = snap(px, tick, np.ceil)
        t = np.where(side < 0, t_dn, t_up)
        off_tick += n_dn
        g = pd.DataFrame({'bin': b[ok], 'side': side, 'px': t.astype(np.int64), 'qty': df['qty'].to_numpy()[ok], 'n': 1})
        parts.append(g.groupby(['bin', 'side', 'px'], sort=False).agg(qty=('qty', 'sum'), n=('n', 'sum')).reset_index())
        del df
    g = pd.concat(parts).groupby(['bin', 'side', 'px']).agg(qty=('qty', 'sum'), n=('n', 'sum')).reset_index()
    if off_tick > 1e-3 * rows:
        raise ValueError(f'{sym} {day}: {off_tick} trade prices off the tick {tick}')
    return dict(off_tick=off_tick, bin=g['bin'].to_numpy(), side=g['side'].to_numpy(), px=g['px'].to_numpy(),
                qty=g['qty'].to_numpy(), n=g['n'].to_numpy(), rows=rows, dropped=dropped, sha=sha[0], url=url)


def convert_day(sym, day, out_dir):
    out_dir = Path(out_dir)
    out = out_dir / f'{sym}_{day}.npz'
    if out.exists():
        return json.loads(str(np.load(out)['meta']))
    t0 = time.time()
    day0_ms = int(pd.Timestamp(day, tz='UTC').value // 1_000_000)
    book = convert_book(sym, day, day0_ms)
    if book['sha'] != published_sha256(book['url']):
        raise ValueError(f'{sym} {day}: bookTicker checksum mismatch')
    tr = convert_trades(sym, day, day0_ms, book['tick'])
    if tr['sha'] != published_sha256(tr['url']):
        raise ValueError(f'{sym} {day}: aggTrades checksum mismatch')
    base = int(book['bid'][book['first_bin']])
    meta = {
        'symbol': sym, 'day': day, 'day0_ms': day0_ms, 'bin_ms': BIN_MS, 'tick': book['tick'],
        'base_ticks': base, 'book_rows': book['rows'], 'book_rows_outside_day': book['dropped'],
        'book_prices_off_tick': book['off_tick'], 'trade_prices_off_tick': tr['off_tick'],
        'first_book_bin': book['first_bin'], 'crossed_bins': book['crossed_bins'],
        'trade_rows': tr['rows'], 'trade_rows_outside_day': tr['dropped'], 'trade_groups': int(len(tr['bin'])),
        'sha256': {'bookTicker': book['sha'], 'aggTrades': tr['sha']},
        'source': [book['url'], tr['url']], 'license': LICENSE, 'convert_secs': round(time.time() - t0, 1),
    }
    def units(q):
        u = np.rint(q / QTY_UNIT)
        if np.abs(q - u * QTY_UNIT).max() > 1e-6 or u.max() >= 2**31:
            raise ValueError(f'{sym} {day}: quantities are not multiples of {QTY_UNIT}')
        return u.astype(np.int32)

    tmp = out_dir / f'.{sym}_{day}.tmp.npz'
    np.savez_compressed(
        tmp, meta=json.dumps(meta),
        bid=(book['bid'] - base).astype(np.int32), spread=(book['ask'] - book['bid']).astype(np.int32),
        bid_qty=units(book['bq']), ask_qty=units(book['aq']),
        upd=np.minimum(book['upd'], 255).astype(np.uint8),
        tr_bin=tr['bin'].astype(np.int32), tr_side=tr['side'].astype(np.int8),
        tr_px=(tr['px'] - base).astype(np.int32), tr_qty=units(tr['qty']), tr_n=np.minimum(tr['n'], 255).astype(np.uint8))
    tmp.rename(out)
    return meta


def load_day(path):
    """Load one converted day as plain numpy arrays (absolute ticks, float64 quantities)."""
    z = np.load(path)
    meta = json.loads(str(z['meta']))
    base = meta['base_ticks']
    bid = z['bid'].astype(np.int64) + base
    return {
        'meta': meta, 'bid': bid, 'ask': bid + z['spread'].astype(np.int64),
        'bid_qty': z['bid_qty'].astype(np.float64) * QTY_UNIT, 'ask_qty': z['ask_qty'].astype(np.float64) * QTY_UNIT,
        'upd': z['upd'].astype(np.int64),
        'tr_bin': z['tr_bin'].astype(np.int64), 'tr_side': z['tr_side'].astype(np.int64),
        'tr_px': z['tr_px'].astype(np.int64) + base, 'tr_qty': z['tr_qty'].astype(np.float64) * QTY_UNIT, 'tr_n': z['tr_n'].astype(np.int64),
    }
