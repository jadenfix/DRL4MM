"""Stage 1: download + verify + convert every planned instrument-day (sequential, resumable)."""
import json
import shutil
import sys
import time
from pathlib import Path

from .config import SYMBOLS, days
from .data import convert_day

MIN_FREE_BYTES = 700 * 1024 * 1024


def fetch_all(out_dir, symbols=SYMBOLS, log=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for day, week, split in days():
        for sym in symbols:
            while shutil.disk_usage(out_dir).free < MIN_FREE_BYTES:
                log(f'waiting: less than {MIN_FREE_BYTES >> 20} MB free disk')
                time.sleep(60)
            meta = convert_day(sym, day, out_dir)
            meta.update(week=week, split=split)
            manifest[f'{sym}_{day}'] = meta
            log(f"{sym} {day} {split:5s} tick={meta['tick']} book_rows={meta['book_rows']} "
                f"trades={meta['trade_rows']} {meta['convert_secs']}s")
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == '__main__':
    fetch_all(sys.argv[1], log=lambda m: print(m, flush=True))
