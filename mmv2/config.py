"""Dataset plan and splits (fixed before any model was trained).

Three 7-day windows chosen from daily high/low ranges of the four perpetuals over the period
for which Binance publishes USD-M ``bookTicker`` archives (2023-05-16 .. 2024-03-30):
the calmest week, a median-volatility week and the most volatile week. Inside every week the
split is chronological: days 1-4 train, day 5 validation, days 6-7 test. SOLUSDT is never used
to train or select the agent; its train/validation days are only used to fit and tune the
baselines (which makes the baselines stronger on SOL, not the agent).
"""
import datetime as _dt

WEEKS = {
    'calm': '2023-08-08',      # mean daily log(high/low) 2.4%, the calmest 7-day window
    'median': '2023-11-21',    # 5.0%, median window
    'volatile': '2024-03-14',  # 10.3%, the most volatile window (BTC ATH then -17%)
}
TRAIN_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'LTCUSDT']
HELDOUT_SYMBOLS = ['SOLUSDT']
SYMBOLS = TRAIN_SYMBOLS + HELDOUT_SYMBOLS
SPLIT_OF_DAY_INDEX = ['train'] * 4 + ['val'] + ['test'] * 2


def days():
    """[(day, week, split)] in calendar order."""
    out = []
    for week, start in WEEKS.items():
        d0 = _dt.date.fromisoformat(start)
        for i, split in enumerate(SPLIT_OF_DAY_INDEX):
            out.append(((d0 + _dt.timedelta(days=i)).isoformat(), week, split))
    return out


def split_days(split):
    return [d for d, _, s in days() if s == split]


# Trading setup shared by every policy
DECISION_BINS = 10          # one decision per second (bins are 100 ms)
EPISODE_SECONDS = 3600      # episodes are clock hours
QUOTE_NOTIONAL = 1000.0     # USDT per quote, converted to base units at each day's first mid
MAX_INV_QUOTES = 5          # a side is not quoted if a fill could push |inventory| past 5 quotes
MAKER_FEE_BPS = 0.0         # primary results; fee sensitivity reported separately
TAKER_FEE_BPS = 0.0
MARKOUT_SECONDS = 5
