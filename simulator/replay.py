"""Run a fixed quote policy on supplied CSV events; no model or network calls."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .market_env import MarketMakingEnv


def _input_hash(path):
    """Hash bounded input incrementally, including when a writer grows it."""
    digest = hashlib.sha256()
    total = 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            total += len(chunk)
            if total > 64 * 1024 * 1024:
                raise ValueError('Replay CSV exceeds the 64 MiB input limit')
            digest.update(chunk)
    return digest.hexdigest()


def run_replay(path, *, symbol='AAPL', quote_bps=10.0, quote_size=1, max_steps=100000):
    """Export event-by-event accounting, with input identity and explicit policy."""
    if not math.isfinite(quote_bps) or not 0 < quote_bps <= 1000:
        raise ValueError('quote_bps must be finite and between 0 and 1000')
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError('max_steps must be a positive integer')
    path = Path(path)
    before = _input_hash(path)
    env = MarketMakingEnv(symbol=symbol, data_source='historical', config={
        'historical_data_path': str(path), 'quote_size': quote_size,
        'max_episode_steps': max_steps, 'latency_mean': 0.0, 'latency_std': 0.0,
    })
    initial = env.reset()
    if not np.isfinite(initial).all():
        raise ValueError('Initial observation is not finite')
    rows, fills = [], []
    done = False
    seen = 0
    while not done:
        observation, reward, done, info = env.step(
            np.array([-quote_bps / 10000, quote_bps / 10000], dtype=np.float32)
        )
        if not np.isfinite(observation).all() or not math.isfinite(reward):
            raise ValueError('Replay produced a nonfinite observation or reward')
        pending = env.fill_count - seen
        if pending > len(env.recent_fills):
            raise ValueError('Fill export exceeded retained history for one step')
        if pending:
            for fill in list(env.recent_fills)[-pending:]:
                fills.append({
                    'timestamp': fill.timestamp, 'price': fill.price,
                    'quantity': fill.quantity,
                    'side': 'buy' if fill.buyer_id.startswith('agent_') else 'sell',
                    'buyer_order': fill.buyer_id, 'seller_order': fill.seller_id,
                })
        seen = env.fill_count
        rows.append({**info, 'reward': reward})
    if _input_hash(path) != before:
        raise ValueError('Source CSV changed during replay; results discarded')
    return {
        'mode': 'historical-order-event-replay', 'input_sha256': before,
        'symbol': symbol, 'policy': {'name': 'fixed-midpoint-quotes',
        'quote_bps': quote_bps, 'quote_size': quote_size},
        'episode_limits': {'max_steps': max_steps, 'max_inventory': env.max_inventory},
        'events_consumed': env._historical_cursor, 'events_total': len(env.market_data),
        'complete': env._historical_cursor == len(env.market_data),
        'rows': rows, 'fills': fills, 'final': rows[-1],
        'limits': 'Book simulation only: no exchange connection, fees, latency or trained policy.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--symbol', default='AAPL')
    parser.add_argument('--quote-bps', type=float, default=10.0)
    parser.add_argument('--quote-size', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=100000)
    args = parser.parse_args()
    if args.output.resolve() == args.csv.resolve():
        parser.error('output must differ from the source CSV')
    try:
        result = run_replay(args.csv, symbol=args.symbol, quote_bps=args.quote_bps,
                            quote_size=args.quote_size, max_steps=args.max_steps)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f'Replay failed: {error}\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"{len(result['rows'])} steps, {len(result['fills'])} fills; complete={result['complete']}")
    print(f"Saved {args.output}")
    if not result['complete']:
        parser.exit(2, 'Replay ended before all input events were consumed\n')


if __name__ == '__main__':
    main()
