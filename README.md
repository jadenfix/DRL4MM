# DRL4MM

Market-making research with a FIFO order book and a TD3 agent. The local
historical replay path consumes your own CSV order events, places quotes before
each next event, and exports fills plus reconciled cash, inventory, and PnL.

## Replay supplied market events

For replay alone, install the lightweight dependencies:

```sh
python -m pip install numpy pandas gym==0.26.2 python-dotenv
python -m simulator.replay /path/to/orders.csv --output runs/replay.json \
  --symbol AAPL --quote-bps 10 --quote-size 1
```

Each CSV represents one instrument. `--symbol` sets the book's metadata; it does
not filter mixed-instrument data. The CSV header is:

```text
timestamp,event,order_id,side,price,quantity
```

- `timestamp`: nonnegative numeric seconds, in chronological input order.
  Equal timestamps keep their original row order.
- `event`: `add` for a limit order, `market` for an immediate market order,
  or `cancel` for a previously active external order.
- `order_id`: nonempty external identifier, reused by its cancellation.
- `side`: `buy` or `sell` for order events; blank for cancellations.
- `price`: positive finite limit price for `add`; blank for other events.
- `quantity`: positive integer shares for orders; blank for cancellations.

Start with events establishing both sides of the book, then include later
events to replay. Export provider data into this format; the project does not
bundle or invent historical market data. Invalid rows, backward timestamps,
duplicate active IDs, and cancellations of inactive orders fail explicitly.
Inputs are limited to 64 MiB and 100,000 events. External IDs are namespaced so
they cannot impersonate agent orders.

The JSON records input SHA-256, the fixed quote-policy parameters, each step,
executed fills, and final accounting. `complete: false` and an unsuccessful CLI
exit identify runs stopped before all events are consumed, including inventory
or step limits. The CLI refuses to overwrite its source CSV.

## Use the data in the RL environment

```python
from simulator.market_env import MarketMakingEnv

env = MarketMakingEnv(data_source="historical", config={
    "historical_data_path": "/path/to/orders.csv",
    "quote_size": 1,
    "latency_mean": 0.0,
    "latency_std": 0.0,
})
observation = env.reset()
```

Training/evaluation factories read `data.source` and `data.historical_data_path`
from `config/td3_config.yaml`. Set your file path there before running them.
The default now requests historical input rather than silently running an
empty synthetic book. Install `requirements.txt` for TD3 training; optional
Weights & Biases is imported only when explicitly enabled.

## Verify

```sh
python -m pip install pytest
python -m pytest tests/test_lob.py tests/test_accounting.py tests/test_historical_replay.py -q
```

These tests exercise actual matching, chronological environment steps, reset,
CLI execution, data rejection, and cash/PnL reconciliation using explicitly
small software fixtures. Fixture results are not trading-performance evidence.
The full `tests/` suite also requires Torch.

## Current scope

This is a book simulation with immediate order placement. The replay CLI uses
zero latency; simulator latency settings only shift order timestamps and do not
delay execution. Fees, queue models calibrated to an exchange, live data,
exchange connectivity, and a validated trained trading policy are not provided.
Unfilled market-order quantity is discarded. The legacy synthetic initializer
remains unfinished; live reset explicitly rejects the unavailable adapter.
Review the separate TD3 target-shape and order-integrity PRs alongside the replay
and accounting changes before relying on training results.

