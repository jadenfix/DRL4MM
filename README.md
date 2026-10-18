# DRL4MM

Market-making research with a FIFO order book and reinforcement-learning quoting
agents. Branch `v2` adds a reproducible pipeline on three weeks of
real Binance order flow across four instruments, a queue-position replay, tuned
baselines and a one-shot held-out test with confidence intervals. The original
(v1) event-replay simulator and TD3 code are unchanged and documented further
down.

## v2 in one command

```sh
python -m pip install torch numpy pandas scipy pytest
python -m mmv2 all --data data/v2 --out runs/v2
```

`all` downloads and verifies the data, fits and tunes the baselines, trains the
final agent (five seeds), freezes it, evaluates the held-out test days once and
writes `runs/v2/results.md`. Stages can be run one at a time (`fetch`,
`baselines`, `ablate`, `train`, `select`, `test`, `report`; see
`python -m mmv2 --help`). It trains one process at a time on four threads and
needs about 0.6 GB of disk for the converted data; nothing raw is kept.

## v2 result, stated plainly

**The v2 agent does not beat the best baseline.** On the held-out test days
(6 per instrument, 144 hourly episodes each) no instrument shows a significant
difference between the agent and the baseline chosen on validation, and the
agent trails, again without significance, the baseline that turned out best on
the test days on every instrument. Pooled over the four instruments the agent
lost -1.17 USDT per hour [-2.98, +0.37] quoting ~1,000 USDT per side, against -1.67 [-4.80, +0.98]
for the validation-chosen baselines (difference +0.50 [-0.17, +1.36], sign-flip p = 0.34).
Every policy in this replay is close to break-even or losing; the only clear
ranking is that the v2 agent improves on the previous TD3 (pooled +9.19 USDT/h,
95% CI [+1.32, +19.64], HAC p = 0.005, day-level sign-flip p = 0.125), and that
averaging five seeds beats every single seed (members ranged from -2.06 to
-10.86 USDT/h on the test days).

| instrument | agent PnL / h [95% CI] | best baseline on validation | its PnL / h | agent - it [95% CI], sign-flip p | best baseline on test (hindsight) | agent - it [95% CI], p |
|---|---|---|---|---|---|---|
| BTCUSDT | -1.06 [-2.86, +0.39] | best fixed half-spread (mid) | +0.37 [-2.36, +2.34] | -1.43 [-3.40, +0.20], 0.250 | repo fixed-midpoint 10 bps | -1.46 [-3.38, +0.13], 0.250 |
| ETHUSDT | -1.01 [-3.28, +0.39] | best fixed half-spread (mid) | -5.23 [-13.10, +0.06] | +4.22 [-0.26, +10.03], 0.250 | Avellaneda-Stoikov | -0.98 [-3.11, +0.33], 0.750 |
| LTCUSDT | -0.75 [-3.40, +1.14] | best fixed half-spread (mid) | +2.55 [-0.45, +6.14] | -3.29 [-8.25, +1.36], 0.344 | repo fixed-midpoint 10 bps | -3.35 [-8.32, +1.10], 0.344 |
| SOLUSDT (never seen by the agent) | -1.86 [-8.08, +3.49] | best fixed half-spread (mid) | -4.38 [-11.15, +1.45] | +2.52 [+0.62, +4.39], 0.125 | repo fixed-midpoint 10 bps | -2.78 [-7.21, +0.05], 0.156 |

PnL in USDT per hourly episode, inventory flattened at the end; 95% day-block bootstrap intervals; exact sign-flip test over the 6 test days (smallest attainable p = 0.031). Most of each policy's PnL comes from a few fills during price spikes (the tuned Avellaneda-Stoikov quotes barely trade at all), so hourly PnL is noisy and the intervals are wide.

The full report, including per-policy drawdowns, fee sensitivity, the
ablations and the baseline tuning tables, is
[`results/v2/results.md`](results/v2/results.md).

## Data

Binance USD-M perpetual futures public data from
[data.binance.vision](https://data.binance.vision) (`bookTicker` and
`aggTrades` daily archives), licensed CC BY-NC-SA 4.0 by Binance. Each archive
is streamed over HTTPS, checked against its published SHA-256, inflated in
memory and reduced to one compact NPZ per instrument-day: the top of book on a
100 ms grid (last update in each bin) and every aggregated trade summed per
(bin, aggressor side, price).

Instruments: BTCUSDT, ETHUSDT, LTCUSDT, SOLUSDT. Binance publishes futures
`bookTicker` archives only for 2023-05-16 to 2024-03-30; within that period the
three weeks were picked from daily high/low ranges before any modelling:

| week | days | regime (mean daily log high/low, 4 instruments) |
|---|---|---|
| calm | 2023-08-08 .. 08-14 | 2.4%, the calmest 7-day window |
| median | 2023-11-21 .. 11-27 | 5.0%, the median window |
| volatile | 2024-03-14 .. 03-20 | 10.3%, the most volatile window |

Inside every week the split is chronological: days 1-4 train, day 5
validation, days 6-7 test. BTC, ETH and LTC train the agent; SOL is never used
to train or select it (its train/validation days only fit and tune SOL's
baselines, which makes them stronger, not the agent). In total 84 instrument-days: 786M bookTicker rows and 87M aggTrades, kept as 0.53 GB of NPZ; 2883 off-grid prints (for example a 0.001 BTC print at 29699.92 on a 0.1 grid) were moved one tick away from the touch and counted. Per-day row counts and checksums are in `results/v2/data_manifest.json`.

## Simulator (mmv2/sim.py)

A queue-position replay for one passive market maker quoting about 1,000 USDT
per side (converted at each day's first mid):

- one decision per second from state up to the end of the previous 100 ms
  bin; new quotes reach the book 100 ms later (the old ones stay live meanwhile);
- orders are post-only and join the back of their level: queue ahead = the
  displayed size at the touch, 0 inside the spread, unknown behind the touch
  (a level becomes visible with everything displayed ahead of us);
- recorded trades at our price first consume the queue ahead, trades through
  our price or a recorded opposite touch crossing it fill us in full;
  cancellations are assumed to come from behind us first;
- re-sending an unchanged price keeps queue priority;
- inventory is capped at five quotes; the episode (a clock hour) ends by
  crossing the spread to flatten. No fees in the primary numbers (VIP0 fees are
  reported as a sensitivity), no market impact.

These are deliberately conservative fill assumptions. Under them, quoting at
the touch is unprofitable for every policy (5 s markouts of about -1 bps per
fill), so the economics come from wide quotes that are hit during short spikes.

## Baselines (tuned on validation days only)

- Best fixed symmetric half-spread around the mid from a 13-point grid (0-30 bps).
- Avellaneda-Stoikov with inventory skew: sigma and the fill-intensity decay k
  fitted on the instrument's training days; gamma (9 values) and the horizon
  (time left in the episode, or a constant 30/120/600 s) tuned on validation.
- The repo's fixed-midpoint policy (`simulator/replay.py`, 10 bps).
- The previous TD3 actor (v1), run through an adapter that rebuilds its
  52-value observation.

## Agent (mmv2/rl.py)

Five SAC seeds (actor 64x64, twin critics 128x128, 600k one-second steps each,
episodes drawn at random from the BTC/ETH/LTC training days, checkpoint every
50k steps scored on half of the validation hours), with actions averaged across
seeds at run time. Reward per second, in bps of the quote notional: each new
fill credited with its markout against the mid 30 s later, minus 0.002 x
(inventory in quotes)^2, plus the liquidation cost on the last step. Rewards may
look ahead (they are only used for learning); observations never do.

Observation: 16 causal market features (spread in ticks and bps, tick size,
queue imbalance at the touch, order-flow imbalance over 1 s and 5 s, trade-flow
imbalance over 1 s and 10 s, traded volume, realized volatility over 10 s and
60 s, mid returns over 1/10/60 s, book update rate, touch depth) and 9 agent
features (inventory, time since last fill, time left, and for each side whether
an order rests, its distance from the touch and its queue position), z-scored
with training-day statistics. Action: a half-spread and a skew around the
microprice.

Ablations (one seed each, 400k steps, scored on the full validation days):

| run (1 seed, 400k steps) | settings | val PnL / h (BTC, ETH, LTC pooled) | BTC | ETH | LTC | fills / h | mean abs inv |
|---|---|---|---|---|---|---|---|
| td3_full | `{}` | -5.34 | -3.86 | -3.48 | -8.66 | 59.5 | 3.75 |
| td3_markout_reward | `{"reward": "markout", "eta": 0.002}` | -1.66 | -1.48 | -0.44 | -3.05 | 22.7 | 1.76 |
| sac_markout_reward | `{"algo": "sac", "reward": "markout", "eta": 0.002}` | +0.08 | +1.66 | -0.35 | -1.08 | 5.8 | 0.33 |
| sac_full | `{"algo": "sac"}` | -3.44 | -2.51 | -3.69 | -4.13 | 25.5 | 0.88 |
| sac_residual_markout | `{"algo": "sac", "action": "residual", "reward": "markout", "eta": 0.002}` | -0.81 | +0.26 | +1.29 | -3.97 | 11.9 | 0.61 |
| td3_residual_markout | `{"action": "residual", "reward": "markout", "eta": 0.002}` | -1.70 | -0.34 | -1.02 | -3.73 | 5.7 | 0.74 |
| td3_residual_pnl | `{"action": "residual", "eta": 0.002}` | -3.10 | -1.47 | -3.18 | -4.64 | 34.4 | 1.77 |
| sac_markout_mid_anchor | `{"algo": "sac", "reward": "markout", "eta": 0.002, "anchor": "mid"}` | -1.17 | -0.25 | -0.88 | -2.39 | 10.4 | 0.58 |
| sac_markout_basic_features | `{"algo": "sac", "reward": "markout", "eta": 0.002, "features": "basic"}` | -2.03 | -2.62 | -0.08 | -3.39 | 10.9 | 0.69 |
| sac_markout_eta01 | `{"algo": "sac", "reward": "markout", "eta": 0.01}` | -0.53 | +0.06 | +0.07 | -1.72 | 8.7 | 0.28 |

What mattered on validation: crediting fills with their markout instead of the raw mark-to-market change (TD3 -5.3 -> -1.7, SAC -3.4 -> +0.1 USDT/h), SAC over TD3, the full feature set over spread/volatility only (+0.1 vs -2.0), and anchoring on the microprice rather than the mid (+0.1 vs -1.2). A residual action around the tuned 7.5 bps half-spread and a larger inventory penalty did not help. All of these differences are within about one standard error (~0.6 USDT/h) of the validation mean.

## Evaluation protocol

- The test days were evaluated once, after the agent and all baseline
  parameters were frozen (`runs/v2/test_episodes.json` records the SHA-256 of
  everything evaluated; `python -m mmv2 test` refuses to run twice).
- The best baseline per instrument is the one with the highest validation PnL;
  the report also shows the comparison against whichever baseline did best on
  the test days.
- Intervals: day-block bootstrap (10,000 resamples of whole days). Paired tests:
  exact sign-flip over days (6 per instrument; smallest attainable two-sided
  p = 0.031) and a Newey-West t-test on hourly differences.
- No look-ahead: `tests/test_v2_lookahead.py` perturbs every book update and
  trade from a decision boundary on and checks that every feature, quote and
  fill before it is unchanged, for every policy and both latency settings; a
  deliberate one-bin peek makes it fail.

## Limitations

- Top-of-book data only: queue positions behind the touch are unknown and
  handled pessimistically; real queue dynamics and hidden liquidity are not
  modelled, nor is our impact on other traders.
- Three weeks and four instruments: six test days per instrument give coarse
  day-level tests.
- No fees, funding or exchange connectivity; not evidence of live performance.

## Verify

```sh
python -m pytest tests/test_v2_sim.py tests/test_v2_lookahead.py tests/test_v2_data_stats.py -q
```

`mmv2/export_site.py` writes the frozen agent, each instrument's best baseline,
two 5-minute held-out slices per instrument and the test numbers for the
browser demo on the author's site, plus Python reference traces; a TypeScript
port of the simulator and policies reproduces every quote, fill and cash value
on those slices exactly (`results/v2/site_parity.txt`).

## Data credits

Market data: Binance, [data.binance.vision](https://data.binance.vision),
CC BY-NC-SA 4.0 (attribution, non-commercial, share-alike). The converted NPZ
files are derived data under the same licence and are not committed; the
pipeline re-creates them from the public archives.

---

# v1: event replay of your own CSV order events

The original (v1) local historical replay path consumes your own CSV order
events, places quotes before each next event, and exports fills plus reconciled
cash, inventory, and PnL.

## Replay demo

![Rendered simulator replay using a labeled software fixture](demos/replay-demo.gif)

[MP4 video](demos/replay-demo.mp4) · [Reproduce and verify this capture](demos/README.md).
This visualizes actual CLI output from a synthetic software fixture; it is not
live market data or evidence of trading performance.

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

### Training budgets and checkpoints

`training.total_timesteps` counts actual training actions. Short historical CSVs
are replayed as separate episodes until that budget is reached; the final
episode is capped to the remaining budget. This budget cutoff retains the
environment's terminal flag rather than treating truncation as a terminal TD3
transition.

`training.eval_frequency` and `agent.save_frequency` are positive integer
action-step intervals. `training.save_frequency`, when supplied, overrides the
legacy agent setting. Evaluation and periodic saving occur at episode
boundaries once the corresponding step threshold is crossed. One boundary
produces one checkpoint, even if its episode crossed several intervals.
Periodic files use `checkpoint_steps_<actual steps>.pth`; the final model is
always saved. Budgets smaller than one episode still train and save that final
model. Zero, negative, fractional, and boolean schedule values are rejected.

`--resume` restores online networks, optimizers, and agent iteration counters;
the requested step budget applies to the new invocation. Replay data, random
states, and lagged target networks are not restored, so this is not exact
continued-training recovery. Evaluation currently replays the configured CSV,
not a separate held-out dataset. Software fixture checks do not establish
market performance.

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

