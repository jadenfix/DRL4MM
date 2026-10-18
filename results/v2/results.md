# DRL4MM v2 results

## Verdict on the held-out test days

- BTCUSDT: the agent is not significantly different from the best baseline (best fixed half-spread (mid)); difference -1.427 USDT/h, 95% CI [-3.398, +0.202], sign-flip p = 0.250, HAC p = 0.183.
- ETHUSDT: the agent is not significantly different from the best baseline (best fixed half-spread (mid)); difference +4.219 USDT/h, 95% CI [-0.259, +10.025], sign-flip p = 0.250, HAC p = 0.119.
- LTCUSDT: the agent is not significantly different from the best baseline (best fixed half-spread (mid)); difference -3.293 USDT/h, 95% CI [-8.254, +1.359], sign-flip p = 0.344, HAC p = 0.134.
- SOLUSDT: the agent is not significantly different from the best baseline (best fixed half-spread (mid)); difference +2.520 USDT/h, 95% CI [+0.622, +4.390], sign-flip p = 0.125, HAC p = 0.369.
- Pooled over the four instruments (calendar-day clusters): difference +0.505 USDT/h, 95% CI [-0.170, +1.357], sign-flip p = 0.344.
- Pooled over the four instruments (instrument-day clusters): difference +0.505 USDT/h, 95% CI [-1.727, +2.848], sign-flip p = 0.682.

## Held-out test (each instrument: 6 days, 144 hourly episodes)

| instrument | policy | mean PnL / h (USDT) [95% CI] | Sharpe-like [95% CI] | max DD (cum.) | mean abs inv (quotes) | fills / h | 5 s markout (bps) | edge at fill (bps) | PnL / h at VIP0 fees |
|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | v2 agent (seed ensemble) | -1.056 [-2.863, +0.391] | -0.106 [-0.24, +0.12] | 218.98 | 0.32 | 1.5 | -0.02 | +4.04 | -1.637 |
| BTCUSDT | best fixed half-spread (mid) `fixed_mid_10bps` | +0.371 [-2.360, +2.337] | +0.032 [-0.18, +0.34] | 229.31 | 0.66 | 2.8 | -0.03 | +6.03 | -0.702 |
| BTCUSDT | Avellaneda-Stoikov `avellaneda_stoikov_g0.3_tau120s` | +0.011 [-0.091, +0.112] | +0.015 [-0.12, +0.19] | 6.02 | 0.00 | 0.8 | +0.42 | +6.65 | -0.141 |
| BTCUSDT | repo fixed-midpoint 10 bps `repo_fixed_midpoint_10bps` | +0.400 [-2.251, +2.320] | +0.033 [-0.17, +0.34] | 226.56 | 0.66 | 2.8 | -0.06 | +6.07 | -0.672 |
| BTCUSDT | previous TD3 (v1) `td3_v1_previous` | -0.672 [-5.200, +2.944] | -0.028 [-0.20, +0.11] | 380.55 | 3.19 | 11.4 | -0.59 | +1.06 | -4.510 |
| BTCUSDT | fixed half-spread around microprice (extra) `fixed_micro_10bps` | +0.424 [-1.931, +2.168] | +0.035 [-0.16, +0.34] | 222.05 | 0.67 | 2.8 | -0.14 | +6.09 | -0.652 |
| ETHUSDT | v2 agent (seed ensemble) | -1.010 [-3.284, +0.388] | -0.044 [-0.13, +0.04] | 356.08 | 0.50 | 2.9 | +0.08 | +4.73 | -2.020 |
| ETHUSDT | best fixed half-spread (mid) `fixed_mid_5bps` | -5.229 [-13.104, +0.063] | -0.173 [-0.30, +0.01] | 924.40 | 1.41 | 47.9 | -0.99 | +2.19 | -15.592 |
| ETHUSDT | Avellaneda-Stoikov `avellaneda_stoikov_g0.3_tau120s` | -0.030 [-0.230, +0.167] | -0.021 [-0.11, +0.18] | 14.40 | 0.00 | 1.4 | -0.78 | +7.11 | -0.304 |
| ETHUSDT | repo fixed-midpoint 10 bps `repo_fixed_midpoint_10bps` | -2.654 [-8.176, +0.404] | -0.102 [-0.22, +0.02] | 472.57 | 0.79 | 5.8 | -0.70 | +5.88 | -4.429 |
| ETHUSDT | previous TD3 (v1) `td3_v1_previous` | -2.950 [-7.117, -0.469] | -0.085 [-0.20, -0.02] | 491.62 | 3.48 | 16.1 | -0.88 | +1.74 | -7.663 |
| ETHUSDT | fixed half-spread around microprice (extra) `fixed_micro_5bps` | -5.193 [-13.081, +0.006] | -0.173 [-0.30, +0.00] | 922.33 | 1.41 | 47.8 | -0.97 | +2.20 | -15.548 |
| LTCUSDT | v2 agent (seed ensemble) | -0.746 [-3.404, +1.137] | -0.045 [-0.19, +0.12] | 299.53 | 0.45 | 3.4 | -0.49 | +3.16 | -1.685 |
| LTCUSDT | best fixed half-spread (mid) `fixed_mid_7.5bps` | +2.547 [-0.454, +6.139] | +0.157 [-0.08, +0.28] | 83.00 | 0.82 | 12.7 | -1.23 | +2.68 | -0.412 |
| LTCUSDT | Avellaneda-Stoikov `avellaneda_stoikov_g0.3` | -0.096 [-0.196, -0.016] | -0.306 [-0.44, -0.16] | 13.86 | 0.00 | 0.5 | -1.65 | +1.98 | -0.263 |
| LTCUSDT | repo fixed-midpoint 10 bps `repo_fixed_midpoint_10bps` | +2.601 [-0.258, +7.004] | +0.123 [-0.05, +0.23] | 144.17 | 0.64 | 4.9 | -1.25 | +4.26 | +1.160 |
| LTCUSDT | previous TD3 (v1) `td3_v1_previous` | -23.604 [-57.106, -1.626] | -0.354 [-0.58, -0.12] | 3418.56 | 3.24 | 195.4 | -1.55 | +0.81 | -57.472 |
| LTCUSDT | fixed half-spread around microprice (extra) `fixed_micro_7.5bps` | +3.159 [-0.404, +7.822] | +0.164 [-0.08, +0.28] | 70.96 | 0.84 | 12.2 | -1.21 | +2.73 | +0.262 |
| SOLUSDT | v2 agent (seed ensemble) | -1.861 [-8.085, +3.486] | -0.057 [-0.19, +0.43] | 851.79 | 1.12 | 9.9 | -0.20 | +5.32 | -4.590 |
| SOLUSDT | best fixed half-spread (mid) `fixed_mid_15bps` | -4.380 [-11.154, +1.452] | -0.137 [-0.25, +0.20] | 865.44 | 1.06 | 5.9 | -0.09 | +9.87 | -6.324 |
| SOLUSDT | Avellaneda-Stoikov `avellaneda_stoikov_g0.03_tau600s` | +0.137 [+0.014, +0.337] | +0.180 [+0.04, +0.28] | 3.31 | 0.00 | 0.5 | +3.20 | +17.66 | +0.046 |
| SOLUSDT | repo fixed-midpoint 10 bps `repo_fixed_midpoint_10bps` | +0.921 [-1.995, +4.196] | +0.027 [-0.04, +0.31] | 671.20 | 1.50 | 24.2 | -0.54 | +5.37 | -4.720 |
| SOLUSDT | previous TD3 (v1) `td3_v1_previous` | -14.196 [-27.237, -2.386] | -0.347 [-0.71, -0.06] | 2286.33 | 3.07 | 238.9 | -0.77 | +0.43 | -56.065 |
| SOLUSDT | fixed half-spread around microprice (extra) `fixed_micro_15bps` | -3.600 [-9.917, +1.447] | -0.112 [-0.22, +0.20] | 783.86 | 1.05 | 6.0 | -0.20 | +9.79 | -5.533 |
| ALL | v2 agent (seed ensemble) | -1.168 [-2.982, +0.368] | -0.053 [-0.11, +0.02] | 1291.46 | 0.60 | 4.4 | -0.30 | +4.55 | -2.483 |
| ALL | best fixed half-spread (mid) | -1.673 [-4.796, +0.985] | -0.069 [-0.15, +0.06] | 1448.81 | 0.99 | 17.3 | -0.95 | +4.25 | -5.757 |
| ALL | Avellaneda-Stoikov | +0.005 [-0.073, +0.089] | +0.006 [-0.08, +0.11] | 23.28 | 0.00 | 0.8 | +0.31 | +8.42 | -0.165 |
| ALL | repo fixed-midpoint 10 bps | +0.317 [-1.739, +2.246] | +0.013 [-0.06, +0.11] | 1140.97 | 0.90 | 9.4 | -0.72 | +5.10 | -2.165 |
| ALL | previous TD3 (v1) | -10.355 [-20.792, -3.079] | -0.229 [-0.35, -0.10] | 6211.36 | 3.25 | 115.5 | -1.04 | +0.56 | -31.428 |
| ALL | fixed half-spread around microprice (extra) | -1.303 [-4.372, +1.358] | -0.052 [-0.14, +0.08] | 1340.74 | 0.99 | 17.2 | -0.95 | +4.32 | -5.368 |
| ALL | best baseline (per instrument, chosen on validation) | -1.673 [-4.796, +0.985] | -0.069 [-0.15, +0.06] | 1448.81 | 0.99 | 17.3 | -0.95 | +4.25 | -5.757 |

| instrument | best baseline (validation) | agent - baseline, mean PnL / h [95% day-block CI] | sign-flip p (days) | HAC p | share of hours agent strictly ahead |
|---|---|---|---|---|---|
| BTCUSDT | best fixed half-spread (mid) | -1.427 [-3.398, +0.202] | 0.250 | 0.183 | 0.16 |
| ETHUSDT | best fixed half-spread (mid) | +4.219 [-0.259, +10.025] | 0.250 | 0.119 | 0.33 |
| LTCUSDT | best fixed half-spread (mid) | -3.293 [-8.254, +1.359] | 0.344 | 0.134 | 0.24 |
| SOLUSDT | best fixed half-spread (mid) | +2.520 [+0.622, +4.390] | 0.125 | 0.369 | 0.38 |
| ALL | pooled, clustered by calendar day | +0.505 [-0.170, +1.357] | 0.344 | 0.663 | 0.28 |
| ALL | pooled, clustered by instrument-day | +0.505 [-1.727, +2.848] | 0.682 | 0.663 | 0.28 |

Against whichever required baseline did best on the test days themselves (chosen ex post, which favours the baseline):

| instrument | best baseline (test, ex post) | agent - baseline [95% CI] | sign-flip p | HAC p |
|---|---|---|---|---|
| BTCUSDT | repo fixed-midpoint 10 bps | -1.455 [-3.384, +0.125] | 0.250 | 0.181 |
| ETHUSDT | Avellaneda-Stoikov | -0.980 [-3.113, +0.327] | 0.750 | 0.635 |
| LTCUSDT | repo fixed-midpoint 10 bps | -3.347 [-8.320, +1.101] | 0.344 | 0.113 |
| SOLUSDT | repo fixed-midpoint 10 bps | -2.782 [-7.207, +0.049] | 0.156 | 0.395 |

Individual ensemble members, pooled test mean PnL / h [95% CI]: seed0 -2.063 [-4.090, -0.390]; seed1 -10.856 [-19.198, -3.871]; seed2 -2.905 [-5.884, -0.388]; seed3 -2.482 [-5.193, +0.238]; seed4 -2.151 [-4.448, -0.311].

PnL is USDT per hourly episode for ~1,000 USDT quotes, after liquidating at the touch; CIs are 95% day-block bootstrap. Sharpe-like = mean / std of hourly PnL. Markout: 5 s mid move after a fill, signed for the agent (negative = adverse selection). Edge at fill: fill price vs the mid just before the fill. "VIP0 fees" subtracts 2 bps maker / 5 bps taker post hoc.

## Agent

Final configuration: `{"algo": "sac", "reward": "markout", "eta": 0.002, "gamma": 0.99, "anchor": "micro", "features": "full", "action": "quadratic", "h0": 7.5, "half_max": 30.0, "skew_max": 10.0, "latency": 1, "actor_hidden": [64, 64], "critic_hidden": [128, 128], "steps": 600000}`

Members (checkpoint picked on the validation subset during training): seed 0 @ 500000 steps, seed 1 @ 200000 steps, seed 2 @ 300000 steps, seed 3 @ 300000 steps, seed 4 @ 400000 steps

Full-validation mean PnL / h: {"seed0": -0.432, "seed1": -1.853, "seed2": -0.228, "seed3": -0.418, "seed4": 0.488, "ensemble": -0.271, "ensemble_per_symbol": {"BTCUSDT": 0.3677416666666744, "ETHUSDT": 0.1219677777777706, "LTCUSDT": -1.301739444444452}}

## Ablations (validation days only)

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

## Baselines (sigma, k fitted on training days; parameters chosen on validation; validation mean PnL / h)

| instrument | sigma (bps/sqrt s) | k (1/bps) | fixed half-spread | Avellaneda-Stoikov | repo 10 bps | previous TD3 | best on validation |
|---|---|---|---|---|---|---|---|
| BTCUSDT | 0.97 | 0.59 | `fixed_mid_10bps` +0.81 | `avellaneda_stoikov_g0.3_tau120s` +0.03 | `repo_fixed_midpoint_10bps` +0.81 | `td3_v1_previous` -1.70 | best fixed half-spread (mid) |
| ETHUSDT | 1.04 | 0.60 | `fixed_mid_5bps` +1.14 | `avellaneda_stoikov_g0.3_tau120s` +0.03 | `repo_fixed_midpoint_10bps` +0.95 | `td3_v1_previous` -2.96 | best fixed half-spread (mid) |
| LTCUSDT | 1.29 | 0.56 | `fixed_mid_7.5bps` +0.45 | `avellaneda_stoikov_g0.3` +0.00 | `repo_fixed_midpoint_10bps` -1.97 | `td3_v1_previous` -11.04 | best fixed half-spread (mid) |
| SOLUSDT | 2.16 | 0.45 | `fixed_mid_15bps` +5.28 | `avellaneda_stoikov_g0.03_tau600s` +0.15 | `repo_fixed_midpoint_10bps` +2.25 | `td3_v1_previous` -9.06 | best fixed half-spread (mid) |

## Data

| day | week | split | BTCUSDT book rows / trades | ETHUSDT book rows / trades | LTCUSDT book rows / trades | SOLUSDT book rows / trades |
|---|---|---|---|---|---|---|
| 2023-08-08 | calm | train | 14.2M / 1.04M | 8.7M / 0.51M | 2.1M / 0.14M | 3.3M / 0.30M |
| 2023-08-09 | calm | train | 13.0M / 0.83M | 7.3M / 0.38M | 1.6M / 0.09M | 3.2M / 0.27M |
| 2023-08-10 | calm | train | 10.8M / 0.60M | 7.3M / 0.32M | 1.4M / 0.08M | 2.3M / 0.18M |
| 2023-08-11 | calm | train | 9.4M / 0.42M | 6.7M / 0.28M | 1.2M / 0.06M | 2.2M / 0.15M |
| 2023-08-12 | calm | val | 6.5M / 0.23M | 5.5M / 0.20M | 1.1M / 0.05M | 1.4M / 0.12M |
| 2023-08-13 | calm | test | 7.7M / 0.29M | 6.2M / 0.26M | 1.3M / 0.08M | 1.6M / 0.12M |
| 2023-08-14 | calm | test | 11.9M / 0.64M | 7.4M / 0.32M | 1.5M / 0.08M | 2.4M / 0.19M |
| 2023-11-21 | median | train | 29.7M / 1.51M | 11.1M / 1.17M | 3.3M / 0.16M | 8.6M / 1.75M |
| 2023-11-22 | median | train | 23.8M / 1.20M | 11.8M / 1.00M | 2.2M / 0.07M | 6.5M / 1.01M |
| 2023-11-23 | median | train | 11.0M / 0.76M | 7.6M / 0.64M | 2.0M / 0.07M | 5.3M / 0.82M |
| 2023-11-24 | median | train | 13.1M / 1.22M | 8.5M / 0.92M | 1.9M / 0.07M | 4.9M / 0.66M |
| 2023-11-25 | median | val | 6.3M / 0.38M | 5.1M / 0.33M | 1.6M / 0.05M | 4.1M / 0.57M |
| 2023-11-26 | median | test | 8.7M / 0.64M | 6.7M / 0.60M | 1.7M / 0.05M | 3.8M / 0.52M |
| 2023-11-27 | median | test | 14.6M / 0.94M | 10.3M / 0.80M | 1.8M / 0.06M | 5.4M / 0.75M |
| 2024-03-14 | volatile | train | 24.6M / 2.86M | 14.7M / 1.93M | 4.1M / 0.23M | 16.4M / 2.37M |
| 2024-03-15 | volatile | train | 26.7M / 4.27M | 16.7M / 2.87M | 5.4M / 0.28M | 19.6M / 4.49M |
| 2024-03-16 | volatile | train | 19.1M / 2.09M | 12.3M / 1.70M | 4.5M / 0.19M | 16.1M / 2.86M |
| 2024-03-17 | volatile | train | 16.8M / 1.95M | 12.3M / 1.68M | 3.5M / 0.13M | 14.1M / 2.18M |
| 2024-03-18 | volatile | val | 19.3M / 2.05M | 13.8M / 1.83M | 3.9M / 0.17M | 17.3M / 2.41M |
| 2024-03-19 | volatile | test | 26.8M / 3.95M | 19.4M / 3.17M | 6.2M / 0.30M | 19.6M / 3.85M |
| 2024-03-20 | volatile | test | 23.7M / 3.70M | 18.6M / 3.53M | 4.8M / 0.22M | 15.4M / 3.00M |

Tick sizes: BTCUSDT 0.1, ETHUSDT 0.01, LTCUSDT 0.01, SOLUSDT 0.001. 786M bookTicker rows and 87M aggTrades over 84 instrument-days, every archive verified against its published SHA-256.

Frozen inputs (SHA-256): `{"agent": "2c97a3b47ef9ada3b7693e4efb8ec1eb103440a193c2881a03ed36d3ab8f617f", "baselines": "1db6079bb0463ca8475fded1097afed248a4a9875313f328ca329b31a06b717a", "v1_actor": "96b427b70cc52be371ac4dbb6358385eac73f9b5c37252ee0f83806801f8dcf3"}`
