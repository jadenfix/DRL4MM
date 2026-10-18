"""No look-ahead: perturbing everything from bin T onward must not change any feature,
observation, quote or fill decided before T (perturb-the-future test)."""
import json

import numpy as np
import pytest

from mmv2.policies import AvellanedaStoikov, FixedSpread, PreviousTD3, RepoFixedMidpoint, run_policy
from mmv2.rl import DEFAULTS, PolicyCall
from mmv2.sim import N_OBS, Day
from mmv2.synth import random_day, write_day


def random_layers(rng, sizes):
    return [{'in': i, 'out': o, 'w': rng.normal(0, 1.0 / np.sqrt(i), i * o).tolist(), 'b': rng.normal(0, 0.1, o).tolist()}
            for i, o in zip(sizes[:-1], sizes[1:])]


@pytest.fixture(scope='module')
def days(tmp_path_factory):
    d = tmp_path_factory.mktemp('la')
    bid, spread, bq, aq, trades = random_day(d / 'a.npz', n_bins=40000, seed=1)
    T = 20000  # perturb from a decision boundary on: the decision at T must still be unchanged
    rng = np.random.default_rng(9)
    bid2 = bid.copy()
    bid2[T:] += np.cumsum(rng.choice([-3, 3], len(bid) - T))
    bq2, aq2, sp2 = bq.copy(), aq.copy(), spread.copy()
    bq2[T:] = rng.integers(1, 90000, len(bid) - T)
    aq2[T:] = rng.integers(1, 90000, len(bid) - T)
    sp2[T:] = rng.choice([1, 2, 3], len(bid) - T)
    tr2 = [t for t in trades if t[0] < T] + [(b, int(rng.choice([-1, 1])), int(bid2[b]), int(rng.integers(1, 90000)))
                                              for b in range(T, len(bid), 3)]
    write_day(d / 'b.npz', bid2, sp2, bq2, aq2, tr2)
    return Day(d / 'a.npz'), Day(d / 'b.npz'), T


def test_market_features_ignore_the_future(days):
    a, b, T = days
    Fa, Fb = a.features(), b.features()
    m_last = T // 10  # row m uses bins <= 10m-1, i.e. < T for every m <= T/10
    assert np.array_equal(Fa[:m_last + 1], Fb[:m_last + 1])
    assert not np.array_equal(Fa[m_last + 1:m_last + 50], Fb[m_last + 1:m_last + 50])  # the perturbation is visible later


def policies(tmp_path):
    rng = np.random.default_rng(3)
    norm = {'mean': [0.0] * N_OBS, 'std': [1.0] * N_OBS, 'clip': 5.0}
    rl = PolicyCall([random_layers(rng, [N_OBS, 64, 64, 2])], norm, {**DEFAULTS, 'half_max': 2.0, 'skew_max': 2.0})
    v1 = tmp_path / 'v1.json'
    v1.write_text(json.dumps({'layers': random_layers(rng, [52, 64, 64, 2]),
                              'normalizer': {'mean': [0.0] * 52, 'std': [10.0] * 52, 'clip': 5.0},
                              'action_map': {'half_max': 0.001}}))
    return [FixedSpread(0.5), FixedSpread(0.0, 'micro'), RepoFixedMidpoint(1.0),
            AvellanedaStoikov(0.01, 0.8, 1.5), PreviousTD3(v1), rl]


@pytest.mark.parametrize('latency', [0, 1])
def test_decisions_and_fills_before_T_are_unchanged(days, tmp_path, latency):
    a, b, T = days
    for pol in policies(tmp_path):
        ra = run_policy(a, 0, pol, record=True, latency_bins=latency)
        rb = run_policy(b, 0, pol, record=True, latency_bins=latency)
        before = [s for s in ra['trace'] if s[0] <= T]  # decided at t <= T from bins <= t-1 < T
        assert before == rb['trace'][:len(before)], pol.name
        assert len(before) > 1000
        fa = [f for f in Episode_fills(a, pol, latency) if f[0] < T]
        fb = [f for f in Episode_fills(b, pol, latency) if f[0] < T]
        assert fa == fb, pol.name
        assert ra['trace'] != rb['trace'], pol.name  # and the future does matter


def Episode_fills(day, pol, latency):
    from mmv2.sim import Episode
    ep = Episode(day, 0, latency_bins=latency)
    pol.reset(ep)
    k = 0
    while not ep.done:
        ep.step(*pol(ep, k))
        k += 1
    return ep.fills
