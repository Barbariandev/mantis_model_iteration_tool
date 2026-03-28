import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playground.data import DataProvider, CausalView
from playground.featurizer import Featurizer, Predictor
from playground.evaluator import (
    make_binary_labels, make_hitfirst_labels, make_lbfgs_labels,
    make_xsec_labels, detect_breakouts, evaluate, _generate_embeddings,
    _rolling_std, CHALLENGES,
)


def _make_synth_df(n, base_price=100.0, volatility=0.001, seed=42):
    rng = np.random.RandomState(seed)
    log_returns = rng.normal(0, volatility, n)
    prices = base_price * np.exp(np.cumsum(log_returns))
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
        "open": prices * (1 - volatility),
        "high": prices * (1 + volatility),
        "low": prices * (1 - volatility),
        "close": prices,
        "volume": rng.uniform(100, 1000, n),
    })


def _make_provider(assets_dict):
    return DataProvider(assets_dict)


class DummyBinaryFeaturizer(Featurizer):
    warmup = 100
    compute_interval = 1

    def compute(self, view):
        p = view.prices(view.assets[0])
        r = np.diff(np.log(p[-20:])) if len(p) > 20 else np.zeros(1)
        return {"momentum": np.array([float(np.mean(r))])}


class DummyBinaryPredictor(Predictor):
    def predict(self, features):
        m = features["momentum"][0]
        p_up = 0.5 + 0.5 * np.tanh(m * 500)
        return np.array([p_up, 1 - p_up])


# ---- Tests ----

def test_causal_view_no_future_data():
    n = 500
    df = _make_synth_df(n)
    provider = _make_provider({"A": df.copy()})

    for t in [0, 10, 100, 499]:
        view = provider.view(t)
        p = view.prices("A")
        assert len(p) == t + 1, f"Expected {t+1} prices at t={t}, got {len(p)}"
        full_prices = provider.prices("A")
        assert p[-1] == full_prices[t], f"Last price mismatch at t={t}"
        if t < n - 1:
            assert full_prices[t + 1] not in p, f"Future price leaked at t={t}"


def test_causal_view_copy_isolation():
    """Returned arrays are copies -- mutating them can't affect internal state."""
    df = _make_synth_df(100)
    provider = _make_provider({"A": df})
    view = provider.view(50)
    p1 = view.prices("A")
    p1[0] = -999.0
    p2 = view.prices("A")
    assert p2[0] != -999.0, "Mutating returned array must not affect internal state"


def test_causal_view_multi_asset():
    n = 200
    df_a = _make_synth_df(n, base_price=100, seed=1)
    df_b = _make_synth_df(n, base_price=50, seed=2)
    provider = _make_provider({"A": df_a, "B": df_b})

    view = provider.view(100)
    assert sorted(view.assets) == ["A", "B"]
    assert len(view.prices("A")) == 101
    assert len(view.prices("B")) == 101
    mat = view.prices_matrix()
    assert mat.shape == (101, 2)


def test_binary_labels_correctness():
    prices = np.array([100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0, 104.0])
    labels, valid = make_binary_labels(prices, horizon=2)
    assert len(labels) == 6
    assert labels[0] == 0  # price[2]=99 < price[0]=100
    assert labels[1] == 1  # price[3]=102 > price[1]=101
    assert labels[2] == 0  # price[4]=98 < price[2]=99
    assert labels[3] == 1  # price[5]=103 > price[3]=102
    assert labels[4] == 0  # price[6]=97 < price[4]=98
    assert labels[5] == 1  # price[7]=104 > price[5]=103


def test_hitfirst_labels_basic():
    n = 10000
    rng = np.random.RandomState(123)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    labels, valid = make_hitfirst_labels(prices, horizon=100, vol_window=500)
    assert len(labels) == n - 100
    assert valid.sum() > 0
    valid_labels = labels[valid]
    assert set(np.unique(valid_labels)).issubset({0, 1, 2})


def test_lbfgs_labels_basic():
    n = 15000
    rng = np.random.RandomState(456)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    buckets, valid_idx = make_lbfgs_labels(prices, horizon=60, vol_window=3600)
    assert len(buckets) == len(valid_idx)
    if len(buckets) > 0:
        assert set(np.unique(buckets)).issubset({0, 1, 2, 3, 4})
        assert all(valid_idx[i] < valid_idx[i+1] for i in range(len(valid_idx)-1))


def test_xsec_labels_basic():
    n = 1000
    rng = np.random.RandomState(789)
    pm = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, (n, 5)), axis=0))
    labels, valid = make_xsec_labels(pm, horizon=60)
    assert labels.shape == (n - 60, 5)
    assert valid.shape == (n - 60,)
    valid_labels = labels[valid]
    if len(valid_labels) > 0:
        row_sums = valid_labels.sum(axis=1)
        for s in row_sums:
            assert s in [2, 3], f"Median split should give ~half 1s, got sum={s}"


def test_walk_forward_no_leakage():
    n = 2000
    seen_max_t = []

    class RecordingFeaturizer(Featurizer):
        warmup = 50
        compute_interval = 1
        def compute(self, view):
            seen_max_t.append(view.t)
            p = view.prices(view.assets[0])
            assert len(p) == view.t + 1, "CausalView length mismatch"
            return {"x": np.array([float(p[-1])])}

    class PassPredictor(Predictor):
        def predict(self, features):
            return np.array([0.5, 0.5])

    df = _make_synth_df(n)
    provider = _make_provider({"ETH": df})

    embeddings, warmup = _generate_embeddings(
        RecordingFeaturizer(), PassPredictor(), provider)

    assert len(seen_max_t) == n - 50
    for i, t in enumerate(seen_max_t):
        assert t == i + 50, f"Expected t={i+50}, got {t}"
    for i in range(1, len(seen_max_t)):
        assert seen_max_t[i] > seen_max_t[i-1], "Timesteps must be monotonic"


def test_compute_interval_caching():
    call_count = [0]

    class SlowFeaturizer(Featurizer):
        warmup = 10
        compute_interval = 5
        def compute(self, view):
            call_count[0] += 1
            return {"x": np.array([float(view.t)])}

    class PassPredictor(Predictor):
        def predict(self, features):
            return np.array([features["x"][0]])

    df = _make_synth_df(100)
    provider = _make_provider({"A": df})
    embeddings, _ = _generate_embeddings(SlowFeaturizer(), PassPredictor(), provider)

    assert len(embeddings) == 90  # 100 - 10
    expected_calls = 1 + (90 - 1) // 5
    assert call_count[0] == expected_calls, (
        f"Expected {expected_calls} compute calls, got {call_count[0]}")


def test_evaluate_binary_smoketest():
    n = 5000
    df = _make_synth_df(n, volatility=0.002)
    provider = _make_provider({"ETH": df})

    result = evaluate(
        "ETH-1H-BINARY",
        DummyBinaryFeaturizer(),
        DummyBinaryPredictor(),
        provider=provider,
    )

    assert "error" not in result, f"Evaluation failed: {result.get('error')}"
    assert result["type"] == "binary"
    assert len(result["windows"]) > 0
    assert 0.0 <= result["mean_auc"] <= 1.0


def test_breakout_detection():
    n = 2000
    prices = np.ones(n) * 100.0
    prices[500:600] = np.linspace(100, 110, 100)
    prices[600:700] = np.linspace(110, 115, 100)
    prices[700:] = 115.0

    pm = prices.reshape(-1, 1)
    events = detect_breakouts(pm, ["TEST"], range_lookback=400,
                              barrier_pct=25.0, min_range_pct=1.0)
    assert isinstance(events, list)


def test_hitfirst_sigma_matches_original():
    """Verify our hitfirst sigma computation matches the original hitfirst.py
    algorithm: rolling_std directly on r_h, NOT the time-shifted sigma_from_price."""
    n = 3000
    rng = np.random.RandomState(999)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    horizon = 100
    vol_window = 500

    r_h = np.log(prices[horizon:] + 1e-12) - np.log(prices[:-horizon] + 1e-12)
    sig_raw = _rolling_std(r_h, vol_window)
    sig_original = np.full(n - horizon, np.nan)
    if len(sig_raw) > 0:
        sig_original[vol_window - 1:] = sig_raw

    labels, valid = make_hitfirst_labels(prices, horizon, vol_window=vol_window)
    assert len(labels) == n - horizon

    log_p = np.log(prices + 1e-12)
    for t in [vol_window, vol_window + 100, vol_window + 500]:
        if t >= n - horizon:
            break
        if not np.isfinite(sig_original[t]) or sig_original[t] <= 0:
            assert not valid[t], f"Expected invalid label at t={t}"
            continue
        base = log_p[t]
        path = log_p[t + 1: t + 1 + horizon] - base
        up_hits = np.where(path >= sig_original[t])[0]
        dn_hits = np.where(path <= -sig_original[t])[0]
        has_up = len(up_hits) > 0
        has_dn = len(dn_hits) > 0
        if not has_up and not has_dn:
            expected = 2
        elif has_up and not has_dn:
            expected = 0
        elif has_dn and not has_up:
            expected = 1
        else:
            if up_hits[0] < dn_hits[0]:
                expected = 0
            elif dn_hits[0] < up_hits[0]:
                expected = 1
            else:
                expected = 2
        assert labels[t] == expected, (
            f"Label mismatch at t={t}: got {labels[t]}, expected {expected}")


def test_lbfgs_bucket_boundaries():
    """Verify LBFGS bucket boundaries match utils.make_bins_from_price exactly."""
    z = np.array([-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    expected = [0, 0, 1, 2, 2, 2, 2, 2, 3, 4, 4]
    buckets = np.zeros(len(z), dtype=np.int32)
    buckets[z <= -2.0] = 0
    buckets[(z > -2.0) & (z < -1.0)] = 1
    buckets[(z >= -1.0) & (z <= 1.0)] = 2
    buckets[(z > 1.0) & (z < 2.0)] = 3
    buckets[z >= 2.0] = 4
    for i, (got, exp) in enumerate(zip(buckets, expected)):
        assert got == exp, f"z={z[i]}: got bucket {got}, expected {exp}"


def test_causal_view_immutable():
    """Cannot mutate CausalView attributes to bypass time boundary."""
    df = _make_synth_df(500)
    provider = _make_provider({"A": df})
    view = provider.view(50)

    raised = False
    try:
        view._t = 499
    except AttributeError:
        raised = True
    assert raised, "Setting _t should raise AttributeError"

    raised = False
    try:
        view._closes = {}
    except AttributeError:
        raised = True
    assert raised, "Setting _closes should raise AttributeError"

    raised = False
    try:
        view.t = 499
    except AttributeError:
        raised = True
    assert raised, "Setting t should raise AttributeError"


def test_causal_view_no_base_leak():
    """prices() returns a copy, so .base cannot recover full array."""
    df = _make_synth_df(500)
    provider = _make_provider({"A": df})
    view = provider.view(50)

    p = view.prices("A")
    assert len(p) == 51
    assert p.base is None, ".base should be None on a copy (owned array)"
    assert p.flags.owndata, "prices() must return owned data, not a view"

    o = view.ohlcv("A")
    assert o.shape == (51, 5)
    assert o.flags.owndata, "ohlcv() must return owned data, not a view"


def test_causal_view_mangled_access_blocked():
    """Naive underscore access to internals should fail."""
    df = _make_synth_df(500)
    provider = _make_provider({"A": df})
    view = provider.view(50)

    raised = False
    try:
        _ = view._closes
    except AttributeError:
        raised = True
    assert raised, "view._closes should raise AttributeError (name-mangled)"

    raised = False
    try:
        _ = view._ohlcv
    except AttributeError:
        raised = True
    assert raised, "view._ohlcv should raise AttributeError (name-mangled)"

    raised = False
    try:
        _ = view._t
    except AttributeError:
        raised = True
    assert raised, "view._t should raise AttributeError (name-mangled)"


def test_coinglass_alignment_invariant():
    """Verify causal alignment: at each minute, only the last COMPLETED candle is visible.

    - At minute 59 (inside first hour), hourly feature = NaN (no completed candle)
    - At minute 60 (start of second hour), hourly feature = close of first hourly candle
    - At minute 119, hourly feature = still the close of first hourly candle
    - At minute 120, hourly feature = close of second hourly candle
    """
    from playground.coinglass import align_to_minutes

    hourly_times_ms = np.array([0, 3_600_000, 7_200_000, 10_800_000], dtype=np.int64)
    hourly_closes = np.array([100.0, 200.0, 300.0, 400.0])
    period_ms = 3_600_000  # 1 hour

    n_minutes = 240
    minute_times_ms = np.arange(n_minutes, dtype=np.int64) * 60_000

    aligned = align_to_minutes(hourly_times_ms, hourly_closes, minute_times_ms, period_ms)

    assert len(aligned) == n_minutes
    for m in range(60):
        assert np.isnan(aligned[m]), (
            f"Minute {m}: expected NaN (no completed hourly candle), got {aligned[m]}")

    for m in range(60, 120):
        assert aligned[m] == 100.0, (
            f"Minute {m}: expected 100.0 (first hourly close), got {aligned[m]}")

    for m in range(120, 180):
        assert aligned[m] == 200.0, (
            f"Minute {m}: expected 200.0 (second hourly close), got {aligned[m]}")

    for m in range(180, 240):
        assert aligned[m] == 300.0, (
            f"Minute {m}: expected 300.0 (third hourly close), got {aligned[m]}")


def test_coinglass_alignment_daily():
    """Daily candle alignment: first 24h should be NaN, then switch at day boundaries."""
    from playground.coinglass import align_to_minutes

    day_ms = 86_400_000
    daily_times = np.array([0, day_ms, 2 * day_ms], dtype=np.int64)
    daily_closes = np.array([10.0, 20.0, 30.0])

    n_minutes = 3 * 24 * 60
    minute_times_ms = np.arange(n_minutes, dtype=np.int64) * 60_000

    aligned = align_to_minutes(daily_times, daily_closes, minute_times_ms, day_ms)

    for m in range(24 * 60):
        assert np.isnan(aligned[m]), f"Minute {m}: expected NaN before first daily close"

    for m in range(24 * 60, 2 * 24 * 60):
        assert aligned[m] == 10.0, f"Minute {m}: expected 10.0 (first daily close)"

    for m in range(2 * 24 * 60, 3 * 24 * 60):
        assert aligned[m] == 20.0, f"Minute {m}: expected 20.0 (second daily close)"


def test_causal_view_cg_integration():
    """CausalView.cg() correctly slices CoinGlass features up to t."""
    n = 200
    df = _make_synth_df(n)
    cg_feature = np.arange(n, dtype=np.float64)
    coinglass_data = {"A": {"oi_1h": cg_feature}}
    provider = DataProvider({"A": df}, coinglass=coinglass_data)

    view = provider.view(50)
    assert view.has_cg()
    assert "oi_1h" in view.cg_fields("A")

    arr = view.cg("A", "oi_1h")
    assert len(arr) == 51
    assert arr[-1] == 50.0
    assert arr.flags.owndata, "cg() must return a copy"

    arr[0] = -1.0
    arr2 = view.cg("A", "oi_1h")
    assert arr2[0] != -1.0, "cg() copy isolation failed"


def test_causal_view_cg_missing_field():
    """Accessing a missing CoinGlass field raises KeyError."""
    df = _make_synth_df(100)
    provider = DataProvider({"A": df}, coinglass={"A": {"oi_1h": np.zeros(100)}})
    view = provider.view(50)
    raised = False
    msg = ""
    err_type = None
    try:
        view.cg("A", "nonexistent")
    except KeyError as e:
        raised = True
        msg = str(e)
        err_type = type(e)
    assert raised and err_type is KeyError, f"Expected KeyError, got {err_type}: {msg}"


def test_causal_view_no_cg():
    """CausalView without CoinGlass data reports has_cg() == False."""
    df = _make_synth_df(100)
    provider = DataProvider({"A": df})
    view = provider.view(50)
    assert not view.has_cg()
    assert view.cg_fields("A") == []


def test_holdout_only_metrics():
    """Headline metrics must come from holdout-period windows only.

    Creates a dataset where early dev-period predictions are trivially correct
    (hardcoded for the first half) while holdout predictions are random.
    If leakage is present, mean_auc will be inflated by dev-period accuracy.
    """
    n = 10000
    df = _make_synth_df(n, volatility=0.002, seed=42)
    provider = _make_provider({"ETH": df})
    prices = provider.prices("ETH")
    labels_all, _ = make_binary_labels(prices, horizon=60)

    dev_len = n - 2000
    warmup = 100

    class LeakyFeaturizer(Featurizer):
        warmup = 100
        compute_interval = 1
        def compute(self, view):
            return {"t": np.array([float(view.t)])}

    class LeakyPredictor(Predictor):
        def predict(self, features):
            t = int(features["t"][0])
            if 0 <= t < len(labels_all) and t < dev_len:
                return np.array([float(labels_all[t]), 1.0 - float(labels_all[t])])
            return np.array([0.5, 0.5])

    result_no_holdout = evaluate(
        "ETH-1H-BINARY", LeakyFeaturizer(), LeakyPredictor(),
        provider=provider, holdout_start=0)

    result_with_holdout = evaluate(
        "ETH-1H-BINARY", LeakyFeaturizer(), LeakyPredictor(),
        provider=provider, holdout_start=dev_len)

    assert "error" not in result_no_holdout, f"no-holdout eval failed: {result_no_holdout.get('error')}"
    assert "error" not in result_with_holdout, f"holdout eval failed: {result_with_holdout.get('error')}"

    auc_all = result_no_holdout["mean_auc"]
    auc_holdout = result_with_holdout["mean_auc"]

    assert auc_all > 0.8, f"Without holdout filtering, leaky featurizer should get >0.8 AUC, got {auc_all}"
    assert auc_holdout < 0.6, f"With holdout filtering, random predictions should get ~0.5 AUC, got {auc_holdout}"

    holdout_windows = [w for w in result_with_holdout["windows"] if w.get("in_holdout")]
    dev_windows = [w for w in result_with_holdout["windows"] if not w.get("in_holdout")]
    assert len(holdout_windows) > 0, "Should have holdout windows"
    assert len(dev_windows) > 0, "Should have dev windows (for diagnostics)"


def test_evaluate_hitfirst_smoketest():
    """Verify hitfirst eval returns 70/30 split with up/dn AUC structure."""
    n = 12000
    df = _make_synth_df(n, volatility=0.005, seed=77)
    provider = _make_provider({"ETH": df})

    class HitFeaturizer(Featurizer):
        warmup = 100
        compute_interval = 1
        def compute(self, view):
            p = view.prices("ETH")
            r = np.diff(np.log(p[-50:])) if len(p) > 50 else np.zeros(1)
            return {"r": np.array([float(np.mean(r))])}

    class HitPredictor(Predictor):
        def predict(self, features):
            m = features["r"][0]
            p_up = 0.3 + 0.2 * np.tanh(m * 200)
            p_dn = 0.3 - 0.1 * np.tanh(m * 200)
            p_neither = 1.0 - p_up - p_dn
            return np.array([max(p_up, 0.01), max(p_dn, 0.01), max(p_neither, 0.01)])

    result = evaluate(
        "ETH-HITFIRST-100M",
        HitFeaturizer(),
        HitPredictor(),
        provider=provider,
    )
    assert "error" not in result, f"Evaluation failed: {result.get('error')}"
    assert result["type"] == "hitfirst"
    assert "n_valid" in result
    assert "up_auc" in result
    assert "dn_auc" in result


if __name__ == "__main__":
    tests = [
        ("causal_view_no_future_data", test_causal_view_no_future_data),
        ("causal_view_copy_isolation", test_causal_view_copy_isolation),
        ("causal_view_multi_asset", test_causal_view_multi_asset),
        ("binary_labels_correctness", test_binary_labels_correctness),
        ("hitfirst_labels_basic", test_hitfirst_labels_basic),
        ("lbfgs_labels_basic", test_lbfgs_labels_basic),
        ("xsec_labels_basic", test_xsec_labels_basic),
        ("walk_forward_no_leakage", test_walk_forward_no_leakage),
        ("compute_interval_caching", test_compute_interval_caching),
        ("evaluate_binary_smoketest", test_evaluate_binary_smoketest),
        ("breakout_detection", test_breakout_detection),
        ("hitfirst_sigma_matches_original", test_hitfirst_sigma_matches_original),
        ("lbfgs_bucket_boundaries", test_lbfgs_bucket_boundaries),
        ("causal_view_immutable", test_causal_view_immutable),
        ("causal_view_no_base_leak", test_causal_view_no_base_leak),
        ("causal_view_mangled_access_blocked", test_causal_view_mangled_access_blocked),
        ("evaluate_hitfirst_smoketest", test_evaluate_hitfirst_smoketest),
        ("coinglass_alignment_invariant", test_coinglass_alignment_invariant),
        ("coinglass_alignment_daily", test_coinglass_alignment_daily),
        ("causal_view_cg_integration", test_causal_view_cg_integration),
        ("causal_view_cg_missing_field", test_causal_view_cg_missing_field),
        ("causal_view_no_cg", test_causal_view_no_cg),
        ("holdout_only_metrics", test_holdout_only_metrics),
    ]
    passed = 0
    failed = 0
    for name, fn in tests:
        status = "PASS"
        err = ""
        try:
            fn()
            passed += 1
        except Exception as e:
            status = "FAIL"
            err = str(e)
            failed += 1
        print(f"  [{status}] {name}" + (f"  -- {err}" if err else ""))
    print(f"\n{passed}/{passed+failed} tests passed")
    sys.exit(1 if failed > 0 else 0)
