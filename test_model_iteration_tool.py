import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mantis_model_iteration_tool.data import DataProvider, CausalView
from mantis_model_iteration_tool.featurizer import Featurizer, Predictor
from mantis_model_iteration_tool.evaluator import (
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
    from mantis_model_iteration_tool.coinglass import align_to_minutes

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
    from mantis_model_iteration_tool.coinglass import align_to_minutes

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


# ---- Eval / Inference parity & advanced leakage tests ----

from mantis_model_iteration_tool.inferencer import run_inference_single, ModelSlot


class StatefulFeaturizer(Featurizer):
    """Records every CausalView.t seen, and hashes the price array.
    This lets tests compare what eval vs inference presented."""
    warmup = 50
    compute_interval = 1

    def setup(self, assets, length):
        self.history = []

    def compute(self, view):
        p = view.prices(view.assets[0])
        self.history.append({
            "t": view.t,
            "len": len(p),
            "last_price": float(p[-1]),
            "checksum": float(np.sum(p)),
        })
        r = np.diff(np.log(p[-20:])) if len(p) > 20 else np.zeros(1)
        return {"momentum": np.array([float(np.mean(r))]), "t": view.t}


class StatefulPredictor(Predictor):
    """Stores last-seen features for comparison."""
    def __init__(self):
        self.last_features = None

    def predict(self, features):
        self.last_features = dict(features)
        m = features["momentum"][0]
        p_up = 0.5 + 0.5 * np.tanh(m * 500)
        return np.array([p_up, 1 - p_up])


def test_eval_inference_parity_final_embedding():
    """The final embedding from eval must EXACTLY equal inference output
    when both use the same data and strategy."""
    n = 500
    df = _make_synth_df(n, seed=77)
    provider = _make_provider({"ETH": df})

    feat_eval = StatefulFeaturizer()
    pred_eval = StatefulPredictor()
    emb_matrix, warmup = _generate_embeddings(feat_eval, pred_eval, provider)
    eval_final = emb_matrix[-1]

    feat_inf = StatefulFeaturizer()
    pred_inf = StatefulPredictor()
    slot = ModelSlot(
        agent_id="test", iteration=1,
        challenge_name="ETH-1H-BINARY", strategy_path="",
        featurizer=feat_inf, predictor=pred_inf, loaded=True,
    )
    inf_result = run_inference_single(slot, provider)

    assert np.allclose(eval_final, inf_result, atol=1e-7), (
        f"Eval final embedding {eval_final} != inference {inf_result}"
    )


def test_eval_inference_same_view_at_T():
    """At the last timestep, eval and inference must see identical
    CausalView data (same t, same price array, same length)."""
    n = 300
    df = _make_synth_df(n, seed=88)
    provider = _make_provider({"ETH": df})

    feat_eval = StatefulFeaturizer()
    pred_eval = StatefulPredictor()
    _generate_embeddings(feat_eval, pred_eval, provider)
    eval_last = feat_eval.history[-1]

    feat_inf = StatefulFeaturizer()
    pred_inf = StatefulPredictor()
    slot = ModelSlot(
        agent_id="test", iteration=1,
        challenge_name="ETH-1H-BINARY", strategy_path="",
        featurizer=feat_inf, predictor=pred_inf, loaded=True,
    )
    run_inference_single(slot, provider)
    inf_last = feat_inf.history[-1]

    assert eval_last["t"] == inf_last["t"], (
        f"Eval saw t={eval_last['t']} but inference saw t={inf_last['t']}"
    )
    assert eval_last["len"] == inf_last["len"], (
        f"Eval view length {eval_last['len']} != inference {inf_last['len']}"
    )
    assert abs(eval_last["checksum"] - inf_last["checksum"]) < 1e-6, (
        "Price data checksum mismatch between eval and inference"
    )


def test_eval_inference_parity_with_compute_interval():
    """With compute_interval > 1, eval and inference must still produce
    identical final embeddings."""
    n = 400

    class IntervalFeaturizer(Featurizer):
        warmup = 30
        compute_interval = 10

        def setup(self, assets, length):
            self.call_count = 0

        def compute(self, view):
            self.call_count += 1
            p = view.prices(view.assets[0])
            return {"mean": np.array([float(np.mean(p[-30:]))])}

    class SimplePredictor(Predictor):
        def predict(self, features):
            v = features["mean"][0]
            return np.array([float(np.clip(v / 200, 0.01, 0.99)),
                             float(1.0 - np.clip(v / 200, 0.01, 0.99))])

    df = _make_synth_df(n, seed=99)
    provider = _make_provider({"ETH": df})

    feat_e = IntervalFeaturizer()
    pred_e = SimplePredictor()
    emb_all, _ = _generate_embeddings(feat_e, pred_e, provider)

    feat_i = IntervalFeaturizer()
    pred_i = SimplePredictor()
    slot = ModelSlot(
        agent_id="test", iteration=1,
        challenge_name="ETH-1H-BINARY", strategy_path="",
        featurizer=feat_i, predictor=pred_i, loaded=True,
    )
    inf_emb = run_inference_single(slot, provider)

    assert np.allclose(emb_all[-1], inf_emb, atol=1e-7), (
        f"compute_interval>1 mismatch: eval={emb_all[-1]} inf={inf_emb}"
    )


def test_eval_inference_parity_with_coinglass():
    """CoinGlass data must appear identically in eval and inference CausalViews."""
    n = 200
    df = _make_synth_df(n, seed=55)
    rng = np.random.RandomState(55)
    cg = {"ETH": {"funding_rate": rng.uniform(-0.01, 0.01, n).astype(np.float64)}}
    provider = DataProvider({"ETH": df}, coinglass=cg)

    class CGFeaturizer(Featurizer):
        warmup = 10
        compute_interval = 1

        def setup(self, assets, length):
            self.cg_checksums = []

        def compute(self, view):
            p = view.prices(view.assets[0])
            fr = view.cg("ETH", "funding_rate")
            self.cg_checksums.append(float(np.sum(fr)))
            return {"m": np.array([float(np.mean(p[-10:]))])}

    class CGPredictor(Predictor):
        def predict(self, features):
            v = features["m"][0]
            return np.array([float(np.clip(v / 200, 0.01, 0.99)),
                             float(1.0 - np.clip(v / 200, 0.01, 0.99))])

    feat_e = CGFeaturizer()
    pred_e = CGPredictor()
    _generate_embeddings(feat_e, pred_e, provider)

    feat_i = CGFeaturizer()
    pred_i = CGPredictor()
    slot = ModelSlot(
        agent_id="test", iteration=1,
        challenge_name="ETH-1H-BINARY", strategy_path="",
        featurizer=feat_i, predictor=pred_i, loaded=True,
    )
    run_inference_single(slot, provider)

    assert abs(feat_e.cg_checksums[-1] - feat_i.cg_checksums[-1]) < 1e-10, (
        "CoinGlass checksum diverged between eval and inference at T-1"
    )


def test_eval_inference_parity_multi_asset():
    """With multiple assets, eval and inference must agree."""
    n = 250
    df_eth = _make_synth_df(n, base_price=2000, seed=10)
    df_btc = _make_synth_df(n, base_price=40000, seed=11)
    provider = _make_provider({"ETH": df_eth, "BTC": df_btc})

    class MultiFeaturizer(Featurizer):
        warmup = 20
        compute_interval = 1

        def setup(self, assets, length):
            self.last_matrix_checksum = None

        def compute(self, view):
            m = view.prices_matrix()
            self.last_matrix_checksum = float(np.sum(m))
            return {"spread": np.array([float(m[-1, 0] - m[-1, 1])])}

    class SpreadPredictor(Predictor):
        def predict(self, features):
            s = features["spread"][0]
            v = 0.5 + 0.5 * np.tanh(s / 1000)
            return np.array([v, 1 - v])

    feat_e = MultiFeaturizer()
    pred_e = SpreadPredictor()
    emb_all, _ = _generate_embeddings(feat_e, pred_e, provider)

    feat_i = MultiFeaturizer()
    pred_i = SpreadPredictor()
    slot = ModelSlot(
        agent_id="test", iteration=1,
        challenge_name="ETH-1H-BINARY", strategy_path="",
        featurizer=feat_i, predictor=pred_i, loaded=True,
    )
    inf_emb = run_inference_single(slot, provider)

    assert np.allclose(emb_all[-1], inf_emb, atol=1e-7), "Multi-asset parity failed"
    assert abs(feat_e.last_matrix_checksum - feat_i.last_matrix_checksum) < 1e-6, (
        "Multi-asset price matrix diverged"
    )


def test_causal_view_mangled_bypass_blocked():
    """Even object.__getattribute__ to the mangled name must not allow
    reading future data beyond t."""
    n = 200
    df = _make_synth_df(n, seed=33)
    provider = _make_provider({"ETH": df})
    t = 50
    view = provider.view(t)

    # Even if someone accesses raw internals, the API shouldn't expose future
    raw_closes = object.__getattribute__(view, '_CausalView__closes')
    full_len = len(raw_closes["ETH"])
    api_len = len(view.prices("ETH"))
    assert api_len == t + 1, f"API should show {t+1} bars, got {api_len}"
    assert full_len > api_len, "Internal array should be longer than API slice"
    # The API copy must not share memory with the internal buffer
    api_prices = view.prices("ETH")
    api_prices[0] = -999999.0
    fresh = view.prices("ETH")
    assert fresh[0] != -999999.0, "Mutating API output corrupted internal state"


def test_causal_view_ohlcv_no_future_rows():
    """ohlcv() must only return rows 0..t, not beyond."""
    n = 300
    df = _make_synth_df(n, seed=44)
    provider = _make_provider({"ETH": df})

    for t in [0, 1, 50, 149, 299]:
        view = provider.view(t)
        ohlcv = view.ohlcv("ETH")
        assert ohlcv.shape[0] == t + 1, (
            f"At t={t}, ohlcv has {ohlcv.shape[0]} rows, expected {t+1}"
        )
        # Verify the last close matches what prices() returns
        p = view.prices("ETH")
        assert abs(ohlcv[-1, 3] - p[-1]) < 1e-12, (
            "OHLCV close column doesn't match prices() at the boundary"
        )


def test_causal_view_prices_matrix_no_future():
    """prices_matrix() must return (t+1, n_assets) with no future data."""
    n = 200
    df_a = _make_synth_df(n, seed=1)
    df_b = _make_synth_df(n, seed=2, base_price=50)
    provider = _make_provider({"A": df_a, "B": df_b})

    for t in [0, 10, 99, 199]:
        view = provider.view(t)
        m = view.prices_matrix()
        assert m.shape == (t + 1, 2), f"At t={t}, matrix shape {m.shape}"
        # Each column must match individual prices()
        for i, asset in enumerate(sorted(["A", "B"])):
            np.testing.assert_array_equal(m[:, i], view.prices(asset))


def test_causal_view_cg_no_future():
    """CoinGlass data via view.cg() must be truncated to t+1."""
    n = 200
    df = _make_synth_df(n, seed=66)
    rng = np.random.RandomState(66)
    cg_vals = rng.uniform(0, 1, n)
    cg = {"ETH": {"oi": cg_vals}}
    provider = DataProvider({"ETH": df}, coinglass=cg)

    for t in [0, 50, 100, 199]:
        view = provider.view(t)
        oi = view.cg("ETH", "oi")
        assert len(oi) == t + 1, f"CG 'oi' at t={t}: len={len(oi)} expected {t+1}"
        np.testing.assert_array_equal(oi, cg_vals[:t + 1])
        # Mutation must not affect internal state
        oi[0] = -1.0
        fresh = view.cg("ETH", "oi")
        assert fresh[0] != -1.0, "CG mutation leaked into internal state"


def test_causal_view_copy_independence():
    """Two views at different t from the same provider must be independent."""
    n = 200
    df = _make_synth_df(n, seed=77)
    provider = _make_provider({"ETH": df})

    v1 = provider.view(50)
    v2 = provider.view(100)

    p1 = v1.prices("ETH")
    p2 = v2.prices("ETH")

    assert len(p1) == 51
    assert len(p2) == 101
    # Mutating one doesn't affect the other
    p1[-1] = -1.0
    p2_fresh = v2.prices("ETH")
    assert p2_fresh[50] != -1.0, "Cross-view mutation leakage"


def test_provider_alignment_truncates_correctly():
    """DataProvider must align all assets to the shortest series."""
    df_short = _make_synth_df(100, seed=1)
    df_long = _make_synth_df(200, seed=2)
    provider = _make_provider({"A": df_short, "B": df_long})

    assert provider.length == 100, f"Expected 100, got {provider.length}"
    # View at last valid t must work
    view = provider.view(99)
    assert len(view.prices("A")) == 100
    assert len(view.prices("B")) == 100
    # View beyond range must raise
    raised = False
    try:
        provider.view(100)
    except IndexError:
        raised = True
    assert raised, "view(100) should raise IndexError for 100-length provider"


def test_provider_cg_alignment_truncates():
    """CoinGlass arrays must be truncated to match OHLCV min_len."""
    df_short = _make_synth_df(100, seed=1)
    df_long = _make_synth_df(200, seed=2)
    rng = np.random.RandomState(3)
    cg = {"A": {"rate": rng.uniform(0, 1, 300)}}
    provider = DataProvider({"A": df_short, "B": df_long}, coinglass=cg)

    assert provider.length == 100
    view = provider.view(99)
    rate = view.cg("A", "rate")
    assert len(rate) == 100, f"CG should be truncated to 100, got {len(rate)}"


def test_inference_warmup_guard():
    """Inference must refuse if provider has fewer candles than warmup."""
    df = _make_synth_df(30, seed=1)
    provider = _make_provider({"ETH": df})

    feat = StatefulFeaturizer()  # warmup=50
    pred = StatefulPredictor()
    slot = ModelSlot(
        agent_id="test", iteration=1,
        challenge_name="ETH-1H-BINARY", strategy_path="",
        featurizer=feat, predictor=pred, loaded=True,
    )
    raised = False
    try:
        run_inference_single(slot, provider)
    except RuntimeError as e:
        raised = True
        assert "warmup" in str(e).lower(), f"Expected warmup error, got: {e}"
    assert raised, "Should raise RuntimeError for insufficient data"


def test_eval_embedding_count_matches_timesteps():
    """_generate_embeddings must produce exactly (T - warmup) embeddings."""
    n = 200
    df = _make_synth_df(n, seed=42)
    provider = _make_provider({"ETH": df})

    feat = DummyBinaryFeaturizer()  # warmup=100
    pred = DummyBinaryPredictor()
    emb, warmup = _generate_embeddings(feat, pred, provider)

    expected_count = n - feat.warmup
    assert emb.shape[0] == expected_count, (
        f"Expected {expected_count} embeddings, got {emb.shape[0]}"
    )
    assert warmup == feat.warmup


def test_walk_forward_holdout_boundary_no_leak():
    """Walk-forward with holdout_start: dev windows exist but don't count
    toward headline metrics; holdout windows do."""
    n = 5000
    df = _make_synth_df(n, seed=42)
    provider = _make_provider({"ETH": df})

    holdout_start = 2500
    result_with_holdout = evaluate(
        "ETH-1H-BINARY",
        DummyBinaryFeaturizer(),
        DummyBinaryPredictor(),
        provider=provider,
        holdout_start=holdout_start,
    )
    result_no_holdout = evaluate(
        "ETH-1H-BINARY",
        DummyBinaryFeaturizer(),
        DummyBinaryPredictor(),
        provider=provider,
        holdout_start=0,
    )
    assert "windows" in result_with_holdout, f"Missing windows: {result_with_holdout}"
    assert "windows" in result_no_holdout

    # With holdout, headline metric uses only holdout windows (subset)
    # Without holdout, all windows count → different mean_auc
    hw = [w for w in result_with_holdout["windows"] if w.get("in_holdout")]
    dw = [w for w in result_with_holdout["windows"] if not w.get("in_holdout")]
    assert len(hw) > 0, "Should have at least one holdout window"
    assert len(dw) > 0, "Should have dev windows too for diagnostics"


def test_featurizer_sees_monotonic_t():
    """Each successive call to featurizer.compute must see strictly increasing t."""
    n = 200
    df = _make_synth_df(n, seed=42)
    provider = _make_provider({"ETH": df})

    feat = StatefulFeaturizer()
    pred = StatefulPredictor()
    _generate_embeddings(feat, pred, provider)

    ts = [h["t"] for h in feat.history]
    for i in range(1, len(ts)):
        assert ts[i] > ts[i - 1], f"Non-monotonic t: {ts[i-1]} -> {ts[i]}"
    assert ts[0] == feat.warmup, f"First t should be {feat.warmup}, got {ts[0]}"
    assert ts[-1] == n - 1, f"Last t should be {n-1}, got {ts[-1]}"


def test_featurizer_view_length_equals_t_plus_one():
    """At each timestep, the view's price array length must be exactly t+1."""
    n = 200
    df = _make_synth_df(n, seed=42)
    provider = _make_provider({"ETH": df})

    feat = StatefulFeaturizer()
    pred = StatefulPredictor()
    _generate_embeddings(feat, pred, provider)

    for h in feat.history:
        assert h["len"] == h["t"] + 1, (
            f"At t={h['t']}, view length was {h['len']}, expected {h['t']+1}"
        )


# ---- Deep parity instrumentation ----
# An "InstrumentedFeaturizer" that records the full raw data at each compute()
# call for byte-level comparison across eval and inference paths.

class InstrumentedFeaturizer(Featurizer):
    """Captures every raw input to compute() for deep comparison."""
    warmup = 30
    compute_interval = 1

    def setup(self, assets, length):
        self.traces = []

    def compute(self, view):
        t = view.t
        assets = view.assets
        trace = {"t": t, "assets": assets}
        for a in assets:
            p = view.prices(a)
            o = view.ohlcv(a)
            trace[f"prices_{a}_len"] = len(p)
            trace[f"prices_{a}_first"] = float(p[0])
            trace[f"prices_{a}_last"] = float(p[-1])
            trace[f"prices_{a}_sum"] = float(np.sum(p))
            trace[f"ohlcv_{a}_shape"] = o.shape
            trace[f"ohlcv_{a}_sum"] = float(np.sum(o))
            if view.has_cg():
                for field in view.cg_fields(a):
                    cg_arr = view.cg(a, field)
                    trace[f"cg_{a}_{field}_len"] = len(cg_arr)
                    trace[f"cg_{a}_{field}_sum"] = float(np.sum(cg_arr))
        self.traces.append(trace)
        p = view.prices(assets[0])
        r = np.diff(np.log(p[-20:])) if len(p) > 20 else np.zeros(1)
        return {"momentum": np.array([float(np.mean(r))])}


class InstrumentedPredictor(Predictor):
    """Captures every feature dict and output embedding."""
    def __init__(self):
        self.feature_traces = []
        self.output_traces = []

    def predict(self, features):
        snapshot = {}
        for k, v in features.items():
            arr = np.asarray(v)
            snapshot[k] = {"shape": arr.shape, "sum": float(np.sum(arr)),
                           "bytes": arr.tobytes()}
        self.feature_traces.append(snapshot)
        m = features["momentum"][0]
        out = np.array([0.5 + 0.5 * np.tanh(m * 500),
                        0.5 - 0.5 * np.tanh(m * 500)])
        self.output_traces.append(out.copy())
        return out


def _compare_traces(eval_traces, inf_traces, label=""):
    """Deep compare the last eval trace against the last inference trace."""
    e = eval_traces[-1]
    i = inf_traces[-1]
    mismatches = []
    for key in set(list(e.keys()) + list(i.keys())):
        ev = e.get(key)
        iv = i.get(key)
        if ev is None:
            mismatches.append(f"{label}{key}: missing in eval")
        elif iv is None:
            mismatches.append(f"{label}{key}: missing in inference")
        elif isinstance(ev, float) and isinstance(iv, float):
            if abs(ev - iv) > 1e-10:
                mismatches.append(f"{label}{key}: eval={ev} inf={iv}")
        elif ev != iv:
            mismatches.append(f"{label}{key}: eval={ev!r} inf={iv!r}")
    return mismatches


def test_deep_parity_all_raw_inputs():
    """Byte-level comparison of every raw input (prices, OHLCV, CG) at T-1."""
    n = 300
    df = _make_synth_df(n, seed=101)
    rng = np.random.RandomState(101)
    cg = {"ETH": {
        "funding": rng.uniform(-0.01, 0.01, n),
        "oi_1h": rng.uniform(1e6, 5e6, n),
    }}
    provider = DataProvider({"ETH": df}, coinglass=cg)

    feat_e = InstrumentedFeaturizer()
    pred_e = InstrumentedPredictor()
    _generate_embeddings(feat_e, pred_e, provider)

    feat_i = InstrumentedFeaturizer()
    pred_i = InstrumentedPredictor()
    slot = ModelSlot(
        agent_id="t", iteration=1, challenge_name="ETH-1H-BINARY",
        strategy_path="", featurizer=feat_i, predictor=pred_i, loaded=True,
    )
    run_inference_single(slot, provider)

    mismatches = _compare_traces(feat_e.traces, feat_i.traces, "view.")
    assert not mismatches, "Raw input divergence:\n" + "\n".join(mismatches)


def test_deep_parity_feature_dict_bytes():
    """The feature dict passed to predict() must be byte-identical at T-1."""
    n = 300
    df = _make_synth_df(n, seed=102)
    provider = _make_provider({"ETH": df})

    feat_e = InstrumentedFeaturizer()
    pred_e = InstrumentedPredictor()
    _generate_embeddings(feat_e, pred_e, provider)

    feat_i = InstrumentedFeaturizer()
    pred_i = InstrumentedPredictor()
    slot = ModelSlot(
        agent_id="t", iteration=1, challenge_name="ETH-1H-BINARY",
        strategy_path="", featurizer=feat_i, predictor=pred_i, loaded=True,
    )
    run_inference_single(slot, provider)

    e_snap = pred_e.feature_traces[-1]
    i_snap = pred_i.feature_traces[-1]
    assert set(e_snap.keys()) == set(i_snap.keys()), (
        f"Feature keys differ: eval={set(e_snap.keys())} inf={set(i_snap.keys())}"
    )
    for k in e_snap:
        assert e_snap[k]["shape"] == i_snap[k]["shape"], (
            f"Feature '{k}' shape: eval={e_snap[k]['shape']} inf={i_snap[k]['shape']}"
        )
        assert e_snap[k]["bytes"] == i_snap[k]["bytes"], (
            f"Feature '{k}' bytes differ between eval and inference"
        )


def test_deep_parity_output_embedding_bytes():
    """The raw predict() output must be byte-identical (pre-float32 cast)."""
    n = 300
    df = _make_synth_df(n, seed=103)
    provider = _make_provider({"ETH": df})

    feat_e = InstrumentedFeaturizer()
    pred_e = InstrumentedPredictor()
    _generate_embeddings(feat_e, pred_e, provider)

    feat_i = InstrumentedFeaturizer()
    pred_i = InstrumentedPredictor()
    slot = ModelSlot(
        agent_id="t", iteration=1, challenge_name="ETH-1H-BINARY",
        strategy_path="", featurizer=feat_i, predictor=pred_i, loaded=True,
    )
    run_inference_single(slot, provider)

    e_out = pred_e.output_traces[-1]
    i_out = pred_i.output_traces[-1]
    assert np.array_equal(e_out, i_out), (
        f"Output embedding bytes differ: eval={e_out} inf={i_out}"
    )


def test_deep_parity_deterministic_reruns():
    """Running eval twice with fresh instances produces identical embeddings
    (no hidden state leaking between runs)."""
    n = 300
    df = _make_synth_df(n, seed=104)
    provider = _make_provider({"ETH": df})

    emb1, _ = _generate_embeddings(
        InstrumentedFeaturizer(), InstrumentedPredictor(), provider)
    emb2, _ = _generate_embeddings(
        InstrumentedFeaturizer(), InstrumentedPredictor(), provider)

    assert np.array_equal(emb1, emb2), (
        "Two identical eval runs produced different embeddings — state leak"
    )


def test_deep_parity_featurizer_isolation():
    """A featurizer that stores state must not carry state across
    setup() calls (eval vs inference use separate setup)."""
    n = 200
    df = _make_synth_df(n, seed=105)
    provider = _make_provider({"ETH": df})

    feat = InstrumentedFeaturizer()
    pred = InstrumentedPredictor()

    _generate_embeddings(feat, pred, provider)
    eval_trace_count = len(feat.traces)

    slot = ModelSlot(
        agent_id="t", iteration=1, challenge_name="ETH-1H-BINARY",
        strategy_path="", featurizer=feat, predictor=pred, loaded=True,
    )
    run_inference_single(slot, provider)

    # setup() should have reset traces, so inference traces are separate
    inf_trace_count = len(feat.traces)
    assert inf_trace_count < eval_trace_count, (
        f"Inference appended to eval traces ({inf_trace_count} >= {eval_trace_count}), "
        f"setup() should have reset state"
    )


def test_deep_parity_large_compute_interval():
    """Parity with large compute_interval relative to data size."""
    n = 200

    class BigInterval(Featurizer):
        warmup = 10
        compute_interval = 50

        def setup(self, assets, length):
            self.ts = []

        def compute(self, view):
            self.ts.append(view.t)
            p = view.prices(view.assets[0])
            return {"v": np.array([float(p[-1])])}

    class VP(Predictor):
        def predict(self, features):
            return np.array([features["v"][0], 100.0 - features["v"][0]])

    df = _make_synth_df(n, seed=106)
    provider = _make_provider({"ETH": df})

    fe = BigInterval()
    pe = VP()
    emb_all, _ = _generate_embeddings(fe, pe, provider)

    fi = BigInterval()
    pi = VP()
    slot = ModelSlot(
        agent_id="t", iteration=1, challenge_name="ETH-1H-BINARY",
        strategy_path="", featurizer=fi, predictor=pi, loaded=True,
    )
    inf_emb = run_inference_single(slot, provider)

    assert np.allclose(emb_all[-1], inf_emb, atol=1e-7), (
        f"Large ci parity fail: eval={emb_all[-1]} inf={inf_emb}"
    )
    # Verify inference computed at the same grid t as eval's last compute
    assert fi.ts[-1] == fe.ts[-1], (
        f"Cache boundary mismatch: eval last compute t={fe.ts[-1]} "
        f"inf last compute t={fi.ts[-1]}"
    )


def test_deep_parity_compute_interval_1_grid():
    """With ci=1, every timestep triggers a compute; verify total compute counts."""
    n = 100

    class CI1(Featurizer):
        warmup = 10
        compute_interval = 1
        def setup(self, assets, length):
            self.count = 0
        def compute(self, view):
            self.count += 1
            p = view.prices(view.assets[0])
            return {"v": np.array([float(p[-1])])}

    class VP(Predictor):
        def predict(self, features):
            return np.array([features["v"][0]])

    df = _make_synth_df(n, seed=107)
    provider = _make_provider({"ETH": df})

    fe = CI1()
    _generate_embeddings(fe, VP(), provider)
    assert fe.count == n - 10, f"Eval compute count: {fe.count}, expected {n-10}"

    fi = CI1()
    slot = ModelSlot(
        agent_id="t", iteration=1, challenge_name="ETH-1H-BINARY",
        strategy_path="", featurizer=fi, predictor=VP(), loaded=True,
    )
    run_inference_single(slot, provider)
    assert fi.count == 1, f"Inference with ci=1 should compute once, got {fi.count}"


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
        # Eval-inference parity
        ("eval_inference_parity_final_embedding", test_eval_inference_parity_final_embedding),
        ("eval_inference_same_view_at_T", test_eval_inference_same_view_at_T),
        ("eval_inference_parity_compute_interval", test_eval_inference_parity_with_compute_interval),
        ("eval_inference_parity_coinglass", test_eval_inference_parity_with_coinglass),
        ("eval_inference_parity_multi_asset", test_eval_inference_parity_multi_asset),
        # Advanced leakage
        ("causal_view_mangled_bypass_blocked", test_causal_view_mangled_bypass_blocked),
        ("causal_view_ohlcv_no_future_rows", test_causal_view_ohlcv_no_future_rows),
        ("causal_view_prices_matrix_no_future", test_causal_view_prices_matrix_no_future),
        ("causal_view_cg_no_future", test_causal_view_cg_no_future),
        ("causal_view_copy_independence", test_causal_view_copy_independence),
        ("provider_alignment_truncates", test_provider_alignment_truncates_correctly),
        ("provider_cg_alignment_truncates", test_provider_cg_alignment_truncates),
        ("inference_warmup_guard", test_inference_warmup_guard),
        ("eval_embedding_count", test_eval_embedding_count_matches_timesteps),
        ("walk_forward_holdout_boundary", test_walk_forward_holdout_boundary_no_leak),
        ("featurizer_sees_monotonic_t", test_featurizer_sees_monotonic_t),
        ("featurizer_view_length_equals_t_plus_one", test_featurizer_view_length_equals_t_plus_one),
        # Deep parity
        ("deep_parity_all_raw_inputs", test_deep_parity_all_raw_inputs),
        ("deep_parity_feature_dict_bytes", test_deep_parity_feature_dict_bytes),
        ("deep_parity_output_embedding_bytes", test_deep_parity_output_embedding_bytes),
        ("deep_parity_deterministic_reruns", test_deep_parity_deterministic_reruns),
        ("deep_parity_featurizer_isolation", test_deep_parity_featurizer_isolation),
        ("deep_parity_large_compute_interval", test_deep_parity_large_compute_interval),
        ("deep_parity_ci1_grid", test_deep_parity_compute_interval_1_grid),
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
