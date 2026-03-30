import logging
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, log_loss

from model_iteration_tool.data import DataProvider, BREAKOUT_ASSETS
from model_iteration_tool.featurizer import Featurizer, Predictor

logger = logging.getLogger(__name__)

EPS = 1e-12


@dataclass
class ChallengeConfig:
    name: str
    challenge_type: str
    primary_asset: str
    assets: list
    embedding_dim: int
    horizon: int
    weight: float = 1.0
    range_lookback: int = 0
    barrier_pct: float = 25.0
    min_range_pct: float = 1.0


CHALLENGES = {
    "ETH-1H-BINARY": ChallengeConfig(
        name="ETH-1H-BINARY", challenge_type="binary",
        primary_asset="ETH", assets=["ETH"],
        embedding_dim=2, horizon=60, weight=1.0,
    ),
    "ETH-HITFIRST-100M": ChallengeConfig(
        name="ETH-HITFIRST-100M", challenge_type="hitfirst",
        primary_asset="ETH", assets=["ETH"],
        embedding_dim=3, horizon=100, weight=2.5,
    ),
    "ETH-LBFGS": ChallengeConfig(
        name="ETH-LBFGS", challenge_type="lbfgs",
        primary_asset="ETH", assets=["ETH"],
        embedding_dim=17, horizon=60, weight=3.5,
    ),
    "BTC-LBFGS-6H": ChallengeConfig(
        name="BTC-LBFGS-6H", challenge_type="lbfgs",
        primary_asset="BTC", assets=["BTC"],
        embedding_dim=17, horizon=360, weight=2.875,
    ),
    "MULTI-BREAKOUT": ChallengeConfig(
        name="MULTI-BREAKOUT", challenge_type="breakout",
        primary_asset="BTC", assets=list(BREAKOUT_ASSETS),
        embedding_dim=2 * len(BREAKOUT_ASSETS), horizon=0,
        weight=5.0, range_lookback=5760,
        barrier_pct=25.0, min_range_pct=1.0,
    ),
    "XSEC-RANK": ChallengeConfig(
        name="XSEC-RANK", challenge_type="xsec_rank",
        primary_asset="BTC", assets=list(BREAKOUT_ASSETS),
        embedding_dim=len(BREAKOUT_ASSETS), horizon=240,
        weight=3.0,
    ),
}


def _rolling_std(x, window):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < window:
        return np.array([], dtype=np.float64)
    c1 = np.concatenate(([0.0], np.cumsum(x)))
    c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    s1 = c1[window:] - c1[:-window]
    s2 = c2[window:] - c2[:-window]
    var = (s2 - (s1 * s1) / window) / max(window - 1, 1)
    return np.sqrt(np.maximum(var, 0.0))


def _sigma_series(prices, horizon, vol_window):
    prices = np.asarray(prices, dtype=np.float64)
    T = len(prices)
    if T <= horizon:
        return np.full(T, np.nan)
    r = np.log(prices[horizon:] + EPS) - np.log(prices[:-horizon] + EPS)
    sig_raw = _rolling_std(r, vol_window)
    sig = np.full(T - horizon, np.nan)
    if len(sig_raw) > 0:
        sig[vol_window - 1:] = sig_raw
    out = np.full(T, np.nan)
    out[horizon:] = sig
    return out


def make_binary_labels(prices, horizon):
    prices = np.asarray(prices, dtype=np.float64)
    T = len(prices)
    if T <= horizon:
        return np.array([], dtype=np.int32), np.array([], dtype=bool)
    n = T - horizon
    labels = (prices[horizon:horizon + n] > prices[:n]).astype(np.int32)
    valid = np.ones(n, dtype=bool)
    return labels, valid


def make_hitfirst_labels(prices, horizon, vol_window=7200):
    """Generate hitfirst labels. Sigma uses forward log-returns (intentional:
    label depends on future path). vol_window=7200 matches hitfirst.py."""
    prices = np.asarray(prices, dtype=np.float64)
    T = len(prices)
    if T <= horizon:
        return np.array([], dtype=np.int32), np.array([], dtype=bool)

    log_p = np.log(prices + EPS)
    n = T - horizon

    r_h = np.log(prices[horizon:] + EPS) - np.log(prices[:-horizon] + EPS)
    if len(r_h) < vol_window:
        return np.full(n, -1, dtype=np.int32), np.zeros(n, dtype=bool)
    sig_raw = _rolling_std(r_h, vol_window)
    sig = np.full(n, np.nan)
    if len(sig_raw) > 0:
        sig[vol_window - 1:] = sig_raw

    labels = np.full(n, -1, dtype=np.int32)
    valid_mask = np.isfinite(sig) & (sig > 0)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) > 0:
        log_p_padded = log_p[:n + horizon]
        for t in valid_indices:
            s = sig[t]
            path = log_p_padded[t + 1: t + 1 + horizon] - log_p[t]
            up_first = -1
            dn_first = -1
            up_mask = path >= s
            dn_mask = path <= -s
            if up_mask.any():
                up_first = np.argmax(up_mask)
            if dn_mask.any():
                dn_first = np.argmax(dn_mask)
            if up_first < 0 and dn_first < 0:
                labels[t] = 2
            elif up_first >= 0 and dn_first < 0:
                labels[t] = 0
            elif dn_first >= 0 and up_first < 0:
                labels[t] = 1
            elif up_first < dn_first:
                labels[t] = 0
            elif dn_first < up_first:
                labels[t] = 1
            else:
                labels[t] = 2

    valid = labels >= 0
    return labels, valid


def make_lbfgs_labels(prices, horizon, vol_window=3600):
    """Vol-normalized return buckets matching utils.make_bins_from_price exactly.

    Sigma is computed via sigma_from_price (causal -- uses returns ending at
    price[t], not future prices).  vol_window=3600 matches the actual call in
    bucket_forecast.compute_linear_salience:
        vol_window = max(MIN_REQUIRED_SAMPLES // 2, 1000) = 3600.

    Bucket boundaries (from utils.py lines 99-104):
        0: z <= -2        (far down)
        1: -2 < z < -1    (slightly down)
        2: -1 <= z <= 1   (neutral)
        3: 1 < z < 2      (slightly up)
        4: z >= 2          (far up)
    """
    prices = np.asarray(prices, dtype=np.float64)
    T = len(prices)
    if T <= horizon:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int64)
    r = np.log(prices[horizon:] + EPS) - np.log(prices[:T - horizon] + EPS)
    sig = _sigma_series(prices, horizon, vol_window)
    sig_start = sig[:len(r)]
    valid_mask = np.isfinite(sig_start) & (sig_start > 0)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int64)
    z = r[valid_idx] / (sig_start[valid_idx] + EPS)
    buckets = np.zeros(len(z), dtype=np.int32)
    buckets[z <= -2.0] = 0
    buckets[(z > -2.0) & (z < -1.0)] = 1
    buckets[(z >= -1.0) & (z <= 1.0)] = 2
    buckets[(z > 1.0) & (z < 2.0)] = 3
    buckets[z >= 2.0] = 4
    return buckets, valid_idx


@dataclass
class BreakoutEvent:
    trigger_t: int
    resolution_t: int
    asset_idx: int
    asset: str
    direction: int
    label: int


def detect_breakouts(prices_matrix, assets, range_lookback, barrier_pct,
                     min_range_pct, max_pending=43200):
    T, n_assets = prices_matrix.shape
    if n_assets != len(assets):
        raise ValueError(f"prices_matrix has {n_assets} columns but {len(assets)} assets given")
    if T <= range_lookback:
        return []

    events = []
    barrier_frac = barrier_pct / 100.0
    range_frac = min_range_pct / 100.0

    for aidx in range(n_assets):
        prices = prices_matrix[:, aidx]
        swv = sliding_window_view(prices, range_lookback)
        rng_high = swv.max(axis=1)
        rng_low = swv.min(axis=1)
        rng_width = rng_high - rng_low

        p_slice = prices[range_lookback:]
        rh_slice = rng_high[:len(p_slice)]
        rl_slice = rng_low[:len(p_slice)]
        rw_slice = rng_width[:len(p_slice)]

        wide_enough = rw_slice >= p_slice * range_frac
        high_break = (p_slice > rh_slice) & wide_enough
        low_break = (p_slice < rl_slice) & wide_enough

        pending_high = None
        pending_low = None
        asset_name = assets[aidx]
        for i in range(len(p_slice)):
            t = range_lookback + i
            p = p_slice[i]
            if pending_high is not None:
                if t - pending_high[0] > max_pending:
                    pending_high = None
                elif p >= pending_high[1]:
                    events.append(BreakoutEvent(
                        pending_high[0], t, aidx, asset_name, 1, 1))
                    pending_high = None
                elif p <= pending_high[2]:
                    events.append(BreakoutEvent(
                        pending_high[0], t, aidx, asset_name, 1, 0))
                    pending_high = None
            if pending_low is not None:
                if t - pending_low[0] > max_pending:
                    pending_low = None
                elif p <= pending_low[1]:
                    events.append(BreakoutEvent(
                        pending_low[0], t, aidx, asset_name, -1, 1))
                    pending_low = None
                elif p >= pending_low[2]:
                    events.append(BreakoutEvent(
                        pending_low[0], t, aidx, asset_name, -1, 0))
                    pending_low = None
            if high_break[i] and pending_high is None:
                bd = rw_slice[i] * barrier_frac
                pending_high = (t, p + bd, p - bd)
            elif low_break[i] and pending_low is None:
                bd = rw_slice[i] * barrier_frac
                pending_low = (t, p - bd, p + bd)

    events.sort(key=lambda e: e.resolution_t)
    return events


def make_xsec_labels(prices_matrix, horizon):
    prices_matrix = np.asarray(prices_matrix, dtype=np.float64)
    T, N = prices_matrix.shape
    if T <= horizon:
        n = 0
        return np.zeros((n, N), dtype=np.int32), np.zeros(n, dtype=bool)
    n = T - horizon
    p0 = prices_matrix[:n]
    p1 = prices_matrix[horizon:horizon + n]
    ok = (p0 > 0) & (p1 > 0)
    ret = np.where(ok, p1 / p0 - 1.0, np.nan)
    med = np.nanmedian(ret, axis=1, keepdims=True)
    labels = np.where(np.isnan(ret), -1, (ret > med).astype(np.int32))
    valid = np.all(labels >= 0, axis=1)
    return labels, valid


def _generate_embeddings(featurizer, predictor, provider, warmup=None):
    if warmup is None:
        warmup = featurizer.warmup
    T = provider.length
    featurizer.setup(provider.assets, T)
    embeddings = []
    cache = None
    cache_t = -999999
    for t in range(warmup, T):
        if cache is None or (t - cache_t) >= featurizer.compute_interval:
            view = provider.view(t)
            cache = featurizer.compute(view)
            cache_t = t
        emb = predictor.predict(cache)
        embeddings.append(np.asarray(emb, dtype=np.float32).ravel())
    if not embeddings:
        return np.zeros((0, 0), dtype=np.float32), warmup
    return np.array(embeddings, dtype=np.float32), warmup


def _walk_forward_binary(embeddings, labels, chunk_size=4000, lag=60,
                         min_train=500, score_col=0, holdout_start=0):
    N = len(labels)
    if len(embeddings) != N:
        raise ValueError(f"embeddings length {len(embeddings)} != labels length {N}")
    results = []
    for test_start in range(min_train + lag, N, chunk_size):
        test_end = min(test_start + chunk_size, N)
        train_end = test_start - lag
        if train_end < min_train:
            continue
        lab_test = labels[test_start:test_end]
        if len(np.unique(lab_test)) < 2:
            continue
        direct_scores = embeddings[test_start:test_end, score_col]
        direct_auc = float(roc_auc_score(lab_test, direct_scores))
        results.append({
            "start": test_start, "end": test_end,
            "auc": direct_auc,
            "in_holdout": test_start >= holdout_start,
        })
    return results


def _balanced_accuracy(y_true, y_pred, K):
    """Per-class accuracy averaged, matching bucket_forecast.py."""
    per_c = []
    for c in range(K):
        mask = y_true == c
        if mask.sum() > 0:
            per_c.append(float((y_pred[mask] == c).sum()) / float(mask.sum()))
    return float(np.mean(per_c)) if per_c else 0.0


def _walk_forward_lbfgs(embeddings, labels, chunk_size=6000, lag=60, min_train=500,
                        holdout_start=0):
    """Walk-forward balanced accuracy on argmax bucket predictions.

    Matches bucket_forecast.py: no meta model, just direct balanced accuracy
    of the 5-bucket probability argmax predictions.
    """
    N = len(labels)
    if len(embeddings) != N:
        raise ValueError(f"embeddings length {len(embeddings)} != labels length {N}")
    K = 5
    random_bal = 1.0 / K
    results = []
    for test_start in range(min_train + lag, N, chunk_size):
        test_end = min(test_start + chunk_size, N)
        if test_end - test_start < 200:
            continue
        lab_test = labels[test_start:test_end]
        if len(np.unique(lab_test)) < 2:
            continue
        emb_test = embeddings[test_start:test_end, :K]
        probs = np.clip(emb_test, 1e-6, None)
        probs = probs / probs.sum(axis=1, keepdims=True)
        preds = probs.argmax(axis=1)
        bal_acc = _balanced_accuracy(lab_test, preds, K)
        results.append({
            "start": test_start, "end": test_end,
            "balanced_accuracy": bal_acc,
            "lift_over_random": bal_acc - random_bal,
            "in_holdout": test_start >= holdout_start,
        })
    return results


def _walk_forward_xsec_spearman(embeddings, returns_2d, valid_mask, n_assets,
                                chunk_size=4000, lag=60, min_train=200,
                                holdout_start=0):
    """Walk-forward Spearman correlation between predicted scores and actual returns.

    At each valid timestep, computes rank correlation across assets between
    the agent's predicted scores and actual forward returns.
    """
    N = len(valid_mask)
    if embeddings.shape[0] != N:
        raise ValueError(f"embeddings rows {embeddings.shape[0]} != valid_mask length {N}")
    if returns_2d.shape != (N, n_assets):
        raise ValueError(f"returns_2d shape {returns_2d.shape} != expected ({N}, {n_assets})")
    results = []
    for test_start in range(min_train + lag, N, chunk_size):
        test_end = min(test_start + chunk_size, N)
        if test_end - test_start < 50:
            continue
        test_valid = valid_mask[test_start:test_end]
        if test_valid.sum() < 50:
            continue
        emb_test = embeddings[test_start:test_end][test_valid]
        ret_test = returns_2d[test_start:test_end][test_valid]
        spearmans = []
        for i in range(len(emb_test)):
            scores_i = emb_test[i]
            returns_i = ret_test[i]
            finite = np.isfinite(returns_i) & np.isfinite(scores_i)
            if finite.sum() < 5:
                continue
            rho, _ = spearmanr(scores_i[finite], returns_i[finite])
            if np.isfinite(rho):
                spearmans.append(rho)
        if len(spearmans) < 20:
            continue
        results.append({
            "start": test_start, "end": test_end,
            "mean_spearman": float(np.mean(spearmans)),
            "n_timesteps": len(spearmans),
            "in_holdout": test_start >= holdout_start,
        })
    return results


def _evaluate_binary(featurizer, predictor, provider, config, holdout_start=0):
    asset = config.primary_asset
    horizon = config.horizon
    prices = provider.prices(asset)
    labels_all, valid_all = make_binary_labels(prices, horizon)
    embeddings, warmup = _generate_embeddings(featurizer, predictor, provider)
    n_usable = len(labels_all) - warmup
    if n_usable <= 0:
        return {"error": "not enough data after warmup"}
    emb = embeddings[:n_usable]
    lab = labels_all[warmup:warmup + n_usable]
    hs = max(0, holdout_start - warmup)
    windows = _walk_forward_binary(emb, lab, holdout_start=hs)
    if not windows:
        return {"error": "no valid walk-forward windows"}
    holdout_windows = [w for w in windows if w.get("in_holdout")]
    if not holdout_windows:
        return {"error": "no walk-forward windows in holdout period"}
    return {
        "challenge": config.name, "type": "binary",
        "n_timesteps": provider.length, "n_evaluated": n_usable,
        "horizon": horizon, "windows": windows,
        "mean_auc": float(np.nanmean([w["auc"] for w in holdout_windows])),
    }


def _logit(p):
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p) - np.log(1.0 - p)


def _walk_forward_hitfirst(up_scores, dn_scores, y_up, y_dn, probs, lab,
                           chunk_size=4000, lag=60, min_train=2000, holdout_start=0):
    """Walk-forward evaluation for hitfirst: train LogReg on expanding window,
    test on next chunk. Returns list of window results."""
    N = len(lab)
    windows = []
    for test_start in range(min_train + lag, N, chunk_size):
        test_end = min(test_start + chunk_size, N)
        if test_end - test_start < 200:
            continue
        train_up_sc = up_scores[:test_start]
        train_dn_sc = dn_scores[:test_start]
        train_y_up = y_up[:test_start]
        train_y_dn = y_dn[:test_start]
        test_up_sc = up_scores[test_start:test_end]
        test_dn_sc = dn_scores[test_start:test_end]
        test_y_up = y_up[test_start:test_end]
        test_y_dn = y_dn[test_start:test_end]
        test_lab = lab[test_start:test_end]
        test_probs = probs[test_start:test_end]

        w = {"start": test_start, "end": test_end, "up_auc": np.nan, "dn_auc": np.nan,
             "in_holdout": test_start >= holdout_start}

        if len(np.unique(train_y_up)) >= 2 and len(np.unique(test_y_up)) >= 2:
            clf_up = LogisticRegression(
                C=1.0, solver="lbfgs", class_weight="balanced",
                max_iter=500, tol=1e-4, random_state=42)
            clf_up.fit(train_up_sc.reshape(-1, 1), train_y_up)
            pred_up = clf_up.decision_function(test_up_sc.reshape(-1, 1))
            w["up_auc"] = float(roc_auc_score(test_y_up, pred_up))

        if len(np.unique(train_y_dn)) >= 2 and len(np.unique(test_y_dn)) >= 2:
            clf_dn = LogisticRegression(
                C=1.0, solver="lbfgs", class_weight="balanced",
                max_iter=500, tol=1e-4, random_state=42)
            clf_dn.fit(train_dn_sc.reshape(-1, 1), train_y_dn)
            pred_dn = clf_dn.decision_function(test_dn_sc.reshape(-1, 1))
            w["dn_auc"] = float(roc_auc_score(test_y_dn, pred_dn))

        if len(test_lab) > 0 and len(np.unique(test_lab)) >= 2:
            w["direct_log_loss"] = float(log_loss(test_lab, test_probs, labels=[0, 1, 2]))
        windows.append(w)
    return windows


def _evaluate_hitfirst(featurizer, predictor, provider, config, holdout_start=0):
    """Walk-forward hitfirst evaluation: separate up/down LogReg
    on logit-transformed probabilities, expanding train window."""
    asset = config.primary_asset
    horizon = config.horizon
    prices = provider.prices(asset)
    labels_all, valid_all = make_hitfirst_labels(prices, horizon)
    embeddings, warmup = _generate_embeddings(featurizer, predictor, provider)
    n_usable = len(labels_all) - warmup
    if n_usable <= 0:
        return {"error": "not enough data after warmup"}
    emb = embeddings[:n_usable]
    lab_raw = labels_all[warmup:warmup + n_usable]
    val_raw = valid_all[warmup:warmup + n_usable]

    valid_orig_indices = np.where(val_raw)[0]
    hs_in_valid = int(np.searchsorted(valid_orig_indices, max(0, holdout_start - warmup)))

    emb = emb[val_raw]
    lab = lab_raw[val_raw]
    if len(lab) < 500:
        return {"error": f"only {len(lab)} valid samples, need >= 500"}
    if emb.ndim < 2 or emb.shape[1] < 3:
        return {"error": f"hitfirst requires embedding dim >= 3, got shape {emb.shape}"}

    probs = np.clip(emb[:, :3], EPS, 1.0 - EPS)
    row_sums = probs.sum(axis=1, keepdims=True)
    probs = probs / np.where(row_sums > 0, row_sums, 1.0)

    up_scores = _logit(probs[:, 0])
    dn_scores = _logit(probs[:, 1])
    up_scores = np.nan_to_num(up_scores, nan=0.0, posinf=0.0, neginf=0.0)
    dn_scores = np.nan_to_num(dn_scores, nan=0.0, posinf=0.0, neginf=0.0)

    y_up = (lab == 0).astype(np.int32)
    y_dn = (lab == 1).astype(np.int32)

    windows = _walk_forward_hitfirst(up_scores, dn_scores, y_up, y_dn, probs, lab,
                                     holdout_start=hs_in_valid)
    if not windows:
        return {"error": "no valid walk-forward windows"}

    holdout_windows = [w for w in windows if w.get("in_holdout")]
    if not holdout_windows:
        return {"error": "no walk-forward windows in holdout period"}
    up_aucs = [w["up_auc"] for w in holdout_windows if np.isfinite(w.get("up_auc", float("nan")))]
    dn_aucs = [w["dn_auc"] for w in holdout_windows if np.isfinite(w.get("dn_auc", float("nan")))]
    ll_vals = [w["direct_log_loss"] for w in holdout_windows if "direct_log_loss" in w and np.isfinite(w["direct_log_loss"])]
    return {
        "challenge": config.name, "type": "hitfirst",
        "n_timesteps": provider.length, "n_valid": len(lab),
        "horizon": horizon, "windows": windows,
        "direct_log_loss": float(np.mean(ll_vals)) if ll_vals else np.nan,
        "up_auc": float(np.mean(up_aucs)) if up_aucs else np.nan,
        "dn_auc": float(np.mean(dn_aucs)) if dn_aucs else np.nan,
    }


def _evaluate_lbfgs(featurizer, predictor, provider, config, holdout_start=0):
    """Evaluate LBFGS challenge using balanced accuracy on argmax bucket predictions.

    Matches bucket_forecast.py: direct balanced accuracy, no meta model.
    Random baseline is 0.2 (1/5 buckets).
    """
    asset = config.primary_asset
    horizon = config.horizon
    prices = provider.prices(asset)
    bucket_labels, valid_idx = make_lbfgs_labels(prices, horizon)
    embeddings, warmup = _generate_embeddings(featurizer, predictor, provider)
    after_warmup_mask = valid_idx >= warmup
    valid_after = valid_idx[after_warmup_mask]
    labs_after = bucket_labels[after_warmup_mask]
    if len(valid_after) < 500:
        return {"error": f"only {len(valid_after)} valid samples"}
    emb_indices = valid_after - warmup
    keep = emb_indices < len(embeddings)
    emb_indices = emb_indices[keep]
    lab = labs_after[keep]
    emb = embeddings[emb_indices]

    hs_in_compact = int(np.searchsorted(valid_after[keep], holdout_start))

    windows = _walk_forward_lbfgs(emb, lab, holdout_start=hs_in_compact)
    if not windows:
        return {"error": "no valid walk-forward windows"}
    holdout_windows = [w for w in windows if w.get("in_holdout")]
    if not holdout_windows:
        return {"error": "no walk-forward windows in holdout period"}
    return {
        "challenge": config.name, "type": "lbfgs",
        "n_timesteps": provider.length, "n_valid": len(lab),
        "horizon": horizon, "windows": windows,
        "mean_balanced_accuracy": float(np.nanmean([w["balanced_accuracy"] for w in holdout_windows])),
        "mean_lift": float(np.nanmean([w["lift_over_random"] for w in holdout_windows])),
    }


def _evaluate_breakout(featurizer, predictor, provider, config, holdout_start=0):
    available = [a for a in config.assets if a in provider.assets]
    if len(available) < 5:
        return {"error": f"only {len(available)} assets available, need >= 5"}
    asset_to_col = {a: i for i, a in enumerate(available)}
    pm = np.column_stack([provider.prices(a) for a in available])
    logger.info("Detecting breakout events across %d assets...", len(available))
    events = detect_breakouts(
        pm, available, config.range_lookback,
        config.barrier_pct, config.min_range_pct)
    if len(events) < 50:
        return {"error": f"only {len(events)} breakout events, need >= 50"}
    logger.info("Found %d breakout events, generating embeddings...", len(events))
    embeddings, warmup = _generate_embeddings(featurizer, predictor, provider)
    n_fallback = 0
    scored_events = []
    for ev in events:
        emb_idx = ev.trigger_t - warmup
        if emb_idx < 0 or emb_idx >= len(embeddings):
            continue
        emb = embeddings[emb_idx]
        col = asset_to_col[ev.asset]
        if col * 2 < len(emb):
            score = float(emb[col * 2])
        else:
            score = 0.5
            n_fallback += 1
        scored_events.append((ev.resolution_t, ev.trigger_t, score, ev.label))
    if n_fallback > 0:
        logger.warning("Breakout: %d/%d events used fallback score 0.5 "
                       "(embedding too short for asset column)", n_fallback, len(scored_events))
    if len(scored_events) < 30:
        return {"error": f"only {len(scored_events)} scoreable events"}
    scored_events.sort(key=lambda x: x[0])
    scores = np.array([s[2] for s in scored_events])
    labels = np.array([s[3] for s in scored_events])
    trigger_times = np.array([s[1] for s in scored_events])
    n_ev = len(labels)

    holdout_event_idx = int(np.searchsorted(trigger_times, holdout_start))

    chunk_size = max(30, n_ev // 5)
    min_train = max(30, n_ev // 4)
    windows = []
    for test_start in range(min_train, n_ev, chunk_size):
        test_end = min(test_start + chunk_size, n_ev)
        if test_end - test_start < 20:
            continue
        lab_test = labels[test_start:test_end]
        sc_test = scores[test_start:test_end]
        if len(np.unique(lab_test)) < 2:
            continue
        windows.append({
            "start": test_start, "end": test_end,
            "auc": float(roc_auc_score(lab_test, sc_test)),
            "in_holdout": test_start >= holdout_event_idx,
        })
    if not windows:
        return {"error": "no valid walk-forward windows for breakout"}
    holdout_windows = [w for w in windows if w.get("in_holdout")]
    if not holdout_windows:
        return {"error": "no walk-forward windows in holdout period for breakout"}
    return {
        "challenge": config.name, "type": "breakout",
        "n_assets": len(available), "n_events": len(events),
        "n_scored": len(scored_events), "n_fallback": n_fallback,
        "windows": windows,
        "mean_auc": float(np.nanmean([w["auc"] for w in holdout_windows])),
    }


def _evaluate_xsec(featurizer, predictor, provider, config, holdout_start=0):
    """Evaluate XSEC-RANK challenge using Spearman correlation.

    At each timestep, the agent produces a score per asset. These are compared
    to actual forward returns via Spearman rank correlation. Higher is better.
    """
    available = [a for a in config.assets if a in provider.assets]
    if len(available) < 5:
        return {"error": f"only {len(available)} assets available, need >= 5"}
    pm = np.column_stack([provider.prices(a) for a in available])
    horizon = config.horizon
    T, N = pm.shape
    if T <= horizon:
        return {"error": "not enough data for horizon"}
    n = T - horizon
    p0 = pm[:n]
    p1 = pm[horizon:horizon + n]
    ok = (p0 > 0) & (p1 > 0)
    returns_2d = np.where(ok, p1 / p0 - 1.0, np.nan)
    valid_mask = np.all(np.isfinite(returns_2d), axis=1)

    embeddings, warmup = _generate_embeddings(featurizer, predictor, provider)
    if embeddings.shape[1] < len(available):
        logger.warning("XSEC: embedding dim %d < %d assets, padding with zeros",
                        embeddings.shape[1], len(available))
        pad = np.zeros((embeddings.shape[0], len(available) - embeddings.shape[1]))
        embeddings = np.hstack([embeddings, pad])
    n_usable = min(n - warmup, len(embeddings))
    if n_usable <= 0:
        return {"error": "not enough data"}
    emb = embeddings[:n_usable, :len(available)]
    ret = returns_2d[warmup:warmup + n_usable]
    val = valid_mask[warmup:warmup + n_usable]
    hs = max(0, holdout_start - warmup)
    windows = _walk_forward_xsec_spearman(emb, ret, val, len(available), holdout_start=hs)
    if not windows:
        return {"error": "no valid walk-forward windows"}
    holdout_windows = [w for w in windows if w.get("in_holdout")]
    if not holdout_windows:
        return {"error": "no walk-forward windows in holdout period"}
    return {
        "challenge": config.name, "type": "xsec_rank",
        "n_assets": len(available), "n_timesteps": provider.length,
        "n_usable": n_usable, "horizon": horizon, "windows": windows,
        "mean_spearman": float(np.nanmean([w["mean_spearman"] for w in holdout_windows])),
    }


_DISPATCH = {
    "binary": _evaluate_binary,
    "hitfirst": _evaluate_hitfirst,
    "lbfgs": _evaluate_lbfgs,
    "breakout": _evaluate_breakout,
    "xsec_rank": _evaluate_xsec,
}


def evaluate(challenge_name, featurizer, predictor, provider=None,
             days_back=90, interval="1m", holdout_start=0):
    """Evaluate a featurizer + predictor. Fetches data if provider is None.

    holdout_start: index in the full price array where the holdout period begins.
    Only walk-forward windows testing within the holdout contribute to reported
    headline metrics (mean_auc, mean_spearman, etc.). Dev-period windows are
    still computed and returned for diagnostics but excluded from the headline.
    """
    if challenge_name not in CHALLENGES:
        raise ValueError(f"Unknown challenge {challenge_name!r}. Available: {list(CHALLENGES.keys())}")
    config = CHALLENGES[challenge_name]
    if provider is None:
        from model_iteration_tool.data import fetch_assets
        data, _ = fetch_assets(
            assets=config.assets, interval=interval, days_back=days_back)
        provider = DataProvider(data)
    eval_fn = _DISPATCH[config.challenge_type]
    return eval_fn(featurizer, predictor, provider, config, holdout_start=holdout_start)
