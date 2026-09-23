"""Pure metric functions. No I/O; everything takes numpy arrays."""
from __future__ import annotations

import numpy as np

THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)
EPS = 1e-4


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, float) - y) ** 2))


def _bins(p, bins):
    edges = np.linspace(0, 1, bins + 1)
    return edges, np.clip(np.digitize(p, edges) - 1, 0, bins - 1)


def ece(p, y, bins: int = 10) -> float:
    """Expected calibration error, equal-width bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    _, idx = _bins(p, bins)
    return float(sum((idx == b).mean() * abs(p[idx == b].mean() - y[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def reliability(p, y, bins: int = 10) -> list[dict]:
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges, idx = _bins(p, bins)
    out = []
    for b in range(bins):
        m = idx == b
        out.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                    "mean_p": float(p[m].mean()) if m.any() else None,
                    "frac_true": float(y[m].mean()) if m.any() else None})
    return out


def prf(p, y, t: float) -> dict:
    y = np.asarray(y).astype(bool)
    pred = np.asarray(p, float) >= t
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    return {"threshold": float(t), "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
            "pass_rate": float(pred.mean())}


def f1_curve(p, y, thresholds=THRESHOLDS) -> np.ndarray:
    p, y = np.asarray(p, float), np.asarray(y).astype(bool)
    pred = p[None, :] >= np.asarray(thresholds)[:, None]
    tp = (pred & y).sum(1)
    fp = (pred & ~y).sum(1)
    fn = (~pred & y).sum(1)
    return np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)


def best_f1(p, y, thresholds=THRESHOLDS) -> dict:
    curve = f1_curve(p, y, thresholds)
    i = int(np.argmax(curve))
    return {"best_f1": float(curve[i]), "best_f1_threshold": float(thresholds[i])}


def choose_gate(p, y, target_recall: float, thresholds=THRESHOLDS) -> float:
    """Highest threshold whose recall still meets the target (most filtering at acceptable loss)."""
    ok = [t for t in thresholds if prf(p, y, t)["recall"] >= target_recall]
    return float(max(ok)) if ok else float(min(thresholds))


def bootstrap(stat, *arrays, n: int = 2000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    m = len(arrays[0])
    vals = [stat(*(a[idx] for a in arrays)) for idx in (rng.integers(0, m, m) for _ in range(n))]
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]


def prevalence(p, y, n_boot: int = 2000, seed: int = 0) -> dict:
    """Sum-of-probabilities estimator vs labeled truth."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    lo, hi = bootstrap(lambda a, b: a.mean() - b.mean(), p, y, n=n_boot, seed=seed)
    gap = float(p.mean() - y.mean())
    return {"estimated": float(p.mean()), "labeled": float(y.mean()), "gap": gap,
            "gap_ci95": [lo, hi], "ci_includes_zero": bool(lo <= 0 <= hi), "n": int(len(p))}


# ---------------------------------------------------------------- recalibration


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _nll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(p, y) -> float:
    z, y = _logit(p), np.asarray(y, float)
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 600))
    losses = [_nll(1 / (1 + np.exp(-z / t)), y) for t in grid]
    return float(grid[int(np.argmin(losses))])


def apply_temperature(p, t: float):
    return 1 / (1 + np.exp(-_logit(p) / t))


def fit_isotonic(x, y) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators. Returns (block upper x, block value), non-decreasing."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ux, inv = np.unique(x, return_inverse=True)  # tied x must share one value
    sums = np.bincount(inv, weights=y)
    wts = np.bincount(inv).astype(float)
    vals, ws, xs = [], [], []
    for xi, s, w in zip(ux, sums, wts):
        vals.append(s / w); ws.append(w); xs.append(xi)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w2 = ws[-2] + ws[-1]
            v2 = (vals[-2] * ws[-2] + vals[-1] * ws[-1]) / w2
            vals[-2:], ws[-2:], xs[-2:] = [v2], [w2], [xs[-1]]
    return np.array(xs), np.array(vals)


def apply_isotonic(model, x):
    xs, vals = model
    idx = np.clip(np.searchsorted(xs, np.asarray(x, float), side="left"), 0, len(vals) - 1)
    return vals[idx]


def cross_fit(p, y, method: str, seed: int = 0) -> tuple[np.ndarray, list]:
    """Two-fold, label-stratified: fit on one half, predict the other, then swap.

    Every uniform comment gets an out-of-fold recalibrated probability.
    """
    p, y = np.asarray(p, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(p), int)
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        fold[idx[len(idx) // 2:]] = 1
    out = np.empty_like(p)
    params = []
    for f in (0, 1):
        tr, te = fold != f, fold == f
        if method == "temperature":
            t = fit_temperature(p[tr], y[tr])
            out[te] = apply_temperature(p[te], t)
            params.append(t)
        elif method == "isotonic":
            m = fit_isotonic(p[tr], y[tr])
            out[te] = apply_isotonic(m, p[te])
            params.append(len(m[0]))
        else:
            raise ValueError(method)
    return out, params


# ---------------------------------------------------------------- agreement


def cohen_kappa(a, b, weights: str | None = None) -> float | None:
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0:
        return None
    cats = sorted(set(a.tolist()) | set(b.tolist()))
    k = len(cats)
    if k < 2:
        return 1.0
    pos = {c: i for i, c in enumerate(cats)}
    obs = np.zeros((k, k))
    for x, z in zip(a, b):
        obs[pos[x], pos[z]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    if weights == "linear":
        i, j = np.indices((k, k))
        w = np.abs(i - j) / (k - 1)
    else:
        w = 1 - np.eye(k)
    denom = (w * exp).sum()
    return float(1 - (w * obs).sum() / denom) if denom else 1.0


# ---------------------------------------------------------------- Jev Score answers


def score_level(answer: dict, n_levels: int) -> tuple[float, str]:
    """Map a Jev Score answer to level units 0..n_levels-1.

    Jev returns a probability-weighted `score` plus a `legend` mapping levels to values. We read the
    numeric level values from the legend (keys or values) and interpolate; if there is no usable
    legend we assume levels are evenly spaced on [0, 1]. Returns (level, rule_used).
    """
    s = float(answer["score"])
    legend = answer.get("legend")
    nums: list[float] = []
    if isinstance(legend, dict):
        for k, v in legend.items():
            for cand in (v, k):
                try:
                    nums.append(float(cand))
                    break
                except (TypeError, ValueError):
                    continue
    elif isinstance(legend, list):
        for item in legend:
            v = item.get("value") if isinstance(item, dict) else item
            try:
                nums.append(float(v))
            except (TypeError, ValueError):
                pass
    if len(nums) == n_levels:
        levels = np.sort(np.array(nums))
        return float(np.interp(s, levels, np.arange(n_levels))), "legend"
    if 0.0 <= s <= 1.0:
        return s * (n_levels - 1), "unit_interval"
    return s, "raw"
