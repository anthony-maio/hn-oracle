import numpy as np
import pytest

from oracle import metrics as M
from oracle.text import clean_html, gradable, regex_hint, resolution_year


def test_brier_and_ece_perfect():
    y = np.array([0, 1, 0, 1])
    assert M.brier(y.astype(float), y) == 0
    assert M.ece(y.astype(float), y) == 0


def test_ece_known_value():
    p = np.array([0.8] * 10)
    y = np.array([1] * 5 + [0] * 5)
    assert M.ece(p, y) == pytest.approx(0.3)


def test_prf_and_best_f1():
    p = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    y = np.array([1, 1, 1, 0, 0])
    r = M.prf(p, y, 0.5)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 0, 1)
    assert r["recall"] == pytest.approx(2 / 3)
    b = M.best_f1(p, y)
    assert b["best_f1"] == pytest.approx(1.0)
    assert 0.2 < b["best_f1_threshold"] <= 0.3


def test_choose_gate_meets_recall():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 500), 0, 1)
    t = M.choose_gate(p, y, 0.9)
    assert M.prf(p, y, t)["recall"] >= 0.9
    assert M.prf(p, y, round(t + 0.01, 2))["recall"] < 0.9 or t == 0.99


def test_isotonic_monotone_and_ties():
    x = np.array([0.1, 0.1, 0.2, 0.5, 0.5, 0.9, 1.0, 1.0])
    y = np.array([0, 1, 0, 1, 0, 1, 1, 1])
    m = M.fit_isotonic(x, y)
    assert np.all(np.diff(m[1]) >= 0)
    q = M.apply_isotonic(m, x)
    assert q[0] == q[1]  # tied inputs share one value
    assert M.apply_isotonic(m, np.array([2.0]))[0] == m[1][-1]


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(1)
    true_p = rng.uniform(0.05, 0.95, 4000)
    y = (rng.uniform(size=4000) < true_p).astype(int)
    z = np.log(true_p / (1 - true_p))
    over = 1 / (1 + np.exp(-3 * z))  # 3x overconfident
    t = M.fit_temperature(over, y)
    assert 2.4 < t < 3.6
    assert M.ece(M.apply_temperature(over, t), y) < M.ece(over, y)


def test_cross_fit_shapes():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.7 + rng.normal(0.1, 0.1, 300), 0, 1)
    for method in ("temperature", "isotonic"):
        q, params = M.cross_fit(p, y, method)
        assert q.shape == p.shape and len(params) == 2
        assert np.all((q >= 0) & (q <= 1))


def test_prevalence_ci():
    p = np.full(600, 0.05)
    y = np.array([1] * 30 + [0] * 570)
    r = M.prevalence(p, y)
    assert r["gap"] == pytest.approx(0.0)
    assert r["ci_includes_zero"]


def test_kappa():
    assert M.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert M.cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)
    assert M.cohen_kappa([0, 1, 2, 3], [0, 1, 2, 2], "linear") > M.cohen_kappa([0, 1, 2, 3], [0, 1, 2, 0], "linear")


@pytest.mark.parametrize("answer,expected,rule", [
    ({"score": 0.5}, 1.5, "unit_interval"),
    ({"score": 0.5, "legend": {"a": 0.0, "b": 1 / 3, "c": 2 / 3, "d": 1.0}}, 1.5, "legend"),
    ({"score": 2.5, "legend": [{"value": 1}, {"value": 2}, {"value": 3}, {"value": 4}]}, 1.5, "legend"),
])
def test_score_level(answer, expected, rule):
    lvl, used = M.score_level(answer, 4)
    assert lvl == pytest.approx(expected) and used == rule


def test_text_helpers():
    assert clean_html("a<p>b &#x27;c&#x27; <i>d</i> &amp; e") == "a\n\nb 'c' d & e"
    assert regex_hint("Mark my words, Rust will win")
    assert not regex_hint("I like Rust")
    assert resolution_year(2012, "1_to_5y") == 2017
    assert resolution_year(2012, "unstated") is None
    assert gradable(2012, "5_to_10y", 2026) and not gradable(2020, "5_to_10y", 2026)
