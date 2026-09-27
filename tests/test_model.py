"""Unit tests for the model and the theoretical results (run: python -m pytest -q)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from difss.model import (YE_L, agg, dispersion_lipschitz, ent_dispersion, ent_ye,  # noqa: E402
                         entropy_weights, orthopairs, weight_bound)

rng = np.random.default_rng(0)


def random_ifns(m, p, eps=0.01):
    mu = rng.uniform(eps, 1, (m, p))
    nu = rng.uniform(0, 1, (m, p)) * (1 - mu)
    return mu, nu


def test_orthopair_feasible():
    x = rng.normal(size=(50, 7))
    h = rng.uniform(0, 1, (50, 7))
    mu, nu, _ = orthopairs(x, True, h, 0.01, 0.25)
    assert np.all(mu >= 0.01 - 1e-12) and np.all(nu >= -1e-12) and np.all(mu + nu <= 1 + 1e-12)


def test_ifwg_closure_idempotency():
    mu, nu = random_ifns(30, 5)
    w = entropy_weights(rng.uniform(0, 1, 5))
    gm, gn = agg(mu, nu, w)
    assert np.all(gm + gn <= 1 + 1e-12)
    m0, n0 = np.repeat(mu[:, :1], 5, 1), np.repeat(nu[:, :1], 5, 1)
    a, b = agg(m0, n0, w)
    assert np.allclose(a, mu[:, 0]) and np.allclose(b, nu[:, 0])


def test_weight_bound_holds_generic():
    """Proposition 5.1: ||w - w~||_1 <= 2 p Lam / (D - p Lam)."""
    for _ in range(2000):
        p = rng.integers(2, 9)
        E = rng.uniform(0.2, 0.99, p)
        Lam = rng.uniform(0, 0.05)
        Et = np.clip(E + rng.uniform(-Lam, Lam, p), 0, 1)
        D = (1 - E).sum()
        b = weight_bound(D, p, Lam)
        if np.isfinite(b):
            assert np.abs(entropy_weights(E) - entropy_weights(Et)).sum() <= b + 1e-12


def test_ye_lipschitz():
    for _ in range(2000):
        mu, nu = random_ifns(40, 1)
        d = rng.uniform(0, 0.05)
        mu2 = np.clip(mu + rng.uniform(-d, d, mu.shape), 0, 1)
        nu2 = np.clip(nu + rng.uniform(-d, d, nu.shape), 0, 1 - mu2)
        assert abs(ent_ye(mu[:, 0], nu[:, 0]) - ent_ye(mu2[:, 0], nu2[:, 0])) <= YE_L * d + 1e-12


def test_dispersion_lipschitz():
    """Lemma 5.2: data-dependent Lipschitz factor for the dispersion entropy."""
    for _ in range(1000):
        mu, nu = random_ifns(60, 1, eps=0.05)
        d = rng.uniform(0, 0.02)
        s = (1 + mu - nu) / 2
        L = dispersion_lipschitz(s, d)
        mu2 = np.clip(mu + rng.uniform(-d, d, mu.shape), 0.05, 1)
        nu2 = np.clip(nu + rng.uniform(-d, d, nu.shape), 0, 1 - mu2)
        assert abs(ent_dispersion(mu[:, 0], nu[:, 0]) - ent_dispersion(mu2[:, 0], nu2[:, 0])) <= L * d + 1e-12


def test_score_sensitivity_bound():
    """Proposition 5.3: |S(w) - S(w~)| <= ln(1/eta) * sum_j omega_j ||w_j - w~_j||_1."""
    eta = 0.01
    for _ in range(500):
        n, p = 3, 5
        parts = [random_ifns(25, p, eps=eta) for _ in range(n)]
        omega = entropy_weights(rng.uniform(0, 1, n))
        W = [entropy_weights(rng.uniform(0, 1, p)) for _ in range(n)]
        Wt = [entropy_weights(rng.uniform(0, 1, p)) for _ in range(n)]

        def S(ws):
            B = [agg(mu, nu, w) for (mu, nu), w in zip(parts, ws)]
            gm, gn = agg(np.column_stack([b[0] for b in B]), np.column_stack([b[1] for b in B]), omega)
            return gm - gn
        bound = np.log(1 / eta) * sum(o * np.abs(a - b).sum() for o, a, b in zip(omega, W, Wt))
        assert np.max(np.abs(S(W) - S(Wt))) <= bound + 1e-12


def test_dispersion_datafree_constant():
    """Lemma 5.2(b), data-free constant: |H - H~| <= delta ln(m/eps) / (eps ln m)."""
    eps = 0.05
    from difss.model import dispersion_lipschitz_worstcase
    for _ in range(1000):
        m = int(rng.integers(5, 80))
        mu, nu = random_ifns(m, 1, eps=eps)
        d = rng.uniform(0, 0.03)
        mu2 = np.clip(mu + rng.uniform(-d, d, mu.shape), eps, 1)
        nu2 = np.clip(nu + rng.uniform(-d, d, nu.shape), 0, 1 - mu2)
        L = dispersion_lipschitz_worstcase(m, eps)
        assert abs(ent_dispersion(mu[:, 0], nu[:, 0]) - ent_dispersion(mu2[:, 0], nu2[:, 0])) <= L * d + 1e-12


def test_score_sensitivity_range_form_and_rank_corollary():
    """Proposition 5.3 (first, range-based inequality) and Corollary 5.4."""
    eta = 0.01
    for _ in range(300):
        n, p, m = 3, 4, 30
        parts = [random_ifns(m, p, eps=eta) for _ in range(n)]
        omega = entropy_weights(rng.uniform(0, 1, n))
        W = [entropy_weights(rng.uniform(0, 1, p)) for _ in range(n)]
        Wt = [np.abs(w + rng.uniform(-0.02, 0.02, p)) for w in W]
        Wt = [w / w.sum() for w in Wt]

        def S(ws):
            B = [agg(mu, nu, w) for (mu, nu), w in zip(parts, ws)]
            gm, gn = agg(np.column_stack([b[0] for b in B]), np.column_stack([b[1] for b in B]), omega)
            return gm - gn
        s0, s1 = S(W), S(Wt)
        for i in range(m):
            b = 0.0
            for j, (mu, nu) in enumerate(parts):
                rm = np.log(mu[i]).max() - np.log(mu[i]).min()
                rn = np.log(1 - nu[i]).max() - np.log(1 - nu[i]).min()
                b += omega[j] * np.abs(W[j] - Wt[j]).sum() * (rm + rn) / 2
            assert abs(s0[i] - s1[i]) <= b + 1e-12
        Delta = np.log(1 / eta) * sum(o * np.abs(a - c).sum() for o, a, c in zip(omega, W, Wt))
        order = np.argsort(-s0)
        gaps = s0[order][:-1] - s0[order][1:]
        for r in np.where(gaps > 2 * Delta)[0]:     # certified adjacent pairs keep their order
            assert s1[order[r]] > s1[order[r + 1]]


def test_dispersion_entropy_properties():
    """Proposition 4.1: range, equality case, permutation/scale invariance, Pigou-Dalton."""
    for _ in range(500):
        m = int(rng.integers(3, 50))
        mu, nu = random_ifns(m, 1)
        mu, nu = mu[:, 0], nu[:, 0]
        H = ent_dispersion(mu, nu)
        assert 0 < H <= 1 + 1e-12
        perm = rng.permutation(m)
        assert np.isclose(H, ent_dispersion(mu[perm], nu[perm]))
        # equal scores -> H = 1
        assert np.isclose(ent_dispersion(np.full(m, 0.6), np.full(m, 0.2)), 1.0)
        # Pigou-Dalton transfer on s = (1 + mu - nu)/2 (implemented through nu) cannot decrease H
        s = (1 + mu - nu) / 2
        hi, lo = int(np.argmax(s)), int(np.argmin(s))
        t = rng.uniform(0, (s[hi] - s[lo]) / 2)
        s2 = s.copy(); s2[hi] -= t; s2[lo] += t
        q1, q2 = s / s.sum(), s2 / s2.sum()
        H1 = -(q1 * np.log(q1)).sum() / np.log(m); H2 = -(q2 * np.log(q2)).sum() / np.log(m)
        assert H2 >= H1 - 1e-12
        assert np.isclose(H1, H)


def test_hesitancy_never_raises_score():
    """Orthopair construction (Section 4.2): for fixed performance values, more
    hesitancy evidence never raises the reduced or aggregated IFWG score, and
    the orthopair score equals the hesitancy-free score minus pi."""
    x = rng.uniform(0, 100, (40, 6))
    h_lo = rng.uniform(0, 0.5, (40, 6))
    h_hi = np.clip(h_lo + rng.uniform(0, 0.5, (40, 6)), 0, 1)
    mu0, nu0, _ = orthopairs(x, True, np.zeros_like(x), 0.01, 0.25, bounds=(0.0, 100.0))
    mu1, nu1, _ = orthopairs(x, True, h_lo, 0.01, 0.25, bounds=(0.0, 100.0))
    mu2, nu2, _ = orthopairs(x, True, h_hi, 0.01, 0.25, bounds=(0.0, 100.0))
    pi1 = 1 - mu1 - nu1
    assert np.allclose((mu1 - nu1), (mu0 - nu0) - pi1)
    assert np.all(mu2 - nu2 <= mu1 - nu1 + 1e-12)
    w = entropy_weights(rng.uniform(0, 1, 6))
    s1 = np.subtract(*agg(mu1, nu1, w)); s2 = np.subtract(*agg(mu2, nu2, w))
    assert np.all(s2 <= s1 + 1e-12)
