"""Numerical checks of MFKPosterior and the MF-KG / MF-MES values.

1. Means and variances match SMT's MFK predictions.
2. The joint covariance over levels and points is positive semidefinite.
3. Fantasy updates: retrain SMT with frozen hyperparameters (hyper_opt="NoOp")
   after adding one observation, and compare the change in the predicted means
   with the Gaussian update implied by the covariance.
4. MC check of the KG value against explicit fantasies on the covariance.
"""
import os
import sys

import numpy as np
from smt.applications import MFK

W = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # repo root
sys.path.insert(0, W)
sys.path.insert(0, os.path.join(W, "examples", "HPC_onsite2"))
from adaptive_computing.drivers.hero_mf_information import MFKPosterior, _sample_min_values, _mes_direct, _mes_correlated
from simulation_files.hartmann import hf_hartmann, lf_hartmann

rng = np.random.default_rng(0)
f_hf = lambda X: np.array([hf_hartmann(x) for x in X]).reshape(-1, 1)
f_lf = lambda X: np.array([lf_hartmann(x) for x in X]).reshape(-1, 1)


class FrozenMFK(MFK):
    """MFK with per-level, per-dimension theta fixed (SMT's theta0 option
    turns a (levels, dims) array into a list and broadcasts one value per level)."""
    frozen_theta = None

    frozen_scaling = None

    def _check_param(self):
        super()._check_param()
        self._theta0 = np.array(self.frozen_theta, dtype=float)

    def _new_train_init(self):
        # SMT re-standardizes X and y over all levels at every train; keep the
        # original scaling so the frozen theta means the same length scales.
        super()._new_train_init()
        self.X_offset, self.X_scale, self.y_mean, self.y_std = self.frozen_scaling
        self.X_norma_all = [(x - self.X_offset) / self.X_scale for x in self.X]
        self.y_norma_all = [(f - self.y_mean) / self.y_std for f in self.y]


def fit(X0, y0, X1, y1, theta0=None):
    kw = dict(print_global=False)
    if theta0 is not None:
        kw.update(hyper_opt="NoOp")
        m = FrozenMFK(**kw)
        m.frozen_theta = theta0
        m.frozen_scaling = SCALING
    else:
        m = MFK(**kw)
    m.set_training_values(X0, y0, name=0)
    m.set_training_values(X1, y1)
    m.train()
    return m


for n_hf, n_lf in [(8, 30), (12, 60)]:
    print(f"=== {n_hf} HF + {n_lf} LF points (HF nested in LF)")
    X1 = rng.random((n_hf, 6))
    X0 = np.vstack([X1, rng.random((n_lf - n_hf, 6))])
    y0, y1 = f_lf(X0), f_hf(X1)
    m = fit(X0, y0, X1, y1)
    post = MFKPosterior(m)
    T = rng.random((200, 6))

    mu0, mu1, v0, v1, c01 = post.marginals(T)
    mse, _ = m.predict_variances_all_levels(T)
    print("1. max |mu1 - SMT|      ", np.abs(mu1 - m.predict_values(T).ravel()).max())
    print("   max |mu0 - SMT|      ", np.abs(mu0 - m._predict_intermediate_values(T, 1).ravel()).max())
    print("   max rel |var0 - SMT| ", np.abs(v0 - mse[:, 0]).max() / mse[:, 0].max())
    print("   max rel |var1 - SMT| ", np.abs(v1 - mse[:, 1]).max() / mse[:, 1].max())

    S = T[:40]
    tS = post.terms(S)
    K00, K01, K10, K11 = post.cross(tS, tS)
    print("   diag(K11) vs var1    ", np.abs(np.diag(K11) - v1[:40]).max() / v1.max())
    J = np.block([[K00, K01], [K10, K11]])
    ev = np.linalg.eigvalsh(0.5 * (J + J.T))
    print("2. joint cov min eig / max eig", ev.min() / ev.max(), " asym", np.abs(J - J.T).max() / np.abs(J).max())

    theta = np.array(m.optimal_theta)
    SCALING = (m.X_offset, m.X_scale, m.y_mean, m.y_std)
    m_same = fit(X0, y0, X1, y1, theta0=theta)
    base1 = m_same.predict_values(T).ravel()
    base0 = m_same._predict_intermediate_values(T, 1).ravel()
    print("3. NoOp refit, same data: max |dmu1|", np.abs(base1 - mu1).max())
    tT = post.terms(T)

    # (a) LF observation at a new point
    x = rng.random((1, 6)); y = f_lf(x)
    mf = fit(np.vstack([X0, x]), np.vstack([y0, y]), X1, y1, theta0=theta)
    tx = post.terms(x)
    _, _, Kt0_x, _ = post.cross(tT, tx)          # Cov(f1(T), f0(x))
    k00x, _, _, _ = post.cross(tT, tx)           # Cov(f0(T), f0(x))
    m0x, m1x, v0x, v1x, c01x = post.marginals(x)
    pred1 = mu1 + Kt0_x[:, 0] / v0x[0] * (y[0, 0] - m0x[0])
    pred0 = mu0 + k00x[:, 0] / v0x[0] * (y[0, 0] - m0x[0])
    d1 = mf.predict_values(T).ravel() - mu1
    print("   (a) LF at new x:   HF mean change, rel error",
          np.abs(mf.predict_values(T).ravel() - pred1).max() / np.abs(d1).max(),
          "| LF mean change rel error",
          np.abs(mf._predict_intermediate_values(T, 1).ravel() - pred0).max() / np.abs(mf._predict_intermediate_values(T, 1).ravel() - mu0).max())

    # (b) HF observation at an existing LF-only point (promotion)
    x = X0[n_hf:n_hf + 1]; y = f_hf(x)
    mf = fit(X0, y0, np.vstack([X1, x]), np.vstack([y1, y]), theta0=theta)
    tx = post.terms(x)
    _, _, _, K11x = post.cross(tT, tx)
    m0x, m1x, v0x, v1x, c01x = post.marginals(x)
    pred1 = mu1 + K11x[:, 0] / v1x[0] * (y[0, 0] - m1x[0])
    d1 = mf.predict_values(T).ravel() - mu1
    print("   (b) HF at LF point: HF mean change, rel error",
          np.abs(mf.predict_values(T).ravel() - pred1).max() / np.abs(d1).max(), " var0 there", v0x[0])

    # (c) Nested LF+HF at a new point (approximate: rho uncertainty)
    x = rng.random((1, 6)); ya, yb = f_lf(x), f_hf(x)
    mf = fit(np.vstack([X0, x]), np.vstack([y0, ya]), np.vstack([X1, x]), np.vstack([y1, yb]), theta0=theta)
    tx = post.terms(x)
    m0x, m1x, v0x, v1x, c01x = post.marginals(x)
    nt = post.nested_terms(tx, tT)
    z1 = (ya[0, 0] - m0x[0]) / np.sqrt(v0x[0])
    mu1p = mu1 + tT["rho_hat"] * nt["d0"][0] * z1
    m1xp = m1x[0] + tx["rho_hat"][0] * nt["d0x"][0] * z1
    Vp = nt["vd"][0] + nt["v1"][0] * z1 + nt["v2"][0] * z1 ** 2
    Cp = nt["kd"][0] + nt["k1"][0] * z1 + nt["k2"][0] * z1 ** 2
    pred1 = mu1p + Cp / Vp * (yb[0, 0] - m1xp)
    d1 = mf.predict_values(T).ravel() - mu1
    print("   (c) nested at new x: HF mean change, rel error",
          np.abs(mf.predict_values(T).ravel() - pred1).max() / np.abs(d1).max())

    # 4. f* samples and MES sanity
    _, mug, _, vg, _ = post.marginals(np.vstack([T, X1]))
    fs = _sample_min_values(mug, np.sqrt(vg), 16)
    print("4. f* samples: min", fs.min().round(3), "max", fs.max().round(3), "best observed HF", y1.min().round(3))
    g = np.array([[-2.0, 0.0, 2.0]])
    print("   MES direct", _mes_direct(g).round(4), " correlated rho=0.999", _mes_correlated(g, np.array([0.999])).round(4),
          " rho=0", _mes_correlated(g, np.array([0.0])).round(4))
    print()

# 5. Level-1 refit at SMT's theta reproduces SMT's fit; sampler runs.
from adaptive_computing.drivers.hero_mf_information import sample_level1
X1 = rng.random((8, 6)); X0 = np.vstack([X1, rng.random((22, 6))])
m = fit(X0, f_lf(X0), X1, f_hf(X1))
post = MFKPosterior(m)
lvl, ll = post._level1_fit(post._lvl[1]["theta"])
ref = post._lvl[1]
print("5. refit at SMT theta: max diff C", np.abs(lvl["C"] - ref["C"]).max(), " beta", np.abs(lvl["beta"] - ref["beta"]).max(),
      " gamma", np.abs(lvl["gamma"] - ref["gamma"]).max(), " s2 rel", abs(lvl["s2"] / ref["s2"] - 1))
T = rng.random((50, 6))
a = post.marginals(T); b = post.with_level1(lvl).marginals(T)
print("   marginals via with_level1, max diff", max(np.abs(x - y).max() for x, y in zip(a, b)))
import time; t0 = time.time()
posts, acc = sample_level1(post, 8, np.random.default_rng(1))
print(f"   sampler: {len(posts)} samples, acceptance {acc:.2f}, {time.time() - t0:.2f}s")
print("   SMT theta1", np.round(post._lvl[1]["theta"], 4))
for q in posts[:4]:
    print("   sample theta1", np.round(q._lvl[1]["theta"], 4), " discrepancy sd", round(float(np.sqrt(q._lvl[1]["s2"]) * q.y_std), 4))
