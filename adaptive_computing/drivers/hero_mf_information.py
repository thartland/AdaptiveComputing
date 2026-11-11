"""Information-based multifidelity selection: MF-KG and MF-MES.

Both policies choose a location *and* a fidelity by value per unit cost, with
no cost exponent or promotion fraction, and by default no streak guard:

* ``acquisition="kg"``: multifidelity knowledge gradient (Poloczek, Wang &
  Frazier 2017; Wu et al. 2019).  The value of observing level ``l`` at ``x``
  is the expected decrease in the minimum of the HF posterior mean.
* ``acquisition="mes"``: multifidelity max-value entropy search (Takeno et
  al. 2020).  The value is the mutual information between the observation
  and the HF minimum value ``f*``.

Both need posterior covariances between levels and between locations.  SMT's
``MFK`` only exposes per-point variances, so :class:`MFKPosterior` extends
SMT's own variance formulas to covariances (see its docstring).  The model
and hyperparameters are exactly the ones SMT fitted; nothing is refitted.

Selecting HF at a location without an LF value submits both levels (nested
submission in the base class), so the HF option is valued as observing both
and costs ``c_LF + c_HF`` there, or ``c_HF`` at a completed LF-only point
(a promotion).  Pending jobs are handled by the kriging-believer model.

The value is maximized over a scrambled Sobol set, Gaussian perturbations
of the points with the lowest HF mean and the completed LF-only points,
followed by a batched local search around the best few candidates per level.
"""

import copy

import numpy as np
from scipy import linalg
from scipy.linalg import solve_triangular
from scipy.stats import norm, qmc

from adaptive_computing.drivers.hero_mfsego import ActiveLoopDriverHeroMFSEGO


class MFKPosterior:
    """Joint posterior of a trained two-level SMT ``MFK`` model.

    With inputs normalized and outputs scaled as SMT does internally:

    level 0, universal Kriging with trend ``f`` and GLS coefficients::

        K0(x, x') = s0 [c0(x, x') - r0(x)' R0^-1 r0(x') + u0(x)' u0(x')]

    level 1, a discrepancy GP whose trend is ``[g(x) mu0(x), f(x)]``, so the
    scale factor rho is one of its GLS coefficients::

        Kd(x, x')        = s1 [c1(x, x') - r1(x)' R1^-1 r1(x') + u1(x)' u1(x')]
        K11(x, x')       = Kd(x, x') + E[rho(x) rho(x')] K0(x, x')
        Cov(f0(x), f1(x')) = rho_hat(x') K0(x, x')

    with ``u = G^-T (f - F' R^-1 r)`` and ``E[rho rho']`` taken from SMT's
    ``sigma2_rho``.  The diagonals reproduce
    ``MFK.predict_variances_all_levels`` (with ``propagate_uncertainty``).
    """

    def __init__(self, model):
        opts = model.options
        if model.nlvl != 2:
            raise ValueError("MFKPosterior supports two fidelity levels.")
        if not model.is_continuous or getattr(model, "_use_pls", False):
            raise ValueError("MFKPosterior needs a continuous, non-PLS MFK.")
        if not opts["propagate_uncertainty"]:
            raise ValueError("MFKPosterior assumes propagate_uncertainty=True.")
        self.m = model
        self.y_std = float(np.ravel(model.y_std)[0])
        self.y_mean = float(np.ravel(model.y_mean)[0])
        self._poly = model._regression_types[opts["poly"]]
        self._rho_regr = model._regression_types[opts["rho_regr"]]
        self._lvl = []
        for i in range(2):
            par = model.optimal_par[i]
            C, G = par["C"], par["G"]
            self._lvl.append(dict(
                C=C, G=G, beta=par["beta"], gamma=par["gamma"],
                Ft=solve_triangular(C, model.F_all[i], lower=True),
                s2=float(np.ravel(par["sigma2"])[0]) / self.y_std ** 2,
                X=model.X_norma_all[i], theta=model.optimal_theta[i],
            ))
        q = model.q_all[1]
        l1 = self._lvl[1]
        self._q = q
        self._srho = (l1["s2"] * linalg.inv(l1["G"].T @ l1["G"])[:q, :q]
                      + l1["beta"][:q] @ l1["beta"][:q].T)

    # -- building blocks (normalized inputs, scaled outputs) -------------
    def _norm(self, X):
        return (np.atleast_2d(X) - self.m.X_offset) / self.m.X_scale

    def _corr(self, lvl, A, B):
        dx = self.m._differences(A, Y=B)
        d = self.m._componentwise_distance(dx)
        self.m.corr.theta = self._lvl[lvl]["theta"]
        return np.asarray(self.m.corr(d)).reshape(A.shape[0], B.shape[0])

    def _terms(self, Xn):
        """Per-level (r_t, u, mean) at normalized points ``Xn``."""
        l0, l1 = self._lvl
        f0 = self._poly(Xn)
        r0 = self._corr(0, Xn, l0["X"])
        mu0 = (f0 @ l0["beta"] + r0 @ l0["gamma"]).ravel()
        rt0 = solve_triangular(l0["C"], r0.T, lower=True)
        u0 = solve_triangular(l0["G"].T, f0.T - l0["Ft"].T @ rt0, lower=True)

        g = self._rho_regr(Xn)
        f1 = np.vstack((g.T * mu0, f0.T))
        r1 = self._corr(1, Xn, l1["X"])
        mu1 = (f1.T @ l1["beta"] + r1 @ l1["gamma"]).ravel()
        rt1 = solve_triangular(l1["C"], r1.T, lower=True)
        u1 = solve_triangular(l1["G"].T, f1 - l1["Ft"].T @ rt1, lower=True)
        rho_hat = (g @ l1["beta"][: self._q]).ravel()
        return dict(rt0=rt0, u0=u0, rt1=rt1, u1=u1, g=g, mu0=mu0, mu1=mu1,
                    rho_hat=rho_hat)

    # -- public API (original units) -------------------------------------
    def marginals(self, X):
        """Means, variances and the LF-HF covariance at each point.

        Returns ``mu0, mu1, var0, var1, cov01`` as 1-D arrays.
        """
        t = self._terms(self._norm(X))
        l0, l1 = self._lvl
        var0 = l0["s2"] * (1.0 - (t["rt0"] ** 2).sum(0) + (t["u0"] ** 2).sum(0))
        vard = l1["s2"] * (1.0 - (t["rt1"] ** 2).sum(0) + (t["u1"] ** 2).sum(0))
        erho2 = ((t["g"] @ self._srho) * t["g"]).sum(1)
        var1 = vard + erho2 * var0
        cov01 = t["rho_hat"] * var0
        s = self.y_std ** 2
        return (t["mu0"] * self.y_std + self.y_mean,
                t["mu1"] * self.y_std + self.y_mean,
                np.maximum(var0 * s, 0.0), np.maximum(var1 * s, 0.0), cov01 * s)

    def terms(self, X):
        """Precompute the terms needed by :meth:`cross` for points ``X``."""
        Xn = self._norm(X)
        t = self._terms(Xn)
        t["Xn"] = Xn
        return t

    def cross(self, ta, tb):
        """Cross-covariances between point sets A and B.

        Returns ``K00, K01, K10, K11`` with ``Kij = Cov(f_i(A), f_j(B))``.
        """
        l0, l1 = self._lvl
        K0 = l0["s2"] * (self._corr(0, ta["Xn"], tb["Xn"])
                         - ta["rt0"].T @ tb["rt0"] + ta["u0"].T @ tb["u0"])
        Kd = l1["s2"] * (self._corr(1, ta["Xn"], tb["Xn"])
                         - ta["rt1"].T @ tb["rt1"] + ta["u1"].T @ tb["u1"])
        erho = ta["g"] @ self._srho @ tb["g"].T
        s = self.y_std ** 2
        return (K0 * s, K0 * tb["rho_hat"][None, :] * s,
                K0 * ta["rho_hat"][:, None] * s, (Kd + erho * K0) * s)

    def nested_terms(self, tx, ta):
        """Exact update for observing LF and then HF at new points ``x``.

        SMT's level-1 trend uses the LF mean, so after LF is observed at x the
        level-1 terms change with the observed value.  With
        ``z1 = (y0 - mu0(x)) / sd0(x)`` (all arrays (n_x, n_a), original units):

        * stage 1 changes ``mu1(a)`` by ``rho_hat(a) * d0 * z1``;
        * then ``Cov(f1(a), f1(x) | y0) = kd + k1 z1 + k2 z1^2`` and
          ``Var(f1(x) | y0) = vd + v1 z1 + v2 z1^2`` (vectors over x).

        Requires a constant rho regression (q = 1).
        """
        if self._q != 1:
            raise ValueError("nested_terms supports a single rho coefficient.")
        l0, l1 = self._lvl
        K0 = l0["s2"] * (self._corr(0, tx["Xn"], ta["Xn"])
                         - tx["rt0"].T @ ta["rt0"] + tx["u0"].T @ ta["u0"])
        var0x = l0["s2"] * (1.0 - (tx["rt0"] ** 2).sum(0) + (tx["u0"] ** 2).sum(0))
        sd0x = np.sqrt(np.maximum(var0x, 1.0e-30))
        kn_a = K0 / sd0x[:, None]          # scaled LF mean change per unit z1
        kn_x = sd0x
        e1 = np.zeros((l1["G"].shape[0], 1))
        e1[0, 0] = 1.0
        h = solve_triangular(l1["G"].T, e1, lower=True).ravel()
        uh_a, uh_x = ta["u1"].T @ h, tx["u1"].T @ h
        ga, gx = ta["g"][:, 0], tx["g"][:, 0]
        hh = float(h @ h)
        s1 = l1["s2"]
        kd = s1 * (self._corr(1, tx["Xn"], ta["Xn"])
                   - tx["rt1"].T @ ta["rt1"] + tx["u1"].T @ ta["u1"])
        k1 = s1 * ((gx * kn_x)[:, None] * uh_a[None, :] + ga[None, :] * kn_a * uh_x[:, None])
        k2 = s1 * (gx * kn_x)[:, None] * ga[None, :] * kn_a * hh
        vd = s1 * (1.0 - (tx["rt1"] ** 2).sum(0) + (tx["u1"] ** 2).sum(0))
        v1 = 2.0 * s1 * gx * kn_x * uh_x
        v2 = s1 * (gx * kn_x) ** 2 * hh
        s = self.y_std ** 2
        return dict(d0=kn_a * self.y_std, d0x=kn_x * self.y_std,
                    kd=kd * s, k1=k1 * s, k2=k2 * s,
                    vd=vd * s, v1=v1 * s, v2=v2 * s)



    # -- level-1 hyperparameters -----------------------------------------
    def _level1_fit(self, theta):
        """GLS fit of the level-1 discrepancy GP at length-scale parameters
        ``theta`` (SMT's ``exp(-sum theta d^2)``), as SMT computes it.

        Returns the level dict and the log integrated likelihood, i.e. with
        the trend coefficients (flat prior) and the process variance
        (Jeffreys prior) integrated out (Berger, De Oliveira & Sanso 2001).
        """
        m, l1 = self.m, self._lvl[1]
        X, F, y = l1["X"], self.m.F_all[1], self.m.y_norma_all[1]
        n, k = F.shape
        old = self._lvl[1]["theta"]
        self._lvl[1]["theta"] = np.asarray(theta, dtype=float)
        R = self._corr(1, X, X)
        self._lvl[1]["theta"] = old
        R[np.diag_indices(n)] += m.options["nugget"]
        C = linalg.cholesky(R, lower=True)
        Ft = solve_triangular(C, F, lower=True)
        Q, G = linalg.qr(Ft, mode="economic")
        Yt = solve_triangular(C, y, lower=True)
        beta = solve_triangular(G, Q.T @ Yt)
        res = Yt - Ft @ beta
        S2 = float((res ** 2).sum())
        lvl = dict(C=C, G=G, beta=beta, gamma=solve_triangular(C.T, res),
                   Ft=Ft, s2=S2 / (n - k), X=X, theta=np.asarray(theta, dtype=float))
        loglik = (-np.log(np.diag(C)).sum() - np.log(np.abs(np.diag(G))).sum()
                  - 0.5 * (n - k) * np.log(max(S2, 1.0e-300)))
        return lvl, loglik

    def with_level1(self, lvl):
        """Copy of this posterior with level-1 parameters ``lvl``."""
        new = copy.copy(self)
        new._lvl = [self._lvl[0], lvl]
        new._srho = (lvl["s2"] * linalg.inv(lvl["G"].T @ lvl["G"])[: self._q, : self._q]
                     + lvl["beta"][: self._q] @ lvl["beta"][: self._q].T)
        return new

    def level1_log_posterior(self, u):
        """Log posterior of ``u = log(beta)`` with ``theta = beta^2``.

        Prior: the jointly robust prior of Gu, Wang & Berger (2018), with
        its default constants, on the inverse ranges ``beta`` of the
        Gaussian kernel ``exp(-(beta d)^2)``.  It vanishes when every range
        grows without bound, which is the degenerate fit (an almost
        constant discrepancy) this is meant to avoid.  Returns the log
        posterior and the level dict.
        """
        X = self._lvl[1]["X"]
        n, p = X.shape
        a = 0.2
        c = n ** (-1.0 / p) * np.maximum(X.max(0) - X.min(0), 1.0e-12)
        b = n ** (-1.0 / p) * (a + p)
        beta = np.exp(u)
        t = float(c @ beta)
        try:
            lvl, loglik = self._level1_fit(beta ** 2)
        except (linalg.LinAlgError, ValueError):
            return -np.inf, None
        return loglik + a * np.log(t) - b * t + u.sum(), lvl


def sample_level1(post, n_samples, rng, n_burn=400, n_steps=800):
    """Posterior samples of the level-1 parameters of ``post``.

    Random-walk Metropolis on ``log(beta)``, started at the posterior mode
    found from SMT's fitted theta, with the step size adapted during burn-in
    towards an acceptance rate of about 0.25.  Returns ``n_samples``
    posteriors (equally spaced after burn-in) and the acceptance rate.
    """
    from scipy.optimize import minimize

    def nlp(u):
        v, _ = post.level1_log_posterior(u)
        return -v if np.isfinite(v) else 1.0e300

    p = post._lvl[1]["X"].shape[1]
    starts = [0.5 * np.log(np.clip(post._lvl[1]["theta"], 1.0e-4, None)), np.full(p, -1.0)]
    u = min((minimize(nlp, s0, method="L-BFGS-B", bounds=[(-8.0, 4.0)] * p).x for s0 in starts),
            key=nlp)
    lp, lvl = post.level1_log_posterior(u)
    step, acc, out = 0.3, 0, []
    keep = set(np.linspace(n_burn, n_burn + n_steps - 1, n_samples).astype(int))
    for i in range(n_burn + n_steps):
        prop = u + step * rng.standard_normal(p)
        lp_new, lvl_new = (post.level1_log_posterior(prop) if np.all(np.abs(prop) < 12.0)
                           else (-np.inf, None))
        ok = np.log(rng.random()) < lp_new - lp
        if ok:
            u, lp, lvl = prop, lp_new, lvl_new
        if i < n_burn:
            step *= np.exp((float(ok) - 0.25) / np.sqrt(i + 1.0))
        else:
            acc += ok
            if i in keep:
                out.append(post.with_level1(lvl))
    return out, acc / n_steps


# ----------------------------------------------------------------------
# Acquisition values
# ----------------------------------------------------------------------

_HALF_LOG_2PI_E = 0.5 * np.log(2.0 * np.pi * np.e)
_ZGRID = np.linspace(-8.0, 8.0, 241)


def _mes_direct(gamma):
    """MI between f(x) and f* when f(x) itself is observed (Wang & Jegelka 2017).

    ``gamma = (mu - f*) / sigma`` for minimization.  Shape-preserving.
    """
    logcdf = norm.logcdf(gamma)
    return 0.5 * gamma * np.exp(norm.logpdf(gamma) - logcdf) - logcdf


def _mes_correlated(gamma, rho):
    """MI between a correlated observation and f* (Takeno et al. 2020).

    The observation z (standardized) has correlation ``rho`` with the HF value
    f(x); conditioning on f(x) >= f* gives the density
    ``phi(z) Phi((gamma + rho z) / sqrt(1 - rho^2)) / Phi(gamma)``, whose
    entropy is integrated numerically.  ``gamma`` is (n, K), ``rho`` is (n,).
    """
    rho = np.clip(rho, -0.999999, 0.999999)[:, None, None]
    s = np.sqrt(1.0 - rho ** 2)
    z = _ZGRID[None, None, :]
    logp = (norm.logpdf(z) + norm.logcdf((gamma[..., None] + rho * z) / s)
            - norm.logcdf(gamma)[..., None])
    ent = -np.trapz(np.exp(logp) * logp, _ZGRID, axis=-1)
    return np.maximum(_HALF_LOG_2PI_E - ent, 0.0)


def _sample_min_values(mu, sd, n_samples):
    """Stratified samples of f* = min f, treating points as independent.

    Solves ``sum log Phi((mu - y) / sd) = log(1 - u)`` for y by bisection,
    i.e. P(f* <= y) = u (Wang & Jegelka 2017, without the Gumbel fit).
    """
    sd = np.maximum(sd, 1.0e-9)
    u = (np.arange(n_samples) + 0.5) / n_samples
    target = np.log1p(-u)
    lo = np.full(n_samples, np.min(mu - 10.0 * sd))
    hi = np.full(n_samples, np.min(mu + 10.0 * sd))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        val = norm.logcdf((mu[None, :] - mid[:, None]) / sd[None, :]).sum(1)
        # val decreases with y; P(f* > y) = exp(val)
        above = val > target
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    return 0.5 * (lo + hi)


class ActiveLoopDriverHeroMFInfo(ActiveLoopDriverHeroMFSEGO):
    """Multifidelity knowledge gradient or max-value entropy search.

    Parameters
    ----------
    acquisition : {"kg", "mes"}
    n_sobol : int
        Size of the scrambled Sobol candidate set per step.
    n_local : int
        Gaussian perturbations around each of the ``n_local_centres`` points
        with the lowest HF posterior mean.
    local_scale : float
        Standard deviation of those perturbations, as a fraction of the range.
    n_kg_set : int
        Points with the lowest HF mean kept in the KG discretization (plus
        the candidate itself and the best observed HF points).
    n_mc : int
        Quasi-Monte Carlo normal samples for the KG expectation.
    n_fstar : int
        Samples of the HF minimum value for MES.
    n_refine : int
        Candidates per level kept in the local search.
    n_refine_points, refine_scales
        Gaussian perturbations per kept candidate, and their standard
        deviations (fraction of the range) for successive rounds.
    seed : int
    max_lf_cost_without_hf : float or None
        Optional guard, in HF-equivalents: once the LF cost spent since the
        last HF selection reaches this, the next selection is the best HF
        option whatever the values say.  It catches a model whose LF-HF
        discrepancy has collapsed, so that HF never looks worth its cost.
        ``None`` (default) disables it.
    hyper : {"mle", "bayes"}
        ``"mle"`` (default) uses SMT's fitted hyperparameters.  ``"bayes"``
        averages the value of each option over ``n_hyper`` posterior samples
        of the level-1 (discrepancy) length scales under the jointly robust
        prior (:func:`sample_level1`); level 0 stays at SMT's fit.
    n_hyper : int
    """

    def __init__(self, *args, acquisition="kg", n_sobol=1024, n_local=20,
                 n_local_centres=10, local_scale=0.05, n_kg_set=50,
                 n_mc=128, n_fstar=16, n_refine=3, n_refine_points=128,
                 refine_scales=(0.03, 0.01, 0.003), seed=0,
                 max_lf_cost_without_hf=None, hyper="mle", n_hyper=8, **kwargs):
        kwargs.setdefault("mfsego_cost_power", 1.0)
        kwargs.setdefault("promotion_ei_fraction", None)
        super().__init__(*args, **kwargs)
        if acquisition not in ("kg", "mes"):
            raise ValueError("acquisition must be 'kg' or 'mes'.")
        if self.n_fidelity != 2:
            raise ValueError("MF-KG / MF-MES are implemented for two levels.")
        self.acquisition = acquisition
        self.n_sobol, self.n_local = int(n_sobol), int(n_local)
        self.n_local_centres, self.local_scale = int(n_local_centres), float(local_scale)
        self.n_kg_set, self.n_mc = int(n_kg_set), int(n_mc)
        self.n_fstar, self.n_refine = int(n_fstar), int(n_refine)
        self.n_refine_points, self.refine_scales = int(n_refine_points), tuple(refine_scales)
        self.seed = int(seed)
        self.max_low_fidelity_streak = None  # base-class guard off
        self.max_lf_cost_without_hf = (None if max_lf_cost_without_hf is None
                                       else float(max_lf_cost_without_hf))
        self._lf_cost_since_hf = 0.0
        if hyper not in ("mle", "bayes"):
            raise ValueError("hyper must be 'mle' or 'bayes'.")
        self.hyper, self.n_hyper = hyper, int(n_hyper)
        self._n_selections = 0
        self.last_info = {}

    # ------------------------------------------------------------------
    # Candidate sets and bookkeeping
    # ------------------------------------------------------------------

    def _bounds(self):
        lim = np.asarray(self.dataset.x_limits, dtype=float)
        return lim[:, 0], lim[:, 1]

    def _existing(self, level):
        """All rows (completed or pending) at a level."""
        x = np.asarray(self.dataset._x_data[level], dtype=float)
        return x.reshape((-1, self.dataset.n_in))

    def _has_point(self, X, level):
        rows = self._existing(level)
        if rows.size == 0:
            return np.zeros(len(X), dtype=bool)
        d = np.abs(X[:, None, :] - rows[None, :, :]).max(-1)
        return (d <= self.nesting_tol).any(1)

    def _candidates(self, post, rng):
        lo, hi = self._bounds()
        n_in = self.dataset.n_in
        sob = qmc.Sobol(d=n_in, scramble=True, seed=rng).random(self.n_sobol)
        X = lo + sob * (hi - lo)
        x_hf = self._existing(1)
        base = np.vstack([X, x_hf]) if x_hf.size else X
        _, mu1, _, _, _ = post.marginals(base)
        centres = base[np.argsort(mu1)[: self.n_local_centres]]
        local = (centres[:, None, :] + self.local_scale * (hi - lo)
                 * rng.standard_normal((len(centres), self.n_local, n_in)))
        X = np.vstack([X, np.clip(local.reshape(-1, n_in), lo, hi)])
        return X

    def _hf_cost(self, X):
        costs = self._get_fidelity_costs()
        return costs[1] + np.where(self._has_point(X, 0), 0.0, costs[0])

    # ------------------------------------------------------------------
    # Values
    # ------------------------------------------------------------------

    def _mes_values(self, post, X, fstar):
        mu0, mu1, var0, var1, cov01 = post.marginals(X)
        sd1 = np.sqrt(np.maximum(var1, 1.0e-18))
        gamma = (mu1[:, None] - fstar[None, :]) / sd1[:, None]
        v_hf = _mes_direct(gamma).mean(1)
        rho = cov01 / np.sqrt(np.maximum(var0 * var1, 1.0e-36))
        v_lf = _mes_correlated(gamma, rho).mean(1)
        # Nothing to learn where the level is already known.
        v_lf = np.where(var0 <= 1.0e-12 * post.y_std ** 2, 0.0, v_lf)
        v_hf = np.where(var1 <= 1.0e-12 * post.y_std ** 2, 0.0, v_hf)
        return np.maximum(v_lf, 0.0), np.maximum(v_hf, 0.0)

    def _kg_values(self, post, X, tA, muA, Z1, Z2):
        """KG of observing LF, and of observing HF (nested where needed)."""
        tX = post.terms(X)
        mu0, mu1, var0, var1, cov01 = post.marginals(X)
        _, _, K10, K11 = post.cross(tA, tX)      # Cov(f1(A), f_j(X))
        n = len(X)
        # Column for the candidate itself: Cov(f1(x), f0(x)), Cov(f1(x), f1(x)).
        c_x0, c_x1 = cov01, var1
        best_now = np.minimum(muA.min(), mu1)

        def expected_min(S, Z):
            # S: (n, nA + 1, m), Z: (nz, m); means over Z of min_a mu_a + S_a Z
            mu_all = np.concatenate([np.broadcast_to(muA, (n, len(muA))),
                                     mu1[:, None]], axis=1)
            vals = mu_all[:, None, :] + np.einsum("nam,zm->nza", S, Z)
            return vals.min(-1).mean(-1)

        # LF observation at x: s = Cov(f1(A), f0(x)) / sd0(x)
        sd0 = np.sqrt(np.maximum(var0, 1.0e-18))
        S_lf = np.concatenate([K10.T, c_x0[:, None]], axis=1)[..., None] / sd0[:, None, None]
        kg_lf = best_now - expected_min(S_lf, Z1)
        kg_lf = np.where(var0 <= 1.0e-12 * post.y_std ** 2, 0.0, kg_lf)

        # HF observation: LF then HF where LF is missing (exact two-stage
        # update of SMT's recursive model), else HF only.
        need_lf = ~self._has_point(X, 0)
        nt = post.nested_terms(tX, tA)
        m = need_lf.astype(float)[:, None]
        d0 = np.concatenate([nt["d0"], nt["d0x"][:, None]], axis=1) * m
        rho_all = np.concatenate([np.broadcast_to(tA["rho_hat"], (n, len(muA))),
                                  tX["rho_hat"][:, None]], axis=1)
        kd = np.concatenate([nt["kd"], nt["vd"][:, None]], axis=1)
        k1 = np.concatenate([nt["k1"], nt["v1"][:, None]], axis=1) * m
        k2 = np.concatenate([nt["k2"], nt["v2"][:, None]], axis=1) * m
        vd, v1, v2 = nt["vd"], nt["v1"] * m[:, 0], nt["v2"] * m[:, 0]
        z1, z2 = Z2[:, 0][None, :, None], Z2[:, 1][None, :, None]
        mu_all = np.concatenate([np.broadcast_to(muA, (n, len(muA))), mu1[:, None]], axis=1)
        mu_p = mu_all[:, None, :] + (rho_all * d0)[:, None, :] * z1
        cov = kd[:, None, :] + k1[:, None, :] * z1 + k2[:, None, :] * z1 ** 2
        var = vd[:, None] + v1[:, None] * Z2[None, :, 0] + v2[:, None] * Z2[None, :, 0] ** 2
        mu_pp = mu_p + cov / np.sqrt(np.maximum(var, 1.0e-30))[:, :, None] * z2
        kg_hf = best_now - mu_pp.min(-1).mean(-1)
        kg_hf = np.where(var1 <= 1.0e-12 * post.y_std ** 2, 0.0, kg_hf)
        return np.maximum(kg_lf, 0.0), np.maximum(kg_hf, 0.0)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def get_next_sample(self):
        kb_surrogate, kb_dataset = self._build_kriging_believer_surrogate()
        post = MFKPosterior(kb_surrogate.surrogate_model[-1])
        rng = np.random.default_rng([self.seed, self._n_selections])
        self._n_selections += 1
        costs = self._get_fidelity_costs()
        lo, hi = self._bounds()

        if self.hyper == "bayes":
            posts, acc = sample_level1(post, self.n_hyper,
                                       np.random.default_rng([self.seed, self._n_selections, 1]))
            hyper_info = dict(hyper_acceptance=float(acc), theta1_median=np.median(
                [q._lvl[1]["theta"] for q in posts], axis=0).tolist())
        else:
            posts, hyper_info = [post], {}

        X = self._candidates(post, rng)
        pool = self._completed_lf_only_points()
        X_hf = np.vstack([X, pool]) if len(pool) else X

        if self.acquisition == "mes":
            grid = np.vstack([X_hf, self._existing(1)]) if self._existing(1).size else X_hf
            fstars = []
            for q in posts:
                _, mu_g, _, var_g, _ = q.marginals(grid)
                fstars.append(_sample_min_values(mu_g, np.sqrt(var_g), self.n_fstar))

            def values(Xc):
                v = [self._mes_values(q, Xc, f) for q, f in zip(posts, fstars)]
                return np.mean([a for a, _ in v], 0), np.mean([b for _, b in v], 0)
        else:
            x_hf_done = self._existing(1)
            _, mu_c, _, _, _ = post.marginals(X_hf)
            kg_set = X_hf[np.argsort(mu_c)[: self.n_kg_set]]
            if x_hf_done.size:
                _, mu_d, _, _, _ = post.marginals(x_hf_done)
                kg_set = np.vstack([kg_set, x_hf_done[np.argsort(mu_d)[:10]]])
            Z1 = norm.ppf(qmc.Sobol(1, scramble=True, seed=rng).random(self.n_mc).clip(1e-9, 1 - 1e-9))
            Z2 = norm.ppf(qmc.Sobol(2, scramble=True, seed=rng).random(self.n_mc).clip(1e-9, 1 - 1e-9))

            sets = [(q, q.terms(kg_set), q.marginals(kg_set)[1]) for q in posts]

            def values(Xc):
                v = [self._kg_values(q, Xc, tA, muA, Z1, Z2) for q, tA, muA in sets]
                return np.mean([a for a, _ in v], 0), np.mean([b for _, b in v], 0)

        v_lf_X, _ = values(X)
        _, v_hf_X = values(X_hf)
        r_lf = v_lf_X / costs[0]
        r_hf = v_hf_X / self._hf_cost(X_hf)

        # Refine the best continuous candidates of each level by a batched
        # local search: Gaussian clouds around the incumbents, shrinking.
        best = {0: (r_lf.max(), X[np.argmax(r_lf)]),
                1: (r_hf.max(), X_hf[np.argmax(r_hf)])}
        n_cont = len(X)
        for level, ratio in ((0, r_lf), (1, r_hf[:n_cont])):
            centres = X[np.argsort(ratio)[::-1][: self.n_refine]]
            for scale in self.refine_scales:
                cloud = (centres[:, None, :] + scale * (hi - lo)
                         * rng.standard_normal((len(centres), self.n_refine_points, len(lo))))
                cloud = np.vstack([centres, np.clip(cloud.reshape(-1, len(lo)), lo, hi)])
                v = values(cloud)[level]
                c = costs[0] if level == 0 else self._hf_cost(cloud)
                r = v / c
                order = np.argsort(r)[::-1]
                centres = cloud[order[: self.n_refine]]
                if r[order[0]] > best[level][0]:
                    best[level] = (float(r[order[0]]), cloud[order[0]].copy())

        level = 1 if best[1][0] >= best[0][0] else 0
        guard = (level == 0 and self.max_lf_cost_without_hf is not None
                 and self._lf_cost_since_hf + costs[0]
                 > self.max_lf_cost_without_hf * costs[1] + 1e-9)
        if guard:
            level = 1
        ratio, x = best[level]
        x = np.atleast_2d(x)
        if not self._required_fidelities(x, level) or not np.isfinite(ratio) or ratio <= 0.0:
            # Degenerate values: fall back to the HF-EI location at HF.
            x = np.atleast_2d(self.sampler.minimize_acq_func(kb_surrogate, kb_dataset, i_fidelity=1))
            level, source = 1, "fallback_hf_ei"
        elif level == 1 and len(pool) and np.any(np.all(np.isclose(pool, x, atol=self.nesting_tol, rtol=0.0), axis=1)):
            source = "promotion"
        else:
            source = "new"
        if guard:
            source = "guard_" + source
        self._lf_cost_since_hf = 0.0 if level == 1 else self._lf_cost_since_hf + costs[0]

        self.last_candidate_source = source
        self.last_selected_fidelity = level
        self.last_mfsego_scores = np.array([best[0][0], best[1][0]])
        self.last_mfsego_eligible = np.ones(self.n_fidelity, dtype=bool)
        self.last_new_candidate = x.copy()
        self.last_promotion_pool_size = len(pool)
        self.last_info = dict(value_per_cost_lf=float(best[0][0]),
                              value_per_cost_hf=float(best[1][0]),
                              decision=source, **hyper_info)
        print(f"[mf-{self.acquisition}] level={level} source={source} "
              f"value/cost LF={best[0][0]:.4g} HF={best[1][0]:.4g} pool={len(pool)}")
        return x, level

    def _selection_metadata(self):
        metadata = super()._selection_metadata()
        metadata.update({"selection_rule": f"mf_{self.acquisition}", **self.last_info})
        return metadata
