"""Promotion-first multifidelity selection for batched HERO execution.

The policy screens new locations cheaply and spends high-fidelity budget on
locations whose low-fidelity value is already known:

* A *new* location (the minimizer of the high-fidelity acquisition) is
  evaluated at the low fidelity only.
* A *promotion* evaluates the high fidelity at an existing, completed
  low-fidelity point that has no high-fidelity value yet.

Each step compares the two options by expected improvement per unit cost on
the same pending-aware high-fidelity posterior.  A promotion buys
``EI_promote`` for ``c_HF``.  A new location eventually needs both levels, so
it buys roughly ``EI_new`` for ``c_LF + c_HF``.  Promote when

    EI_promote / c_HF >= EI_new / (c_LF + c_HF).

The threshold therefore comes from the fidelity costs rather than from a
tuned exponent.  The promotion pool is every completed LF-only point, so the
decision uses the whole sampling history, not only the current candidate.

Two guards keep the policy from wasting work:

* If the new location is already known at low fidelity (its LF posterior
  variance is below ``snap_variance_fraction`` of the LF prior variance), a
  new LF job there would teach little.  The step then chooses between HF at
  the new location (cost of the missing levels) and the best promotion by
  the same EI-per-cost rule.
* After ``max_low_fidelity_streak`` consecutive LF-only selections the step
  must evaluate HF.  It promotes the best pool point or evaluates HF at the
  new location, whichever has the larger EI per cost.

For more than two fidelity levels the comparison is made between the
highest level and the level just below it.
"""

import numpy as np

from adaptive_computing.drivers.hero_mfsego import ActiveLoopDriverHeroMFSEGO


class ActiveLoopDriverHeroMFPromotion(ActiveLoopDriverHeroMFSEGO):
    """Promotion-first fidelity selection.

    Parameters
    ----------
    snap_variance_fraction : float
        A new location whose LF posterior variance is below this fraction of
        the LF prior variance counts as already sampled at low fidelity.
    max_low_fidelity_streak : int or None
        Force a promotion after this many consecutive LF-only selections.
        ``None`` disables the guard.
    """

    def __init__(
        self,
        *args,
        snap_variance_fraction=1.0e-3,
        max_low_fidelity_streak=5,
        **kwargs,
    ):
        # The MF-SEGO score and promotion fraction are not used by this
        # policy; pass neutral values so the parent's validation passes.
        kwargs.setdefault("mfsego_cost_power", 1.0)
        kwargs.setdefault("promotion_ei_fraction", None)
        super().__init__(*args, **kwargs)

        if snap_variance_fraction < 0.0:
            raise ValueError("snap_variance_fraction must be nonnegative.")
        self.snap_variance_fraction = float(snap_variance_fraction)
        self.max_low_fidelity_streak = max_low_fidelity_streak

        self.last_decision = None
        self.last_threshold_ratio = None
        self.last_lf_variance_ratio = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _hf_ei(self, x, surrogate, dataset):
        """Expected improvement (not its log) under the HF posterior."""
        # expected_improvement returns -log(EI + 1e-30).
        acquisition = self._hf_acquisition_value(x, surrogate, dataset)
        return float(np.exp(-acquisition))

    def _best_promotion(self, surrogate, dataset):
        """Return (x, EI, pool size) for the best completed LF-only point."""
        pool = self._completed_lf_only_points()
        if len(pool) == 0:
            return None, 0.0, 0

        best_x, best_ei = None, -np.inf
        for row in pool:
            x = np.atleast_2d(row)
            ei = self._hf_ei(x, surrogate, dataset)
            if ei > best_ei:
                best_x, best_ei = x.copy(), ei
        return best_x, best_ei, len(pool)

    def _lf_variance_ratio(self, x, surrogate):
        """LF posterior variance at ``x`` relative to the LF prior variance."""
        lf_level = self.n_fidelity - 2
        lf_model = surrogate.surrogate_model[lf_level]
        var_x = float(np.ravel(lf_model.predict_variances(np.atleast_2d(x)))[0])
        # Far outside the unit-scaled design every correlation is zero, so
        # the predicted variance there is the prior variance.
        far = np.atleast_2d(
            np.asarray(self.dataset.x_limits, dtype=float)[:, 1]
            + 1.0e3
            * np.ptp(np.asarray(self.dataset.x_limits, dtype=float), axis=1)
        )
        var_prior = float(np.ravel(lf_model.predict_variances(far))[0])
        if var_prior <= 0.0:
            return np.inf
        return max(var_x, 0.0) / var_prior

    # ------------------------------------------------------------------
    # Selection policy
    # ------------------------------------------------------------------

    def get_next_sample(self):
        kb_surrogate, kb_dataset = self._build_kriging_believer_surrogate()
        highest = self.n_fidelity - 1
        screen = highest - 1

        costs = self._get_fidelity_costs()
        c_hf = float(costs[highest])
        c_new = float(np.sum(costs[: highest + 1]))
        # Promote when EI_p / c_hf >= EI_new / c_new.
        threshold_ratio = c_hf / c_new

        new_x = np.atleast_2d(
            self.sampler.minimize_acq_func(
                kb_surrogate, kb_dataset, i_fidelity=highest
            )
        )
        new_ei = self._hf_ei(new_x, kb_surrogate, kb_dataset)
        promo_x, promo_ei, pool_size = self._best_promotion(
            kb_surrogate, kb_dataset
        )
        lf_ratio = self._lf_variance_ratio(new_x, kb_surrogate)
        new_is_known_at_lf = (
            lf_ratio <= self.snap_variance_fraction
            or not self._required_fidelities(new_x, screen)
        )

        can_promote = promo_x is not None and promo_ei > 0.0

        if self._must_select_high_fidelity():
            # The guard forces an HF evaluation; choose where by EI per cost.
            # Promoting a pool point whose HF EI is essentially zero (the
            # model already knows it is poor) would waste the HF run.
            c_new_hf = float(sum(
                costs[level] for level in self._required_fidelities(new_x, highest)
            )) or c_new
            if can_promote and promo_ei / c_hf >= new_ei / c_new_hf:
                decision = "promote_streak"
            else:
                decision = "new_hf_streak"
        elif can_promote and promo_ei >= threshold_ratio * new_ei:
            decision = "promote_ei"
        elif new_is_known_at_lf:
            # Screening new_x at LF would teach little, and new_x beats the
            # best promotion on EI per cost: evaluate HF at new_x (only the
            # missing levels are submitted).
            decision = "new_hf"
        else:
            decision = "new_lf"

        if decision.startswith("promote"):
            x, i_fidelity = promo_x, highest
            self.last_candidate_source = "promotion"
        elif decision.startswith("new_hf"):
            x, i_fidelity = new_x, highest
            self.last_candidate_source = "new"
        else:
            x, i_fidelity = new_x, screen
            self.last_candidate_source = "new"

        self.last_decision = decision
        self.last_threshold_ratio = threshold_ratio
        self.last_lf_variance_ratio = lf_ratio
        self.last_new_candidate = new_x.copy()
        self.last_new_candidate_ei = new_ei
        self.last_promotion_candidate = (
            None if promo_x is None else promo_x.copy()
        )
        self.last_promotion_candidate_ei = promo_ei
        self.last_promotion_pool_size = pool_size
        self.last_mfsego_scores = np.full(self.n_fidelity, np.nan)
        self.last_mfsego_eligible = np.ones(self.n_fidelity, dtype=bool)
        self.last_selected_fidelity = i_fidelity

        print(
            f"[promotion-first] decision={decision} new_EI={new_ei:.4g} "
            f"promo_EI={promo_ei:.4g} (needs >= {threshold_ratio:.3f}*new_EI) "
            f"pool={pool_size} lf_var_ratio={lf_ratio:.3g} "
            f"lf_streak={self._low_fidelity_streak}"
        )
        return x, i_fidelity

    def _selection_metadata(self):
        metadata = super()._selection_metadata()
        metadata.update(
            {
                "selection_rule": "mf_promotion_first",
                "decision": self.last_decision,
                "threshold_ratio": self.last_threshold_ratio,
                "lf_variance_ratio": self.last_lf_variance_ratio,
            }
        )
        return metadata
