"""MF-SEGO-style fidelity selection for batched HERO execution."""

import numpy as np

from adaptive_computing.drivers.hero_batched_mf import (
    ActiveLoopDriverHeroBatchedMF,
)


class ActiveLoopDriverHeroMFSEGO(ActiveLoopDriverHeroBatchedMF):
    """HERO driver using MF-SEGO-style fidelity selection.

    The acquisition function chooses a location using the highest-fidelity
    co-Kriging model.  MF-SEGO then chooses the fidelity from its predicted
    contribution to high-fidelity variance, penalized by normalized
    cumulative cost.

    Parameters
    ----------
    mfsego_cost_power : float
        Exponent applied to normalized cumulative cost.  The MF-SEGO-style
        default is 2.0.
    """

    def __init__(self, *args, mfsego_cost_power=2.0, 
                             promotion_ei_fraction=0.8, **kwargs):
        super().__init__(*args, **kwargs)

        if mfsego_cost_power <= 0.0:
            raise ValueError("mfsego_cost_power must be positive.")

        if promotion_ei_fraction is not None:
            promotion_ei_fraction = float(promotion_ei_fraction)
            if not 0.0 <= promotion_ei_fraction <= 1.0:
                raise ValueError("promotion_ei_fraction must be in [0, 1] or be None")


        self.mfsego_cost_power = float(mfsego_cost_power)
        self.promotion_ei_fraction = promotion_ei_fraction
        self.last_mfsego_scores = None
        self.last_mfsego_eligible = None
        self.last_selected_fidelity = None

        self.last_candidate_source = None
        self.last_new_candidate = None
        self.last_new_candidate_ei = None
        self.last_promotion_candidate = None
        self.last_promotion_candidate_ei = None
        self.last_promotion_pool_size = 0

    def _mfsego_scores(self, x, surrogate):
        """Return the MF-SEGO fidelity scores at ``x``.

        For component ``k`` the score is its variance contribution at the
        highest level divided by normalized cumulative cost raised to
        ``mfsego_cost_power``.
        """
        x = np.atleast_2d(x)

        try:
            model = surrogate.surrogate_model[-1]
        except (AttributeError, IndexError) as exc:
            raise TypeError(
                "MF-SEGO requires AdaptiveComputing's SMT multifidelity "
                "surrogate with an MFK model at surrogate_model[-1]."
            ) from exc

        if not hasattr(model, "predict_variances_all_levels"):
            raise TypeError(
                "The highest-fidelity surrogate must be an SMT MFK model."
            )

        level_variances, rho_squared = (
            model.predict_variances_all_levels(x)
        )
        level_variances = np.clip(
            np.asarray(level_variances, dtype=float),
            0.0,
            np.inf,
        )

        if level_variances.ndim != 2:
            raise ValueError(
                "MFK variance output must have one column per fidelity."
            )
        if level_variances.shape[1] != self.n_fidelity:
            raise ValueError(
                "MFK variance output does not match the number of "
                "fidelity levels."
            )

        transfer = np.ones_like(level_variances)

        # Transfer the contribution from level k to the highest level.
        for k in range(self.n_fidelity):
            for j in range(k, self.n_fidelity - 1):
                rho_j = np.asarray(
                    rho_squared[j],
                    dtype=float,
                ).reshape(-1)

                if rho_j.size == 1:
                    rho_j = np.full(x.shape[0], rho_j.item())
                elif rho_j.size != x.shape[0]:
                    raise ValueError(
                        "Unexpected rho output from "
                        "predict_variances_all_levels."
                    )

                transfer[:, k] *= np.clip(rho_j, 0.0, np.inf)

        variance_contributions = level_variances * transfer
        variance_reduction = np.cumsum(variance_contributions, axis=1)
       
        cumulative_reduction = variance_reduction
        total_reduction = cumulative_reduction[:,-1]
        lf_fraction = np.divide(
                cumulative_reduction[:,0],
                total_reduction,
                out=np.zeros_like(total_reduction),
                where=total_reduction>0.0)




        # Selecting level k requests every missing lower level as well.
        costs = self._get_fidelity_costs()
        cumulative_cost = np.cumsum(costs)
        normalized_cost = cumulative_cost / cumulative_cost[-1]
        cost_penalty = normalized_cost**self.mfsego_cost_power
        scores = variance_reduction / cost_penalty[np.newaxis, :]

        # At a point where every variance is numerically zero, prefer the
        # highest level rather than silently defaulting to level zero.
        all_zero = np.all(np.isclose(scores, 0.0), axis=1)
        scores[all_zero, -1] = np.inf
        
        print("level variances:", level_variances[0])
        print("rho squared:", rho_squared)
        print("variance_reduction:", variance_reduction[0])
        print("cumulative_reduction:", cumulative_reduction[0])
        print("LF variance fraction:", lf_fraction[0])
        print("normalized_cost:", normalized_cost)
        print("cost_penalty:", cost_penalty)
        print("MFSEGO scores:", scores[0])
        return scores


    @staticmethod
    def _as_scalar(value):
        values = np.asarray(value, dtype=float).reshape(-1)

        if values.size != 1:
            raise ValueError("acqf must return one value per candidate")
        return float(values[0])

    def _completed_lf_only_points(self):
        """Return completed LF points without an HF value or pending HF job."""

        highest_fidelity = self.n_fidelity - 1

        if highest_fidelity == 0:
            return np.empty((0, self.dataset.n_in))

        # For a two-fidelity problem this is fidelity zero. For more levels,
        # use the level immediately below the highest level as the promotion
        # pool.
        promotion_source = highest_fidelity - 1

        x_lower, _ = self.dataset.get_unmasked_data(
            i_fidelity=promotion_source,
            i_output=self.objective_output,
        )

        candidates = []

        for row in np.asarray(x_lower):
            x = np.atleast_2d(row)

            # _point_exists includes completed and pending HF evaluations.
            if self._point_exists(x, highest_fidelity):
                continue

            # Avoid repeated locations in the promotion pool.
            duplicate = any(
                np.allclose(
                    row,
                    previous,
                    rtol=0.0,
                    atol=self.nesting_tol,
                )
                for previous in candidates
            )

            if not duplicate:
                candidates.append(np.asarray(row).copy())

        if not candidates:
            return np.empty((0, self.dataset.n_in))

        return np.vstack(candidates)


    def _hf_acquisition_value(
        self,
        x,
        surrogate,
        dataset,
    ):
        """Evaluate the acquisition using the highest-fidelity posterior."""
    
        highest_fidelity = self.n_fidelity - 1
    
        return self._as_scalar(
            self.sampler.acq_func(
                np.atleast_2d(x),
                surrogate,
                dataset,
                highest_fidelity,
            )
        )


    def _best_promotion_candidate(
        self,
        surrogate,
        dataset,
    ):
        """Return the LF-only point having the best HF acquisition value."""
    
        promotion_points = self._completed_lf_only_points()
    
        if len(promotion_points) == 0:
            return None, None, 0
    
        best_x = None
        best_acquisition = np.inf
    
        for row in promotion_points:
            x = np.atleast_2d(row)
            acquisition = self._hf_acquisition_value(
                x,
                surrogate,
                dataset,
            )
    
            # AdaptiveComputing minimizes acquisition functions.
            if acquisition < best_acquisition:
                best_acquisition = acquisition
                best_x = x.copy()
    
        return best_x, best_acquisition, len(promotion_points)


    def get_next_sample(self):
        # The same pending-aware model makes both decisions.
        kb_surrogate, kb_dataset = (
            self._build_kriging_believer_surrogate()
        )
        highest_fidelity = self.n_fidelity - 1

        new_x = self.sampler.minimize_acq_func(
            kb_surrogate,
            kb_dataset,
            i_fidelity=highest_fidelity,
        )
        new_x = np.atleast_2d(new_x)

        new_acquisition = self._hf_acquisition_value(new_x, kb_surrogate, kb_dataset,)

        promotion_x, promotion_acquisition, pool_size = (
            self._best_promotion_candidate(
                kb_surrogate,
                kb_dataset,
            )
        )

        # expected_improvement.py returns negative EI because the sampler
        # minimizes acquisition functions.
        new_ei = max(-new_acquisition, 0.0)

        if promotion_acquisition is None:
            promotion_ei = None
        else:
            promotion_ei = max(-promotion_acquisition, 0.0)

        promote = False

        if (
            self.promotion_ei_fraction is not None
            and promotion_x is not None
            and promotion_ei > 0.0
        ):
            promote = (
                new_ei <= 0.0
                or promotion_ei
                >= self.promotion_ei_fraction * new_ei
            )

        self.last_new_candidate = new_x.copy()
        self.last_new_candidate_ei = new_ei
        self.last_promotion_candidate = (
            None if promotion_x is None else promotion_x.copy()
        )
        self.last_promotion_candidate_ei = promotion_ei
        self.last_promotion_pool_size = pool_size

        if promote:
            # The LF value already exists. _required_fidelities() will
            # therefore request only the missing HF job.
            x = promotion_x
            i_fidelity = highest_fidelity
            self.last_candidate_source = "promotion"

            scores = self._mfsego_scores(x, kb_surrogate)
            eligible = np.asarray(
                [
                    bool(self._required_fidelities(x, level))
                    for level in range(self.n_fidelity)
                ],
                dtype=bool,
            )

        else:
            x = new_x
            self.last_candidate_source = "new"

            scores = self._mfsego_scores(x, kb_surrogate)

            eligible = np.asarray(
                [
                    bool(self._required_fidelities(x, level))
                    for level in range(self.n_fidelity)
                ],
                dtype=bool,
            )

            selectable_scores = scores[0].copy()
            selectable_scores[~eligible] = -np.inf
            selectable_scores[np.isnan(selectable_scores)] = -np.inf

            if np.any(selectable_scores > -np.inf):
                i_fidelity = int(np.argmax(selectable_scores))
            elif np.any(eligible):
                raise RuntimeError(
                    "MF-SEGO produced no usable fidelity score "
                    "at the selected location."
                )
            else:
                i_fidelity = int(np.argmax(scores[0]))

        self.last_mfsego_scores = scores[0].copy()
        self.last_mfsego_eligible = eligible.copy()
        self.last_selected_fidelity = i_fidelity

        # Preserve your existing max_low_fidelity_streak update here if it
        # currently lives inside get_next_sample().
        return x, i_fidelity

        #scores = self._mfsego_scores(x, kb_surrogate)
        #eligible = np.asarray(
        #    [
        #        bool(self._required_fidelities(x, i_fidelity))
        #        for i_fidelity in range(self.n_fidelity)
        #    ],
        #    dtype=bool,
        #)

        #selectable_scores = scores[0].copy()
        #selectable_scores[~eligible] = -np.inf
        #selectable_scores[np.isnan(selectable_scores)] = -np.inf
        #if np.any(selectable_scores > -np.inf):
        #    i_fidelity = int(np.argmax(selectable_scores))
        #elif np.any(eligible):
        #    raise RuntimeError(
        #        "MF-SEGO produced no usable fidelity score at the selected "
        #        "location."
        #    )
        #else:
        #    # The base class will retry and then report a clear duplicate
        #    # error if the acquisition optimizer repeatedly returns this x.
        #    i_fidelity = int(np.argmax(scores[0]))

        #if self._must_select_high_fidelity():
        #    i_fidelity = self.n_fidelity - 1

        #self.last_mfsego_scores = scores[0].copy()
        #self.last_mfsego_eligible = eligible.copy()
        #self.last_selected_fidelity = i_fidelity
        #return x, i_fidelity

    def _selection_metadata(self):
        return {
        "selection_rule": "mfsego",
        "candidate_source": self.last_candidate_source,
        "mfsego_scores": self.last_mfsego_scores.tolist(),
        "eligible_fidelities": (
            self.last_mfsego_eligible.tolist()
        ),
        "promotion_ei_fraction": (
            self.promotion_ei_fraction
        ),
        "promotion_pool_size": (
            self.last_promotion_pool_size
        ),
        "new_candidate": (
            None
            if self.last_new_candidate is None
            else self.last_new_candidate.tolist()
        ),
        "new_candidate_ei": (
            self.last_new_candidate_ei
        ),
        "best_promotion_candidate": (
            None
            if self.last_promotion_candidate is None
            else self.last_promotion_candidate.tolist()
        ),
        "best_promotion_candidate_ei": (
            self.last_promotion_candidate_ei
        ),
    }
