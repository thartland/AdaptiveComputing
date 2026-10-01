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

    def __init__(self, *args, mfsego_cost_power=2.0, **kwargs):
        super().__init__(*args, **kwargs)

        if mfsego_cost_power <= 0.0:
            raise ValueError("mfsego_cost_power must be positive.")

        self.mfsego_cost_power = float(mfsego_cost_power)
        self.last_mfsego_scores = None
        self.last_mfsego_eligible = None
        self.last_selected_fidelity = None

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
        return scores

    def get_next_sample(self):
        # The same pending-aware model makes both decisions.
        kb_surrogate, kb_dataset = (
            self._build_kriging_believer_surrogate()
        )
        highest_fidelity = self.n_fidelity - 1

        x = self.sampler.minimize_acq_func(
            kb_surrogate,
            kb_dataset,
            i_fidelity=highest_fidelity,
        )
        x = np.atleast_2d(x)

        scores = self._mfsego_scores(x, kb_surrogate)
        eligible = np.asarray(
            [
                bool(self._required_fidelities(x, i_fidelity))
                for i_fidelity in range(self.n_fidelity)
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
                "MF-SEGO produced no usable fidelity score at the selected "
                "location."
            )
        else:
            # The base class will retry and then report a clear duplicate
            # error if the acquisition optimizer repeatedly returns this x.
            i_fidelity = int(np.argmax(scores[0]))

        if self._must_select_high_fidelity():
            i_fidelity = self.n_fidelity - 1

        self.last_mfsego_scores = scores[0].copy()
        self.last_mfsego_eligible = eligible.copy()
        self.last_selected_fidelity = i_fidelity
        return x, i_fidelity

    def _selection_metadata(self):
        return {
            "selection_rule": "mfsego",
            "mfsego_scores": self.last_mfsego_scores.tolist(),
            "eligible_fidelities": self.last_mfsego_eligible.tolist(),
        }
