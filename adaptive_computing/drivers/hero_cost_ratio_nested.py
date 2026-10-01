"""Nested, pending-aware cost-ratio selection for batched HERO runs."""

import numpy as np

from adaptive_computing.drivers.hero_batched_mf import (
    ActiveLoopDriverHeroBatchedMF,
)


class ActiveLoopDriverHeroCostRatioNested(
    ActiveLoopDriverHeroBatchedMF
):
    """Cost-ratio policy using the shared batched multifidelity machinery.

    Each fidelity gets its own acquisition-minimizing candidate.  All
    candidates and acquisition values use the same pending-aware
    Kriging-believer surrogate.  The winning candidate minimizes acquisition
    value divided by its *incremental* nested cost, meaning already available
    lower-fidelity values are not charged again.

    The existing :class:`ActiveLoopDriverHeroCostRatio` remains available for
    reproducing legacy behavior.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_candidate_locations = None
        self.last_acquisition_values = None
        self.last_incremental_costs = None
        self.last_cost_ratio_scores = None
        self.last_required_fidelities = None
        self.last_selected_fidelity = None

    @staticmethod
    def _as_scalar(value):
        values = np.asarray(value, dtype=float).reshape(-1)
        if values.size != 1:
            raise ValueError(
                "The acquisition function must return one value for one "
                "candidate."
            )
        return float(values[0])

    def get_next_sample(self):
        kb_surrogate, kb_dataset = (
            self._build_kriging_believer_surrogate()
        )
        fidelity_costs = self._get_fidelity_costs()

        candidates = []
        acquisition_values = np.full(self.n_fidelity, np.nan)
        incremental_costs = np.full(self.n_fidelity, np.inf)
        ratio_scores = np.full(self.n_fidelity, np.inf)
        required_fidelities = []

        for i_fidelity in range(self.n_fidelity):
            x = self.sampler.minimize_acq_func(
                kb_surrogate,
                kb_dataset,
                i_fidelity=i_fidelity,
            )
            x = np.atleast_2d(x)
            candidates.append(x)

            required = self._required_fidelities(x, i_fidelity)
            required_fidelities.append(tuple(required))
            if not required:
                continue

            acquisition = self._as_scalar(
                self.sampler.acq_func(
                    x,
                    kb_surrogate,
                    kb_dataset,
                    i_fidelity,
                )
            )
            incremental_cost = float(
                sum(fidelity_costs[level] for level in required)
            )

            acquisition_values[i_fidelity] = acquisition
            incremental_costs[i_fidelity] = incremental_cost
            ratio_scores[i_fidelity] = acquisition / incremental_cost

        selectable = ~np.isnan(ratio_scores) & (
            ratio_scores < np.inf
        )
        if not np.any(selectable):
            raise RuntimeError(
                "Cost-ratio selection found no candidate requiring a new "
                "simulation."
            )

        selectable_scores = np.where(
            selectable,
            ratio_scores,
            np.inf,
        )
        if self._must_select_high_fidelity():
            i_fidelity = self.n_fidelity - 1
        else:
            i_fidelity = int(np.argmin(selectable_scores))
        self.last_candidate_locations = np.vstack(candidates)
        self.last_acquisition_values = acquisition_values.copy()
        self.last_incremental_costs = incremental_costs.copy()
        self.last_cost_ratio_scores = ratio_scores.copy()
        self.last_required_fidelities = tuple(required_fidelities)
        self.last_selected_fidelity = i_fidelity


        return candidates[i_fidelity], i_fidelity

    def _selection_metadata(self):
        return {
            "selection_rule": "cost_ratio_nested",
            "candidate_locations": (
                self.last_candidate_locations.tolist()
            ),
            "acquisition_values": (
                self.last_acquisition_values.tolist()
            ),
            "incremental_costs": (
                self.last_incremental_costs.tolist()
            ),
            "cost_ratio_scores": (
                self.last_cost_ratio_scores.tolist()
            ),
            "required_fidelities_by_candidate": (
                self.last_required_fidelities
            ),
        }
