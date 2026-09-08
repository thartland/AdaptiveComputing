from copy import deepcopy

import numpy as np

from adaptive_computing.datasets.base import _KBDataset
from adaptive_computing.drivers.hero import ActiveLoopDriverHero


class ActiveLoopDriverHeroMFSEGO(ActiveLoopDriverHero):
    """
    HERO driver using MF-SEGO-style fidelity selection.

    The acquisition function selects x using the highest-fidelity model.
    The fidelity is then selected using the predicted contribution to
    highest-fidelity variance, normalized by cumulative squared cost.

    Parameters
    ----------
    mfsego_cost_power : float
        Exponent applied to normalized cumulative cost. The MF-SEGO
        implementation uses 2.0.
    nesting_tol : float
        Tolerance used when deciding whether x is already present at a
        fidelity.
    """

    def __init__(
        self,
        *args,
        mfsego_cost_power=2.0,
        nesting_tol=1.0e-12,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if mfsego_cost_power <= 0.0:
            raise ValueError("mfsego_cost_power must be positive.")

        self.mfsego_cost_power = float(mfsego_cost_power)
        self.nesting_tol = float(nesting_tol)

        self.last_mfsego_scores = None
        self.last_selected_fidelity = None
        self.last_batch = None

    def _get_fidelity_costs(self):
        if self.fidelity_costs is None:
            raise ValueError(
                "MF-SEGO requires one positive fidelity cost per level."
            )

        if isinstance(self.fidelity_costs, dict):
            costs = np.asarray(
                [self.fidelity_costs[i] for i in range(self.n_fidelity)],
                dtype=float,
            )
        else:
            costs = np.asarray(self.fidelity_costs, dtype=float)

        if costs.shape != (self.n_fidelity,):
            raise ValueError(
                f"Expected {self.n_fidelity} fidelity costs; "
                f"received shape {costs.shape}."
            )

        if np.any(costs <= 0.0):
            raise ValueError("All fidelity costs must be positive.")

        return costs

    def _build_kriging_believer_surrogate(self):
        """
        Construct the pending-aware model used for both location and
        fidelity selection.
        """
        # Train the persistent model using only completed observations.
        self.surrogate.train(self.dataset)

        phantom_x = [
            np.empty((0, self.dataset.n_in))
            for _ in range(self.dataset.n_fidelity)
        ]
        phantom_y = [
            np.empty((0, self.dataset.n_out))
            for _ in range(self.dataset.n_fidelity)
        ]

        has_pending_data = False

        for i_fidelity in range(self.dataset.n_fidelity):
            y_data = self.dataset._y_data[i_fidelity]
            mask = np.ma.getmaskarray(y_data)

            if mask.ndim == 1:
                pending_idx = mask
            else:
                pending_idx = np.all(mask, axis=1)

            if not np.any(pending_idx):
                continue

            has_pending_data = True

            x_pending = np.asarray(
                self.dataset._x_data[i_fidelity]
            )[pending_idx]

            # Important: fantasize using the corresponding fidelity.
            y_pending = self.surrogate.predict_values(
                x_pending,
                fidelity_level=i_fidelity,
            )

            phantom_x[i_fidelity] = x_pending
            phantom_y[i_fidelity] = y_pending

        if has_pending_data:
            training_dataset = _KBDataset(
                self.dataset,
                phantom_x,
                phantom_y,
            )
        else:
            training_dataset = self.dataset

        pending_surrogate = deepcopy(self.surrogate)
        pending_surrogate.train(training_dataset)

        return pending_surrogate, training_dataset

    def _mfsego_scores(self, x, surrogate):
        """
        Return the MF-SEGO fidelity score at x.

        For component k:

            score[k] =
                sigma_delta[k]^2
                * product(rho[j]^2, j=k,...,L-2)
                / normalized_cumulative_cost[k]^p
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

        variance_reduction = level_variances * transfer

        # MF-SEGO uses cumulative cost because selecting level k also
        # requests every lower fidelity.
        costs = self._get_fidelity_costs()
        cumulative_cost = np.cumsum(costs)
        normalized_cost = cumulative_cost / cumulative_cost[-1]
        cost_penalty = normalized_cost**self.mfsego_cost_power

        scores = variance_reduction / cost_penalty[np.newaxis, :]

        # At a point where every variance is numerically zero, prefer
        # the highest level rather than defaulting to level zero.
        all_zero = np.all(np.isclose(scores, 0.0), axis=1)
        scores[all_zero, -1] = np.inf

        return scores

    def get_next_sample(self):
        # One pending-aware model must make both decisions.
        kb_surrogate, kb_dataset = (
            self._build_kriging_believer_surrogate()
        )

        highest_fidelity = self.n_fidelity - 1

        # Acquisition function determines the location.
        x = self.sampler.minimize_acq_func(
            kb_surrogate,
            kb_dataset,
            i_fidelity=highest_fidelity,
        )

        # MF-SEGO determines which fidelity to evaluate there.
        scores = self._mfsego_scores(x, kb_surrogate)
        i_fidelity = int(np.argmax(scores[0]))

        self.last_mfsego_scores = scores[0].copy()
        self.last_selected_fidelity = i_fidelity

        return np.atleast_2d(x), i_fidelity

    def _point_exists(self, x, i_fidelity):
        """Include both completed and currently pending observations."""
        x_data = np.asarray(self.dataset._x_data[i_fidelity])

        if x_data.size == 0:
            return False

        x_data = x_data.reshape((-1, self.dataset.n_in))
        x = np.asarray(x).reshape((1, self.dataset.n_in))

        return bool(
            np.any(
                np.all(
                    np.isclose(
                        x_data,
                        x,
                        rtol=0.0,
                        atol=self.nesting_tol,
                    ),
                    axis=1,
                )
            )
        )

    def _required_fidelities(self, x, selected_fidelity):
        """
        Preserve nesting without resubmitting levels already completed
        or pending at x.
        """
        return [
            level
            for level in range(selected_fidelity + 1)
            if not self._point_exists(x, level)
        ]

    def _submit_candidate(self, x, fidelities):
        for i_fidelity in fidelities:
            self.dataset.add_samples(
                np.atleast_2d(x),
                i_fidelity=i_fidelity,
            )

    def _finish_batch(self, active_fidelities):
        if self.inline_manager is not None:
            # This is the joint-queue manager interface discussed
            # previously. It must process these queues concurrently.
            self.inline_manager.run_until_done(
                i_fidelities=sorted(active_fidelities)
            )

        # For external managers this waits for HERO results. It is one
        # synchronization per batch, not one synchronization per job.
        self.hero_wait_for_data_and_train()

    def _ensure_bo_is_initialized(self):
        if self._bopt_initialized:
            return

        if (
            self.surrogate is not None
            and self.surrogate._has_adequate_data(self.dataset)
        ):
            self._bopt_initialized = True
            return

        raise RuntimeError(
            "The initial multifidelity DOE has not completed. Queue the "
            "initial samples, process all fidelity queues, and call "
            "hero_wait_for_data_and_train() before run()."
        )

    def run(self, N_steps=None, batch_size=1):
        """
        Run batched MF-SEGO.

        N_steps counts selected design locations.
        batch_size limits actual simulation jobs submitted concurrently.
        """
        if N_steps is None:
            raise ValueError(
                "N_steps must be finite for batched HERO execution."
            )

        N_steps = int(N_steps)
        batch_size = int(batch_size)

        if N_steps < 0:
            raise ValueError("N_steps cannot be negative.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least one.")

        self._ensure_bo_is_initialized()

        n_locations = 0

        while n_locations < N_steps:
            jobs_in_batch = 0
            active_fidelities = set()
            batch_records = []

            while n_locations < N_steps:
                # A pending-aware acquisition should normally avoid
                # duplicates. Limit retries for numerical edge cases.
                for _ in range(10):
                    x, selected_fidelity = self.get_next_sample()
                    fidelities = self._required_fidelities(
                        x,
                        selected_fidelity,
                    )
                    if fidelities:
                        break
                else:
                    raise RuntimeError(
                        "MF-SEGO repeatedly selected an existing point."
                    )

                n_jobs = len(fidelities)

                if n_jobs > batch_size:
                    raise ValueError(
                        f"Selected fidelity {selected_fidelity} requires "
                        f"{n_jobs} nested jobs, but batch_size is "
                        f"{batch_size}. Increase batch_size."
                    )

                # Do not partially submit a nested fidelity group.
                if jobs_in_batch + n_jobs > batch_size:
                    break

                self._submit_candidate(x, fidelities)

                jobs_in_batch += n_jobs
                n_locations += 1
                active_fidelities.update(fidelities)

                batch_records.append(
                    {
                        "x": x.copy(),
                        "selected_fidelity": selected_fidelity,
                        "submitted_fidelities": tuple(fidelities),
                        "scores": self.last_mfsego_scores.copy(),
                    }
                )

            if jobs_in_batch == 0:
                raise RuntimeError("Unable to construct a nonempty batch.")

            self.last_batch = batch_records
            self._finish_batch(active_fidelities)
