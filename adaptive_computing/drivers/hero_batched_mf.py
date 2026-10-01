"""Shared batched execution machinery for multifidelity HERO drivers."""

from copy import deepcopy

import numpy as np

from adaptive_computing.datasets.base import _KBDataset
from adaptive_computing.drivers.hero import ActiveLoopDriverHero


class ActiveLoopDriverHeroBatchedMF(ActiveLoopDriverHero):
    """Base class for batched multifidelity Bayesian optimization with HERO.

    A subclass supplies only the selection policy through
    :meth:`get_next_sample`.  This class owns the execution policy shared by
    MF-SEGO and cost-ratio selection: pending-point Kriging-believer models,
    nested submissions, a batch limit measured in simulation jobs, joint
    fidelity-queue processing, and cost/performance history.

    Parameters
    ----------
    nesting_tol : float
        Absolute tolerance for recognizing the same design location.
    reference_minimum : float or None
        Known high-fidelity minimum used to form simple-regret histories.
    objective_output : int or None
        Output column used as the objective.  By default this is inferred from
        ``surrogate.i_output`` and otherwise defaults to zero.
    """

    def __init__(
        self,
        *args,
        nesting_tol=1.0e-12,
        reference_minimum=None,
        objective_output=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if nesting_tol < 0.0:
            raise ValueError("nesting_tol must be nonnegative.")

        if objective_output is None:
            objective_output = getattr(self.surrogate, "i_output", 0)
        objective_output = int(objective_output)
        if objective_output < 0 or objective_output >= self.dataset.n_out:
            raise ValueError(
                f"objective_output={objective_output} is invalid for "
                f"n_out={self.dataset.n_out}."
            )
        surrogate_output = getattr(self.surrogate, "i_output", None)
        if (
            surrogate_output is not None
            and objective_output != surrogate_output
        ):
            raise ValueError(
                "objective_output must match surrogate.i_output for a "
                "single-output surrogate."
            )

        self.nesting_tol = float(nesting_tol)
        self.reference_minimum = (
            None
            if reference_minimum is None
            else float(reference_minimum)
        )
        self.objective_output = objective_output

        # Selection records retain submission order across fidelity levels.
        self.selection_history = []
        # Regret is recorded at batch boundaries, when new data are visible.
        self.batch_history = []
        self.cost_best_hf_pairs = []
        self.cumulative_adaptive_cost = 0.0
        self.initial_design_cost = None
        self._batch_index = 0
        self.last_batch = None

        self.max_low_fidelity_streak = 5
        self._low_fidelity_streak = 0

    def _must_select_high_fidelity(self):
        return (self.max_low_fidelity_streak is not None
                and self._low_fidelity_streak >= self.max_low_fidelity_streak)
    
    # ------------------------------------------------------------------
    # Fidelity costs and performance history
    # ------------------------------------------------------------------

    

    def _get_fidelity_costs(self):
        if self.fidelity_costs is None:
            raise ValueError(
                "A positive cost is required for every fidelity level."
            )

        if isinstance(self.fidelity_costs, dict):
            try:
                costs = np.asarray(
                    [
                        self.fidelity_costs[i]
                        for i in range(self.n_fidelity)
                    ],
                    dtype=float,
                )
            except KeyError as exc:
                raise ValueError(
                    "fidelity_costs must contain every fidelity index."
                ) from exc
        else:
            costs = np.asarray(self.fidelity_costs, dtype=float)

        if costs.shape != (self.n_fidelity,):
            raise ValueError(
                f"Expected {self.n_fidelity} fidelity costs; "
                f"received shape {costs.shape}."
            )
        if np.any(~np.isfinite(costs)) or np.any(costs <= 0.0):
            raise ValueError("All fidelity costs must be finite and positive.")

        return costs

    def _best_observed_high_fidelity(self):
        highest_fidelity = self.n_fidelity - 1
        _, y_data = self.dataset.get_unmasked_data(
            i_fidelity=highest_fidelity,
            i_output=self.objective_output,
        )

        if len(y_data) == 0:
            return np.inf

        values = np.asarray(y_data)[:, self.objective_output]
        values = values[np.isfinite(values)]
        if values.size == 0:
            return np.inf

        return float(np.min(values))

    def _initialize_performance_history(self):
        if self.initial_design_cost is not None:
            return

        costs = self._get_fidelity_costs()
        self.initial_design_cost = float(
            sum(
                len(self.dataset._x_data[i_fidelity])
                * costs[i_fidelity]
                for i_fidelity in range(self.n_fidelity)
            )
        )

        initial_best = self._best_observed_high_fidelity()
        self.cost_best_hf_pairs.append((0.0, initial_best))

    @property
    def cost_regret_pairs(self):
        """Return ``(adaptive cost, observed HF simple regret)`` pairs."""
        if self.reference_minimum is None:
            raise ValueError(
                "Set reference_minimum before requesting cost_regret_pairs."
            )

        return [
            (cost, best_hf - self.reference_minimum)
            for cost, best_hf in self.cost_best_hf_pairs
        ]

    @property
    def total_cost_regret_pairs(self):
        """Return regret pairs including the initial-design cost."""
        if self.initial_design_cost is None:
            raise RuntimeError("The performance history is not initialized.")

        return [
            (cost + self.initial_design_cost, regret)
            for cost, regret in self.cost_regret_pairs
        ]

    def reset_performance_history(self):
        """Start a new history from the dataset's current contents."""
        self.selection_history = []
        self.batch_history = []
        self.cost_best_hf_pairs = []
        self.cumulative_adaptive_cost = 0.0
        self.initial_design_cost = None
        self._batch_index = 0
        self.last_batch = None

    # ------------------------------------------------------------------
    # Pending-aware Kriging-believer model
    # ------------------------------------------------------------------

    def _pending_indices(self, i_fidelity):
        if not hasattr(self.dataset, "_hero_todo"):
            raise TypeError(
                "Batched HERO drivers require a dataset with _hero_todo."
            )

        pending = np.asarray(
            self.dataset._hero_todo[i_fidelity],
            dtype=bool,
        ).reshape(-1)
        n_samples = len(self.dataset._x_data[i_fidelity])

        if pending.size != n_samples:
            raise RuntimeError(
                f"Inconsistent HERO dataset at fidelity {i_fidelity}: "
                f"{pending.size} todo flags for {n_samples} samples."
            )

        return pending

    def _phantom_outputs(self, predictions):
        """Expand single-output predictions to the dataset output shape."""
        predictions = np.asarray(predictions)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        if predictions.shape[1] == self.dataset.n_out:
            return predictions

        if predictions.shape[1] == 1 and hasattr(
            self.surrogate,
            "i_output",
        ):
            values = np.full(
                (predictions.shape[0], self.dataset.n_out),
                np.nan,
            )
            values[:, self.surrogate.i_output] = predictions[:, 0]
            return values

        raise ValueError(
            "Surrogate prediction shape is incompatible with the dataset "
            "output shape."
        )

    def _build_kriging_believer_surrogate(self):
        """Build one pending-aware model for location and fidelity choice."""
        # The persistent surrogate always represents completed observations.
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
            pending_idx = self._pending_indices(i_fidelity)
            if not np.any(pending_idx):
                continue

            has_pending_data = True
            x_pending = np.asarray(
                self.dataset._x_data[i_fidelity]
            )[pending_idx]
            y_pending = self.surrogate.predict_values(
                x_pending,
                fidelity_level=i_fidelity,
            )

            phantom_x[i_fidelity] = x_pending
            phantom_y[i_fidelity] = self._phantom_outputs(y_pending)

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

    # ------------------------------------------------------------------
    # Nested submissions and batch execution
    # ------------------------------------------------------------------

    def _point_exists(self, x, i_fidelity):
        """Return whether a valid or pending point exists at a level."""
        x_data = np.asarray(self.dataset._x_data[i_fidelity])
        if x_data.size == 0:
            return False

        x_data = x_data.reshape((-1, self.dataset.n_in))
        x = np.asarray(x).reshape((1, self.dataset.n_in))
        same_location = np.all(
            np.isclose(
                x_data,
                x,
                rtol=0.0,
                atol=self.nesting_tol,
            ),
            axis=1,
        )

        if hasattr(self.surrogate, "i_output"):
            valid = np.asarray(
                self.dataset._unmasked_data[i_fidelity][
                    :, self.surrogate.i_output
                ],
                dtype=bool,
            )
        else:
            valid = np.all(
                np.asarray(
                    self.dataset._unmasked_data[i_fidelity],
                    dtype=bool,
                ),
                axis=1,
            )

        pending = self._pending_indices(i_fidelity)
        return bool(np.any(same_location & (valid | pending)))

    def _required_fidelities(self, x, selected_fidelity):
        selected_fidelity = int(selected_fidelity)
        if selected_fidelity < 0 or selected_fidelity >= self.n_fidelity:
            raise ValueError(
                f"Invalid selected fidelity {selected_fidelity}."
            )

        return [
            level
            for level in range(selected_fidelity + 1)
            if not self._point_exists(x, level)
        ]

    def _required_cost(self, x, selected_fidelity):
        costs = self._get_fidelity_costs()
        return float(
            sum(
                costs[level]
                for level in self._required_fidelities(
                    x,
                    selected_fidelity,
                )
            )
        )

    def _submit_candidate(self, x, fidelities):
        row_indices = {}

        for i_fidelity in fidelities:
            row_indices[i_fidelity] = len(
                self.dataset._x_data[i_fidelity]
            )
            self.dataset.add_samples(
                np.atleast_2d(x),
                i_fidelity=i_fidelity,
            )

        return row_indices

    def _finish_batch(self, active_fidelities):
        # Include work that was already pending when run() was entered.
        active_fidelities = set(active_fidelities)
        active_fidelities.update(
            i_fidelity
            for i_fidelity in range(self.n_fidelity)
            if np.any(self._pending_indices(i_fidelity))
        )

        if self.inline_manager is not None:
            self.inline_manager.run_until_done(
                i_fidelities=sorted(active_fidelities)
            )

        # External managers process queues concurrently while this waits.
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

    def _selection_metadata(self):
        """Return subclass-specific diagnostics for the latest selection."""
        return {}

    def _complete_batch_records(self, batch_records):
        costs = self._get_fidelity_costs()
        batch_cost = 0.0

        for record in batch_records:
            outputs = {}
            for i_fidelity, row in record["row_indices"].items():
                outputs[i_fidelity] = np.asarray(
                    self.dataset._y_data[i_fidelity][row]
                ).tolist()
                batch_cost += costs[i_fidelity]

            record["outputs"] = outputs

        self.cumulative_adaptive_cost += float(batch_cost)
        best_hf = self._best_observed_high_fidelity()

        for record in batch_records:
            record["cumulative_adaptive_cost_after_batch"] = (
                self.cumulative_adaptive_cost
            )
            record["best_high_fidelity_after_batch"] = best_hf

        self.selection_history.extend(batch_records)
        self.cost_best_hf_pairs.append(
            (self.cumulative_adaptive_cost, best_hf)
        )
        self.batch_history.append(
            {
                "batch": self._batch_index,
                "batch_cost": float(batch_cost),
                "cumulative_adaptive_cost": (
                    self.cumulative_adaptive_cost
                ),
                "best_high_fidelity": best_hf,
                "selections": deepcopy(batch_records),
            }
        )
        self.last_batch = batch_records
        self._batch_index += 1

    def run(self, N_steps=None, batch_size=1):
        """Run a finite batched multifidelity BO campaign.

        ``N_steps`` counts selected design locations.  ``batch_size`` limits
        actual simulation jobs, so a nested high-fidelity selection can occupy
        more than one slot.
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
        self._initialize_performance_history()

        n_locations = 0

        while n_locations < N_steps:
            jobs_in_batch = 0
            active_fidelities = set()
            batch_records = []

            while n_locations < N_steps:
                # Pending fantasies should normally prevent duplicates.  Keep
                # a bounded retry for optimizer roundoff and local minima.
                for _ in range(10):
                    x, selected_fidelity = self.get_next_sample()
                    x = np.atleast_2d(x)
                    if x.shape != (1, self.dataset.n_in):
                        raise ValueError(
                            "get_next_sample() must return exactly one point "
                            f"with shape (1, {self.dataset.n_in})."
                        )
                    selected_fidelity = int(selected_fidelity)
                    fidelities = self._required_fidelities(
                        x,
                        selected_fidelity,
                    )
                    if fidelities:
                        break
                else:
                    raise RuntimeError(
                        "The selection policy repeatedly selected a location "
                        "that is already available or pending."
                    )

                n_jobs = len(fidelities)
                if n_jobs > batch_size:
                    raise ValueError(
                        f"Selected fidelity {selected_fidelity} requires "
                        f"{n_jobs} nested jobs, but batch_size is "
                        f"{batch_size}. Increase batch_size."
                    )

                # Never split a nested group across two batches.
                if jobs_in_batch + n_jobs > batch_size:
                    break

                metadata = deepcopy(self._selection_metadata())
                row_indices = self._submit_candidate(x, fidelities)
                submitted_cost = float(
                    sum(
                        self._get_fidelity_costs()[level]
                        for level in fidelities
                    )
                )
               
                highest_fidelity = self.n_fidelity - 1
                if highest_fidelity in fidelities:
                    self._low_fidelity_streak = 0
                else:
                    self._low_fidelity_streak += 1


                batch_records.append(
                    {
                        "selection": len(self.selection_history)
                        + len(batch_records),
                        "batch": self._batch_index,
                        "x": x.copy(),
                        "selected_fidelity": selected_fidelity,
                        "submitted_fidelities": tuple(fidelities),
                        "submitted_cost": submitted_cost,
                        "row_indices": row_indices,
                        "selection_metadata": metadata,
                    }
                )

                jobs_in_batch += n_jobs
                n_locations += 1
                active_fidelities.update(fidelities)

            if jobs_in_batch == 0:
                raise RuntimeError("Unable to construct a nonempty batch.")

            self._finish_batch(active_fidelities)
            self._complete_batch_records(batch_records)

    def step(self):
        """Run one synchronized selection with room for a nested group."""
        self.run(N_steps=1, batch_size=self.n_fidelity)
