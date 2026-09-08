from adaptive_computing.drivers.active_cost_ratio import ActiveLoopDriverCostRatio
from adaptive_computing.drivers.hero import ActiveLoopDriverHero

class ActiveLoopDriverHeroCostRatio(ActiveLoopDriverHero, ActiveLoopDriverCostRatio):
    def run(self, N_steps=None, batch_size=1):
        if N_steps is None:
            raise ValueError(
                "A finite N_steps is required for batched Hero execution."
            )
        if N_steps < 0:
            raise ValueError("N_steps must be nonnegative.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least one.")

        # Preserve ActiveLoopDriver.run() initialization behavior.
        if not self._bopt_initialized:
            adequate_data = (
                self.surrogate is not None
                and self.surrogate._has_adequate_data(self.dataset)
            )

            if adequate_data:
                self._bopt_initialized = True
            else:
                self.initialize()

        n_completed = 0

        while n_completed < N_steps:
            current_batch_size = min(
                batch_size,
                N_steps - n_completed,
            )

            active_fidelities = set()

            # Queue the entire batch without waiting.
            for _ in range(current_batch_size):
                x, fi_eval = self.get_next_sample()
                self.dataset.add_samples(
                    x,
                    i_fidelity=fi_eval,
                )
                active_fidelities.add(int(fi_eval))

            # Process every fidelity queue that received work.
            if self.inline_manager is not None:
                for fi_eval in sorted(active_fidelities):
                    self.inline_manager.run_until_done(
                        i_fidelity=fi_eval
                    )

            # Synchronize and retrain once per batch.
            self.hero_wait_for_data_and_train()

            n_completed += current_batch_size


    def step(self):
        x, fi_eval = self.get_next_sample()
        self.dataset.add_samples(x, i_fidelity=fi_eval)
        if self.inline_manager is not None:
            self.inline_manager.run_until_done(i_fidelity=fi_eval)
            self.hero_wait_for_data_and_train()
        elif self.retrain:
            self.surrogate.train(self.dataset)
