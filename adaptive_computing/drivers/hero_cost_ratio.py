from adaptive_computing.drivers.active_cost_ratio import ActiveLoopDriverCostRatio
from adaptive_computing.drivers.hero import ActiveLoopDriverHero

class ActiveLoopDriverHeroCostRatio(ActiveLoopDriverHero, ActiveLoopDriverCostRatio):
    def step(self):
        x, fi_eval = self.get_next_sample()
        self.dataset.add_samples(x, i_fidelity=fi_eval)

        if self.inline_manager is not None:
            self.inline_manager.run_until_done(i_fidelity=fi_eval)
            self.hero_wait_for_data_and_train()
        elif self.retrain:
            self.surrogate.train(self.dataset)
