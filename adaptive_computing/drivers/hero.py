from copy import deepcopy

from adaptive_computing.drivers import ActiveLoopDriver
from adaptive_computing.datasets import HeroDataset, _KBDataset
from time import sleep

import numpy as np


class ActiveLoopDriverHero(ActiveLoopDriver):
    def __init__(self, simulations, params, machine_names, output_field_path, surrogate=None, dataset=None,
                 nan_behavior='fail', fidelity_costs=None, acq_func='expected_improvement', blocking=False,
                 task_formatter=None, inline_manager=None, hero_client=None):
        self.use_hero = True
        if dataset is None:
            if isinstance(simulations, list):
                n_fidelity = len(simulations)
            else:
                n_fidelity = 1
            dataset = HeroDataset(params, machine_names, output_field_path, n_fidelity=n_fidelity, blocking=blocking,
                                task_formatter=task_formatter, nan_behavior=nan_behavior,
                                hero_client=hero_client)
        self.dataset = dataset
        if blocking:
            retrain = True
        else:
            retrain = False # only retrain when wait hero_wait_for_data_and_train is called
        super().__init__(simulations, params, surrogate=surrogate, dataset=self.dataset,
                         nan_behavior=nan_behavior, fidelity_costs=fidelity_costs, acq_func=acq_func, retrain=retrain)

        for sim_i in simulations:
            assert(sim_i is None) # since the user has opted to use Hero, simulations should be set to a list of Nones of length n_fidelity and the definition of the simulations should be implemented in the manager script.
        self.evaluators = None

        # Optional LocalHPCManager for the noSSH workflow (controller runs on the
        # same node as the scheduler).  When set, query() calls
        # inline_manager.run_until_done() between task submission and waiting for
        # results.  Not needed when a background manager daemon is already running
        # (e.g. the SSH+tmux approach in examples/hero_HPC_managers).
        # Not serialized to pickle — set this attribute after loading a saved driver:
        #   ac_driver = pickle.load(f)
        #   ac_driver.inline_manager = manager
        self.inline_manager = inline_manager

    def _initialize_fidelity(self, i_fidelity, N_samples_init=3):
        """
        Initializes a fidelity level by queuing random LHS samples in the Hero task system.

        Args:
            i_fidelity (int): Fidelity level index.
            N_samples_init (int, optional): Number of initial samples to generate. Defaults to 3.
        """
        x = self.init_sampler.get_sample(N_samples=N_samples_init)
        self.dataset.add_samples(x, i_fidelity=i_fidelity)

    def add_samples(self, points, i_fidelity=0):
        """
        Queues input points in the Hero task system for asynchronous evaluation.
        Non-blocking: returns immediately after creating the Hero tasks.
        Call hero_wait_for_data_and_train() to wait for results.

        Args:
            points (list or np.ndarray): Points to queue for evaluation.
            i_fidelity (int): Fidelity level index.
        """
        for x in points:
            x = np.atleast_2d(x)
            self.dataset.add_samples(x, i_fidelity)

    def step(self):
        """
        Executes one step of the active learning loop: selects the next sample
        using the acquisition function and queues it as a Hero task.
        """
        x, fi_eval = self.get_next_sample()
        self.dataset.add_samples(x, i_fidelity=fi_eval)
        if self.inline_manager is not None:
            self.inline_manager.run_until_done(i_fidelity=fi_eval)
            self.hero_wait_for_data_and_train()
        if self.retrain:
            self.surrogate.train(self.dataset)

    def _kb_select(self, x, threshold):
        """Return a boolean mask of which points in x need a real Hero simulation.

        Implements the Kriging Believer batch strategy: iteratively selects the
        highest-variance pending point, assumes it will return the surrogate mean
        (the 'belief'), retrains the surrogate with that phantom value, then
        re-evaluates variance for the remaining points.  Points that drop below
        the threshold after retraining are handled by the surrogate alone.

        The real dataset is never modified.  The surrogate is retrained with
        phantom data during the loop; hero_wait_for_data_and_train() will retrain
        it on real data once simulation results arrive.

        Args:
            x:         Query points, shape (N, n_in).
            threshold: Variance threshold.

        Returns:
            np.ndarray: Boolean mask of length N; True where a real simulation is needed.
        """
        tmp_surrogate = deepcopy(self.surrogate)
        pending = list(range(len(x)))
        to_simulate = []
        phantom_x = []
        phantom_y = []

        while pending:
            vars_pending = np.array([
                float(tmp_surrogate.predict_variances(x[[i]])[0][0])
                for i in pending
            ])

            if vars_pending.max() <= threshold:
                break

            # Select the highest-variance pending point
            best_local = int(np.argmax(vars_pending))
            best_global = pending[best_local]

            to_simulate.append(best_global)
            pending.pop(best_local)

            # Kriging Believer: treat the predicted mean as the placeholder response
            y_phantom = tmp_surrogate.predict_values(x[[best_global]])
            phantom_x.append(x[best_global])
            phantom_y.append(y_phantom[0])

            if pending:
                kb_dataset = _KBDataset(
                    self.dataset,
                    [np.array(phantom_x)],  # list of one array for fidelity 0
                    [np.array(phantom_y)],
                )
                tmp_surrogate.train(kb_dataset)

        mask = np.zeros(len(x), dtype=bool)
        mask[to_simulate] = True
        return mask

    def query(self, points, error_criterion, threshold):
        """Query the surrogate; use Kriging Believer to select simulations, then run in parallel.

        Overrides ActiveLoopDriver.query() to use Hero task submission instead of
        local evaluators (self.evaluators is None for Hero drivers).

        Uses the Kriging Believer (KB) batch strategy to decide which points need
        real simulations before submitting anything.  KB iteratively selects the
        highest-variance point, assumes it returns the surrogate mean, retrains the
        surrogate, and checks whether remaining points have dropped below the
        threshold.  Only the points still above threshold after KB planning are
        submitted as a parallel Hero batch (wall time = slowest single job).
        This can reduce the number of HPC jobs compared to submitting all
        above-threshold points blindly.

        For the noSSH workflow, set self.inline_manager to a LocalHPCManager
        instance (or assign it after loading from pickle) so that run_until_done()
        is called automatically between task submission and the Hero wait:

            ac_driver = pickle.load(f)
            ac_driver.inline_manager = manager
            y = ac_driver.query(x_queries, 'absolute_variance', threshold)

        For the SSH+tmux workflow, leave inline_manager as None — the background
        manager daemon processes the tasks while hero_wait_for_data_and_train() waits.

        Args:
            points:          Query points, shape (N, n_inputs).
            error_criterion: Must be 'absolute_variance'.
            threshold:       Points with predicted variance above this value are
                             evaluated via Hero rather than the surrogate alone.

        Returns:
            np.ndarray: Final surrogate predictions at all query points, shape (N, 1).
        """
        assert error_criterion == 'absolute_variance', \
            f"Hero driver query only supports 'absolute_variance', got '{error_criterion}'"

        x = np.asarray(points)
        #print(x)
        #print(self.surrogate.predict_variances(x[[0]]))
        variances = np.array([
            float(self.surrogate.predict_variances(x[[i]])[0][0])
            for i in range(len(x))
        ])

        if np.any(variances > threshold):
            n_naive = int(np.sum(variances > threshold))
            sim_mask = self._kb_select(x, threshold)
            n_sim = int(sim_mask.sum())
            n_saved = n_naive - n_sim
            print(f"Kriging Believer: {n_sim} simulation(s) needed "
                  f"({n_saved} point(s) dropped below threshold after KB retraining, "
                  f"down from {n_naive} naive):")
            for i in np.where(sim_mask)[0]:
                print(f"  x={x[i]}, initial variance={variances[i]:.2e}")
            self.add_samples(x[sim_mask], i_fidelity=0)
            inline_manager = getattr(self, 'inline_manager', None)
            if inline_manager is not None:
                inline_manager.run_until_done(i_fidelity=0)
            self.hero_wait_for_data_and_train()
        else:
            print("All query points are below the variance threshold; using surrogate only.")

        return self.surrogate.predict_values(x)

    def hero_wait_for_data_and_train(self):
        self.dataset.hero_wait_for_data()
        self.surrogate.train(self.dataset)

    def hero_update_avail_data_and_train(self):
        for i_fl in range(self.dataset.n_fidelity):
            self.dataset.hero_update_avail_data(i_fl)
        self.surrogate.train(self.dataset)
