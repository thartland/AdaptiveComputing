# In-process HPC controller — runs on any node that has both scheduler access
# (sbatch/qsub/squeue) and outbound internet access to the Hero API.  In
# practice this is usually an HPC login node, but a compute node with internet
# access works equally well.
#
# Differences from examples/hero_HPC_managers/controller_offline_training.py (the multi-cluster version):
#   - No SSH, no tmux, no hpc_config.py.
#   - Uses LocalHeroClient instead of the real Hero service — no Hero
#     credentials or internet access required.
#   - manager.run_until_done() replaces run_remote_managers() + wait_for_managers()
#     + the Hero wait loop.  The controller and manager are the same process.
#   - hero_wait_for_data_and_train() still collects results and trains the
#     surrogate; it returns almost immediately since run_until_done() already
#     marked all tasks done.
#
# Usage:
#   python controller_offline_training.py [--scheduler {slurm,pbs}]

import os
import pickle
import shutil
import sys
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np

from adaptive_computing.datasets import ContinuousVariable
from adaptive_computing.drivers import ActiveLoopDriverHero
from adaptive_computing.local_hero import LocalHeroClient
from manager import create_manager

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def controller_offline_training(scheduler_type='slurm', mock_mode=None):
    """Run offline Bayesian optimization training using the local HPC manager.

    Args:
        scheduler_type: ``'slurm'`` (default) or ``'pbs'``.
        mock_mode:      When ``True``, bypass the scheduler and run simulations
                        directly.  When ``None`` (default), auto-detects: uses
                        mock mode if ``sbatch``/``qsub`` is not on PATH.

    Returns:
        The trained :class:`ActiveLoopDriverHero` instance.
    """
    if mock_mode is None:
        mock_mode = not bool(shutil.which("sbatch") or shutil.which("qsub"))

    # Unique db path so parallel test runs don't share state.
    db_path = os.path.join(SCRIPT_DIR, f"local_hero_db_{uuid.uuid4().hex[:8]}.json")

    local_hero = LocalHeroClient(db_path=db_path)
    manager = create_manager(
        scheduler_type=scheduler_type,
        hero_client=local_hero,
        mock_mode=mock_mode,
    )

    params = [ContinuousVariable(min=0.8, max=2.0)]
    ac_driver = ActiveLoopDriverHero(
        simulations=[None],
        params=params,
        machine_names=[manager.machine_name],
        output_field_path='y_data',
        surrogate='SMT_GP',
        acq_func='maximum_variance',
        blocking=False,
        hero_client=local_hero,
    )
    ac_driver.inline_manager = manager

    ac_driver.dataset.add_known_samples(
        np.array([[0.938], [1.443], [1.641]]),
        np.array([[2.03],  [3.51],  [3.81]]),
        0,
    )

    ac_driver.add_samples(np.array([[0.8], [2.0]]), i_fidelity=0)

    print('Before first manager run:')
    print(f'_hero_todo = {ac_driver.dataset._hero_todo}')

    manager.run_until_done(i_fidelity=0)
    ac_driver.hero_wait_for_data_and_train()

    print('After first manager run:')
    print(f'_x_data       = {ac_driver.dataset._x_data}')
    print(f'_y_data       = {ac_driver.dataset._y_data}')

    ac_driver.add_samples(np.array([[1.3], [1.7]]), i_fidelity=0)
    ac_driver.run(N_steps=2)

    print('After add_samples + run(N_steps=2):')
    print(f'_hero_todo = {ac_driver.dataset._hero_todo}')

    manager.run_until_done(i_fidelity=0)
    ac_driver.hero_wait_for_data_and_train()

    print('After second manager run:')
    print(f'_x_data       = {ac_driver.dataset._x_data}')
    print(f'_y_data       = {ac_driver.dataset._y_data}')

    if os.path.exists(db_path):
        os.remove(db_path)

    return ac_driver


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scheduler', default='slurm', choices=['slurm', 'pbs'])
    args = parser.parse_args()

    ac_driver = controller_offline_training(scheduler_type=args.scheduler)

    with open('offline_training.pkl', 'wb') as f:
        pickle.dump(ac_driver, f)
    print('Saved offline_training.pkl')
