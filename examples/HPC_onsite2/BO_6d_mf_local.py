import matplotlib
import os
import pickle
import sys

# Ensure appropriate backend for plotting
def set_matplotlib_backend():
    if os.environ.get('DISPLAY'):
        # Try backends in order of preference for X11 forwarding
        backends_to_try = ['TkAgg', 'Qt5Agg', 'GTK3Agg']
        for backend in backends_to_try:
            try:
                matplotlib.use(backend, force=True)
                # Test if backend actually works
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                plt.close(fig)
                print(f"Using interactive matplotlib backend: {backend}")
                return
            except (ImportError, Exception):
                continue
        
        # If no GUI backends work, fall back to Agg
        print("No working interactive backends, using Agg backend")
        matplotlib.use('Agg')
    else:
        print("No DISPLAY detected, using Agg backend")
        matplotlib.use('Agg')

set_matplotlib_backend()

import matplotlib.pyplot as plt
from adaptive_computing.datasets import ContinuousVariable
from adaptive_computing.drivers import ActiveLoopDriverHeroMFSEGO
from adaptive_computing.drivers import ActiveLoopDriverHeroCostRatioNested
from adaptive_computing.local_hero import LocalHeroClient
from local_manager import LocalFunctionManager
from simulation_files.hartmann import hf_hartmann as func_hf
from simulation_files.hartmann import lf_hartmann as func_lf

import numpy as np
import argparse

Y_TRUE_MIN = -3.32237
ndim=6


def bayesian_mf(seed, N_steps, batch_size, mfsego=True):


    # Single shared client — the dataset and manager both read/write the same
    # JSON file, giving them a consistent view without any network calls.
    if mfsego:
        local_hero = LocalHeroClient(
            db_path=os.path.join(os.path.dirname(__file__), 'local_hero_db.json'),
        )
    else:
        local_hero = LocalHeroClient(
            db_path=os.path.join(os.path.dirname(__file__), 'local_hero_cost_ratio_db.json'),
        )

    manager = LocalFunctionManager(hero_client=local_hero, simulation_functions=[func_lf, func_hf], machine_name="local", max_workers=4,)


    params = [ContinuousVariable(min=0, max=1),]*ndim

    if mfsego:
        ac_driver = ActiveLoopDriverHeroMFSEGO(simulations=[None, None],
                                       fidelity_costs=[1.,3.],
                                       params=params,
                                       machine_names = [manager.machine_name],
                                       output_field_path='y_data',
                                       surrogate='SMT_GP',
                                       acq_func='expected_improvement',
                                       blocking=False,
                                       inline_manager=manager,
                                       hero_client=local_hero,
                                       reference_minimum=Y_TRUE_MIN,
                                       mfsego_cost_power=1.0,
                                       promotion_ei_fraction=1.0,
                                       )
    else:
        ac_driver = ActiveLoopDriverHeroCostRatioNested(simulations=[None, None],
                                       fidelity_costs=[1.,3.],
                                       params=params,
                                       machine_names = [manager.machine_name],
                                       output_field_path='y_data',
                                       surrogate='SMT_GP',
                                       acq_func='expected_improvement',
                                       blocking=False,
                                       inline_manager=manager,
                                       hero_client=local_hero,
                                       reference_minimum=Y_TRUE_MIN,
                                       )

   
    ac_driver.sampler._rand_seed = seed
    ac_driver.init_sampler._rand_seed = seed
    np.random.seed(seed)

    xHF = np.atleast_2d(np.random.rand(8, ndim))
    xLF = np.vstack([
        xHF,
        np.array(np.random.rand(2, ndim))])

    # scale and shift to map rand samples in [0, 1]^ndim --> to actual problem bounds
    scale_shift = lambda x: 1. * x + np.array([0.,] * ndim) 
    xHF = scale_shift(xHF)
    xLF = scale_shift(xLF)


    xHF = 1. * xHF + np.array([0.,]*ndim) # scale and shift to map samples in [0, 1]^ndim --> to actual bounds
    xLF = 1. * xLF + np.array([0.,]*ndim)
    ac_driver.add_samples(xLF, i_fidelity=0)
    ac_driver.add_samples(xHF, i_fidelity=1)

    print('Before first manager run:')
    print(f'_hero_todo = {ac_driver.dataset._hero_todo}')

    active_fidelities = [0, 1]
    manager.run_until_done(i_fidelities=sorted(active_fidelities))
    ac_driver.hero_wait_for_data_and_train()


    print('After first manager run:')
    print(f'_x_data       = {ac_driver.dataset._x_data}')
    print(f'_y_data       = {ac_driver.dataset._y_data}')
    print(f'_hero_todo    = {ac_driver.dataset._hero_todo}')
    print(f'_unmasked_data = {ac_driver.dataset._unmasked_data}')
    ac_driver.run(N_steps = N_steps, batch_size=batch_size)
    ac_driver.hero_wait_for_data_and_train()
    print(f'cost/regret pairs = {ac_driver.total_cost_regret_pairs}')
    return ac_driver

if __name__ == "__main__":
    #seeds = [11, 15, 19, 23, 27, 31, 35, 39, 43]
    #seeds = [36 + i for i in range(6)]
    seeds = [38]
    N_steps = 30
    batch_size = 2
    save_dir = "data/"
    for seed in seeds:
        #ac_driver_sego = bayesian_mf(seed, N_steps, batch_size, True)
        ac_driver_cost_ratio = bayesian_mf(seed, N_steps, batch_size, False)
        #history_sego = np.asarray(ac_driver_sego.total_cost_regret_pairs, dtype=float) 
        history_cost_ratio = np.asarray(ac_driver_cost_ratio.total_cost_regret_pairs, dtype=float) 
        #plt.plot(history_sego[:,0], history_sego[:,1], label="MFSEGO")
        plt.plot(history_cost_ratio[:,0], history_cost_ratio[:,1], label="MF cost ratio")
        plt.grid()
        plt.legend()
        plt.yscale('log')
        plt.show()
        #np.savetxt(save_dir + "regret_vs_cost_sego_seed"+str(seed)+".dat", [(history_sego[i, 0], history_sego[i, 1]) for i in range(len(history_sego))])
        np.savetxt(save_dir + "regret_vs_cost_cost_ratio_seed"+str(seed)+".dat", [(history_cost_ratio[i, 0], history_cost_ratio[i, 1]) for i in range(len(history_cost_ratio))])
        for ac_driver in [ac_driver_cost_ratio]:#, ac_driver_cost_ratio]:
            for record in ac_driver.selection_history:
                #metadata = record["selection_metadata"]
                print(
                    f"Selection={record['selection']}, "
                    f"batch={record['batch']}, "
                    #f"candidate_source={metadata['candidate_source']}, "
                    f"x={record['x'].ravel()}, "
                    f"selected fidelity = {record['selected_fidelity']}, "
                    f"evaluated fidelities = {record['submitted_fidelities']}, "
                    #f"promotion pool size ={metadata['promotion_pool_size']}, "
                    f"outputs={record['outputs']}")
