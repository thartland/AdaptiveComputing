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
from adaptive_computing.local_hero import LocalHeroClient
from manager import create_manager
import numpy as np
import argparse



def bayesian_1d_mf(seed):


    # Single shared client — the dataset and manager both read/write the same
    # JSON file, giving them a consistent view without any network calls.
    local_hero = LocalHeroClient(
        db_path=os.path.join(os.path.dirname(__file__), 'local_hero_db.json'),
    )

    scheduler_type = 'slurm'
    script_names = ['script_lf_slurm.sh', 'script_hf_slurm.sh']
    manager = create_manager(scheduler_type=scheduler_type, hero_client=local_hero, script_names=script_names)

    params = [ContinuousVariable(min=0, max=1)]

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
                                   )
   
    ac_driver.sampler._rand_seed = seed
    ac_driver.init_sampler._rand_seed = seed
    np.random.seed(seed)

    xHF = np.array([[0.05], [0.5], [0.95]])
    xLF = np.vstack([
        xHF,
        np.array([[0.37]])])
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
    ac_driver.run(N_steps = 4, batch_size=2)
    ac_driver.hero_wait_for_data_and_train()

    # plot the result
    #plt.figure(figsize=(10, 6))
    #plt.scatter(ac_driver.dataset.x_data[0], ac_driver.dataset.y_data[0], marker='o', color='b', label='Low fidelity')
    #plt.scatter(ac_driver.dataset.x_data[1], ac_driver.dataset.y_data[1], marker='s', color='r', label='High fidelity')
    #plt.xlabel('x_data')
    #plt.ylabel('y_data')
    #plt.title('Bayesian 1D Multi-Fidelity Optimization')
    #plt.legend()
    #plt.savefig('bayesian_1d_mf_result.png', dpi=150, bbox_inches='tight')
    #print("Plot saved as 'bayesian_1d_mf_result.png'")
    #
    ## Try to show plot if backend supports it
    #try:
    #    plt.show()
    #except Exception as e:
    #    print(f"Interactive display failed ({e}), but plot was saved")
    
    
    




    return ac_driver

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LF synthetic objective simulation"
    )
    parser.add_argument(
        "--seed",
        "-seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    args = parser.parse_args()
    
    ac_driver = bayesian_1d_mf(args.seed)
    print(ac_driver.dataset.x_data[0])
    print(ac_driver.dataset.x_data[1])
    print(ac_driver.dataset.y_data[0])
    print(ac_driver.dataset.y_data[1])

    y_true_min = -6.02074005576708
    print(abs(min(ac_driver.dataset.y_data[1]) - y_true_min))

