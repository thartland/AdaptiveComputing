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
from adaptive_computing.drivers import ActiveLoopDriverHero
from adaptive_computing.local_hero import LocalHeroClient
from manager import create_manager
import numpy as np

def bayesian_1d_mf():


    # Single shared client — the dataset and manager both read/write the same
    # JSON file, giving them a consistent view without any network calls.
    local_hero = LocalHeroClient(
        db_path=os.path.join(os.path.dirname(__file__), 'local_hero_db.json'),
    )

    scheduler_type = 'slurm'
    script_names = ['script_lf_slurm.sh', 'script_hf_slurm.sh']
    manager = create_manager(scheduler_type=scheduler_type, hero_client=local_hero, script_names=script_names)

    params = [ContinuousVariable(min=0, max=10)]

    ac_driver = ActiveLoopDriverHero(simulations=[None, None],
                                   fidelity_costs=[1,10],
                                   params=params,
                                   machine_names = [manager.machine_name],
                                   output_field_path='y_data',
                                   surrogate='SMT_GP',
                                   acq_func='maximum_variance',
                                   blocking=False,
                                   inline_manager=manager,
                                   hero_client=local_hero,
                                   )
    
    ac_driver.add_samples(np.array([[0.5], [5.0], [9.5]]), i_fidelity=0)
    ac_driver.add_samples(np.array([[0.5], [5.0], [9.5]]), i_fidelity=1)

    print('Before first manager run:')
    print(f'_hero_todo = {ac_driver.dataset._hero_todo}')

    manager.run_until_done(i_fidelity=0)
    ac_driver.hero_wait_for_data_and_train()
    manager.run_until_done(i_fidelity=1)
    ac_driver.hero_wait_for_data_and_train()

    print('After first manager run:')
    print(f'_x_data       = {ac_driver.dataset._x_data}')
    print(f'_y_data       = {ac_driver.dataset._y_data}')
    print(f'_hero_todo    = {ac_driver.dataset._hero_todo}')
    print(f'_unmasked_data = {ac_driver.dataset._unmasked_data}')
    #ac_driver.run(N_steps = 4)

    ## plot the result
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
    bayesian_1d_mf()
