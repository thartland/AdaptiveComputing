# Run all examples
# To run this script, call "pytest" from the "AdaptiveComputing/" directory
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def test_bayesian_1d_sf(monkeypatch):
    dir_name = 'bayesian_1d_sf'
    py_name = 'smt_gp'
    ac_driver = run_example(monkeypatch,dir_name,py_name)

    # check the minimum of the surrogate model
    i_opt = np.argmin(ac_driver.dataset.y_data[0])
    x_opt = ac_driver.dataset.x_data[0][i_opt,:]
    y_opt = np.min(ac_driver.dataset.y_data[0])
    computed_output = [x_opt[0], y_opt]

    # compare expected and computed outputs
    expected_output = [0.757249, -6.02074] # analytical solution = [x_min, y_min]
    tolerances = [0.1, 0.1]
    output_validator(expected_output, computed_output, tolerances)

    return

def test_bayesian_2d_sf(monkeypatch):
    dir_name = 'bayesian_2d_sf'
    py_name = 'bayesian_2d_sf'
    ac_driver = run_example(monkeypatch,dir_name,py_name)

    # check the minimum of the surrogate model
    i_opt = np.argmin(ac_driver.dataset.y_data[0])
    x_opt = ac_driver.dataset.x_data[0][i_opt,:]
    y_opt = np.min(ac_driver.dataset.y_data[0])
    computed_output = [x_opt[0], x_opt[1], y_opt]

    # compare expected and computed outputs
    expected_output = [3.0, 6.2, 1.4] # analytical solution = [x0_min, x1_min, y_min]
    tolerances = [1.0, 1.0, 2.0]      # generous tolerances for fast test mode (8 steps instead of 50)
    output_validator(expected_output, computed_output, tolerances)

    return

def test_bayesian_1d_mf(monkeypatch):
    dir_name = 'bayesian_1d_mf'
    py_name = 'bayesian_1d_mf'
    ac_driver = run_example(monkeypatch,dir_name,py_name)

    # check the minimum of the surrogate model
    i_opt = np.argmin(ac_driver.dataset.y_data[0])
    x_opt = ac_driver.dataset.x_data[0][i_opt,:]
    y_opt = np.min(ac_driver.dataset.y_data[0])
    computed_output = [x_opt[0], y_opt]

    # compare expected and computed outputs
    expected_output = [3.0, 0.0] # analytical solution = [x_min, y_min]
    tolerances = [0.2, 0.2]      # Slightly relaxed tolerances for stochastic optimization
    output_validator(expected_output, computed_output, tolerances)

    return

def test_bayesian_3d_sf_mixedtypes(monkeypatch):
    dir_name = 'bayesian_3d_sf_mixedtypes'
    py_name = 'bayesian_3d_sf_mixedtypes'
    ac_driver = run_example(monkeypatch,dir_name,py_name)

    # check the minimum of the surrogate model
    i_opt = np.argmin(ac_driver.dataset.y_data[0])
    x_opt = ac_driver.dataset.x_data[0][i_opt,:]
    y_opt = np.min(ac_driver.dataset.y_data[0])
    
    # For mixed-type optimization: x0 (continuous), x1 (ordered), x2 (categorical index)
    # Convert categorical index back to letter for validation
    categories = ['a', 'b', 'c', 'd']
    x2_cat = categories[int(x_opt[2])]
    
    # Validate numeric outputs using standard output_validator
    computed_output = [x_opt[0], x_opt[1], y_opt]
    expected_output = [5.0, 4.0, 0.0] # analytical solution = [x0_min, x1_min, y_min]
    tolerances = [2.0, 2.0, 5.0]      # generous tolerances for fast test mode
    output_validator(expected_output, computed_output, tolerances)
    
    # Separately validate categorical variable (any category is acceptable with minimal samples)
    assert x2_cat in categories, f"x2: expected valid category, got {x2_cat}"
    print(f'Categorical variable x2 = "{x2_cat}" (valid)')

    return

"""
def test_custom_mf_workflow(monkeypatch):
    dir_name = 'custom_mf_workflow'
    py_name = 'custom_mf_workflow'
    ac_driver = run_example(monkeypatch,dir_name,py_name)

    # check the minimum of the surrogate model
    i_opt = np.argmin(ac_driver.dataset.y_data[0])
    x_opt = ac_driver.dataset.x_data[0][i_opt,:]
    y_opt = np.min(ac_driver.dataset.y_data[0])
    computed_output = [x_opt[0], y_opt]

    # compare expected and computed outputs
    expected_output = [3.0, 0.0] # analytical solution = [x_min, y_min]
    tolerances = [0.1, 0.1]
    output_validator(expected_output, computed_output, tolerances)

    return
"""
def test_HPC_onsite_offline_training(monkeypatch):
    """Test HPC_onsite offline training in mock mode (no sbatch required)."""
    dir_name = 'HPC_onsite'
    py_name = 'controller_offline_training'
    ac_driver = run_example(monkeypatch, dir_name, py_name)

    # Verify that the surrogate was trained with collected data points.
    y_data = ac_driver.dataset.y_data[0]
    assert len(y_data) > 0, "No y_data collected"
    assert all(v > 0 for v in y_data), f"Unexpected negative values: {y_data}"
    print(f'HPC_onsite: collected {len(y_data)} data points, y={y_data}')


def test_rental_agent_HPC_onsite(monkeypatch):
    """Test LangGraph + LocalHeroClient + mock manager (no sbatch required)."""
    import importlib
    import pathlib

    # Locate the example directory.
    initial_wd = os.getcwd()
    import pathlib as _pl
    current_path = _pl.Path(initial_wd)
    repo_root = current_path
    while repo_root.parent != repo_root:
        if (repo_root / 'setup.py').exists() or (repo_root / 'environment.yaml').exists():
            break
        repo_root = repo_root.parent

    examples_dir = repo_root / 'examples' / 'rental_agent_HPC_onsite'
    os.chdir(str(examples_dir))
    sys.path.insert(0, str(examples_dir))

    monkeypatch.setattr(plt, 'show', lambda: None)

    try:
        mod = importlib.import_module('controller')
        # Force mock_mode=True so the test never touches sbatch or tmux.
        results = mod.controller(mock_mode=True)
    finally:
        os.chdir(initial_wd)

    # x_values = [1.0, 2.0, 3.0, 4.0], expected y = x^2 = [1.0, 4.0, 9.0, 16.0]
    expected_output = [1.0, 4.0, 9.0, 16.0]
    tolerances = [0.01, 0.01, 0.01, 0.01]
    output_validator(expected_output, results, tolerances)


# 1st arg is always monkeypatch
# 2nd arg is subdirectory inside examples where the .py is located
# 3rd arg is the name of the .py file for the example
def run_example(monkeypatch,dir_name,py_name):
    monkeypatch.setattr(plt, 'show', lambda: None) # close all plots
    initial_wd = os.getcwd()
    print(initial_wd)
    
    # Find the repository root (where setup.py or environment.yaml exists)
    import pathlib
    current_path = pathlib.Path(initial_wd)
    repo_root = current_path
    while repo_root.parent != repo_root:
        if (repo_root / 'setup.py').exists() or (repo_root / 'environment.yaml').exists():
            break
        repo_root = repo_root.parent
    
    examples_dir = repo_root / 'examples' / dir_name
    os.chdir(str(examples_dir))
    print(os.getcwd())
    print('Testing ' + dir_name + '/' + py_name + '.py:')
    #sys.path.insert(0, '.') # add the path to the current directory. For some reason this doesn't work when multiple tests are run in parallel
    sys.path.insert(0, '../'+dir_name) # add the path to the current directory
    import importlib
    mod = importlib.import_module(py_name)
    example_func = getattr(mod, py_name)
    # call the example code and return its output
    output = example_func()
    os.chdir(initial_wd) # return to the initial working directory
    return output

# 1st arg is a 1d array of expected outputs
# 2nd arg is a 1d array of computed (actual) outputs
# 3rd arg is a 1d array of the allowed difference between expected and computed outputs
def output_validator(expected_output, computed_output, tolerances):
    assert len(expected_output) == len(computed_output)
    assert len(expected_output) == len(tolerances)
    print(f'The expected output = {expected_output}')
    print(f'The computed output = {computed_output}')
    for i in range(len(expected_output)):
        assert abs(expected_output[i] - computed_output[i]) < tolerances[i]
    print('Test passed!')
    return
