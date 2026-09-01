# HPC Onsite — No-Credential Single-Node Workflows

**Advanced Example** — Complete [examples/hero/](../hero/) first to understand the basic workflow, then read this.

This example runs the controller directly on the HPC cluster — the same node that can submit Slurm or PBS jobs.  Because the controller and the manager share the same process and use a local JSON file as the task queue, **no Hero credentials, no SSH setup, and no external services are required**.

| Feature | HPC_onsite | hero_HPC_managers |
|---|---|---|
| Hero credentials | **Not required** | Required |
| Controller location | On the HPC cluster itself | Anywhere with SSH to HPC |
| Task queue | Local JSON file | Hero cloud service |
| Manager startup | Inline call in controller | SSH + tmux daemon |
| Config file | Not needed | `hpc_config.py` required |
| Multi-cluster support | No (single node) | Yes |
| Fault tolerance | Controller process must stay alive | Daemon survives SSH disconnect |

**When to use this example**: You are logged into the HPC cluster and want the simplest possible setup — one terminal, one process, no accounts to create.

**When to use [hero_HPC_managers](../hero_HPC_managers/) instead**: You need to run the controller from a machine that cannot submit jobs directly (e.g. your laptop), distribute work across multiple clusters, or want a persistent manager daemon that survives network interruptions.

---

# Prerequisites

## 1. Complete the Basic Hero Example

**First complete**: [examples/hero/](../hero/) to understand the controller/manager workflow before adding HPC job submission.

## 2. Repository Setup

Install AC on the HPC cluster where you will run the controller:

```bash
# On the HPC cluster:
git clone <repo-url> AdaptiveComputing
cd AdaptiveComputing
mamba activate AC
pip install -e .
```

## 3. Edit the Batch Script

Open `simulation_files/script_generic_slurm.sh` and update the two cluster-specific directives:

```bash
#SBATCH --partition=<YOUR_PARTITION>   # e.g. 'debug', 'standard', 'compute'
#SBATCH --account=<YOUR_ACCOUNT>       # your HPC billing account/allocation
```

Run `sinfo` to list available partitions. Run `sacctmgr show user $USER` to find your account name.

For PBS systems, use `simulation_files/script_generic_pbs.sh` instead and pass `scheduler_type='pbs'` to `create_manager()` in `manager.py`.

**No Hero credentials are needed.** The local JSON task queue (`local_hero_db.json`) is created automatically in the example directory on first run.

---

# Architecture

```
HPC cluster (login node or internet-accessible compute node)
├── controller_offline_training.py   ← you run this
│   ├── LocalHeroClient("local_hero_db.json")  ← shared task queue (JSON file)
│   ├── manager.run_until_done()     ← submits + monitors Slurm jobs inline
│   └── hero_wait_for_data_and_train() ← reads results, retrains surrogate
│
└── Slurm / PBS scheduler
    └── Compute nodes run script_generic_slurm.sh → mock_simulation.py
        └── writes result_<task_id>.txt back to simulation_files/
```

The controller and manager run in the **same Python process**.  `LocalHeroClient` writes tasks to a JSON file; the manager reads from the same file — no network calls between them.  `run_until_done()` blocks until all queued tasks reach `done` or `error`, then returns control to the controller.

**No daemon, no tmux, no SSH.** If the controller process is killed, any running Slurm jobs continue on the compute nodes and their result files are picked up automatically via startup reconciliation on the next run.

---

# Available Controllers

Run in this order — each step produces a `.pkl` file consumed by the next:

### 1. `controller_offline_training.py`

Builds the surrogate model from scratch using manual sampling and Bayesian optimization.  All Slurm jobs are submitted and monitored inline.

```bash
cd examples/HPC_onsite
mamba activate AC
python controller_offline_training.py
# Produces: offline_training.pkl, local_hero_db.json
```

### 2. `controller_offline_inference.py`

Loads `offline_training.pkl` and queries the trained surrogate **locally** — no Slurm jobs, no scheduler access needed.

```bash
python controller_offline_inference.py
```

### 3. `controller_online_inference.py`

Loads `offline_training.pkl` and refines the surrogate with live Slurm jobs when prediction variance exceeds a threshold.  High-confidence points are answered by the surrogate; uncertain points trigger new simulations.

```bash
python controller_online_inference.py
# Produces: online_training.pkl
```

---

# How to Adapt for Your Simulation

The key file to modify is `manager.py`.  The two methods you implement control job submission and result reading:

```python
class MySimManager(LocalHPCManager):

    def submit_job(self, task, machine_name, i_fidelity):
        # Extract parameters from task metadata
        params = task['metadata']['x_data']
        task_id = task['id']
        script = self.batch_scripts[i_fidelity]
        cmd = f"sbatch {script} {params[0]} {params[1]} {task_id}"
        return self._run_submit(cmd)

    def read_result(self, task_id):
        result_file = f"result_{task_id}.txt"
        if os.path.exists(result_file):
            value = open(result_file).read().strip()
            os.remove(result_file)
            return value
        return "-1"   # signals the manager to mark the task as error
```

Then update `script_generic_slurm.sh` to run your real simulation and write the result to `result_${task_id}.txt`.

In the controller, create a `LocalHeroClient` pointing at a JSON file and pass it to both the manager and the driver:

```python
from adaptive_computing.local_hero import LocalHeroClient

local_hero = LocalHeroClient("local_hero_db.json")
manager    = create_manager(scheduler_type='slurm', hero_client=local_hero)
ac_driver  = ActiveLoopDriverHero(..., hero_client=local_hero)
```

If you later want to scale to multiple clusters using the real Hero service, replace `LocalHeroClient` with the real `HeroClient` and switch to [hero_HPC_managers](../hero_HPC_managers/).

---

# manager.py Details

`MockSimManager` subclasses `LocalHPCManager` from `adaptive_computing.hpc.local_manager`.
`LocalHPCManager.run_until_done()` implements the full event loop:

1. **Startup reconciliation** — reset stale job IDs from any previous interrupted run; retry tasks in `error` state.
2. **Pass 1** — for tasks with a Slurm job ID, check `sacct`/`squeue` status.  On RUNNING → call `hero_initialize` to claim the task.  On COMPLETED → read result file → call `hero_finalize` to mark task `done`.
3. **Pass 2** — for tasks without a job ID, call `submit_job()` → `sbatch`.  Stops early if the per-user job limit is reached (retries next cycle).
4. **Running tasks block** — cancel any duplicate jobs.
5. **Exit check** — if `ready + running == 0`, all tasks are done or error; return to caller.

The working directory is temporarily changed to `simulation_dir` during `run_until_done()` so that `SLURM_SUBMIT_DIR` points to `simulation_files/` (where `mock_simulation.py` lives and result files are written).  The original working directory is restored on return.

---

# Troubleshooting

**"sbatch: command not found"**: The node you are running on cannot submit to the scheduler.  This example must run on a node with access to `sbatch`/`qsub`/`squeue` — typically an HPC login node.

**"simulation_dir not found"**: The `simulation_files/` directory must be present relative to `manager.py`.  If you moved files, update `SIMULATION_DIR` in `manager.py`.

**Jobs complete but results not picked up**: Check that `script_generic_slurm.sh` writes `result_${task_id}.txt` to `$SLURM_SUBMIT_DIR` (the directory from which `sbatch` is called, which is `simulation_files/` here).

**Ctrl+C during run_until_done()**: Any Slurm jobs already submitted continue running on the compute nodes.  Re-running the controller will pick up completed results via startup reconciliation.

**Stale task database**: `local_hero_db.json` persists between runs.  If you want a clean slate, delete it before running — or call `ac_driver.dataset.clear_hero_queue()` at the start of your controller script (as `controller_online_inference.py` already does).

---

# Learning Path

**New to AC or Hero concepts?** → [examples/hero/](../hero/)

**Need multiple clusters, real Hero service, or remote controller?** → [examples/hero_HPC_managers/](../hero_HPC_managers/)
