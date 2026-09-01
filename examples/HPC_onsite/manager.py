"""
manager.py — Concrete LocalHPCManager for the mock-simulation example.

Defines MockSimManager, which submits jobs to SLURM or PBS and reads result
files written by the batch script.  The controller imports this module and
calls manager.run_until_done() directly — no SSH or tmux involved.

To adapt for a real simulation:
  - Replace submit_job with your sbatch/qsub command (add case directories,
    config files, additional arguments, etc.)
  - Replace read_result with logic that parses your simulation's output file.
  - Adjust the #SBATCH / #PBS directives in the batch script for your system.
  - Pass scheduler_type='pbs' to create_manager() for PBS systems.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from adaptive_computing.hpc.local_manager import LocalHPCManager

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
SIMULATION_DIR = os.path.join(SCRIPT_DIR, 'simulation_files')


class MockSimManager(LocalHPCManager):
    """Manager for the mock conductivity simulation (conductivity = T² / 1000).

    Submits one Slurm or PBS job per task.  The batch script writes the result
    to ``result_<task_id>.txt`` in the simulation directory; read_result picks
    it up and returns it as a string.

    When *mock_mode* is ``True`` the batch script is run directly as a plain
    bash script (bypassing sbatch/qsub) so the example can be tested without
    an HPC scheduler.
    """

    def __init__(self, *args, mock_mode: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.mock_mode = mock_mode

    def submit_job(self, task, machine_name, i_fidelity):
        t       = task['metadata']['x_data'][0]
        task_id = task['id']
        script  = self.batch_scripts[i_fidelity]
        if self.mock_mode:
            # Run the batch script directly without a scheduler so tests work
            # on machines that don't have sbatch/qsub.  The #SBATCH/#PBS
            # directives are silently ignored by bash.
            cmd = f"bash {script} {t} {task_id}"
            subprocess.run(cmd, shell=True, cwd=SIMULATION_DIR)
            return f"mock_{task_id}"
        if self.scheduler_type == 'pbs':
            cmd = f"qsub -v 'temp={t},task_id={task_id}' {script}"
        else:
            cmd = f"sbatch {script} {t} {task_id}"
        return self._run_submit(cmd)

    def read_result(self, task_id):
        result_file = f"result_{task_id}.txt"
        if os.path.exists(result_file):
            with open(result_file) as f:
                value = f.read().strip()
            os.remove(result_file)
            return value
        print(f"WARNING: result file not found for task {task_id}, using -1")
        return "-1"


def create_manager(scheduler_type='slurm', hero_client=None, mock_mode=False):
    """Return a MockSimManager configured for this machine.

    Args:
        scheduler_type: ``'slurm'`` (default) or ``'pbs'``.
        hero_client:    Optional Hero client.  Pass a
                        :class:`~adaptive_computing.local_hero.LocalHeroClient`
                        to use a local JSON file instead of the real Hero
                        service.
        mock_mode:      When ``True``, run simulations directly without a
                        scheduler (useful for testing without HPC access).

    Returns:
        A ready-to-use :class:`MockSimManager` instance.
    """
    if scheduler_type == 'pbs':
        batch_script = os.path.join(SIMULATION_DIR, 'script_generic_pbs.sh')
    else:
        batch_script = os.path.join(SIMULATION_DIR, 'script_generic_slurm.sh')

    poll_interval = 1 if mock_mode else 5

    return MockSimManager(
        machine_name='local',
        batch_scripts=[batch_script],
        scheduler_type=scheduler_type,
        simulation_dir=SIMULATION_DIR,
        poll_interval=poll_interval,
        hero_client=hero_client,
        mock_mode=mock_mode,
    )
