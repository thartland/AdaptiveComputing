"""
manager.py — NEB-style LocalHPCManager for the rental_agent_HPC_onsite example.

Demonstrates the pattern:
  - LangGraph controller submits tasks to LocalHeroClient JSON queue
  - This manager daemon reads the queue, submits SLURM jobs, polls sacct,
    and marks tasks done/error — all independently of the controller process
  - run_forever() keeps the daemon alive between controller restarts

Computation: y = x² (trivial mock, no GPU required).

To adapt for a real application:
  - Replace submit_job with your domain-specific sbatch command
  - Replace read_result with logic that parses your output file
  - Run this script in a tmux session via local_launcher.ensure_manager_running()
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from adaptive_computing.hpc.local_manager import LocalHPCManager
from adaptive_computing.local_hero import LocalHeroClient

SCRIPT_DIR = Path(__file__).parent.resolve()
MANAGER_SCRIPT = Path(__file__).resolve()


class RentalAgentManager(LocalHPCManager):
    """Manager for the x² mock simulation.

    In real mode, submits ``job.sh`` via sbatch.
    In mock mode, computes the result directly (no scheduler required).
    """

    def __init__(self, *args, mock_mode: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.mock_mode = mock_mode

    def submit_job(self, task: dict, machine_name: str, i_fidelity: int) -> str:
        x = task["metadata"]["x_value"]
        task_id = task["id"]
        simulation_dir = str(self.simulation_dir or SCRIPT_DIR)

        if self.mock_mode:
            result = x ** 2
            Path(simulation_dir, f"result_{task_id}.txt").write_text(str(result))
            return f"mock_{task_id}"

        script = self.batch_scripts[i_fidelity]
        cmd = f"sbatch {script} {x} {task_id} {simulation_dir!r}"
        return self._run_submit(cmd)

    def read_result(self, task_id: str) -> str:
        result_file = Path(str(self.simulation_dir or ".")) / f"result_{task_id}.txt"
        if result_file.exists():
            value = result_file.read_text().strip()
            result_file.unlink()
            return value
        return "-1"


def create_manager(
    machine_name: str,
    work_dir: str,
    hero_client: LocalHeroClient,
    mock_mode: bool = False,
) -> RentalAgentManager:
    """Return a configured :class:`RentalAgentManager`.

    Args:
        machine_name: Logical machine name (e.g. ``"kestrel"``).
        work_dir:     Absolute path to the working directory where result
                      files and the Hero JSON db live.
        hero_client:  Shared :class:`LocalHeroClient` instance.
        mock_mode:    Skip sbatch and compute results directly for testing.
    """
    batch_script = str(SCRIPT_DIR / "job.sh")
    poll_interval = 1 if mock_mode else 10

    return RentalAgentManager(
        machine_name=machine_name,
        batch_scripts=[batch_script],
        scheduler_type="slurm",
        simulation_dir=work_dir,
        poll_interval=poll_interval,
        hero_client=hero_client,
        mock_mode=mock_mode,
    )


if __name__ == "__main__":
    work_dir = sys.argv[1] if len(sys.argv) > 1 else str(SCRIPT_DIR)
    machine_name = sys.argv[2] if len(sys.argv) > 2 else "kestrel"

    hero_client = LocalHeroClient(
        db_path=str(Path(work_dir) / "hero_db.json"),
        queue_name="jobs",
        application_id="rental_agent_HPC_onsite",
    )
    mgr = create_manager(
        machine_name=machine_name,
        work_dir=work_dir,
        hero_client=hero_client,
    )
    mgr.run_forever()
