"""
local_manager.py — In-process HPC manager for same-node execution.

Overview
--------
``LocalHPCManager`` implements the Hero/HPC manager event loop as an
in-process call rather than a persistent daemon.  Use this when the
controller runs on a node that has both scheduler access (``sbatch``/``qsub``)
and outbound internet access to the Hero API — no SSH or tmux required.
This is typically an HPC login node, but a compute node with internet access
works equally well.

The key method is :meth:`run_until_done`, which submits and monitors
scheduler jobs until all Hero tasks in the queue reach a terminal state
(``done`` or ``error``).

Subclassing
-----------
Override two abstract methods::

    class MyAppManager(LocalHPCManager):

        def submit_job(self, task, machine_name, i_fidelity):
            t = task['metadata']['x_data'][0]
            script = self.batch_scripts[i_fidelity]
            cmd = f"sbatch {script} {t} {task['id']}"
            return self._run_submit(cmd)

        def read_result(self, task_id):
            result_file = f"result_{task_id}.txt"
            if os.path.exists(result_file):
                value = open(result_file).read().strip()
                os.remove(result_file)
                return value
            return "-1"

Usage
-----
Instantiate in the controller and call :meth:`run_until_done` after queueing
tasks::

    manager = MyAppManager(
        machine_name='local',
        batch_scripts=['/abs/path/to/simulation_files/script.sh'],
        scheduler_type='slurm',
        simulation_dir='/abs/path/to/simulation_files',
    )

    ac_driver.add_samples(x_data, i_fidelity=0)
    manager.run_until_done(i_fidelity=0)       # blocks until all tasks done/error
    ac_driver.hero_wait_for_data_and_train()   # collects results, retrains surrogate
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from abc import ABC, abstractmethod

from .manager_base import (
    JobLimitError,
    TaskError,
    _call_hero_initialize,
    _call_hero_finalize,
)
from .scheduler import (
    cancel_job,
    get_job_status,
    is_job_limit_error,
    parse_job_id,
)


class LocalHPCManager(ABC):
    """In-process Hero/HPC manager for controllers with local scheduler access.

    Unlike :class:`~adaptive_computing.hpc.manager_base.HeroHPCManager` (which
    runs as a daemon in a tmux session started via SSH), ``LocalHPCManager``
    integrates directly into the controller process.  Call
    :meth:`run_until_done` after queuing tasks; it processes them through the
    local scheduler and returns only when all tasks are complete.

    Requires the controller to run on a node that has both scheduler access
    (``sbatch``/``qsub``/``squeue``) and outbound internet access to the Hero
    API.  In practice this is usually an HPC login node, but a compute node
    with internet access works equally well.

    No ``hpc_config.py`` is needed: since the controller runs on the same node
    as the scheduler, there are no remote credentials, remote paths, or Python
    interpreter paths to configure.

    Attributes:
        machine_name:   Logical name for this machine stored in Hero task
                        metadata (e.g. ``'local'`` or the cluster hostname).
        batch_scripts:  List of batch script *absolute* paths indexed by
                        fidelity level.  ``batch_scripts[0]`` is used for
                        ``i_fidelity=0``.
        scheduler_type: ``'slurm'`` or ``'pbs'`` (default: ``'slurm'``).
        simulation_dir: Absolute path to the directory containing batch
                        scripts and result files.  :meth:`run_until_done`
                        temporarily ``chdir``\s there so that
                        ``SLURM_SUBMIT_DIR`` and result file paths are
                        consistent with the batch script.  Set to ``None``
                        to leave the working directory unchanged.
        poll_interval:  Seconds to sleep between polling cycles (default 5).
        hero_client:    Optional Hero client instance.  Pass a
                        :class:`~adaptive_computing.local_hero.LocalHeroClient`
                        to use a local JSON file instead of the real Hero
                        service.  When ``None`` the real ``HeroClient`` is used.
    """

    def __init__(
        self,
        machine_name: str,
        batch_scripts: list[str],
        scheduler_type: str = "slurm",
        simulation_dir: str | None = None,
        poll_interval: int = 5,
        hero_client=None,
    ) -> None:
        self.machine_name = machine_name
        self.batch_scripts = batch_scripts
        self.scheduler_type = scheduler_type
        self.simulation_dir = simulation_dir
        self.poll_interval = poll_interval
        self.hero_client = hero_client

    # ------------------------------------------------------------------
    # Abstract interface — implement these two methods in your subclass
    # ------------------------------------------------------------------

    @abstractmethod
    def submit_job(self, task: dict, machine_name: str, i_fidelity: int) -> str:
        """Submit *task* to the local scheduler and return the job ID.

        Implementations should:

        1. Extract simulation parameters from ``task['metadata']``.
        2. Build the ``sbatch``/``qsub`` command (use absolute script path).
        3. Call :meth:`_run_submit` to execute it and return the job ID.

        Args:
            task:         Hero task dict (keys: ``'id'``, ``'name'``, ``'metadata'``).
            machine_name: Same as ``self.machine_name``.
            i_fidelity:   Fidelity level index (0 for single-fidelity).

        Returns:
            Scheduler job ID string (e.g. ``"12345"``).

        Raises:
            :class:`~adaptive_computing.hpc.manager_base.JobLimitError`
            :class:`~adaptive_computing.hpc.manager_base.TaskError`
        """

    @abstractmethod
    def read_result(self, task_id: str) -> str:
        """Read the simulation result for *task_id* and return it as a string.

        Return ``"-1"`` if the result file is missing or unreadable.
        Implementations should delete the result file after reading to prevent
        stale data from appearing in subsequent polling cycles.

        When :attr:`simulation_dir` is set, the working directory is
        ``simulation_dir`` during :meth:`run_until_done`, so plain relative
        paths (``f"result_{task_id}.txt"``) resolve there.

        Args:
            task_id: Hero task ID string.

        Returns:
            Result string to pass to ``hero_finalize``.
        """

    # ------------------------------------------------------------------
    # Scheduler helper available to subclasses
    # ------------------------------------------------------------------

    def _run_submit(self, command: str) -> str:
        """Run an ``sbatch``/``qsub`` command and return the job ID.

        Args:
            command: Full submission command string.

        Returns:
            Scheduler job ID string.

        Raises:
            :class:`~adaptive_computing.hpc.manager_base.JobLimitError`
            :class:`~adaptive_computing.hpc.manager_base.TaskError`
        """
        print(f"Running: {command}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            if is_job_limit_error(result.stderr):
                raise JobLimitError(
                    f"Job limit reached — will retry next cycle.\n"
                    f"  STDERR: {result.stderr.strip()}"
                )
            raise TaskError(
                f"Job submission failed (rc={result.returncode}).\n"
                f"  STDOUT: {result.stdout.strip()}\n"
                f"  STDERR: {result.stderr.strip()}"
            )
        return parse_job_id(result.stdout)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_until_done(self, i_fidelity: int = 0) -> None:
        """Process all Hero tasks in the queue until none remain active.

        Authenticates with Hero, runs startup reconciliation to reset any
        stale job IDs from a previous run, then polls the queue in a loop.
        Exits when the count of ``ready`` + ``running`` tasks drops to zero
        (all tasks are ``done`` or ``error``).

        If :attr:`simulation_dir` is set, the working directory is temporarily
        changed there for the duration of this call so that ``SLURM_SUBMIT_DIR``
        equals ``simulation_dir`` and result files land in the expected location.
        The original working directory is restored on return (even on error).

        Args:
            i_fidelity: Fidelity level index (0 for single-fidelity).
        """
        if self.hero_client is not None:
            hero           = self.hero_client
            hero_queue     = getattr(self.hero_client, 'queue_name', 'local')
            application_id = getattr(self.hero_client, 'application_id', 'local')
        else:
            from hero import HeroClient, get_env_variable
            from adaptive_computing.hero_utils.set_hero_env_vars import set_hero_env_vars
            set_hero_env_vars()
            try:
                hero_env     = get_env_variable("HERO_ENV", "dev")
                hero_project = get_env_variable("HERO_PROJECT")
                hero_queue   = get_env_variable("HERO_QUEUE")
            except EnvironmentError as e:
                print(e)
                sys.exit(1)
            application_id = f"{hero_env}-{hero_project}"
            hero = HeroClient()

        queue_name = hero_queue if i_fidelity == 0 else hero_queue + str(i_fidelity)
        machine_name = self.machine_name
        scheduler_type = self.scheduler_type

        task_engine = hero.TaskEngine(application_id)
        try:
            hero.authenticate()
        except Exception as e:
            print(f"ERROR: Hero authentication failed: {e}")
            sys.exit(1)

        try:
            queue_record = task_engine.read_queue_by_name(name=queue_name, state="active")
            print(f"Found existing active queue: {queue_name}")
        except Exception:
            print(f"No active queue found, creating new queue: {queue_name}")
            queue_record = task_engine.add_queue(name=queue_name)

        print(f"Scheduler type: {scheduler_type}")

        # Temporarily chdir into simulation_dir so SLURM_SUBMIT_DIR matches
        # the location of mock_simulation.py and result files.
        original_cwd = os.getcwd()
        try:
            if self.simulation_dir is not None:
                if os.path.isdir(self.simulation_dir):
                    os.chdir(self.simulation_dir)
                else:
                    print(
                        f"WARNING: simulation_dir '{self.simulation_dir}' not found; "
                        "skipping chdir."
                    )

            self._reconcile(task_engine, queue_record, machine_name, scheduler_type)

            print(f"Processing queue — polling every {self.poll_interval}s...")
            while True:
                self._poll_cycle(
                    task_engine, queue_record, machine_name, i_fidelity, scheduler_type
                )

                n_ready = len(task_engine.read_tasks(
                    queue_id=queue_record["id"], metatype="Task", state="ready"
                ))
                n_running = len(task_engine.read_tasks(
                    queue_id=queue_record["id"], metatype="Task", state="running"
                ))
                if n_ready + n_running == 0:
                    print("All tasks complete (done or error). Exiting manager loop.")
                    break

                time.sleep(self.poll_interval)

        finally:
            os.chdir(original_cwd)

    def run_forever(self, i_fidelity: int = 0) -> None:
        """Process Hero tasks indefinitely — daemon mode for a tmux manager session.

        Loops forever: runs :meth:`run_until_done` until the queue empties,
        sleeps :attr:`poll_interval` seconds, then checks again.  Tasks added
        by the controller while the queue is idle are picked up on the next
        iteration.  Interrupt with Ctrl-C or ``SIGTERM`` to stop.

        Args:
            i_fidelity: Fidelity level index (0 for single-fidelity).
        """
        print(
            f"Starting daemon mode — polling every {self.poll_interval}s. "
            "Press Ctrl-C to stop."
        )
        while True:
            self.run_until_done(i_fidelity=i_fidelity)
            print(
                f"Queue empty — sleeping {self.poll_interval}s before next check..."
            )
            time.sleep(self.poll_interval)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reconcile(self, task_engine, queue_record, machine_name, scheduler_type):
        """Reset stale scheduler job IDs and requeue error tasks."""
        print("Running startup reconciliation...")
        for state in ("ready", "error"):
            stale_tasks = task_engine.read_tasks(
                queue_id=queue_record["id"], metatype="Task", state=state
            )
            for task in stale_tasks:
                try:
                    meta   = task.get("metadata") or {}
                    job_id = meta.get("scheduler_job_id", {}).get(machine_name, -1)
                    needs_reset = False

                    if state == "error":
                        print(f"  Resetting error task {task['id']} to ready for retry")
                        needs_reset = True
                    elif job_id != -1:
                        check_cmd = (
                            f"qstat -x {job_id}"
                            if scheduler_type == "pbs"
                            else f"squeue -j {job_id} --noheader"
                        )
                        check = subprocess.run(
                            check_cmd, shell=True, capture_output=True, text=True
                        )
                        if check.returncode != 0 or not check.stdout.strip():
                            print(
                                f"  Stale job ID {job_id} for task {task['id']} "
                                "not found in scheduler — resetting"
                            )
                            needs_reset = True

                    if needs_reset:
                        task["metadata"]["scheduler_job_id"][machine_name] = -1
                        task["metadata"]["running"][machine_name] = False
                        task_engine.update_task(
                            task_id=task["id"], state="ready",
                            name=task["name"], metadata=task["metadata"],
                        )
                except Exception as e:
                    print(f"  WARNING: reconciliation failed for task {task['id']}: {e}")
        print("Startup reconciliation complete.")

    def _poll_cycle(self, task_engine, queue_record, machine_name, i_fidelity, scheduler_type):
        """Execute one pass of the event loop (Pass 1, Pass 2, running tasks)."""
        ready_tasks = task_engine.read_tasks(
            queue_id=queue_record["id"], metatype="Task", state="ready"
        )

        # Split 'ready' into unsubmitted vs submitted-but-pending in the scheduler.
        # Hero has no separate "queued" state, so both live under 'ready'.
        n_queued = sum(
            1 for t in ready_tasks
            if t.get("metadata", {}).get("scheduler_job_id", {}).get(machine_name, -1) != -1
        )
        n_unsubmitted = len(ready_tasks) - n_queued
        print(f"  {n_unsubmitted} task(s) in 'ready' state (not yet submitted to scheduler).")
        print(f"  {n_queued} task(s) in 'queued' state (submitted to scheduler, awaiting execution).")
        for state in ("running", "error", "done"):
            n = len(task_engine.read_tasks(
                queue_id=queue_record["id"], metatype="Task", state=state
            ))
            print(f"  {n} task(s) in '{state}' state.")

        pass1_processed: set[str] = set()

        # ----------------------------------------------------------
        # Pass 1: check status of already-submitted jobs
        # ----------------------------------------------------------
        for task in ready_tasks:
            meta   = task["metadata"]
            job_id = meta.get("scheduler_job_id", {}).get(machine_name, -1)
            if job_id == -1:
                continue  # not yet submitted — Pass 2 handles this

            result_file_path = f"result_{task['id']}.txt"
            status = get_job_status(job_id, scheduler_type, result_file=result_file_path)

            if status == "RUNNING" and not meta.get("running", {}).get(machine_name, False):
                rc = _call_hero_initialize(task["id"], machine_name, i_fidelity, task_engine)
                if rc == 0:
                    meta["running"][machine_name] = True
                    task_engine.update_task(
                        task_id=task["id"], state="running",
                        name=task["name"], metadata=meta,
                    )
                    print(f"Task {task['id']}: claimed, state = running")
                elif rc == 2:
                    print(
                        f"Task {task['id']}: already claimed by another machine. "
                        f"Canceling job {job_id}."
                    )
                    cancel_job(job_id, scheduler_type)
                    meta["scheduler_job_id"][machine_name] = -1
                    task_engine.update_task(
                        task_id=task["id"], state="running",
                        name=task["name"], metadata=meta,
                    )
                    pass1_processed.add(task["id"])
                else:
                    print(f"hero_initialize failed (rc={rc}) for task {task['id']}. Canceling job.")
                    cancel_job(job_id, scheduler_type)
                    meta["scheduler_job_id"][machine_name] = -1
                    task_engine.update_task(
                        task_id=task["id"], state="error",
                        name=task["name"], metadata=meta,
                    )
                    pass1_processed.add(task["id"])

            elif status == "COMPLETED":
                result_value = self.read_result(task["id"])
                if not meta.get("running", {}).get(machine_name, False):
                    rc = _call_hero_initialize(task["id"], machine_name, i_fidelity, task_engine)
                    if rc == 2:
                        print(f"Task {task['id']}: already claimed by another machine (job completed).")
                        meta["scheduler_job_id"][machine_name] = -1
                        task_engine.update_task(
                            task_id=task["id"], state="running",
                            name=task["name"], metadata=meta,
                        )
                        pass1_processed.add(task["id"])
                        continue
                    if rc != 0:
                        print(
                            f"hero_initialize failed (rc={rc}) for completed job "
                            f"{job_id}. Marking error."
                        )
                        meta["scheduler_job_id"][machine_name] = -1
                        task_engine.update_task(
                            task_id=task["id"], state="error",
                            name=task["name"], metadata=meta,
                        )
                        pass1_processed.add(task["id"])
                        continue
                print(
                    f"Job {job_id} completed for task {task['id']}, "
                    f"result={result_value}. Calling hero_finalize."
                )
                _call_hero_finalize(result_value, task["id"], machine_name, i_fidelity, task_engine)
                pass1_processed.add(task["id"])

            elif status == "UNKNOWN":
                print(
                    f"Job {job_id} not found in scheduler for task {task['id']} "
                    "— stale, resetting to unsubmitted."
                )
                meta["scheduler_job_id"][machine_name] = -1
                meta["running"][machine_name] = False
                task_engine.update_task(
                    task_id=task["id"], state="ready",
                    name=task["name"], metadata=meta,
                )

            elif status == "FAILED":
                print(f"Job {job_id} failed for task {task['id']}.")
                meta["scheduler_job_id"][machine_name] = -1
                meta["running"][machine_name] = False
                task_engine.update_task(
                    task_id=task["id"], state="error",
                    name=task["name"], metadata=meta,
                )
                pass1_processed.add(task["id"])

        # ----------------------------------------------------------
        # Pass 2: submit new jobs for tasks without a job ID
        # ----------------------------------------------------------
        for task in ready_tasks:
            if task["id"] in pass1_processed:
                continue
            meta   = task["metadata"]
            job_id = meta.get("scheduler_job_id", {}).get(machine_name, -1)
            if job_id != -1:
                continue  # already submitted

            # Ensure bookkeeping fields exist in task metadata
            needs_update = False
            if "scheduler_job_id" not in meta:
                meta["scheduler_job_id"] = {machine_name: -1}
                needs_update = True
            elif machine_name not in meta["scheduler_job_id"]:
                meta["scheduler_job_id"][machine_name] = -1
                needs_update = True
            if "running" not in meta:
                meta["running"] = {machine_name: False}
                needs_update = True
            elif machine_name not in meta["running"]:
                meta["running"][machine_name] = False
                needs_update = True
            if needs_update:
                task_engine.update_task(
                    task_id=task["id"], state="ready",
                    name=task["name"], metadata=meta,
                )

            try:
                new_job_id = self.submit_job(task, machine_name, i_fidelity)
            except JobLimitError as e:
                print(f"  {e}")
                break  # stop Pass 2 for this cycle; retry next time
            except TaskError as e:
                print(f"Task {task['id']}: submission error — marking as error.\n  {e}")
                meta["scheduler_job_id"][machine_name] = -1
                meta["running"][machine_name] = False
                task_engine.update_task(
                    task_id=task["id"], state="error",
                    name=task["name"], metadata=meta,
                )
                continue

            meta["scheduler_job_id"][machine_name] = new_job_id
            try:
                task_engine.update_task(
                    task_id=task["id"], state="ready",
                    name=task["name"], metadata=meta,
                )
            except Exception as e:
                # Hero update failed after a successful submission — cancel the
                # job immediately to avoid an orphaned job.
                print(
                    f"WARNING: Hero update failed after submitting job {new_job_id}: {e}\n"
                    "  Canceling job to avoid orphan."
                )
                cancel_job(new_job_id, scheduler_type)
                continue

            print(f"Task {task['id']}: job {new_job_id} queued on {machine_name}")

        # ----------------------------------------------------------
        # Running tasks: cancel duplicates; finalize completed jobs
        # ----------------------------------------------------------
        running_tasks = task_engine.read_tasks(
            queue_id=queue_record["id"], metatype="Task", state="running"
        )
        for task in running_tasks:
            meta   = task["metadata"]
            job_id = meta.get("scheduler_job_id", {}).get(machine_name, -1)
            meta.setdefault("scheduler_job_id", {}).setdefault(machine_name, -1)
            meta.setdefault("running", {}).setdefault(machine_name, False)

            if not meta["running"][machine_name]:
                # Task claimed by another machine — cancel our pending job if any
                if job_id != -1:
                    print(
                        f"Canceling job {job_id} for task {task['id']} "
                        "(task claimed by another machine)."
                    )
                    cancel_job(job_id, scheduler_type)
                    meta["scheduler_job_id"][machine_name] = -1
                    task_engine.update_task(
                        task_id=task["id"], state="running",
                        name=task["name"], metadata=meta,
                    )
                continue

            # Running on this machine — check for completion
            result_file_path = f"result_{task['id']}.txt"
            status = get_job_status(job_id, scheduler_type, result_file=result_file_path)

            if status == "COMPLETED":
                result_value = self.read_result(task["id"])
                print(
                    f"Job {job_id} completed for task {task['id']}, "
                    f"result={result_value}. Calling hero_finalize."
                )
                if not _call_hero_finalize(result_value, task["id"], machine_name, i_fidelity, task_engine):
                    print(f"WARNING: hero_finalize failed for task {task['id']}")
                meta["scheduler_job_id"][machine_name] = -1
                meta["running"][machine_name] = False

            elif status == "FAILED":
                print(f"Job {job_id} failed for running task {task['id']}.")
                meta["scheduler_job_id"][machine_name] = -1
                meta["running"][machine_name] = False
                task_engine.update_task(
                    task_id=task["id"], state="error",
                    name=task["name"], metadata=meta,
                )

            elif status == "UNKNOWN":
                print(
                    f"Job {job_id} not found in scheduler for running task {task['id']} "
                    "— marking as error."
                )
                meta["scheduler_job_id"][machine_name] = -1
                meta["running"][machine_name] = False
                task_engine.update_task(
                    task_id=task["id"], state="error",
                    name=task["name"], metadata=meta,
                )
