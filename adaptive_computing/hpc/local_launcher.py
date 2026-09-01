"""
local_launcher.py — Launch a LocalHPCManager daemon in a local tmux session.

Onsite variant of remote_manager.py — no SSH required.  Use this when the
controller and the manager both run on the same HPC login node.

The controller calls :func:`ensure_manager_running` once before queuing tasks;
the daemon picks up work from the shared LocalHeroClient JSON file and keeps
running until stopped, so the controller can die and restart (via LangGraph
checkpointing or otherwise) without losing track of submitted SLURM jobs.

Usage::

    from adaptive_computing.hpc.local_launcher import ensure_manager_running

    ensure_manager_running(
        work_dir="/abs/path/to/workdir",
        manager_script="/abs/path/to/manager.py",
        machine_name="kestrel",
    )
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

DEFAULT_SESSION_NAME = "ac-manager"


# ---------------------------------------------------------------------------
# Internal helpers (same pattern as remote_manager.py)
# ---------------------------------------------------------------------------

def _make_env() -> dict:
    """Return an environment dict with TERM set (required by tmux)."""
    env = dict(os.environ)
    env.setdefault("TERM", "xterm-256color")
    return env


def _ensure_tmux(env: dict) -> None:
    """Load tmux via the HPC module system if it is not already in PATH."""
    if shutil.which("tmux"):
        return
    result = subprocess.run(
        "bash -l -c 'module load tmux 2>/dev/null && echo \"$PATH\"'",
        shell=True, capture_output=True, text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        env["PATH"] = result.stdout.strip()


def _tmux(*args: str, env: dict) -> subprocess.CompletedProcess:
    """Run a tmux subcommand."""
    return subprocess.run(["tmux"] + list(args), env=env, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_manager_running(session_name: str = DEFAULT_SESSION_NAME) -> bool:
    """Return ``True`` if the tmux session exists and is running python.

    Args:
        session_name: Name of the tmux session to check.
    """
    env = _make_env()
    _ensure_tmux(env)
    result = _tmux(
        "list-panes", "-t", session_name, "-F", "#{pane_current_command}",
        env=env,
    )
    return result.returncode == 0 and "python" in result.stdout


def launch_manager_in_tmux(
    work_dir: str,
    manager_script: str,
    machine_name: str,
    session_name: str = DEFAULT_SESSION_NAME,
    fidelity: int = 0,
    python_executable: str | None = None,
) -> None:
    """Start *manager_script* in a detached local tmux session.

    Uses ``setsid`` so the tmux server survives systemd ``KillUserProcesses=yes``
    when the parent shell exits.  Log output goes to ``{work_dir}/manager.log``.

    Args:
        work_dir:          Absolute path to the working directory (chdir'd into
                           before running the manager).
        manager_script:    Absolute path to the manager Python script.
        machine_name:      Logical machine name passed as ``argv[1]`` to the script.
        session_name:      Name for the tmux session (default ``"ac-manager"``).
        fidelity:          Fidelity level index passed as ``argv[2]`` (default 0).
        python_executable: Python binary to use.  Defaults to ``sys.executable``
                           (same Python as the calling process).
    """
    env = _make_env()
    _ensure_tmux(env)

    python = python_executable or sys.executable
    log_file = os.path.join(work_dir, "manager.log")

    # Kill any stale session to start clean.
    _tmux("kill-session", "-t", session_name, env=env)

    # setsid prevents systemd KillUserProcesses=yes from killing the tmux
    # server when the short-lived parent process that spawned it exits.
    result = subprocess.run(
        f"setsid tmux new-session -d -s {session_name}",
        shell=True, env=env, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to create tmux session '{session_name}': {result.stderr.strip()}"
        )

    inner_cmd = (
        f"cd {work_dir!r} && "
        f"{python!r} -u {manager_script!r} {work_dir!r} {machine_name} {fidelity} "
        f"> {log_file!r} 2>&1"
    )
    _tmux("send-keys", "-t", session_name, inner_cmd, "Enter", env=env)
    print(f"[local_launcher] Started tmux session '{session_name}' → {manager_script}")


def ensure_manager_running(
    work_dir: str,
    manager_script: str,
    machine_name: str,
    session_name: str = DEFAULT_SESSION_NAME,
    python_executable: str | None = None,
) -> bool:
    """Launch the manager in tmux if it is not already running.

    Safe to call repeatedly — does nothing if the session is alive.

    Args:
        work_dir:          Absolute path to the working directory.
        manager_script:    Absolute path to the manager Python script.
        machine_name:      Logical machine name.
        session_name:      Name for the tmux session.
        python_executable: Python binary override (default: ``sys.executable``).

    Returns:
        ``True`` if a new session was launched, ``False`` if already running.
    """
    if is_manager_running(session_name):
        print(f"[local_launcher] Manager already running in session '{session_name}'")
        return False
    launch_manager_in_tmux(
        work_dir=work_dir,
        manager_script=manager_script,
        machine_name=machine_name,
        session_name=session_name,
        python_executable=python_executable,
    )
    return True


def stop_manager(session_name: str = DEFAULT_SESSION_NAME) -> None:
    """Gracefully stop the manager tmux session.

    Sends Ctrl-C to interrupt the running loop, waits one second, then kills
    the session.

    Args:
        session_name: Name of the tmux session to stop.
    """
    env = _make_env()
    _ensure_tmux(env)
    _tmux("send-keys", "-t", session_name, "C-c", env=env)
    time.sleep(1)
    _tmux("kill-session", "-t", session_name, env=env)
    print(f"[local_launcher] Stopped tmux session '{session_name}'")
