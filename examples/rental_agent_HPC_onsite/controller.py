"""
controller.py — LangGraph controller for the rental_agent_HPC_onsite example.

Demonstrates the checkpoint-safe pattern:
  1. submit_jobs_node — adds tasks to LocalHeroClient queue, stores task IDs
     in LangGraph state, ensures tmux manager is running (or runs inline in
     mock mode)
  2. wait_for_results_node — polls the queue until all tasks are done/error,
     collects results

If the controller process is killed mid-poll, the SQLite checkpointer saves
the task IDs.  On restart with the same thread_id the graph resumes at
wait_for_results_node and finds the tasks already done (the tmux manager kept
running and processed them).

Usage:
    python controller.py            # auto-detects mock vs real mode
    python controller.py --mock     # force mock mode (no sbatch)
    python controller.py --real     # force real SLURM mode
"""

from __future__ import annotations

import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, List, Optional, TypedDict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from adaptive_computing.local_hero import LocalHeroClient
from adaptive_computing.hpc.local_launcher import ensure_manager_running
from manager import MANAGER_SCRIPT, create_manager

SCRIPT_DIR = Path(__file__).parent.resolve()
QUEUE_NAME = "jobs"
APP_ID = "rental_agent_HPC_onsite"
SESSION_NAME = "ac-manager-example"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ExampleState(TypedDict, total=False):
    x_values: List[float]
    task_ids: Optional[List[str]]
    results: Optional[List[float]]
    hero_db_path: str
    machine_name: str
    mock_mode: bool
    work_dir: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_or_create_queue(engine, queue_name: str) -> dict:
    try:
        return engine.read_queue_by_name(queue_name, state="active")
    except (ValueError, KeyError):
        return engine.add_queue(queue_name)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def submit_jobs_node(state: ExampleState) -> ExampleState:
    """Submit one task per x_value to the Hero queue."""
    hero_client = LocalHeroClient(db_path=state["hero_db_path"])
    engine = hero_client.TaskEngine(application_id=APP_ID)
    queue = _get_or_create_queue(engine, QUEUE_NAME)

    # Resume case: tasks already submitted — skip re-submission.
    if state.get("task_ids"):
        print(f"[submit_jobs] Resuming — {len(state['task_ids'])} tasks already queued")
        return {}

    task_ids = []
    for x in state["x_values"]:
        task = engine.add_task(
            queue_id=queue["id"],
            name=f"x={x}",
            metatype="Task",
            metadata={"x_value": x},
        )
        task_ids.append(task["id"])
    print(f"[submit_jobs] Queued {len(task_ids)} tasks")

    mock_mode = state.get("mock_mode", False)
    work_dir = state["work_dir"]

    if mock_mode:
        # Run manager inline for testing — no tmux, no sbatch.
        # Must use the same queue_name as the controller so the manager finds the tasks.
        mgr = create_manager(
            machine_name=state["machine_name"],
            work_dir=work_dir,
            hero_client=LocalHeroClient(
                db_path=state["hero_db_path"],
                queue_name=QUEUE_NAME,
                application_id=APP_ID,
            ),
            mock_mode=True,
        )
        mgr.run_until_done()
    else:
        ensure_manager_running(
            work_dir=work_dir,
            manager_script=str(MANAGER_SCRIPT),
            machine_name=state["machine_name"],
            session_name=SESSION_NAME,
        )

    return {"task_ids": task_ids}


def wait_for_results_node(state: ExampleState) -> ExampleState:
    """Poll the Hero queue until all tasks reach a terminal state."""
    hero_client = LocalHeroClient(db_path=state["hero_db_path"])
    engine = hero_client.TaskEngine(application_id=APP_ID)
    queue = _get_or_create_queue(engine, QUEUE_NAME)

    task_ids = set(state["task_ids"])
    mock_mode = state.get("mock_mode", False)
    poll_interval = 1 if mock_mode else 30

    while True:
        done = engine.read_tasks(queue["id"], metatype="Task", state="done")
        error = engine.read_tasks(queue["id"], metatype="Task", state="error")
        terminal_ids = {t["id"] for t in done + error}

        if task_ids <= terminal_ids:
            break

        remaining = len(task_ids - terminal_ids)
        print(f"[wait_for_results] {remaining} task(s) still pending...")
        time.sleep(poll_interval)

    # Collect results in submission order.
    done_map = {t["id"]: t["metadata"].get("y_data", [-1])[0] for t in done}
    error_ids = {t["id"] for t in error}
    if error_ids:
        print(f"[wait_for_results] WARNING: {len(error_ids)} task(s) failed: {error_ids}")

    results = [done_map.get(tid, -1.0) for tid in state["task_ids"]]
    print(f"[wait_for_results] Results: {list(zip(state['x_values'], results))}")
    return {"results": results}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    g = StateGraph(ExampleState)
    g.add_node("submit_jobs", submit_jobs_node)
    g.add_node("wait_for_results", wait_for_results_node)
    g.add_edge(START, "submit_jobs")
    g.add_edge("submit_jobs", "wait_for_results")
    g.add_edge("wait_for_results", END)
    return g


def controller(mock_mode: bool | None = None) -> list:
    """Run the example workflow and return the result list [y0, y1, ...].

    Args:
        mock_mode: Skip sbatch/tmux and compute directly.  When ``None``
                   (default), auto-detects: mock if ``sbatch`` is not on PATH.
    """
    if mock_mode is None:
        mock_mode = not bool(shutil.which("sbatch"))

    work_dir = str(SCRIPT_DIR)
    run_id = uuid.uuid4().hex[:8]
    hero_db_path = str(SCRIPT_DIR / f"hero_db_{run_id}.json")
    checkpoint_db = str(SCRIPT_DIR / f"checkpoint_{run_id}.db")

    x_values = [1.0, 2.0, 3.0, 4.0]

    initial_state: ExampleState = {
        "x_values": x_values,
        "hero_db_path": hero_db_path,
        "machine_name": "kestrel",
        "mock_mode": mock_mode,
        "work_dir": work_dir,
    }

    g = build_graph()
    thread_id = f"example-{run_id}"

    with SqliteSaver.from_conn_string(checkpoint_db) as checkpointer:
        graph = g.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        final_state = graph.invoke(initial_state, config=config)

    results = final_state.get("results", [])
    print(f"[controller] Final results: {list(zip(x_values, results))}")

    # Cleanup temp files.
    for path in [hero_db_path, checkpoint_db]:
        if os.path.exists(path):
            os.remove(path)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mock", action="store_true", help="Force mock mode")
    group.add_argument("--real", action="store_true", help="Force real SLURM mode")
    args = parser.parse_args()

    mock = True if args.mock else (False if args.real else None)
    results = controller(mock_mode=mock)
    print(f"\nFinal: x={[1,2,3,4]}  y={results}")
