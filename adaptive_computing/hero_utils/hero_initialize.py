import sys

from adaptive_computing.hero_utils.set_hero_env_vars import set_hero_env_vars


class TaskAlreadyClaimed(Exception):
    """Raised by hero_initialize when the task is already running on another machine."""


def hero_initialize(task_id, machine_name, i_fidelity=0, task_engine=None):
    """Mark a Hero task as running on machine_name.

    Args:
        task_id:      Hero task ID string.
        machine_name: Logical machine name to record in task metadata.
        i_fidelity:   Fidelity level index (used only when task_engine is None).
        task_engine:  An already-authenticated Hero TaskEngine.  When provided the
                      function reuses it directly, avoiding a redundant authentication
                      round-trip.  When None a fresh HeroClient is created (standalone
                      script usage).

    Raises:
        TaskAlreadyClaimed: The task is already in the ``running`` state — it was
                            claimed by another machine before this call.
        RuntimeError:       Authentication failed or the Hero API call failed.
    """
    if task_engine is None:
        from hero import HeroClient, get_env_variable
        set_hero_env_vars()
        try:
            hero_env     = get_env_variable('HERO_ENV', 'dev')
            hero_project = get_env_variable('HERO_PROJECT')
        except EnvironmentError as e:
            raise RuntimeError(f"Hero environment variables not set: {e}") from e
        application_id = f'{hero_env}-{hero_project}'
        hero = HeroClient()
        try:
            hero.authenticate()
        except Exception as e:
            raise RuntimeError(f"Hero authentication failed: {e}") from e
        task_engine = hero.TaskEngine(application_id)

    current_task = task_engine.read_task(task_id)
    if current_task['state'] == 'running':
        raise TaskAlreadyClaimed(
            f"Task {task_id} is already running on another machine."
        )

    current_task['metadata']['running'][machine_name] = True
    task_engine.update_task(
        task_id=task_id, state='running',
        name=current_task['name'], metadata=current_task['metadata'],
    )
    print(f"Task {task_id}: state = running, metadata = {current_task['metadata']}")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Usage: python hero_initialize.py <task_id> <machine_name> [i_fidelity]")
        sys.exit(1)

    task_id      = sys.argv[1]
    machine_name = sys.argv[2]
    i_fidelity   = int(sys.argv[3]) if len(sys.argv) == 4 else 0

    try:
        hero_initialize(task_id, machine_name, i_fidelity)
    except TaskAlreadyClaimed as e:
        print(e)
        sys.exit(2)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
