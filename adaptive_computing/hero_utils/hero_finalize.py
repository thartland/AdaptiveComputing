import sys

from adaptive_computing.hero_utils.set_hero_env_vars import set_hero_env_vars


def hero_finalize(cond, task_id, machine_name, i_fidelity=0, task_engine=None):
    """Publish a simulation result to Hero and mark the task done (or error).

    Args:
        cond:         Result value as a float.  Pass ``-1`` to mark the task as
                      error instead of done.
        task_id:      Hero task ID string.
        machine_name: Logical machine name recorded in task metadata.
        i_fidelity:   Fidelity level index (used only when task_engine is None).
        task_engine:  An already-authenticated Hero TaskEngine.  When provided the
                      function reuses it directly, avoiding a redundant authentication
                      round-trip.  When None a fresh HeroClient is created (standalone
                      script usage).

    Raises:
        RuntimeError: Authentication failed or the Hero API call failed.
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

    cond = float(cond)
    current_task = task_engine.read_task(task_id)

    if cond == -1:
        print(f"Task {task_id}: state = error, metadata = {current_task['metadata']}")
        current_task['metadata']['y_data'] = cond
        current_task['metadata']['running'][machine_name] = False
        task_engine.update_task(
            task_id=task_id, state='error',
            name=current_task['name'], metadata=current_task['metadata'],
        )
    else:
        print(f"Task {task_id}: state = done, metadata = {current_task['metadata']}")
        current_task['metadata']['y_data'] = [cond]
        current_task['metadata']['running'][machine_name] = False
        task_engine.update_task(
            task_id=task_id, state='done',
            name=current_task['name'], metadata=current_task['metadata'],
        )


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        print("Usage: python hero_finalize.py <cond> <task_id> <machine_name> [i_fidelity]")
        sys.exit(1)

    try:
        cond = float(sys.argv[1])
    except ValueError:
        print("Error: <cond> must be a valid number.")
        sys.exit(1)

    task_id      = sys.argv[2]
    machine_name = sys.argv[3]
    i_fidelity   = int(sys.argv[4]) if len(sys.argv) == 5 else 0

    try:
        hero_finalize(cond, task_id, machine_name, i_fidelity)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
