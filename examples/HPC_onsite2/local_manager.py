from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np


class LocalFunctionManager:
    """Evaluate queued LocalHero tasks directly with Python functions."""

    def __init__(
        self,
        hero_client,
        simulation_functions,
        machine_name="local",
        max_workers=2,
    ):
        self.hero_client = hero_client
        self.simulation_functions = simulation_functions
        self.machine_name = machine_name
        self.max_workers = max_workers

    def _set_running(self, task_engine, task):
        metadata = task["metadata"]
        metadata["running"][self.machine_name] = True

        task_engine.update_task(
            task_id=task["id"],
            state="running",
            name=task["name"],
            metadata=metadata,
        )

    def _set_result(self, task_engine, task_id, result=None):
        task = task_engine.read_task(task_id)
        metadata = task["metadata"]
        metadata["running"][self.machine_name] = False

        if result is None:
            metadata["y_data"] = -1
            state = "error"
        else:
            metadata["y_data"] = [float(result)]
            state = "done"

        task_engine.update_task(
            task_id=task_id,
            state=state,
            name=task["name"],
            metadata=metadata,
        )

    @staticmethod
    def _evaluate(function, x_data):
        x_data = np.asarray(x_data, dtype=float)
        result = np.asarray(function(x_data)).reshape(-1)

        if result.size != 1:
            raise ValueError(
                "Each simulation function must return one scalar output."
            )

        return float(result[0])

    def run_until_done(
        self,
        i_fidelity=None,
        *,
        i_fidelities=None,
    ):
        if i_fidelity is not None and i_fidelities is not None:
            raise ValueError(
                "Specify either i_fidelity or i_fidelities, not both."
            )

        if i_fidelities is None:
            fidelities = [
                0 if i_fidelity is None else int(i_fidelity)
            ]
        else:
            fidelities = sorted(set(map(int, i_fidelities)))

        task_engine = self.hero_client.TaskEngine(
            self.hero_client.application_id
        )

        work = []

        for level in fidelities:
            queue_name = (
                self.hero_client.queue_name
                if level == 0
                else f"{self.hero_client.queue_name}{level}"
            )

            queue = task_engine.read_queue_by_name(
                name=queue_name,
                state="active",
            )

            tasks = task_engine.read_tasks(
                queue_id=queue["id"],
                metatype="Task",
                state="ready",
            )

            for task in tasks:
                self._set_running(task_engine, task)

                work.append(
                    (
                        task,
                        self.simulation_functions[level],
                        task["metadata"]["x_data"],
                    )
                )

        if not work:
            return

        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(work))
        ) as executor:
            futures = {
                executor.submit(
                    self._evaluate,
                    function,
                    x_data,
                ): task
                for task, function, x_data in work
            }

            for future in as_completed(futures):
                task = futures[future]

                try:
                    result = future.result()
                except Exception as exc:
                    print(
                        f"Local simulation failed for task "
                        f"{task['id']}: {exc}"
                    )
                    self._set_result(
                        task_engine,
                        task["id"],
                        result=None,
                    )
                else:
                    self._set_result(
                        task_engine,
                        task["id"],
                        result=result,
                    )
