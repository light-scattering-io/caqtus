import concurrent.futures
import contextlib
import threading
from typing import Callable, TypeVar, ParamSpec, SupportsFloat, Self, Optional, Literal

from ._task_group import TaskGroup

P = ParamSpec("P")
T = TypeVar("T")


class BackgroundScheduler:
    """Runs tasks periodically in background threads."""

    def __init__(
        self,
        on_error: Literal["stop_all", "stop_current", "ignore"] = "stop_all",
        max_workers: Optional[int] = None,
    ):
        """Initialize a new instance.

        Args:
            on_error: Indicates how the runner should behave when an error occurs in one the tasks.
            max_workers: The maximum number of threads to use to run the tasks. Refer to the documentation of
            concurrent.futures.ThreadPoolExecutor for more information.
        """

        self._exit_stack = contextlib.ExitStack()
        self._on_error = on_error

        self._thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        )
        self._task_group = TaskGroup(self._thread_pool)
        self._must_stop = threading.Event()

    def __enter__(self) -> Self:
        self._exit_stack.__enter__()
        self._exit_stack.enter_context(self._thread_pool)
        self._exit_stack.enter_context(self._task_group)
        self._exit_stack.callback(self.shutdown)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stops repeating tasks.

        When the bottom of the with statement is reached, all periodically running tasks will be stopped. If errors
        occurred in any of the tasks, they will be raised here, wrapped in an ExceptionGroup. When reaching the end of
        the with statement, this will stop repeating tasks, but it will still need to wait for the previous ones to stop
        before exiting.
        """

        return self._exit_stack.__exit__(exc_type, exc_val, exc_tb)

    def schedule_task(
        self,
        func: Callable[P, T],
        interval: SupportsFloat,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        """Schedule a task to be run periodically.

        Args:
            func: The function to execute.
            interval: The interval between executions, in seconds. This is the duration between the end of the previous
                execution and the start of the next one. It does not include the time the task takes to execute.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
        """

        if self.was_stopped():
            raise RuntimeError("Cannot schedule task after shutdown was requested")

        self._task_group.create_task(
            self._call_repetitively, func, float(interval), *args, **kwargs
        )

    def _call_repetitively(
        self, func: Callable[P, T], interval: float, *args: P.args, **kwargs: P.kwargs
    ) -> None:
        while not self._must_stop.is_set():
            try:
                func(*args, **kwargs)
            except Exception:
                if self._on_error == "stop_all":
                    self.shutdown()
                    raise
                elif self._on_error == "stop_current":
                    raise
                elif self._on_error == "ignore":
                    pass
                else:
                    raise AssertionError(
                        f"on_error has an unexpected value {self._on_error}"
                    )
            self._must_stop.wait(interval)

    def is_running(self) -> bool:
        """Indicates if there is any task currently running."""

        return bool(self._task_group.get_running_tasks())

    def was_stopped(self) -> bool:
        return self._must_stop.is_set()

    def shutdown(self) -> None:
        """Indicates that the scheduler should stop as soon as possible.

        This method is not blocking and does not wait for the scheduler to stop.
        """

        self._must_stop.set()