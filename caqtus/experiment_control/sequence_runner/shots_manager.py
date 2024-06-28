from __future__ import annotations

import contextlib
import datetime
import functools
import logging
import weakref
from collections.abc import AsyncIterable
from typing import Mapping, Any, Protocol

import anyio
import anyio.to_process
import attrs
from anyio import TASK_STATUS_IGNORED
from anyio.abc import TaskStatus
from anyio.streams.memory import MemoryObjectSendStream, MemoryObjectReceiveStream

from caqtus.device import DeviceName
from caqtus.shot_compilation import VariableNamespace
from caqtus.types.data import DataLabel, Data
from caqtus.utils.logging import log_async_cm_decorator, log_async_cm

logger = logging.getLogger(__name__)

_log_async_cm = functools.partial(log_async_cm, logger=logger)


class ShotRunner(Protocol):
    async def run_shot(
        self, device_parameters: Mapping[DeviceName, Mapping[str, Any]], timeout: float
    ) -> Mapping[DataLabel, Data]: ...


class ShotCompiler(Protocol):
    def compile_shot(
        self, shot_parameters: VariableNamespace
    ) -> tuple[Mapping[DeviceName, Mapping[str, Any]], float]: ...


class ShotExecutionQueue:
    """Wraps a memory object send stream to ensure that shots are executed in order."""

    def __init__(self, shot_execution_stream: MemoryObjectSendStream[DeviceParameters]):
        self._shot_execution_stream = shot_execution_stream
        self._next_shot = 0
        self._can_push_events = weakref.WeakValueDictionary[int, anyio.Event]()

    async def push(self, shot_parameters: DeviceParameters) -> None:
        """Pushes a shot to the execution queue.

        Push the shot parameters to the execution queue when the shot index matches the
        next shot to run.
        """

        shot_index = shot_parameters.index
        if shot_index != self._next_shot:
            assert shot_index > self._next_shot
            try:
                event = self._can_push_events[shot_index]
            except KeyError:
                event = anyio.Event()
                self._can_push_events[shot_index] = event
            await event.wait()

        assert shot_index == self._next_shot

        await self._shot_execution_stream.send(shot_parameters)
        self._next_shot += 1
        try:
            self._can_push_events[self._next_shot].set()
        except KeyError:
            pass


class ShotManager:
    """Manages the execution of shots.

    This object acts as an execution queue for shots on the experiment.

    When entered as a context manager, it returns two objects:
    - A stream of shot data.
    - A context manager that allows to schedule shots.

    The context manager must be entered and closed before the ShotManager is closed.
    This is necessary to know when all shots have been scheduled.

    Examples:
        async with (ShotManager(...) as (data_stream, scheduler_ctx), scheduler_ctx as scheduler:
            # Schedule shots and collect data.
    """

    def __init__(
        self,
        shot_runner: ShotRunner,
        shot_compiler: ShotCompiler,
        shot_retry_config: ShotRetryConfig,
    ):
        self._shot_runner = shot_runner
        self._shot_compiler = shot_compiler
        self._shot_retry_config = shot_retry_config

        self._exit_stack = contextlib.AsyncExitStack()

    async def __aenter__(
        self,
    ) -> tuple[
        AsyncIterable[ShotData], contextlib.AbstractAsyncContextManager[ShotScheduler]
    ]:
        await self._exit_stack.__aenter__()
        (
            shot_data_send_stream,
            shot_data_receive_stream,
        ) = anyio.create_memory_object_stream[ShotData](1)
        await self._exit_stack.enter_async_context(
            _log_async_cm(shot_data_receive_stream, name="shot_data_receive_stream")
        )
        task_group = await self._exit_stack.enter_async_context(
            create_task_group_message("Errors occurred while processing shots")
        )
        (
            device_parameters_send_stream,
            device_parameters_receive_stream,
        ) = anyio.create_memory_object_stream[DeviceParameters]()
        await task_group.start(
            self.run_shots,
            self._shot_runner,
            device_parameters_receive_stream,
            shot_data_send_stream,
        )
        (
            self._shot_parameters_send_stream,
            shot_parameters_receive_stream,
        ) = anyio.create_memory_object_stream[ShotParameters]()
        await task_group.start(
            self.compile_shots,
            self._shot_compiler,
            shot_parameters_receive_stream,
            device_parameters_send_stream,
        )

        return shot_data_receive_stream, self.scheduler()

    async def __aexit__(self, exc_type, exc_value, traceback):
        return await self._exit_stack.__aexit__(exc_type, exc_value, traceback)

    async def run_shots(
        self,
        shot_runner: ShotRunner,
        device_parameters_output_stream: MemoryObjectReceiveStream[DeviceParameters],
        shot_data_input_stream: MemoryObjectSendStream[ShotData],
        *,
        task_status: TaskStatus[None] = TASK_STATUS_IGNORED,
    ) -> None:
        async with shot_data_input_stream, device_parameters_output_stream:
            task_status.started()
            async for device_parameters in device_parameters_output_stream:
                shot_data = await self._run_shot_with_retry(
                    device_parameters, shot_runner
                )
                logger.debug("Shot %d executed.", device_parameters.index)
                await shot_data_input_stream.send(shot_data)
                logger.debug("Shot %d data sent.", device_parameters.index)

    @log_async_cm_decorator(logger)
    @contextlib.asynccontextmanager
    async def scheduler(self):
        """Returns an object that allows to schedule shots.

        Warnings:
            It does NOT support being called several times.
        """

        async with _log_async_cm(
            self._shot_parameters_send_stream, name="shot_parameters_input_stream"
        ):
            yield ShotScheduler(self._shot_parameters_send_stream)

    async def compile_shots(
        self,
        shot_compiler: ShotCompiler,
        shot_params_receive_stream: MemoryObjectReceiveStream[ShotParameters],
        device_parameters_send_stream: MemoryObjectSendStream[DeviceParameters],
        *,
        task_status: TaskStatus[None] = TASK_STATUS_IGNORED,
    ):
        async with (
            device_parameters_send_stream,
            create_task_group_message("Errors occurred during shot compilation") as tg,
        ):
            shot_execution_queue = ShotExecutionQueue(device_parameters_send_stream)
            async with shot_params_receive_stream:
                for i in range(4):
                    await tg.start(
                        self._compile_shots,
                        shot_compiler,
                        shot_params_receive_stream.clone(),
                        shot_execution_queue,
                    )
            task_status.started()

    @staticmethod
    async def _compile_shots(
        shot_compiler: ShotCompiler,
        shot_params_receive_stream: MemoryObjectReceiveStream[ShotParameters],
        shot_execution_queue: ShotExecutionQueue,
        *,
        task_status: TaskStatus[None] = TASK_STATUS_IGNORED,
    ) -> None:
        async with shot_params_receive_stream:
            task_status.started()
            async for shot_params in shot_params_receive_stream:
                result = await anyio.to_process.run_sync(
                    _compile_shot, shot_params, shot_compiler
                )
                logger.debug("Pushing shot %d to execution queue.", shot_params.index)
                await shot_execution_queue.push(result)

    async def _run_shot_with_retry(
        self, device_parameters: DeviceParameters, shot_runner: ShotRunner
    ) -> ShotData:
        exceptions_to_retry = self._shot_retry_config.exceptions_to_retry
        number_of_attempts = self._shot_retry_config.number_of_attempts
        if number_of_attempts < 1:
            raise ValueError("number_of_attempts must be >= 1")

        errors: list[Exception] = []

        for attempt in range(number_of_attempts):
            try:
                start_time = datetime.datetime.now(tz=datetime.timezone.utc)
                data = await shot_runner.run_shot(
                    device_parameters.device_parameters, device_parameters.timeout
                )
                end_time = datetime.datetime.now(tz=datetime.timezone.utc)
            except* exceptions_to_retry as e:
                errors.extend(e.exceptions)
                # We sleep a bit to allow to recover from the error, for example if it
                # is a timeout.
                await anyio.sleep(0.1)
                logger.warning(
                    f"Attempt {attempt+1}/{number_of_attempts} failed", exc_info=e
                )
            else:
                return ShotData(
                    index=device_parameters.index,
                    start_time=start_time,
                    end_time=end_time,
                    variables=device_parameters.shot_parameters,
                    data=data,
                )
        raise ExceptionGroup(
            f"Could not execute shot after {number_of_attempts} attempts", errors
        )


@attrs.frozen(order=True)
class ShotParameters:
    """Holds information necessary to compile a shot."""

    index: int
    parameters: VariableNamespace = attrs.field(eq=False)


@attrs.frozen(order=True)
class DeviceParameters:
    """Holds information necessary to run a shot."""

    index: int
    shot_parameters: VariableNamespace = attrs.field(eq=False)
    device_parameters: Mapping[DeviceName, Mapping[str, Any]] = attrs.field(eq=False)
    timeout: float = attrs.field()


@attrs.frozen(order=True)
class ShotData:
    """Holds information necessary to store a shot."""

    index: int
    start_time: datetime.datetime = attrs.field(eq=False)
    end_time: datetime.datetime = attrs.field(eq=False)
    variables: VariableNamespace = attrs.field(eq=False)
    data: Mapping[DataLabel, Data] = attrs.field(eq=False)


def _compile_shot(
    shot_parameters: ShotParameters, shot_compiler: ShotCompiler
) -> DeviceParameters:
    compiled, shot_duration = shot_compiler.compile_shot(shot_parameters.parameters)
    return DeviceParameters(
        index=shot_parameters.index,
        shot_parameters=shot_parameters.parameters,
        device_parameters=compiled,
        timeout=shot_duration + 10,
    )


@attrs.define
class ShotRetryConfig:
    """Specifies how to retry a shot if an error occurs.

    Attributes:
        exceptions_to_retry: If an exception occurs while running a shot, it will be
        retried if it is an instance of one of the exceptions in this tuple.
        number_of_attempts: The number of times to retry a shot if an error occurs.
    """

    exceptions_to_retry: tuple[type[Exception], ...] = attrs.field(
        factory=tuple,
        eq=False,
        on_setattr=attrs.setters.validate,
    )
    number_of_attempts: int = attrs.field(default=1, eq=False)


class ShotScheduler:
    def __init__(
        self, shot_parameters_input_stream: MemoryObjectSendStream[ShotParameters]
    ):
        self._shot_parameters_input_stream = shot_parameters_input_stream
        self._current_shot = 0

    async def schedule_shot(self, shot_variables: VariableNamespace) -> None:
        shot_parameters = ShotParameters(
            index=self._current_shot, parameters=shot_variables
        )
        try:
            await self._shot_parameters_input_stream.send(shot_parameters)
        except anyio.BrokenResourceError:
            # We ignore this error because the error that caused it will anyway be
            # raised, and we don't want to clutter the exception traceback.
            pass
        self._current_shot += 1


@contextlib.contextmanager
def renamed_exception_group(message: str):
    try:
        yield
    except ExceptionGroup as e:
        raise ExceptionGroup(message, e.exceptions) from None


@contextlib.asynccontextmanager
async def create_task_group_message(message: str):
    with renamed_exception_group(message):
        async with anyio.create_task_group() as tg:
            yield tg
