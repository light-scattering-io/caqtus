from __future__ import annotations

import contextlib
import datetime
import logging
import weakref
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Mapping, Any, Protocol, Self

import anyio
import anyio.to_process
import attrs
from anyio.streams.memory import MemoryObjectSendStream, MemoryObjectReceiveStream

from caqtus.device import DeviceName
from caqtus.shot_compilation import VariableNamespace
from caqtus.types.data import DataLabel, Data
from caqtus.utils.logging import log_async_cm

logger = logging.getLogger(__name__)


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


@log_async_cm(logger)
class ShotManager:
    def __init__(
        self,
        shot_runner: ShotRunner,
        shot_compiler: ShotCompiler,
        shot_retry_config: ShotRetryConfig,
    ):
        self._shot_runner = shot_runner
        self._shot_compiler = shot_compiler
        self._shot_retry_config = shot_retry_config
        # These streams must be closed in the order defined here.
        (
            self._shot_parameters_input_stream,
            self._shot_parameters_output_stream,
        ) = anyio.create_memory_object_stream[ShotParameters]()
        (
            self._shot_execution_input_stream,
            self._shot_execution_output_stream,
        ) = anyio.create_memory_object_stream[DeviceParameters]()
        (
            self._shot_data_input_stream,
            self._shot_data_output_stream,
        ) = anyio.create_memory_object_stream(1)

        self._task_group = anyio.create_task_group()

    async def __aenter__(self) -> Self:
        await self._task_group.__aenter__()
        self._task_group.start_soon(self._compile_shots_in_order)
        self._task_group.start_soon(self._run_shots, self._shot_runner)
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return await self._task_group.__aexit__(exc_type, exc_value, traceback)

    @log_async_cm(logger)
    @contextlib.asynccontextmanager
    async def data_output_stream(self) -> AsyncGenerator[AsyncIterable[ShotData], None]:
        """Returns an iterable of the data produced by the shots."""

        # We suppress BrokenResourceError because the error that caused it will anyway
        # be raised, and we don't want to clutter the exception traceback.
        with contextlib.suppress(anyio.BrokenResourceError):
            async with self._shot_data_output_stream:
                yield self._shot_data_output_stream

    @log_async_cm(logger)
    @contextlib.asynccontextmanager
    async def scheduler(self):
        """Returns an object that allows to schedule shots.

        Warnings:
            This method should be used as a context manager.
            It does NOT support being called several times.
        """

        async with self._shot_parameters_input_stream:
            yield ShotScheduler(self._shot_parameters_input_stream)

    async def _compile_shots_in_order(self):
        async with (
            self._shot_execution_input_stream,
            anyio.create_task_group() as tg,
            self._shot_parameters_output_stream,
        ):
            shot_execution_queue = ShotExecutionQueue(self._shot_execution_input_stream)
            for _ in range(4):
                tg.start_soon(
                    self._compile_shots,
                    self._shot_compiler,
                    self._shot_parameters_output_stream.clone(),
                    shot_execution_queue,
                )

    @staticmethod
    async def _compile_shots(
        shot_compiler: ShotCompiler,
        shot_params_output_stream: MemoryObjectReceiveStream[ShotParameters],
        shot_execution_queue: ShotExecutionQueue,
    ):
        async with shot_params_output_stream:
            async for shot_params in shot_params_output_stream:
                result = await anyio.to_process.run_sync(
                    _compile_shot, shot_params, shot_compiler
                )
                await shot_execution_queue.push(result)

    async def _run_shots(self, shot_runner: ShotRunner):
        async with (
            self._shot_data_input_stream,
            self._shot_execution_output_stream as receiver,
        ):
            async for device_parameters in receiver:
                shot_data = await self._run_shot_with_retry(
                    device_parameters, shot_runner
                )
                await self._shot_data_input_stream.send(shot_data)

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
