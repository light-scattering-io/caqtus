import asyncio
import contextlib
import datetime
import logging
import pprint
from collections.abc import Mapping
from functools import singledispatchmethod
from threading import Thread, Event
from typing import TYPE_CHECKING, Any

import numpy as np

from camera.runtime import CameraTimeoutError
from device.configuration import DeviceName, DeviceParameter
from device.runtime import RuntimeDevice
from duration_timer import DurationTimer
from experiment.configuration import (
    CameraConfiguration,
    NI6738SequencerConfiguration,
    SpincoreSequencerConfiguration,
)
from experiment.session import ExperimentSessionMaker
from experiment_control.compute_device_parameters import (
    compute_shot_parameters,
    get_devices_initialization_parameters,
    compute_parameters_on_variables_update,
)
from remote_device_client import RemoteDeviceClientManager
from sequence.configuration import (
    ShotConfiguration,
    Step,
    SequenceSteps,
    ArangeLoop,
    LinspaceLoop,
    VariableDeclaration,
    ExecuteShot,
    OptimizationLoop,
    UserInputLoop,
    VariableRange,
)
from sequence.runtime import SequencePath, Sequence, Shot, State
from units import Quantity, units, get_unit, magnitude_in_unit, DimensionalityError
from units.analog_value import add_unit, AnalogValue
from variable.name import DottedVariableName
from variable.namespace import VariableNamespace
from .device_context_manager import DeviceContextManager
from .device_servers import (
    create_device_servers,
    connect_to_device_servers,
    create_devices,
)
from .sequence_context import SequenceContext, SequenceTaskGroup
from .sequence_context import StepContext
from .user_input_loop.exec_user_input import ExecUserInput
from .user_input_loop.input_widget import RawVariableRange, EvaluatedVariableRange

if TYPE_CHECKING:
    from ni6738_analog_card.runtime import NI6738AnalogCard
    from camera.runtime import Camera
    from spincore_sequencer.runtime import SpincorePulseBlaster

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

WATCH_FOR_INTERRUPTION_INTERVAL = 0.1


class SequenceRunnerThread(Thread):
    def __init__(
        self,
        experiment_config_name: str,
        sequence_path: SequencePath,
        session_maker: ExperimentSessionMaker,
        must_interrupt: Event,
    ):
        super().__init__(name=f"thread_{str(sequence_path)}")
        self._session = session_maker()
        self._save_session = session_maker()
        self._sequence = Sequence(sequence_path)
        self._devices: dict[DeviceName, RuntimeDevice] = {}

        # We watch this event while running the sequence and raise SequenceInterrupted if it becomes set.
        self._must_interrupt = must_interrupt

        # These locks are used to prevent concurrent access to the hardware and the database. Since lock access is fair,
        # the first shot to require the locks will be the first to get them, so shot execution order and save order will
        # be preserved. There should be no interleaving between lock acquisitions and releases within a shot.
        self._hardware_lock = asyncio.Lock()
        self._database_lock = asyncio.Lock()

        # This stack is used to ensure that proper cleanup is done when an error occurs.
        # For now, it is only used to close the devices properly.
        self._shutdown_stack = contextlib.ExitStack()

        with self._session.activate() as session:
            self._experiment_config = session.get_experiment_config(
                experiment_config_name
            )
            self._sequence_config = self._sequence.get_config(session)
            self._sequence.set_experiment_config(experiment_config_name, session)
            self._sequence.set_state(State.PREPARING, session)

    def run(self):
        try:
            self._run()
        except* SequenceInterruptedException:
            self.finish(State.INTERRUPTED)
            logger.info("Sequence interrupted")
        except* Exception:
            self.finish(State.CRASHED)
            logger.error("An error occurred while running the sequence", exc_info=True)
            raise
        else:
            self.finish(State.FINISHED)
            logger.info("Sequence finished")

    def _run(self):
        with self._shutdown_stack:
            asyncio.run(self.async_run())

    async def async_run(self):
        await self.prepare()
        await self.run_sequence()

    async def prepare(self):
        devices = self._create_uninitialized_devices()

        for device_name, device in devices.items():
            # We initialize the devices through the stack to unsure that they are closed if an error occurs.
            self._devices[device_name] = self._shutdown_stack.enter_context(
                DeviceContextManager(device)
            )

        async with asyncio.TaskGroup() as task_group:
            for device in self._devices.values():
                task_group.create_task(asyncio.to_thread(initialize_device, device))

        with self._session.activate() as session:
            self._sequence.set_state(State.RUNNING, session)

    def _create_uninitialized_devices(self) -> dict[DeviceName, RuntimeDevice]:
        """Create the devices on their respective servers.

        The devices are created with the initial parameters specified in the experiment and sequence configs, but the
        connection to the devices is not established. The device objects are proxies to the actual devices that are
        running in other processes, possibly on other computers.
        """

        remote_device_servers = create_device_servers(
            self._experiment_config.device_servers
        )
        connect_to_device_servers(remote_device_servers)

        initialization_parameters = get_devices_initialization_parameters(
            self._experiment_config, self._sequence_config
        )
        devices = create_devices(
            initialization_parameters,
            remote_device_servers,
            self._experiment_config.mock_experiment,
        )
        return devices

    def finish(self, state: State):
        with self._session as session:
            self._sequence.set_state(state, session)

    async def run_sequence(self):
        """Execute the sequence header and program"""

        context = StepContext[AnalogValue]()

        async with SequenceTaskGroup() as sequence_task_group:
            watch_interruption = sequence_task_group.create_background_task(
                self.watch_for_interruption()
            )
            context = await self.run_step(
                self._experiment_config.header, context, sequence_task_group
            )
            await self.run_step(
                self._sequence_config.program, context, sequence_task_group
            )
            await sequence_task_group.wait_shots_completed()
            watch_interruption.cancel()

    async def watch_for_interruption(self):
        """Raise SequenceInterrupted if the sequence must be interrupted."""

        while True:
            await asyncio.sleep(WATCH_FOR_INTERRUPTION_INTERVAL)
            if self._must_interrupt.is_set():
                raise SequenceInterruptedException()

    @singledispatchmethod
    async def run_step(
        self, step: Step, context: StepContext, task_group: SequenceTaskGroup
    ) -> StepContext:
        """Execute a given step of the sequence

        This function should be implemented for each Step type that can be run on the
        experiment.

        Args:
            step: the step of the sequence currently executed
            context: Contains the values of the variables before this step.
            task_group: A task group that can be used to create long-running tasks. Tasks in this group will be awaited
            before the sequence is completed in any way. If the sequence is interrupted or an error occurs, the
            remaining tasks will be cancelled. Tasks that must not be cancelled should be shielded.

        Returns:
            A new context object that contains the values of the variables after this step. This context object must be
            a new object.
        """

        raise NotImplementedError(f"run_step is not implemented for {type(step)}")

    @run_step.register
    async def _(
        self,
        steps: SequenceSteps,
        context: StepContext,
        task_group: SequenceTaskGroup,
    ) -> StepContext:
        """Execute the steps of a SequenceSteps.

        This function executes the child steps of a SequenceSteps in order. The context is updated after each step and
        the updated context is passed to the next step.
        """

        for step in steps.children:
            context = await self.run_step(step, context, task_group)
        return context

    @run_step.register
    async def _(
        self,
        declaration: VariableDeclaration,
        context: StepContext,
        _: SequenceTaskGroup,
    ) -> StepContext:
        """Execute a VariableDeclaration step.

        This function evaluates the expression of the declaration and updates the value of the variable in the context.
        """

        value = Quantity(declaration.expression.evaluate(context.variables | units))
        return context.update_variable(declaration.name, value)

    @run_step.register
    async def _(
        self,
        arange_loop: ArangeLoop,
        context: StepContext,
        task_group: SequenceTaskGroup,
    ):
        """Loop over a variable in a numpy arange like loop"""

        variables = context.variables | units

        start = Quantity(arange_loop.start.evaluate(variables))
        stop = Quantity(arange_loop.stop.evaluate(variables))
        step = Quantity(arange_loop.step.evaluate(variables))
        unit = start.units

        start = start.to(unit)
        try:
            stop = stop.to(unit)
        except DimensionalityError:
            raise ValueError(
                f"Stop units of arange loop '{arange_loop.name}' ({stop.units}) is not"
                f" compatible with start units ({unit})"
            )
        try:
            step = step.to(unit)
        except DimensionalityError:
            raise ValueError(
                f"Step units of arange loop '{arange_loop.name}' ({step.units}) are not"
                f" compatible with start units ({unit})"
            )

        for value in np.arange(start.magnitude, stop.magnitude, step.magnitude):
            context = context.update_variable(arange_loop.name, value * unit)
            for step in arange_loop.children:
                context = await self.run_step(step, context, task_group)
        return context

    @run_step.register
    async def _(
        self,
        linspace_loop: LinspaceLoop,
        context: StepContext,
        task_group: SequenceTaskGroup,
    ):
        """Loop over a variable in a numpy linspace like loop"""

        variables = context.variables | units

        try:
            start = Quantity(linspace_loop.start.evaluate(variables))
        except Exception as error:
            raise ValueError(
                f"Could not evaluate start of linspace loop {linspace_loop.name}"
            ) from error
        unit = start.units
        try:
            stop = Quantity(linspace_loop.stop.evaluate(variables))
        except Exception as error:
            raise ValueError(
                f"Could not evaluate stop of linspace loop {linspace_loop.name}"
            ) from error
        try:
            stop = stop.to(unit)
        except DimensionalityError:
            raise ValueError(
                f"Stop units of linspace loop '{linspace_loop.name}' ({stop.units}) is not"
                f" compatible with start units ({unit})"
            )
        num = int(linspace_loop.num)

        for value in np.linspace(start.magnitude, stop.magnitude, num):
            context = context.update_variable(linspace_loop.name, value * unit)
            for step in linspace_loop.children:
                context = await self.run_step(step, context, task_group)
        return context

    @run_step.register
    async def _(
        self, shot: ExecuteShot, context: StepContext, task_group: SequenceTaskGroup
    ) -> StepContext:
        """Execute a shot on the experiment."""

        shot_configuration = self._sequence_config.shot_configurations[shot.name]

        async def compute_parameters():
            change_params = await self.compute_change_parameters(
                shot_configuration, context
            )
            shot_params = await self.compute_shot_parameters(
                shot_configuration, context
            )
            return change_params, shot_params

        computation_task = await task_group.create_computation_task(
            compute_parameters()
        )
        change_parameters, shot_parameters = await computation_task

        task_group.create_hardware_task(
            self.do_shot_with_retry(
                shot.name,
                context.variables,
                change_parameters,
                shot_parameters,
                task_group,
            )
        )
        return context.reset_history()

    @run_step.register
    async def _(
        self, loop: UserInputLoop, context: StepContext, task_group: SequenceTaskGroup
    ) -> StepContext:
        """Repeat its child steps while asking the user the value of some variables."""

        evaluated_variable_ranges = evaluate_variable_ranges(
            loop.iteration_variables, context.variables | units
        )
        raw_variable_ranges = strip_unit_from_variable_ranges(evaluated_variable_ranges)
        variable_units = {
            name: value.unit for name, value in raw_variable_ranges.items()
        }

        runner = ExecUserInput(
            title=str(self._sequence.path),
            variable_ranges=raw_variable_ranges,
        )

        async with asyncio.TaskGroup() as background_task_group:
            task = background_task_group.create_task(asyncio.to_thread(runner.run))

            child_step_index = 0
            while not task.done():
                raw_values = runner.get_current_values()
                for variable_name, raw_value in raw_values.items():
                    minimum = evaluated_variable_ranges[variable_name].minimum
                    maximum = evaluated_variable_ranges[variable_name].maximum
                    value = add_unit(raw_value, variable_units[variable_name])
                    if not (minimum <= value <= maximum):
                        raise ValueError(
                            f"Value {value} for variable {variable_name} is not in the "
                            f"range [{minimum}, {maximum}]"
                        )
                    context = context.update_variable(variable_name, value)

                if child_step_index < len(loop.children):
                    context = await self.run_step(
                        loop.children[child_step_index], context, task_group
                    )
                    child_step_index += 1
                else:
                    child_step_index = 0
                await task_group.wait_shots_completed()
        return context

    @run_step.register
    async def _(
        self,
        optimization_loop: OptimizationLoop,
        context: SequenceContext,
        task_group: SequenceTaskGroup,
    ):
        raise NotImplementedError
        # optimizer_config = self._experiment_config.get_optimizer_config(
        #      optimization_loop.optimizer_name
        # )
        # optimizer = Optimizer(optimization_loop.variables, context.variables | units)
        # await task_group.wait_shots_completed()
        # with CostEvaluatorProcess(self._sequence, optimizer_config) as evaluator:
        #     while not evaluator.is_ready():
        #         if self.is_waiting_to_interrupt():
        #             evaluator.interrupt()
        #             return
        #     for loop_iteration in range(optimization_loop.repetitions):
        #         old_shots = shot_saver.saved_shots
        #         new_values = optimizer.suggest_values()
        #         for name, value in new_values.items():
        #             self.update_variable_value(name, value, context)
        #
        #         for step in optimization_loop.children:
        #             self.run_step(step, context, shot_saver)
        #             if self.is_waiting_to_interrupt():
        #                 return
        #         shot_saver.wait()
        #
        #         new_shots = shot_saver.saved_shots[len(old_shots) :]
        #         score = evaluator.compute_score(new_shots)
        #         logger.info(f"Values for iteration {loop_iteration}: {new_values}")
        #         logger.info(f"Score for iteration {loop_iteration}: {score}")
        #         optimizer.register(new_values, score)
        #         with self._session.activate():
        #             for shot in new_shots:
        #                 shot.add_scores(
        #                     {optimization_loop.optimizer_name: score}, self._session
        #                 )

    async def update_device_parameters(
        self, device_parameters: dict[DeviceName, dict[DeviceParameter, Any]]
    ):
        if self._experiment_config.mock_experiment:
            return

        async with asyncio.TaskGroup() as update_group:
            # There is no need to shield the tasks from cancellation because they are running synchronous functions
            # in other threads and cannot be cancelled in middle of execution.
            # Some devices might be updated while others not if an exception is raised, but I don't think it is a
            # problem.
            for device_name, parameters in device_parameters.items():
                task = asyncio.to_thread(
                    update_device, self._devices[device_name], parameters
                )
                update_group.create_task(task)

    async def compute_change_parameters(
        self, shot: ShotConfiguration, context: StepContext
    ) -> dict[DeviceName, dict[DeviceParameter, Any]]:

        change_parameters = await asyncio.to_thread(
            compute_parameters_on_variables_update,
            context.updated_variables,
            context.variables,
            self._experiment_config,
        )
        return change_parameters

    async def compute_shot_parameters(
        self,
        shot: ShotConfiguration,
        context: StepContext,
    ) -> dict[DeviceName, dict[DeviceParameter, Any]]:
        with DurationTimer() as timer:
            shot_parameters = await asyncio.to_thread(
                compute_shot_parameters,
                self._experiment_config,
                shot,
                context.variables,
            )
        logger.info(
            "Shot parameters computation duration:" f" {timer.duration_in_ms:.1f} ms"
        )
        return shot_parameters

    async def do_shot_with_retry(
        self,
        shot_name: str,
        variables: VariableNamespace,
        change_parameters: dict[DeviceName, dict[DeviceParameter, Any]],
        shot_parameters: dict[DeviceName, dict[DeviceParameter, Any]],
        task_group: SequenceTaskGroup,
    ) -> None:
        number_of_attempts = 2  # must >= 1
        async with self._hardware_lock:
            for attempt in range(number_of_attempts):
                errors: list[Exception] = []
                try:
                    with DurationTimer() as timer:
                        data = await self.do_shot(change_parameters, shot_parameters)
                except* CameraTimeoutError as e:
                    errors.extend(e.exceptions)
                    logger.warning(
                        "A camera timeout error occurred, attempting to redo the failed shot"
                    )
                else:
                    task_group.create_database_task(
                        self.save_shot_with_lock(
                            shot_name, timer.start_time, timer.end_time, variables, data
                        )
                    )
                    return
                logger.warning(f"Attempt {attempt+1}/{number_of_attempts} failed")
        raise ExceptionGroup(
            f"Could not execute shot after {number_of_attempts} attempts", errors
        )

    async def do_shot(
        self,
        change_parameters: dict[DeviceName, dict[DeviceParameter, Any]],
        device_parameters: dict[DeviceName, dict[DeviceParameter, Any]],
    ) -> dict[str, Any]:

        with DurationTimer() as timer:
            await self.update_device_parameters(change_parameters)
            await self.update_device_parameters(device_parameters)
        logger.info(
            "Device parameters update duration:" f" {timer.duration_in_ms:.1f} ms"
        )

        with DurationTimer() as timer:
            await self.run_shot()
        logger.info("Shot execution duration:" f" {timer.duration_in_ms:.1f} ms")
        data = self.extract_data()
        return data

    async def run_shot(self) -> None:
        """Execute the shot by controlling each device asynchronously."""

        if self._experiment_config.mock_experiment:
            await asyncio.sleep(0.5)
            return

        for ni6738_card in self.get_ni6738_cards().values():
            ni6738_card.run()

        async with asyncio.TaskGroup() as run_group:
            for camera in self.get_cameras().values():
                run_group.create_task(asyncio.to_thread(camera.acquire_all_pictures))
            for spincore_sequencer in self.get_spincore_sequencers().values():
                run_group.create_task(asyncio.to_thread(spincore_sequencer.run))

    def extract_data(self) -> dict[DeviceName, Any]:
        if self._experiment_config.mock_experiment:
            return {
                "image": np.random.uniform(0, 2**15, (100, 100)).astype(np.uint16)
            }

        data = {}
        for camera_name, camera in self.get_cameras().items():
            data[camera_name] = camera.read_all_pictures()
        return data

    async def save_shot_with_lock(
        self,
        shot_name: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        parameters: VariableNamespace,
        measures: dict[DeviceName, Any],
    ):
        async with self._database_lock:
            return await asyncio.to_thread(
                self.save_shot, shot_name, start_time, end_time, parameters, measures
            )

    def save_shot(
        self,
        shot_name: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        parameters: VariableNamespace,
        measures: dict[DeviceName, Any],
    ) -> Shot:
        with self._save_session as session:
            params = {str(name): value for name, value in parameters.to_dict().items()}
            return self._sequence.create_shot(
                shot_name, start_time, end_time, params, measures, session
            )

    def get_ni6738_cards(self) -> dict[str, "NI6738AnalogCard"]:
        return {
            device_name: device  # type: ignore
            for device_name, device in self._devices.items()
            if isinstance(
                self._experiment_config.get_device_config(device_name),
                NI6738SequencerConfiguration,
            )
        }

    def get_spincore_sequencers(self) -> dict[str, "SpincorePulseBlaster"]:
        return {
            device_name: device  # type: ignore
            for device_name, device in self._devices.items()
            if isinstance(
                self._experiment_config.get_device_config(device_name),
                SpincoreSequencerConfiguration,
            )
        }

    def get_cameras(self) -> dict[str, "Camera"]:
        return {
            device_name: device  # type: ignore
            for device_name, device in self._devices.items()
            if isinstance(
                self._experiment_config.get_device_config(device_name),
                CameraConfiguration,
            )
        }


def update_device(device: RuntimeDevice, parameters: dict[DeviceParameter, Any]):
    try:
        if parameters:
            device.update_parameters(**parameters)
    except Exception as error:
        raise RuntimeError(
            f"Failed to update device {device.get_name()} with parameters:\n"
            f"{pprint.pformat(parameters)}"
        ) from error


def evaluate_variable_ranges(
    variable_ranges: Mapping[DottedVariableName, VariableRange],
    context_variables: Mapping[DottedVariableName, Any],
) -> dict[DottedVariableName, EvaluatedVariableRange]:
    """Replace expressions in variable ranges with their real values."""

    evaluated_variable_ranges: dict[DottedVariableName, EvaluatedVariableRange] = {}
    for variable_name, variable_range in variable_ranges.items():
        initial_value = variable_range.initial_value.evaluate(context_variables)

        first_bound = variable_range.first_bound.evaluate(context_variables)
        second_bound = variable_range.second_bound.evaluate(context_variables)

        minimum = min(first_bound, second_bound)
        maximum = max(first_bound, second_bound)
        evaluated_range = EvaluatedVariableRange(
            initial_value=initial_value,
            minimum=minimum,
            maximum=maximum,
        )
        evaluated_variable_ranges[variable_name] = evaluated_range
    return evaluated_variable_ranges


def strip_unit_from_variable_ranges(
    variable_ranges: dict[DottedVariableName, EvaluatedVariableRange],
) -> dict[DottedVariableName, RawVariableRange]:
    """Replace expressions in variable ranges with their real values."""

    raw_variable_ranges: dict[DottedVariableName, RawVariableRange] = {}
    for variable_name, variable_range in variable_ranges.items():
        initial_value = variable_range.initial_value
        unit = get_unit(initial_value)
        initial_value = magnitude_in_unit(initial_value, unit)

        minimum = variable_range.minimum
        minimum = magnitude_in_unit(minimum, unit)
        maximum = variable_range.maximum
        maximum = magnitude_in_unit(maximum, unit)

        evaluated_range = RawVariableRange(
            initial_value=initial_value,
            minimum=minimum,
            maximum=maximum,
            unit=unit,
        )
        raw_variable_ranges[variable_name] = evaluated_range
    return raw_variable_ranges


def initialize_device(device: RuntimeDevice):
    try:
        device.initialize()
        logger.debug(f"Device '{device.get_name()}' started.")
    except Exception as error:
        raise RuntimeError(f"Could not start device '{device.get_name()}'") from error


# these exceptions are used to interrupt the sequence and inherit from BaseException to prevent them from being caught
# by 'except Exception' which would prevent the sequence from being interrupted.
class SequenceInterruptedException(BaseException):
    pass


class SequenceFinishedException(BaseException):
    pass
