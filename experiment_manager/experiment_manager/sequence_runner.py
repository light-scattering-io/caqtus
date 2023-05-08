import asyncio
import contextlib
import datetime
import logging
import pprint
from collections.abc import Mapping
from functools import singledispatchmethod
from multiprocessing.managers import RemoteError
from threading import Thread, Event
from typing import TYPE_CHECKING, Any, Self

import numpy
import numpy as np

from camera.configuration import CameraConfiguration
from camera.runtime import CameraTimeoutError
from device import RuntimeDevice, DeviceName
from experiment.configuration import (
    SpincoreSequencerConfiguration,
    DeviceServerConfiguration,
    DeviceParameter,
)
from experiment.session import ExperimentSessionMaker
from ni6738_analog_card.configuration import NI6738SequencerConfiguration
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
from sequence.runtime import SequencePath, Sequence, Shot
from sql_model import State
from units import Quantity, units, get_unit, magnitude_in_unit, DimensionalityError
from units.analog_value import add_unit, AnalogValue
from variable.name import DottedVariableName
from variable.namespace import VariableNamespace
from .compute_shot_parameters import compute_shot_parameters
from .initialize_devices import get_devices_initialization_parameters
from .run_optimization import Optimizer, CostEvaluatorProcess
from .sequence_context import SequenceContext, SequenceTaskGroup
from .sequence_context import StepContext
from .shot_saver import ShotSaver
from .user_input_loop.exec_user_input import ExecUserInput
from .user_input_loop.input_widget import RawVariableRange, EvaluatedVariableRange
from .variable_change import compute_parameters_on_variables_update

if TYPE_CHECKING:
    from ni6738_analog_card.runtime import NI6738AnalogCard
    from camera.runtime import CCamera
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
        self._remote_device_managers: dict[str, RemoteDeviceClientManager] = {}
        self._devices: dict[str, RuntimeDevice] = {}

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
        except* SequenceInterrupted:
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
        async with asyncio.TaskGroup() as task_group:
            task_group.create_task(self.prepare())
            task_group.create_task(self.run_sequence())

    async def prepare(self):
        async with self._hardware_lock:
            self._remote_device_managers = create_remote_device_managers(
                self._experiment_config.device_servers
            )
            self.connect_to_device_servers()

            devices = self.create_devices()

            for device_name, device in devices.items():
                # We initialize the devices through the stack to unsure that they are closed if an error occurs.
                self._devices[device_name] = await asyncio.to_thread(
                    self._shutdown_stack.enter_context, DeviceContextManager(device)
                )

        with self._session.activate() as session:
            self._sequence.set_state(State.RUNNING, session)

    def connect_to_device_servers(self):
        """Start the connection to the device servers."""

        if self._experiment_config.mock_experiment:
            return

        for server_name, server in self._remote_device_managers.items():
            logger.info(f"Connecting to device server {server_name}...")
            try:
                server.connect()
            except ConnectionRefusedError as error:
                raise ConnectionRefusedError(
                    f"The remote server '{server_name}' rejected the connection. It is"
                    " possible that the server is not running or that the port is not"
                    " open."
                ) from error
            except TimeoutError as error:
                raise TimeoutError(
                    f"The remote server '{server_name}' did not respond to the"
                    " connection request. It is possible that the server is not"
                    " running or that the port is not open."
                ) from error
            logger.info(f"Connection established to {server_name}")

    def create_devices(self) -> dict[DeviceName, RuntimeDevice]:
        """Instantiate the devices on their respective remote server.

        This function computes the parameters necessary to instantiate the device objects and then creates them on
        the remote servers. The device objects are then returned as a dictionary matching the device names to a proxy to
        the associated device.

        This function only creates the device objects but does not start them. No communication with the actual devices
        is performed at this stage.
        """

        if self._experiment_config.mock_experiment:
            return {}

        initialization_parameters = get_devices_initialization_parameters(
            self._experiment_config, self._sequence_config
        )

        devices: dict[DeviceName, RuntimeDevice] = {}

        for device_name, parameters in initialization_parameters.items():
            server = self._remote_device_managers[parameters["server"]]
            try:
                remote_class = getattr(server, parameters["type"])
            except AttributeError:
                raise ValueError(
                    f"The device '{device_name}' is of type '{parameters['type']}' but"
                    " this type is not registered for the remote device client."
                )

            try:
                devices[device_name] = remote_class(**parameters["init_kwargs"])
            except RemoteError as error:
                raise RuntimeError(
                    f"Remote servers {parameters['server']} could not instantiate"
                    f" device '{device_name}'"
                ) from error

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
                raise SequenceInterrupted()

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

        for value in numpy.arange(start.magnitude, stop.magnitude, step.magnitude):
            context = context.update_variable(arange_loop.name, value * unit)
            for step in arange_loop.children:
                context = await self.run_step(step, context, task_group)
        return context

    @run_step.register
    async def _(
        self,
        linspace_loop: LinspaceLoop,
        context: StepContext,
        shot_saver: ShotSaver,
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

        for value in numpy.linspace(start.magnitude, stop.magnitude, num):
            context = context.update_variable(linspace_loop.name, value * unit)
            for step in linspace_loop.children:
                await self.run_step(step, context, shot_saver)
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
    def _(
        self,
        optimization_loop: OptimizationLoop,
        context: SequenceContext,
        shot_saver: ShotSaver,
    ):
        raise NotImplementedError()
        optimizer_config = self._experiment_config.get_optimizer_config(
            optimization_loop.optimizer_name
        )
        optimizer = Optimizer(optimization_loop.variables, context.variables | units)
        shot_saver.wait()
        with CostEvaluatorProcess(self._sequence, optimizer_config) as evaluator:
            while not evaluator.is_ready():
                if self.is_waiting_to_interrupt():
                    evaluator.interrupt()
                    return
            for loop_iteration in range(optimization_loop.repetitions):
                old_shots = shot_saver.saved_shots
                new_values = optimizer.suggest_values()
                for name, value in new_values.items():
                    self.update_variable_value(name, value, context)

                for step in optimization_loop.children:
                    self.run_step(step, context, shot_saver)
                    if self.is_waiting_to_interrupt():
                        return
                shot_saver.wait()

                new_shots = shot_saver.saved_shots[len(old_shots) :]
                score = evaluator.compute_score(new_shots)
                logger.info(f"Values for iteration {loop_iteration}: {new_values}")
                logger.info(f"Score for iteration {loop_iteration}: {score}")
                optimizer.register(new_values, score)
                with self._session.activate():
                    for shot in new_shots:
                        shot.add_scores(
                            {optimization_loop.optimizer_name: score}, self._session
                        )

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
                try:
                    with DurationTimer() as timer:
                        data = await self.do_shot(change_parameters, shot_parameters)
                except CameraTimeoutError:
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
        raise CameraTimeoutError(
            f"Could not execute shot after {number_of_attempts} attempts"
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

    def get_cameras(self) -> dict[str, "CCamera"]:
        return {
            device_name: device  # type: ignore
            for device_name, device in self._devices.items()
            if isinstance(
                self._experiment_config.get_device_config(device_name),
                CameraConfiguration,
            )
        }


def create_remote_device_managers(
    device_server_configs: dict[str, DeviceServerConfiguration]
) -> dict[str, RemoteDeviceClientManager]:
    remote_device_managers: dict[str, RemoteDeviceClientManager] = {}
    for server_name, server_config in device_server_configs.items():
        address = (server_config.address, server_config.port)
        authkey = bytes(server_config.authkey.get_secret_value(), encoding="utf-8")
        remote_device_managers[server_name] = RemoteDeviceClientManager(
            address=address, authkey=authkey
        )
    return remote_device_managers


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


class DeviceContextManager(contextlib.AbstractContextManager[RuntimeDevice]):
    def __init__(self, device: RuntimeDevice):
        self._device = device

    def __enter__(self) -> RuntimeDevice:
        self.initialize()
        return self._device

    def initialize(self):
        try:
            self._device.initialize()
            logger.debug(f"Device '{self._device.get_name()}' started.")
        except Exception as error:
            raise RuntimeError(
                f"Could not start device '{self._device.get_name()}'"
            ) from error

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self):
        try:
            self._device.close()
            logger.debug(f"Device '{self._device.get_name()}' shut down.")
        except Exception as error:
            raise RuntimeError(
                f"An error occurred while closing '{self._device.get_name()}'"
            ) from error


class SequenceInterrupted(Exception):
    pass


class SequenceFinished(Exception):
    pass


class DurationTimer(contextlib.AbstractContextManager):
    def __init__(self):
        self._start_time = None
        self._end_time = None

    def __enter__(self) -> Self:
        self._start_time = datetime.datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end_time = datetime.datetime.now()

    @property
    def duration(self) -> datetime.timedelta:
        return self.end_time - self.start_time

    @property
    def duration_in_s(self) -> float:
        return self.duration.total_seconds()

    @property
    def duration_in_ms(self) -> float:
        return self.duration_in_s * 1000

    @property
    def start_time(self) -> datetime.datetime:
        if self._start_time is None:
            raise RuntimeError("Timer has not been started yet.")
        else:
            return self._start_time

    @property
    def end_time(self) -> datetime.datetime:
        if self._end_time is None:
            raise RuntimeError("Timer has not been stopped yet.")
        else:
            return self._end_time
