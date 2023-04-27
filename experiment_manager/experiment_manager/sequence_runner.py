import collections.abc
import datetime
import logging
import pprint
import time
import typing
from concurrent.futures import ThreadPoolExecutor, Future
from functools import singledispatchmethod
from multiprocessing.managers import RemoteError
from threading import Thread, Event
from typing import Any

import numpy
import numpy as np

from camera.configuration import CameraConfiguration
from camera.runtime import CameraTimeoutError
from device import RuntimeDevice
from experiment.configuration import (
    SpincoreSequencerConfiguration,
    DeviceServerConfiguration,
    DeviceName,
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
from sequence.runtime import SequencePath, Sequence
from sql_model import State
from units import Quantity, units, get_unit, magnitude_in_unit
from units.analog_value import add_unit
from variable.name import DottedVariableName
from variable.namespace import VariableNamespace
from .compute_shot_parameters import compute_shot_parameters
from .initialize_devices import get_devices_initialization_parameters
from .run_optimization import Optimizer, CostEvaluatorProcess
from .sequence_context import SequenceContext
from .shot_saver import ShotSaver
from .user_input_loop.exec_user_input import ExecUserInput
from .user_input_loop.input_widget import RawVariableRange, EvaluatedVariableRange
from .variable_change import compute_parameters_on_variable_update

if typing.TYPE_CHECKING:
    from ni6738_analog_card.runtime import NI6738AnalogCard
    from camera.runtime import CCamera
    from spincore_sequencer.runtime import SpincorePulseBlaster

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SequenceRunnerThread(Thread):
    def __init__(
        self,
        experiment_config_name: str,
        sequence_path: SequencePath,
        session_maker: ExperimentSessionMaker,
        waiting_to_interrupt: Event,
    ):
        super().__init__(name=f"thread_{str(sequence_path)}")
        self._session = session_maker()
        self._session_maker = session_maker
        self._sequence = Sequence(sequence_path)
        self._waiting_to_interrupt = waiting_to_interrupt
        self._remote_device_managers: dict[str, RemoteDeviceClientManager] = {}
        self._devices: dict[str, RuntimeDevice] = {}

        with self._session.activate() as session:
            self._experiment_config = session.get_experiment_config(
                experiment_config_name
            )
            self._sequence_config = self._sequence.get_config(session)
            self._sequence.set_experiment_config(experiment_config_name, session)
            self._sequence.set_state(State.PREPARING, session)

    def run(self):
        try:
            self.prepare()
            self.run_sequence()
            self.finish()
        except Exception:
            self.record_exception()
            logger.error("An error occurred while running the sequence", exc_info=True)
            raise
        finally:
            self.shutdown()

    def prepare(self):
        self._remote_device_managers = create_remote_device_managers(
            self._experiment_config.device_servers
        )
        self.connect_to_device_servers()

        self._devices = self.create_devices()
        self.start_devices()

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

    def start_devices(self):
        for device_name, device in self._devices.items():
            try:
                device.start()
            except Exception:
                logger.error(f"An error occurred while starting device {device_name}")
                raise

    def finish(self):
        with self._session as session:
            if self.is_waiting_to_interrupt():
                self._sequence.set_state(State.INTERRUPTED, session)
            else:
                self._sequence.set_state(State.FINISHED, session)

    def record_exception(self):
        with self._session:
            self._sequence.set_state(State.CRASHED, self._session)

    def shutdown(self):
        exceptions = []
        for device in self._devices.values():
            try:
                device.shutdown()
            except Exception as error:
                exceptions.append(error)
        if exceptions:
            raise_multiple_exceptions(exceptions, "Errors occurred while shutting down")
        logger.info("Sequence finished")

    def run_sequence(self):
        """Execute the sequence header and program"""

        context = SequenceContext(variables=VariableNamespace())
        with ShotSaver(self._sequence, self._session_maker) as shot_saver:
            self.run_step(self._experiment_config.header, context, shot_saver)
            self.run_step(self._sequence_config.program, context, shot_saver)

    def update_variable_value(
        self, name: DottedVariableName, value: Any, context: SequenceContext
    ):
        """Update the value of a variable.

        This method update the value of a variable in the dictionary-like context. It also gives a chance to the devices
         to update their state if needed.
        """

        context.variables[name] = value
        parameters = compute_parameters_on_variable_update(
            name, self._experiment_config, context.variables
        )
        self.update_device_parameters(parameters)

    @singledispatchmethod
    def run_step(self, step: Step, context: SequenceContext, shot_saver: ShotSaver):
        """Execute a given step of the sequence

        This function should be implemented for each Step type that can be run on the
        experiment. It should also return as soon as possible if the sequence needs to
        be interrupted.

        Args:
            step: the step of the sequence currently executed
            context: a mutable object that holds information about the sequence being
            run, such as the values of the variables. Step that update variables should
            reflect this by modifying the context.
            shot_saver: a ShotSaver object that is passed down to the children steps and is used to store shot data in
            order.

        """

        raise NotImplementedError(f"run_step is not implemented for {type(step)}")

    @run_step.register
    def _(self, steps: SequenceSteps, context: SequenceContext, shot_saver: ShotSaver):
        """Execute each child step sequentially"""

        for step in steps.children:
            if self.is_waiting_to_interrupt():
                return
            else:
                self.run_step(step, context, shot_saver)

    @run_step.register
    def _(
        self, declaration: VariableDeclaration, context: SequenceContext, _: ShotSaver
    ):
        """Add or update a variable declaration in the context"""

        self.update_variable_value(
            declaration.name,
            Quantity(declaration.expression.evaluate(context.variables | units)),
            context,
        )

    @run_step.register
    def _(
        self, arange_loop: ArangeLoop, context: SequenceContext, shot_saver: ShotSaver
    ):
        """Loop over a variable in a numpy arange like loop"""

        start = Quantity(arange_loop.start.evaluate(context.variables | units))
        stop = Quantity(arange_loop.stop.evaluate(context.variables | units))
        step = Quantity(arange_loop.step.evaluate(context.variables | units))

        unit = start.units

        for value in numpy.arange(
            start.to(unit).magnitude, stop.to(unit).magnitude, step.to(unit).magnitude
        ):
            self.update_variable_value(arange_loop.name, value * unit, context)
            for step in arange_loop.children:
                if self.is_waiting_to_interrupt():
                    return
                else:
                    self.run_step(step, context, shot_saver)

    @run_step.register
    def _(
        self,
        linspace_loop: LinspaceLoop,
        context: SequenceContext,
        shot_saver: ShotSaver,
    ):
        """Loop over a variable in a numpy linspace like loop"""

        try:
            start = Quantity(linspace_loop.start.evaluate(context.variables | units))
        except Exception as error:
            raise ValueError(
                f"Could not evaluate start of linspace loop {linspace_loop.name}"
            ) from error
        try:
            stop = Quantity(linspace_loop.stop.evaluate(context.variables | units))
        except Exception as error:
            raise ValueError(
                f"Could not evaluate stop of linspace loop {linspace_loop.name}"
            ) from error
        num = int(linspace_loop.num)

        unit = start.units

        for value in numpy.linspace(
            start.to(unit).magnitude, stop.to(unit).magnitude, num
        ):
            self.update_variable_value(linspace_loop.name, value * unit, context)
            for step in linspace_loop.children:
                if self.is_waiting_to_interrupt():
                    return
                else:
                    self.run_step(step, context, shot_saver)

    @run_step.register
    def _(
        self,
        optimization_loop: OptimizationLoop,
        context: SequenceContext,
        shot_saver: ShotSaver,
    ):
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

    @run_step.register
    def _(self, shot: ExecuteShot, context: SequenceContext, shot_saver: ShotSaver):
        """Execute a shot on the experiment"""

        start_time = datetime.datetime.now()
        try:
            data = self.do_shot(
                self._sequence_config.shot_configurations[shot.name], context.variables
            )
        except CameraTimeoutError as error:
            logger.warning(
                "A camera timeout error occurred:\n"
                f"{error}\n"
                "Attempting to redo the failed shot"
            )
            data = self.do_shot(
                self._sequence_config.shot_configurations[shot.name], context.variables
            )

        end_time = datetime.datetime.now()
        logger.info(
            f"Shot total duration: {(end_time - start_time).total_seconds()*1e3:.1f} ms"
        )

        shot_saver.push_shot(shot.name, start_time, end_time, context.variables, data)

    @run_step.register
    def _(self, loop: UserInputLoop, context: SequenceContext, shot_saver: ShotSaver):
        """Repeat its child steps while asking the user the value of some variables."""

        evaluated_variable_ranges = evaluate_variable_ranges(
            loop.iteration_variables, context.variables | units
        )
        raw_variable_ranges = strip_unit_from_variable_ranges(evaluated_variable_ranges)
        variable_units = {
            name: value.unit for name, value in raw_variable_ranges.items()
        }

        with ThreadPoolExecutor() as executor:
            runner = ExecUserInput(
                title=str(self._sequence.path),
                variable_ranges=raw_variable_ranges,
            )
            result = executor.submit(runner.run)
            child_step_index = 0
            while not result.done():
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
                    self.update_variable_value(variable_name, value, context)

                if self.is_waiting_to_interrupt():
                    result.cancel()
                    break
                else:
                    if child_step_index < len(loop.children):
                        self.run_step(
                            loop.children[child_step_index], context, shot_saver
                        )
                        child_step_index += 1
                    else:
                        child_step_index = 0

        if result.exception():
            raise result.exception()

    def do_shot(
        self, shot: ShotConfiguration, context: VariableNamespace
    ) -> dict[str, Any]:
        self.prepare_shot(shot, context)
        self.run_shot()
        data = self.extract_data()
        return data

    def prepare_shot(self, shot: ShotConfiguration, context: VariableNamespace):
        initial_time = datetime.datetime.now()
        device_parameters = compute_shot_parameters(
            self._experiment_config, shot, context
        )
        computation_time = datetime.datetime.now()
        logger.info(
            "Shot parameters computation duration:"
            f" {(computation_time - initial_time).total_seconds() * 1e3:.1f} ms"
        )
        self.update_device_parameters(device_parameters)
        update_time = datetime.datetime.now()
        logger.info(
            "Device parameters update duration:"
            f" {(update_time - computation_time).total_seconds() * 1e3:.1f} ms"
        )

    def update_device_parameters(self, device_parameters: dict[str, dict[str, Any]]):
        if self._experiment_config.mock_experiment:
            return
        future_updates: dict[str, Future] = {}

        with ThreadPoolExecutor() as update_executor:
            for device_name, parameters in device_parameters.items():
                if parameters:
                    future_updates[device_name] = update_executor.submit(
                        self._devices[device_name].update_parameters, **parameters
                    )

        exceptions = []
        for device_name, update in future_updates.items():
            if (exception := update.exception()) is not None:
                new_exception = RuntimeError(
                    f"Failed to update device {device_name} with parameters:\n"
                    f"{pprint.pformat(device_parameters[device_name])}"
                )
                new_exception.__cause__ = exception
                exceptions.append(new_exception)

        if exceptions:
            raise raise_multiple_exceptions(
                exceptions, "Errors occurred when updating device parameters"
            )

    def run_shot(self) -> None:
        start_time = datetime.datetime.now()
        if self._experiment_config.mock_experiment:
            time.sleep(0.5)
            return
        for ni6738_card in self.get_ni6738_cards().values():
            ni6738_card.run()

        future_acquisitions: dict[str, Future] = {}
        with ThreadPoolExecutor() as acquisition_executor:
            for camera_name, camera in self.get_cameras().items():
                future_acquisitions[camera_name] = acquisition_executor.submit(
                    camera.acquire_all_pictures
                )
            for spincore_sequencer in self.get_spincore_sequencers().values():
                spincore_sequencer.run()

        for acquisition in future_acquisitions.values():
            if exception := acquisition.exception():
                raise exception
        stop_time = datetime.datetime.now()
        logger.info(
            "Shot execution duration:"
            f" {(stop_time - start_time).total_seconds() * 1e3:.1f} ms"
        )

    def extract_data(self):
        if self._experiment_config.mock_experiment:
            return {
                "image": np.random.uniform(0, 2**15, (100, 100)).astype(np.uint16)
            }

        data = {}
        for camera_name, camera in self.get_cameras().items():
            data[camera_name] = camera.read_all_pictures()
        return data

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

    def is_waiting_to_interrupt(self) -> bool:
        return self._waiting_to_interrupt.is_set()


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


def raise_multiple_exceptions(
    exceptions: collections.abc.Sequence[Exception], message: str
):
    if not exceptions:
        return
    if len(exceptions) == 1:
        raise exceptions[0]
    raise ExceptionGroup(message, exceptions)


def evaluate_variable_ranges(
    variable_ranges: dict[DottedVariableName, VariableRange],
    context_variables: dict[str, Any],
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
