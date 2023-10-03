import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, ExitStack
from threading import Lock
from typing import (
    TYPE_CHECKING,
    Mapping,
    Any,
)

from aod_tweezer_arranger.configuration import AODTweezerArrangerConfiguration
from camera.configuration import CameraConfiguration
from device.configuration import DeviceName, DeviceParameter
from device.runtime import RuntimeDevice
from experiment.configuration import ExperimentConfig
from experiment_control.sequence_runner.device_context_manager import (
    DeviceContextManager,
)
from sequencer.configuration import SequencerConfiguration
from sequencer.runtime import Sequencer
from .task_group import TaskGroup

if TYPE_CHECKING:
    from camera.runtime import Camera
    from aod_tweezer_arranger.runtime import AODTweezerArranger

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DevicesHandler(AbstractContextManager):
    def __init__(
        self,
        devices: dict[DeviceName, RuntimeDevice],
        experiment_config: ExperimentConfig,
    ):
        self._devices = devices
        self._exit_stack = ExitStack()
        self._lock = Lock()

        self.sequencers = get_sequencers_in_use(devices, experiment_config)
        self.cameras = get_cameras_in_use(devices, experiment_config)
        self.tweezer_arrangers = get_tweezer_arrangers_in_use(
            devices, experiment_config
        )

    def __enter__(self):
        self._exit_stack.__enter__()
        try:
            self._start_devices()
        except Exception:
            self._exit_stack.close()
            raise
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        return self._exit_stack.__exit__(exc_type, exc_value, exc_traceback)

    def _start_devices(self):
        with ThreadPoolExecutor() as thread_pool, TaskGroup(thread_pool) as g:
            for device_name, device in self._devices.items():
                # We initialize the devices through the stack to unsure that they are closed if an error occurs.
                g.add_task(self._exit_stack.enter_context, DeviceContextManager(device))

    def update_device_parameters(
        self, device_parameters: Mapping[DeviceName, dict[DeviceParameter, Any]]
    ):
        with ThreadPoolExecutor() as thread_pool, TaskGroup(thread_pool) as g:
            for device_name, parameters in device_parameters.items():
                g.add_task(update_device, self._devices[device_name], parameters)

    def start_shot(self):
        with ThreadPoolExecutor() as thread_pool, TaskGroup(thread_pool) as g:
            for tweezer_arrangers in self.tweezer_arrangers.values():
                g.add_task(tweezer_arrangers.start_sequence)
            for camera in self.cameras.values():
                g.add_task(camera.start_acquisition)
        # we need the sequencers to be correctly triggered, so we start them in their priority order
        for sequencer in self.sequencers.values():
            sequencer.start_sequence()


def get_sequencers_in_use(
    devices: Mapping[DeviceName, RuntimeDevice], experiment_config: ExperimentConfig
) -> dict[DeviceName, Sequencer]:
    """Return the sequencer devices used in the experiment.

    The sequencers are sorted by trigger priority, with the highest priority first.
    """

    # Here we can't test the type of the runtime device itself because it is actually a proxy and not an instance of
    # the actual device class, that's why we need to test the type of the configuration instead.
    sequencers: dict[DeviceName, Sequencer] = {
        device_name: device
        for device_name, device in devices.items()
        if isinstance(
            experiment_config.get_device_config(device_name),
            SequencerConfiguration,
        )
    }
    sorted_by_trigger_priority = sorted(
        sequencers.items(), key=lambda x: x[1].get_trigger_priority(), reverse=True
    )
    return dict(sorted_by_trigger_priority)


def get_cameras_in_use(
    devices: Mapping[DeviceName, RuntimeDevice], experiment_config: ExperimentConfig
) -> dict[DeviceName, "Camera"]:
    return {
        device_name: device  # type: ignore
        for device_name, device in devices.items()
        if isinstance(
            experiment_config.get_device_config(device_name),
            CameraConfiguration,
        )
    }


def get_tweezer_arrangers_in_use(
    devices: Mapping[DeviceName, RuntimeDevice], experiment_config: ExperimentConfig
) -> dict[DeviceName, "AODTweezerArranger"]:
    return {
        device_name: device  # type: ignore
        for device_name, device in devices.items()
        if isinstance(
            experiment_config.get_device_config(device_name),
            AODTweezerArrangerConfiguration,
        )
    }


def update_device(device: RuntimeDevice, parameters: Mapping[DeviceParameter, Any]):
    try:
        if parameters:
            device.update_parameters(**parameters)
    except Exception as error:
        raise RuntimeError(f"Failed to update device {device.get_name()}") from error
