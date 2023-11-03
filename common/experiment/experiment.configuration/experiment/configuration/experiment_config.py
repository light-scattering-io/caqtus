import logging
from collections.abc import Iterable
from typing import Optional, Type

from pydantic import Field, validator
from pydantic.color import Color

from analog_lane.configuration import AnalogLane
from atom_detector.configuration import AtomDetectorConfiguration
from atom_detector_lane.configuration import AtomDetectorLane
from camera.configuration import CameraConfiguration
from camera_lane.configuration import CameraLane
from device.configuration import (
    DeviceName,
    DeviceConfiguration,
    DeviceConfigurationAttrs,
    DeviceConfigType,
)
from device_server.name import DeviceServerName
from digital_lane.configuration import DigitalLane
from lane.configuration import Lane
from sequence.configuration import SequenceSteps
from sequencer.configuration import (
    ChannelName,
    SequencerConfiguration,
    ChannelSpecialPurpose,
    AnalogMapping,
    DigitalChannelConfiguration,
    AnalogChannelConfiguration,
)
from settings_model import VersionedSettingsModel, Version
from tweezer_arranger.configuration import TweezerArrangerConfiguration
from tweezer_arranger_lane.configuration import TweezerArrangerLane
from validate_arguments import validate_arguments
from .device_server_config import DeviceServerConfiguration
from .optimization_config import OptimizerConfiguration

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


class ExperimentConfig(VersionedSettingsModel):
    """Holds static configuration of the experiment.

    This configuration is used to instantiate the devices and to run the experiment. It
    contains information about the machine that should change rarely (not at each
    sequence).

    Fields:
        device_servers: The configurations of the servers that will actually instantiate
            devices.
        header: Steps that are always executed before a sequence. At the moment, it is
            only used to pre-define constant before running the sequences.
        device_configurations: All the static configurations of the devices present on
            the experiment.
        optimization_configurations: Possible configurations to choose from when running
            an optimization loop.
        mock_experiment: If True, the experiment will not run the real hardware. It will
         not connect to the device servers but will still compute all devices parameters
         if possible. Parameters will be saved and random images will be generated, but
         there will be no actual data acquisition. This is meant to be used for testing.
    """

    __version__ = "1.0.0"

    device_servers: dict[DeviceServerName, DeviceServerConfiguration] = Field(
        default_factory=dict,
    )

    header: SequenceSteps = Field(
        default_factory=SequenceSteps,
    )

    device_configurations: dict[
        DeviceName, DeviceConfiguration | DeviceConfigurationAttrs
    ] = Field(default_factory=dict)

    optimization_configurations: dict[str, OptimizerConfiguration] = Field(
        default_factory=dict,
    )

    mock_experiment: bool = False

    @classmethod
    def update_parameters_version(cls, config: dict) -> dict:
        if "version" not in config:
            config["version"] = Version(major=1, minor=0, patch=0)
        return config

    @validator("device_configurations")
    def validate_device_configurations(
        cls, device_configurations: dict[DeviceName, DeviceConfiguration]
    ):
        channel_names: set[ChannelName] = set()
        for device_name, device_configuration in device_configurations.items():
            if isinstance(device_configuration, SequencerConfiguration):
                device_channels = device_configuration.get_lane_channels()
                device_channel_names = {
                    str(channel.description) for channel in device_channels
                }
                if channel_names.isdisjoint(device_channel_names):
                    channel_names |= device_channel_names
                else:
                    raise ValueError(
                        f"RuntimeDevice '{device_name}' has channel names that are"
                        " already used by an other device:"
                        f" {channel_names & device_channel_names}"
                    )
        return device_configurations

    def get_color(
        self, channel: ChannelName | ChannelSpecialPurpose
    ) -> Optional[Color]:
        color = None
        channel_exists = False
        for device_config in self.device_configurations.values():
            if isinstance(device_config, SequencerConfiguration):
                try:
                    index = device_config.get_channel_index(channel)
                    channel_exists = True
                    color = device_config.channels[index].color
                    break
                except KeyError:
                    pass
        if channel_exists:
            return color
        else:
            raise ValueError(f"Channel {channel} doesn't exists in the configuration")

    def get_input_units(self, channel: ChannelName) -> Optional[str]:
        units = None
        channel_exists = False
        for device_config in self.device_configurations.values():
            if isinstance(device_config, SequencerConfiguration):
                try:
                    index = device_config.get_channel_index(channel)
                    channel_exists = True
                except KeyError:
                    pass
                else:
                    mapping = device_config.channels[index].output_mapping
                    if isinstance(mapping, AnalogMapping):
                        units = mapping.get_input_units()
                        break
                    else:
                        raise ValueError(
                            f"Channel {channel} has no defined units mapping"
                        )
        if channel_exists:
            return units
        else:
            raise ValueError(f"Channel {channel} doesn't exists in the configuration")

    def get_available_lane_names(self, lane_type: Type[Lane]) -> set[str]:
        lanes = set()

        for device_name, device_config in self.device_configurations.items():
            if isinstance(device_config, SequencerConfiguration):
                for channel in device_config.channels:
                    if not channel.has_special_purpose():
                        if lane_type == DigitalLane and isinstance(
                            channel, DigitalChannelConfiguration
                        ):
                            lanes.add(str(channel.description))
                        elif lane_type == AnalogLane and isinstance(
                            channel, AnalogChannelConfiguration
                        ):
                            lanes.add(str(channel.description))
            elif lane_type == CameraLane and isinstance(
                device_config, CameraConfiguration
            ):
                lanes.add(device_name)
            elif lane_type == TweezerArrangerLane and isinstance(
                device_config, TweezerArrangerConfiguration
            ):
                lanes.add(device_name)
            elif lane_type == AtomDetectorLane and isinstance(
                device_config, AtomDetectorConfiguration
            ):
                lanes.add(device_name)
        return lanes

    def get_device_names(self) -> Iterable[DeviceName]:
        return iter(self.device_configurations.keys())

    def get_device_configs(
        self, config_type: Type[DeviceConfigType]
    ) -> dict[DeviceName, DeviceConfigType]:
        """Return a dictionary of all device configurations matching a given type."""

        return {
            device_name: config
            for device_name, config in self.device_configurations.items()
            if isinstance(config, config_type)
        }

    def get_device_config(self, device_name: DeviceName) -> DeviceConfiguration:
        """Return a copy of the configuration of a given device.

        Args:
            device_name: The name of the device to get the configuration for.
        Returns:
            A copy of the device configuration. Changing the returned device
            configuration will not affect the experiment configuration.
        Raises:
            DeviceConfigNotFoundError: If there is no device configuration with this
            name.
        """

        try:
            config = self.device_configurations[device_name]
        except KeyError:
            raise DeviceConfigNotFoundError(
                f"Could not find a device named {device_name}"
            )
        return config.copy(deep=True)

    def set_device_config(self, device_name: DeviceName, config: DeviceConfiguration):
        """Change a device configuration in the experiment configuration.

        Args:
            device_name: The name of the device to change the configuration for.
            config: The new configuration for the device. A copy of this object is made
                and stored in the experiment configuration.
        Raises:
            TypeError: If config is not an instance of <DeviceConfiguration>.
            DeviceConfigNotFoundError: If there is no device configuration with this
                name.
        """

        if not isinstance(config, DeviceConfiguration):
            raise TypeError(
                "config must be an instance of <DeviceConfiguration>, got"
                f" {type(config)}"
            )

        if device_name not in self.device_configurations:
            raise DeviceConfigNotFoundError(
                f"Could not find a device named '{device_name}'"
            )

        self.device_configurations[device_name] = config.copy(deep=True)

    @validate_arguments
    def add_device_config(self, name: DeviceName, config: DeviceConfiguration):
        """Add a new device configuration to the experiment configuration.

        Raises:
            ValueError: If a device configuration with the same name already exists.
        """

        if name in self.device_configurations:
            raise ValueError(f"Device name '{name}' is already being used")

        self.device_configurations[name] = config.copy(deep=True)

    def get_optimizer_config(self, optimizer_name: str) -> OptimizerConfiguration:
        try:
            return self.optimization_configurations[optimizer_name]
        except KeyError:
            raise ValueError(
                f"Could not find an optimizer configuration named {optimizer_name}"
            )

    def get_device_runtime_type(self, device_name: DeviceName) -> str:
        """Return the runtime type of device."""

        device_config = self.get_device_config(device_name)
        device_type = device_config.get_device_type()
        return device_type

    def get_device_server_names(self) -> Iterable[str]:
        """Return the names of all device servers registered in the configuration."""

        return list(self.device_servers.keys())


class DeviceConfigNotFoundError(RuntimeError):
    pass
