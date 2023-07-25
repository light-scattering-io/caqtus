import datetime
from typing import Any, TypedDict, NewType

from configuration_holder import ConfigurationHolder
from device.configuration import DeviceConfiguration, DeviceParameter
from single_atom_detector import SingleAtomDetector
from .atom_label import AtomLabel

ImagingConfigurationName = NewType("ImagingConfigurationName", str)

DetectorConfiguration = dict[AtomLabel, SingleAtomDetector]


class DetectorConfigurationInfo(TypedDict):
    configuration: dict[AtomLabel, SingleAtomDetector]
    modification_date: datetime.datetime


class AtomDetectorConfiguration(
    DeviceConfiguration, ConfigurationHolder[ImagingConfigurationName, DetectorConfiguration]
):
    """Holds the information needed to initialize an AtomDetector device."""

    def get_device_init_args(
        self, configuration_name: ImagingConfigurationName
    ) -> dict[DeviceParameter, Any]:
        return super().get_device_init_args() | {
            "single_atom_detectors": self[configuration_name]
        }

    def get_device_type(self) -> str:
        return "AtomDetector"

    def remove_configuration(self, configuration_name: ImagingConfigurationName):
        """Remove a configuration from the configuration dictionary."""

        del self.detector_configurations[configuration_name]
