__version__ = "0.1.0"

from device_config.channel_config import ChannelSpecialPurpose
from imaging_source.configuration import ImagingSourceCameraDMK33GR0134Configuration
from orca_quest.configuration import OrcaQuestCameraConfiguration
from siglent_sdg6000x.configuration import SiglentSDG6000XConfiguration
from .experiment_config import (
    ExperimentConfig,
    DeviceConfigNotFoundError,
    SpincoreSequencerConfiguration,
    NI6738SequencerConfiguration,
    CameraConfiguration,
    DeviceServerConfiguration,
)
from .optimization_config import OptimizerConfiguration

device_configs = [
    SpincoreSequencerConfiguration,
    NI6738SequencerConfiguration,
    OrcaQuestCameraConfiguration,
    SiglentSDG6000XConfiguration,
    ImagingSourceCameraDMK33GR0134Configuration,
]

__all__ = [
    ChannelSpecialPurpose,
    ExperimentConfig,
    DeviceConfigNotFoundError,
    CameraConfiguration,
    DeviceServerConfiguration,
    OptimizerConfiguration,
    *device_configs,
]
