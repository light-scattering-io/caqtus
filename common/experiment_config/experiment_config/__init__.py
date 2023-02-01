__version__ = "0.1.0"

from device_config.channel_config import ChannelSpecialPurpose
from orca_quest.configuration import OrcaQuestCameraConfiguration
from siglent_sdg6000x.configuration import SiglentSDG6000XConfiguration
from .experiment_config import (
    ExperimentConfig,
    DeviceConfigNotFoundError,
    get_config_path,
    SpincoreSequencerConfiguration,
    NI6738SequencerConfiguration,
    CameraConfiguration,
    DeviceServerConfiguration,
)

device_configs = [
    SpincoreSequencerConfiguration,
    NI6738SequencerConfiguration,
    OrcaQuestCameraConfiguration,
    SiglentSDG6000XConfiguration,
]

__all__ = [
    ChannelSpecialPurpose,
    ExperimentConfig,
    DeviceConfigNotFoundError,
    get_config_path,
    CameraConfiguration,
    DeviceServerConfiguration,
    *device_configs,
]
