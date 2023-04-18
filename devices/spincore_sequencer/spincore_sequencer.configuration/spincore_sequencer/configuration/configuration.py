from pydantic import Field

from device_config import DeviceConfiguration
from device_config.channel_config import DigitalChannelConfiguration


class SpincoreSequencerConfiguration(DeviceConfiguration, DigitalChannelConfiguration):
    """Holds the static configuration of a spincore sequencer device.

    Attributes:
        board_number: The number of the board to use. With only one board connected,
            this number is usually 0.
        time_step: The quantization time step used. All times during a run are multiples
            of this value.
    """
    # noinspection PyPropertyDefinition
    @classmethod
    @property
    def number_channels(cls) -> int:
        return 24

    board_number: int
    time_step: float = Field(
        default=50e-9,
        ge=50e-9,
        units="s",
    )

    def get_device_type(self) -> str:
        return "SpincorePulseBlaster"

    def get_device_init_args(self) -> dict[str]:
        extra = {
            "board_number": self.board_number,
            "time_step": self.time_step,
        }
        return super().get_device_init_args() | extra
