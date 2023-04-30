import copy
import ctypes
import logging
import math
from enum import Enum
from typing import ClassVar

import numpy as np
from pydantic import Field, validator

from device import RuntimeDevice
from settings_model import SettingsModel
from spectum_awg_m4i66xx_x8.configuration import ChannelSettings
from .pyspcm import pyspcm as spcm
from .pyspcm.spcm_tools import pvAllocMemPageAligned
from .pyspcm.py_header import spcerr
from .pyspcm.py_header.regs import ERRORTEXTLEN

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

AMPLITUDE_REGISTERS = (
    spcm.SPC_AMP0,
    spcm.SPC_AMP1,
    spcm.SPC_AMP2,
    spcm.SPC_AMP3,
)


class SpectrumAWGM4i66xxX8(RuntimeDevice):
    """Class to control the Spectrum M4i.66xx.x8 AWG

    Only sequence mode is implemented.
    """

    NUMBER_CHANNELS: ClassVar[int] = 2

    board_id: str = Field(
        description="An identifier to find the board. ex: /dev/spcm0",
        allow_mutation=False,
    )
    sampling_rate: int = Field(allow_mutation=False, units="Hz")
    channel_settings: tuple["ChannelSettings", ...] = Field(
        description="The configuration of the output channels", allow_mutation=False
    )
    segment_names: frozenset[str] = Field(
        description="The names of the segments to split the AWG memory into",
        allow_mutation=False,
    )

    first_step: str = Field(allow_mutation=False)

    _board_handle: spcm.drv_handle
    _segment_indices: dict[str, int]
    _steps: dict[str, "StepConfiguration"]
    _step_indices: dict[str, int]
    _step_names: dict[int, str]
    _bytes_per_sample: int

    def __init__(
        self,
        name: str,
        board_id: str,
        sampling_rate: int,
        channel_settings: tuple["ChannelSettings", ...],
        segment_names: frozenset[str],
        steps: dict[str, "StepConfiguration"],
        first_step: str,
    ):
        super().__init__(
            name=name,
            board_id=board_id,
            sampling_rate=sampling_rate,
            channel_settings=channel_settings,
            segment_names=segment_names,
            steps=steps,
            first_step=first_step,
        )
        self._steps = copy.deepcopy(steps)
        self._segment_indices = {
            name: index for index, name in enumerate(self.segment_names)
        }
        self._step_indices = {name: index for index, name in enumerate(self._steps)}
        self._step_names = {index: name for name, index in self._step_indices.items()}

    @validator("channel_settings")
    def validate_channel_settings(cls, channel_settings):
        if len(channel_settings) != cls.NUMBER_CHANNELS:
            raise ValueError(
                f"Expected {cls.NUMBER_CHANNELS} channel settings, but got {len(channel_settings)}"
            )
        return channel_settings

    def initialize(self) -> None:
        super().initialize()
        self._board_handle = spcm.spcm_hOpen(self.board_id)
        if not self._board_handle:
            raise RuntimeError(f"Could not find {self.board_id} board")

        spcm.spcm_dwSetParam_i64(
            self._board_handle, spcm.SPC_M2CMD, spcm.M2CMD_CARD_RESET
        )
        bytes_per_sample = ctypes.c_int64(-1)
        spcm.spcm_dwGetParam_i64(
            self._board_handle,
            spcm.SPC_MIINST_BYTESPERSAMPLE,
            ctypes.byref(bytes_per_sample),
        )
        self._bytes_per_sample = bytes_per_sample.value
        self.check_error()

        self._setup_channels()
        self._set_sampling_rate(self.sampling_rate)
        self._setup_card_mode()
        self._setup_sequence()
        self._setup_trigger()
        self._enable_output()

    def _setup_channels(self):
        enable = 0
        if self.channel_settings[0].enabled:
            enable |= spcm.CHANNEL0
        if self.channel_settings[1].enabled:
            enable |= spcm.CHANNEL1

        spcm.spcm_dwSetParam_i64(self._board_handle, spcm.SPC_CHENABLE, enable)

        for channel in range(self.NUMBER_CHANNELS):
            channel_name = self.channel_settings[channel].name
            if self.channel_settings[channel].enabled:
                amplitude = int(self.channel_settings[channel].amplitude * 1e3)
                spcm.spcm_dwSetParam_i64(
                    self._board_handle, AMPLITUDE_REGISTERS[channel], amplitude
                )

                set_amplitude = ctypes.c_int64(-1)
                spcm.spcm_dwGetParam_i64(
                    self._board_handle,
                    AMPLITUDE_REGISTERS[channel],
                    ctypes.byref(set_amplitude),
                )
                if set_amplitude.value != amplitude:
                    raise RuntimeError(
                        f"Could not set amplitude of channel {channel_name} to {amplitude} mV"
                    )
                logger.debug(
                    f"Channel {channel_name} amplitude: {set_amplitude.value} mV"
                )
        self.check_error()

    def _set_sampling_rate(self, sampling_rate):
        spcm.spcm_dwSetParam_i64(self._board_handle, spcm.SPC_SAMPLERATE, sampling_rate)

        set_sampling_rate = ctypes.c_int64(-1)
        spcm.spcm_dwGetParam_i64(
            self._board_handle, spcm.SPC_SAMPLERATE, ctypes.byref(set_sampling_rate)
        )

        if set_sampling_rate.value != sampling_rate:
            raise RuntimeError(f"Could not set sampling rate to {sampling_rate} Hz")
        self.check_error()

    def _setup_card_mode(self):
        spcm.spcm_dwSetParam_i64(
            self._board_handle, spcm.SPC_CARDMODE, spcm.SPC_REP_STD_SEQUENCE
        )

        # The card memory can only be divided by a power of two, so we round up to the next power of two
        number_actual_segments = 2 ** math.ceil(math.log2(len(self.segment_names)))
        if number_actual_segments < 2:
            number_actual_segments = 2
        spcm.spcm_dwSetParam_i64(
            self._board_handle, spcm.SPC_SEQMODE_MAXSEGMENTS, number_actual_segments
        )
        self.check_error()

    def _setup_sequence(self):
        for step_name, step_config in self._steps.items():
            self._setup_step(step_name, step_config)

        self._set_first_step(self.first_step)

    def setup_step(self, step_name: str, config: "StepConfiguration") -> None:
        self._setup_step(step_name, config)
        self._steps[step_name] = config

    def _setup_step(self, step_name: str, config: "StepConfiguration"):
        step_index = self._step_indices[step_name]
        segment_index = self._segment_indices[config.segment]
        next_step_index = self._step_indices[config.next_step]

        assert segment_index <= spcm.SPCSEQ_SEGMENTMASK
        assert (next_step_index << 16) <= spcm.SPCSEQ_NEXTSTEPMASK
        mask_lower = segment_index | (next_step_index << 16)

        assert config.repetition <= spcm.SPCSEQ_LOOPMASK
        mask_upper = config.repetition | config.change_condition.value
        logger.debug(config.change_condition.value)

        mask = mask_lower | (mask_upper << 32)

        logger.debug(f"{step_name=}")
        logger.debug(f"{step_index=}")
        logger.debug(f"{mask & spcm.SPCSEQ_NEXTSTEPMASK=}")

        spcm.spcm_dwSetParam_i64(
            self._board_handle,
            spcm.SPC_SEQMODE_STEPMEM0 + step_index,
            mask,
        )
        self.check_error()

    def _set_first_step(self, first_step: str):
        first_step_index = self._step_indices[first_step]

        spcm.spcm_dwSetParam_i64(
            self._board_handle, spcm.SPC_SEQMODE_STARTSTEP, first_step_index
        )

    def _setup_trigger(self):
        spcm.spcm_dwSetParam_i64(
            self._board_handle, spcm.SPC_TRIG_ORMASK, spcm.SPC_TMASK_SOFTWARE
        )
        self.check_error()

    def _enable_output(self):
        if self.channel_settings[0].enabled:
            spcm.spcm_dwSetParam_i64(self._board_handle, spcm.SPC_ENABLEOUT0, 1)

        if self.channel_settings[1].enabled:
            spcm.spcm_dwSetParam_i64(self._board_handle, spcm.SPC_ENABLEOUT1, 1)

        self.check_error()

    def write_segment_data(
        self,
        segment_name: str,
        data: np.ndarray[("NUMBER_CHANNELS", "number_samples"), np.int16],
    ):
        data = np.array(data, dtype=np.int16)
        if data.shape[0] != self.number_channels_enabled:
            raise ValueError(
                f"Expected values for {self.number_channels_enabled} channels, but got {data.shape[0]=}"
            )

        if data.shape[1] % 32 != 0:
            raise ValueError(
                f"Expected number of samples to be a multiple of 32, but got {data.shape[1]=}"
            )

        for channel in range(self.number_channels_enabled):
            channel_settings = self.channel_settings[channel]
            power = self._measure_mean_power(data[channel], channel_settings.amplitude)
            power_dbm = 10 * math.log10(power / 1e-3) if power > 0 else -np.inf
            logger.info(
                f"Channel {channel_settings.name} power for segment {segment_name}: {power_dbm:.2f} dBm"
            )
            if power_dbm > channel_settings.maximum_power:
                raise ValueError(
                    f"Power of {power_dbm:.2f} dBm exceeds maximum of "
                    f"{channel_settings.maximum_power:.2f} dBm for channel "
                    f"{channel_settings.name}"
                )

        segment_index = self._get_segment_index(segment_name)
        self._write_segment_data(segment_index, data)

    def _get_segment_index(self, segment_name: str) -> int:
        try:
            return self._segment_indices[segment_name]
        except KeyError:
            raise ValueError(f"There is no segment named {segment_name}")

    @staticmethod
    def _measure_mean_power(data: np.ndarray[np.int16], amplitude: float):
        voltages = data * amplitude / (2**15 - 1)

        output_load = 50  # Ohms

        mean_power = np.mean(voltages**2 / output_load)
        return mean_power

    def _write_segment_data(
        self,
        segment_index: int,
        data: np.ndarray[("NUMBER_CHANNELS", "number_samples"), np.int16],
    ):
        spcm.spcm_dwSetParam_i64(
            self._board_handle, spcm.SPC_SEQMODE_WRITESEGMENT, segment_index
        )
        spcm.spcm_dwSetParam_i64(
            self._board_handle, spcm.SPC_SEQMODE_SEGMENTSIZE, data.shape[1]
        )
        self.check_error()

        flattened_data = np.dstack(tuple(data)).flatten(order="C")
        self._transfer_data(flattened_data)

    def _transfer_data(self, data: np.ndarray[np.int16]):
        data_length_bytes = len(data) * self._bytes_per_sample

        buffer = pvAllocMemPageAligned(data_length_bytes)
        ctypes.memmove(buffer, data.ctypes.data, data_length_bytes)

        spcm.spcm_dwDefTransfer_i64(
            self._board_handle,
            spcm.SPCM_BUF_DATA,
            spcm.SPCM_DIR_PCTOCARD,
            0,
            buffer,
            0,
            data_length_bytes,
        )
        spcm.spcm_dwSetParam_i64(
            self._board_handle,
            spcm.SPC_M2CMD,
            spcm.M2CMD_DATA_STARTDMA | spcm.M2CMD_DATA_WAITDMA,
        )
        self.check_error()

        spcm.spcm_dwSetParam_i64(
            self._board_handle, spcm.SPC_M2CMD, spcm.M2CMD_DATA_STOPDMA
        )
        spcm.spcm_dwInvalidateBuf(self._board_handle, spcm.SPCM_BUF_DATA)
        self.check_error()

    def _get_segment_size(self, segment_index: int) -> int:
        """Return the number of samples in the segment."""

        spcm.spcm_dwSetParam_i64(
            self._board_handle, spcm.SPC_SEQMODE_WRITESEGMENT, segment_index
        )

        segment_size = ctypes.c_int64(-1)
        spcm.spcm_dwGetParam_i64(
            self._board_handle, spcm.SPC_SEQMODE_SEGMENTSIZE, ctypes.byref(segment_size)
        )
        self.check_error()
        return segment_size.value

    def run(self):
        spcm.spcm_dwSetParam_i64(self._board_handle, spcm.SPC_TIMEOUT, 0)
        spcm.spcm_dwSetParam_i64(
            self._board_handle,
            spcm.SPC_M2CMD,
            spcm.M2CMD_CARD_START | spcm.M2CMD_CARD_ENABLETRIGGER,
        )
        self.check_error()

    def get_current_step(self) -> str:
        step_index = ctypes.c_int64(-1)
        spcm.spcm_dwGetParam_i64(
            self._board_handle, spcm.SPC_SEQMODE_STATUS, ctypes.byref(step_index)
        )
        self.check_error()
        return self._step_names[step_index.value]

    def stop(self):
        spcm.spcm_dwSetParam_i64(
            self._board_handle, spcm.SPC_M2CMD, spcm.M2CMD_CARD_STOP
        )
        self.check_error()

    def close(self):
        try:
            spcm.spcm_vClose(self._board_handle)
        except Exception as error:
            raise error
        finally:
            super().close()

    def check_error(self):
        buffer = ctypes.create_string_buffer(ERRORTEXTLEN)
        if (
            spcm.spcm_dwGetErrorInfo_i32(self._board_handle, None, None, buffer)
            != spcerr.ERR_OK
        ):
            error_message = bytes(buffer.value).decode("utf-8")
            raise RuntimeError(
                f"An error occurred when programming the board {self.board_id}\n{error_message}"
            )

    def get_maximum_number_segments(self) -> int:
        result = ctypes.c_int64(-1)
        spcm.spcm_dwGetParam_i64(
            self._board_handle, spcm.SPC_SEQMODE_AVAILMAXSEGMENT, ctypes.byref(result)
        )
        self.check_error()
        return result.value

    @property
    def number_channels_enabled(self):
        return sum(channel.enabled for channel in self.channel_settings)


class StepChangeCondition(Enum):
    ALWAYS = 0x0
    ON_TRIGGER = 0x40000000
    END = 0x80000000


class StepConfiguration(SettingsModel):
    segment: str
    next_step: str
    repetition: int
    change_condition: StepChangeCondition = StepChangeCondition.ALWAYS


SpectrumAWGM4i66xxX8.update_forward_refs()
