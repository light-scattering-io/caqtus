import logging
import time

import numpy as np

from spectum_awg_m4i66xx_x8.runtime import (
    SpectrumAWGM4i66xxX8,
    ChannelSettings,
    StepConfiguration,
    StepChangeCondition,
)
from trap_signal_generator.configuration import StaticTrapConfiguration
from trap_signal_generator.runtime import StaticTrapGenerator

logging.basicConfig()

with open("./config_x.yaml", "r") as f:
    config_x = StaticTrapConfiguration.from_yaml(f.read())

with open("./config_y.yaml", "r") as f:
    config_y = StaticTrapConfiguration.from_yaml(f.read())

amplitude_one_tone = 0.165
scale_x = np.sqrt(config_x.number_tones) * amplitude_one_tone
scale_y = np.sqrt(config_y.number_tones) * amplitude_one_tone

assert config_x.number_samples == config_y.number_samples
assert config_x.sampling_rate == config_y.sampling_rate

static_trap_generator_x = StaticTrapGenerator.from_configuration(config_x)
static_trap_generator_y = StaticTrapGenerator.from_configuration(config_y)

with SpectrumAWGM4i66xxX8(
        name="AWG",
        board_id="/dev/spcm0",
        channel_settings=(
                ChannelSettings(name="X", enabled=True, amplitude=scale_x, maximum_power=-4),
                ChannelSettings(name="Y", enabled=True, amplitude=scale_y, maximum_power=-4),
        ),
        segment_names=frozenset(["segment_0"]),
        steps={
            "step_0": StepConfiguration(
                segment="segment_0",
                next_step="step_0",
                repetition=1,
                change_condition=StepChangeCondition.ALWAYS,
            ),
        },
        first_step="step_0",
        sampling_rate=static_trap_generator_x.sampling_rate,
) as awg:
    data_0 = np.int16(
        (
            static_trap_generator_x.compute_signal(),
            static_trap_generator_y.compute_signal(),
        )
    )
    t0 = time.perf_counter()
    awg.update_parameters(segment_data={"segment_0": data_0})
    t1 = time.perf_counter()
    print(f"Time to update parameters: {(t1 - t0) * 1e3} ms")
    awg.start_sequence()
    input()
    awg.stop_sequence()
