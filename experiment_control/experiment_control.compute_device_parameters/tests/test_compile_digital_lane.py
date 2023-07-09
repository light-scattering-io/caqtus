from experiment_control.compute_device_parameters import (
    compile_step_durations,
    compile_digital_lane,
)
from sequence.configuration import ShotConfiguration, DigitalLane
from sequencer.channel import Concatenate, Repeat, ChannelPattern
from variable.namespace import VariableNamespace


def test_digital_lane_compilation(
    shot_config: ShotConfiguration, variables: VariableNamespace
) -> None:
    durations = compile_step_durations(
        step_durations=shot_config.step_durations,
        step_names=shot_config.step_names,
        variables=variables,
    )

    time_step = 50e-9

    lane = shot_config.find_lane("421 cell (AOM)")
    assert isinstance(lane, DigitalLane)
    instruction = compile_digital_lane(durations, lane, time_step)
    assert len(instruction) == int(sum(durations) / time_step)
    result = Concatenate(
        (
            Repeat(ChannelPattern((True,)), 200000),
            Repeat(ChannelPattern((False,)), 1600000),
            Repeat(ChannelPattern((True,)), 4620000),
            Repeat(ChannelPattern((True,)), 160000),
            Repeat(ChannelPattern((True,)), 3),
            Repeat(ChannelPattern((True,)), 20),
            Repeat(ChannelPattern((True,)), 10),
            Repeat(ChannelPattern((True,)), 3),
            Repeat(ChannelPattern((True,)), 100000),
            Repeat(ChannelPattern((True,)), 600000),
            Repeat(ChannelPattern((False,)), 20000),
            Repeat(ChannelPattern((True,)), 200000),
        )
    )
    assert instruction == result
