from caqtus.device import DeviceName
from caqtus.device.sequencer import SequencerCompiler
from caqtus.device.sequencer.timing import number_time_steps
from caqtus.shot_compilation import SequenceContext, ShotContext
from caqtus.shot_compilation.timing import to_time
from caqtus.types.iteration import StepsConfiguration
from caqtus.types.parameter import ParameterNamespace
from .fixtures import (
    spincore_config,
    swabian_configuration,
    ni6738_configuration,
    time_lanes,
    variables,
)


def test_single_digital_lane(
    time_lanes, spincore_config, swabian_configuration, ni6738_configuration, variables
):
    sequence_context = SequenceContext._new(
        device_configurations={
            DeviceName("Spincore"): spincore_config,
            DeviceName("Swabian pulse streamer"): swabian_configuration,
            DeviceName("NI6738"): ni6738_configuration,
        },
        iterations=StepsConfiguration.empty(),
        constants=ParameterNamespace.empty(),
        time_lanes=time_lanes,
    )
    compilers = {
        name: SequencerCompiler(name, sequence_context)
        for name in sequence_context.get_all_device_configurations()
    }

    shot_context = ShotContext(
        sequence_context,
        variables,
        compilers,
    )
    spincore_sequence = compilers[DeviceName("Spincore")].compile_shot_parameters(
        shot_context
    )["sequence"]
    swabian_sequence = compilers[
        DeviceName("Swabian pulse streamer")
    ].compile_shot_parameters(shot_context)["sequence"]
    ni6738_sequence = compilers[DeviceName("NI6738")].compile_shot_parameters(
        shot_context
    )["sequence"]

    shot_duration = shot_context.get_shot_duration()

    assert len(spincore_sequence) == number_time_steps(
        shot_duration, spincore_config.time_step
    )
    assert len(swabian_sequence) == number_time_steps(
        to_time(
            shot_duration + to_time(1.1e-6)
        ),  # Need to tak into account time shift when computing shot duration
        swabian_configuration.time_step,
    )
    assert len(ni6738_sequence) == number_time_steps(
        shot_duration, ni6738_configuration.time_step
    )
