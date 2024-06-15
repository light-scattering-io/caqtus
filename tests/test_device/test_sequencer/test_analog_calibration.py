import numpy as np
from hypothesis import given
from hypothesis.strategies import floats, tuples, lists

# noinspection PyProtectedMember
from caqtus.device.sequencer.channel_commands._calibrated_analog_mapping import (
    Calibration,
)
from caqtus.device.sequencer.instructions import SequencerInstruction
from .instruction_strategy import instruction

calibration = lists(
    tuples(
        floats(allow_nan=False, allow_infinity=False),
        floats(allow_nan=False, allow_infinity=False),
    ),
    min_size=1,
).map(Calibration)


@given(calibration, instruction(np.float64))
def test_calibration_apply(cal: Calibration, instr: SequencerInstruction[np.floating]):
    computed = cal.apply(instr).to_pattern()
    expected = cal.apply(instr.to_pattern())
    assert computed == expected
