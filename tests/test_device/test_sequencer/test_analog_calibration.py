import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import floats, tuples, lists

# noinspection PyProtectedMember
from caqtus.device.sequencer.channel_commands._calibrated_analog_mapping import (
    Calibration,
)
from caqtus.device.sequencer.instructions import SequencerInstruction, Ramp
from .generate_pattern import pattern
from .instruction_strategy import analog_instruction
from .ramp_strategy import ramp

calibration = lists(
    tuples(
        floats(allow_nan=False, allow_infinity=False),
        floats(allow_nan=False, allow_infinity=False),
    ),
    min_size=2,
    max_size=50,
    unique_by=lambda x: x[0],
).map(Calibration)


@given(calibration, pattern(np.float64, min_length=1, max_length=100))
def test_calibration_pattern(cal: Calibration, p: SequencerInstruction[np.floating]):
    computed = cal.apply(p).to_pattern().array
    assert np.all(np.isfinite(computed))
    assert np.all(computed >= min(cal.output_values))
    assert np.all(computed <= max(cal.output_values))


@given(calibration, ramp())
def test_calibration_ramp(cal: Calibration, instr: Ramp):
    computed = cal.apply(instr).to_pattern()
    expected = cal.apply(instr.to_pattern())
    assert computed == expected


@pytest.mark.parametrize(
    "cal, instr",
    [
        (Calibration([(0, 0), (1, 2)]), Ramp(0, 1, 10)),
        (Calibration([(0, 0), (1, 2)]), Ramp(0.5, 0.5, 10)),
        (Calibration([(0.0, 0.0), (2.0, 0.0)]), Ramp(start=1.0, stop=2.0, length=1)),
        (Calibration([(0.0, 0.0), (1.0, 0.0)]), Ramp(start=1.0, stop=-1.0, length=1)),
        (Calibration([(0.0, 0.0), (1.0, 0.0)]), Ramp(start=-1.0, stop=+1.0, length=1)),
        (Calibration([(0.0, 1.0), (1.0, 0.0)]), Ramp(start=4.0, stop=0.0, length=2)),
    ],
)
def test_calibration_on_ramp(cal: Calibration, instr: Ramp):
    computed = cal.apply(instr).to_pattern()
    expected = cal.apply(instr.to_pattern())
    assert computed == expected


@given(calibration, analog_instruction(max_leaves=5))
def test_calibration_apply(cal: Calibration, instr: SequencerInstruction[np.floating]):
    computed = cal.apply(instr).to_pattern()
    expected = cal.apply(instr.to_pattern())
    assert computed == expected
