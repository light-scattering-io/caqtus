import numpy as np
from hypothesis import given
from hypothesis.strategies import floats, tuples, lists

# noinspection PyProtectedMember
from caqtus.device.sequencer.channel_commands._calibrated_analog_mapping import (
    Calibration,
)
from caqtus.device.sequencer.instructions import SequencerInstruction, Ramp
from .generate_pattern import pattern
from .ramp_strategy import ramp
from .instruction_strategy import analog_instruction

calibration = lists(
    tuples(
        floats(allow_nan=False, allow_infinity=False),
        floats(allow_nan=False, allow_infinity=False),
    ),
    min_size=2,
    max_size=50,
).map(Calibration)


@given(calibration, pattern(np.float64, min_length=1, max_length=100))
def test_calibration_pattern(cal: Calibration, p: SequencerInstruction[np.floating]):
    computed = cal.apply(p).to_pattern().array
    assert np.all(np.isfinite(computed))
    assert np.all(computed >= min(cal.output_values))
    assert np.all(computed <= max(cal.output_values))


def test_ramp_0():
    cal = Calibration([(0, 0), (1, 2)])
    ramp = Ramp(0, 1, 10)
    result = cal.apply(ramp)
    assert result == Ramp(0, 2, 10)


def validate_ramp(cal: Calibration, instr: Ramp):
    computed = cal.apply(instr).to_pattern()
    expected = cal.apply(instr.to_pattern())
    assert computed == expected


@given(calibration, ramp())
def test_calibration_ramp(cal: Calibration, instr: Ramp):
    validate_ramp(cal, instr)


def test_ramp_1():
    validate_ramp(
        cal=Calibration([(0.0, 0.0), (0.0, 0.0)]),
        instr=Ramp(start=1.0, stop=2.0, length=1),
    )


@given(calibration, analog_instruction(max_leaves=5))
def test_calibration_apply(cal: Calibration, instr: SequencerInstruction[np.floating]):
    computed = cal.apply(instr).to_pattern()
    expected = cal.apply(instr.to_pattern())
    assert computed == expected
