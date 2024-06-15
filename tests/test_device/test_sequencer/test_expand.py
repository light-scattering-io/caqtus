import numpy as np
from hypothesis import given
from hypothesis.strategies import integers
from numpy.typing import NDArray

from caqtus.device.sequencer.channel_commands.timing.broaden import _broaden_left
from caqtus.device.sequencer.instructions import Concatenated, Pattern
from .generate_concatenate import concatenation
from .generate_pattern import pattern
from .generate_repeat import repeated
from .instruction_strategy import digital_instruction

np.typing.NDArray = np.ndarray


@given(pattern(dtype=np.bool_, min_length=1), integers(min_value=0))
def test_pattern(p, n):
    expanded, excess = _broaden_left(p, n)
    assert len(expanded) == len(p)
    for i in range(len(expanded)):
        assert expanded.array[i] == any(p.array[i : i + n + 1])


def test_pattern_0():
    pattern = Pattern([False, True])
    expanded, excess = _broaden_left(pattern, 1)
    assert expanded == Pattern([True, True])
    assert excess == 0


@given(concatenation(pattern(np.bool_, min_length=1)), integers(min_value=0))
def test_concatenation(concatenated, n):
    expanded, excess = _broaden_left(concatenated, n)
    assert len(expanded) == len(concatenated)
    obtained = expanded.to_pattern().array
    expected = _broaden_left(concatenated.to_pattern(), n)[0].to_pattern().array
    assert np.array_equal(
        obtained, expected
    ), f"Obtained: {obtained}\nExpected: {expected}"


def test_0():
    instr = Concatenated(Pattern([False]), Pattern([False, True]))
    expanded, excess = _broaden_left(instr, 1)
    assert expanded == Pattern([False, True, True])
    assert excess == 0


@given(repeated(pattern(np.bool_, min_length=1, max_length=100)), integers(min_value=0))
def test_repeated(r, n):
    expanded, excess = _broaden_left(r, n)
    assert len(expanded) == len(r)
    obtained = expanded.to_pattern().array
    expected = _broaden_left(r.to_pattern(), n)[0].to_pattern().array
    assert np.array_equal(
        obtained, expected
    ), f"Obtained: {obtained}\nExpected: {expected}"


def test_1():
    instr = Pattern([False, True]) * 10
    expanded, excess = _broaden_left(instr, 1)
    assert expanded == Pattern([True, True]) * 10


def test_2():
    instr = Pattern([False, False, True]) * 10 + Pattern([False, True])
    expanded, excess = _broaden_left(instr, 1)
    assert excess == 0
    assert expanded == Pattern([False, True, True]) * 10 + Pattern([True, True])


def test_3():
    instr = (
        Pattern([False]) * 100_000_000
        + Pattern([True]) * 50_000_000
        + Pattern([False]) * 100_000_000
    )
    expanded, excess = _broaden_left(instr, 10_000_000)
    assert excess == 0


@given(digital_instruction(), integers(min_value=0))
def test_4(instr, n):
    expanded, excess = _broaden_left(instr, n)
    assert len(expanded) == len(instr)
    obtained = expanded.to_pattern().array
    expected = _broaden_left(instr.to_pattern(), n)[0].to_pattern().array
    assert np.array_equal(
        obtained, expected
    ), f"Obtained: {obtained}\nExpected: {expected}"
