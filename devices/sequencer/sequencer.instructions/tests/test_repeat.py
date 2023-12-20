import numpy as np
from hypothesis import given
from hypothesis.strategies import composite, integers

from sequencer.instructions.struct_array_instruction import Repeat, Pattern
from .generate_repeat import generate_repeat, generate_repeat_fixed_length


@composite
def interval(draw, length: int) -> tuple[int, int]:
    start = draw(integers(min_value=0, max_value=length))
    stop = draw(integers(min_value=start, max_value=length))
    return start, stop


@composite
def repeat_and_interval(draw) -> tuple[Repeat, tuple[int, int]]:
    instr = draw(generate_repeat(100, 100))
    s = draw(interval(len(instr)))
    return instr, s


@composite
def draw_two_repeat(draw, max_length: int) -> tuple[Repeat, Repeat]:
    length = draw(integers(min_value=2, max_value=max_length))
    instr1 = draw(generate_repeat_fixed_length(length))
    instr2 = draw(generate_repeat_fixed_length(length))
    return instr1, instr2


@given(draw_two_repeat(100))
def test_merge_1(args):
    repeat1 = args[0].as_type(np.dtype([("f0", np.int64)]))
    repeat2 = args[1].as_type(np.dtype([("f1", np.int64)]))
    merged = repeat1.merge_channels(repeat2)
    assert merged.get_channel("f0").to_pattern() == repeat1.to_pattern()
    assert merged.get_channel("f1").to_pattern() == repeat2.to_pattern()


def test_merge_2():
    repeat1 = 4 * Pattern([0, 1]).as_type(np.dtype([("f0", np.int64)]))
    repeat2 = 2 * Pattern([0, 1, 2, 3]).as_type(np.dtype([("f1", np.int64)]))
    merged = repeat1.merge_channels(repeat2)
    assert merged.get_channel("f0").to_pattern() == repeat1.to_pattern()
    assert merged.get_channel("f1").to_pattern() == repeat2.to_pattern()


@given(repeat_and_interval())
def test_slicing(args):
    instr, (start, stop) = args
    assert instr[start:stop].to_pattern() == instr.to_pattern()[start:stop]


def test_1():
    b = 2 * Pattern([0])
    assert b[0:1] == Pattern([0])
