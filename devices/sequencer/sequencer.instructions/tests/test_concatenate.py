import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import composite, integers

from sequencer.instructions.struct_array_instruction import Pattern, Concatenate


@composite
def flat_concatenation(draw, length: int) -> Concatenate:
    if length <= 1:
        raise ValueError("Length must be strictly greater than 1.")
    else:
        left_length = draw(integers(min_value=1, max_value=length - 1))
        right_length = length - left_length

        left = Pattern([i for i in range(left_length)])
        if right_length == 1:
            right = Pattern([left_length])
        else:
            right = draw(flat_concatenation(right_length))
        return left + right


@composite
def interval(draw, length: int) -> tuple[int, int]:
    start = draw(integers(min_value=0, max_value=length))
    stop = draw(integers(min_value=start, max_value=length))
    return start, stop


@composite
def concatenation_and_interval(draw) -> tuple[Concatenate, tuple[int, int]]:
    length = draw(integers(min_value=2, max_value=100))
    instr = draw(flat_concatenation(length))
    s = draw(interval(length))
    return instr, s


@given(concatenation_and_interval())
def test_slicing(args):
    instr, (start, stop) = args
    assert instr[start:stop].to_pattern() == instr.to_pattern()[start:stop]


def test_slicing_1():
    instr = Pattern([0, 1]) + Pattern([0])
    assert instr[1:2].to_pattern() == Pattern([1])


def test_slicing_2():
    instr = Pattern([0]) + Pattern([1])
    assert instr[0:2].to_pattern() == Pattern([0, 1])


def test_slicing_3():
    instr = Pattern([0]) + Pattern([1])
    assert instr[2:2].to_pattern() == instr.to_pattern()[2:2]


def test():
    dtype = np.dtype([("a", np.int32)])
    a = Pattern([1, 2, 3], dtype=dtype)
    b = Pattern([4, 5, 6])
    c = Pattern([7, 8, 9])
    with pytest.raises(TypeError):
        a + b
    assert (b + c + b).depth == 1
