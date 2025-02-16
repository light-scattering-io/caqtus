import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import composite, integers

from caqtus.shot_compilation.timed_instructions import (
    Pattern,
    Concatenated,
    Repeated,
    merge_instructions,
)
from .generate_concatenate import generate_concatenate
from .generate_repeat import generate_repeat


@composite
def interval(draw, length: int) -> tuple[int, int]:
    start = draw(integers(min_value=0, max_value=length))
    stop = draw(integers(min_value=start, max_value=length))
    return start, stop


@composite
def concatenation_and_interval(draw) -> tuple[Concatenated, tuple[int, int]]:
    length = draw(integers(min_value=2, max_value=100))
    instr = draw(generate_concatenate(length))
    s = draw(interval(length))
    return instr, s


@composite
def two_concatenations(draw) -> tuple[Concatenated, Concatenated]:
    length = draw(integers(min_value=2, max_value=100))
    instr1 = draw(generate_concatenate(length))
    instr2 = draw(generate_concatenate(length))
    return instr1, instr2


@composite
def draw_concatenation_and_repeat(
    draw,
) -> tuple[Concatenated, Pattern]:
    repeat = draw(generate_repeat(100, 100))
    concatenation = draw(generate_concatenate(len(repeat)))
    return concatenation, repeat


@given(two_concatenations())
def test_merge(args):
    instr1 = args[0]
    instr2 = args[1]
    merged = merge_instructions(f0=instr1, f1=instr2)
    assert merged["f0"].to_pattern() == instr1.to_pattern()
    assert merged["f1"].to_pattern() == instr2.to_pattern()


@given(draw_concatenation_and_repeat())
def test_merge_2(args):
    concatenation = args[0]
    repeat = args[1]
    merged = merge_instructions(f0=concatenation, f1=repeat)
    assert merged["f0"].to_pattern() == concatenation.to_pattern()
    assert merged["f1"].to_pattern() == repeat.to_pattern()


def test_merge_3():
    # We need a large number of repetitions to trigger the bug.
    # Merging was hitting a recursion limit.
    instr1 = Pattern([0]) + 2000 * Pattern([1])
    instr2 = Pattern([2]) + 2000 * Pattern([3])
    merged = merge_instructions(f0=instr1, f1=instr2)
    assert merged["f0"].to_pattern() == instr1.to_pattern()
    assert merged["f1"].to_pattern() == instr2.to_pattern()
    assert merged.depth == 2


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


def test_slicing_4():
    instr = Pattern([0]) + 3 * Pattern([1])
    assert instr[1:] == Repeated(3, Pattern([1])), str(instr[1:])


def test():
    dtype = np.dtype([("a", np.int32)])
    a = Pattern([1, 2, 3], dtype=dtype)
    b = Pattern([4, 5, 6])
    with pytest.raises(TypeError):
        a + b  # pyright: ignore[reportUnusedExpression]
