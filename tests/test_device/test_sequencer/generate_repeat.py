from typing import TypeVar

import numpy as np
from hypothesis.strategies import (
    composite,
    integers,
    permutations,
    SearchStrategy,
    builds,
)
from numpy.typing import DTypeLike
from sympy import factorint

from caqtus.device.sequencer.instructions import Repeated, SequencerInstruction
from .generate_pattern import generate_pattern


@composite
def generate_repeat(draw, max_repetitions: int, max_sub_instruction_length) -> Repeated:
    repetitions = draw(integers(min_value=2, max_value=max_repetitions))
    sub_instruction_length = draw(
        integers(min_value=1, max_value=max_sub_instruction_length)
    )

    instr = generate_pattern(sub_instruction_length)
    return repetitions * instr


@composite
def generate_repeat_fixed_length(draw, length: int) -> Repeated:
    a, b = draw(factorize(length))
    return generate_pattern(a) * b


@composite
def factorize(draw, number: int) -> tuple[int, int]:
    if number <= 1:
        raise ValueError("Number must be greater than 1.")
    factors = factorint(number)
    flat_factors = []
    for factor, exponent in factors.items():
        flat_factors.extend([factor] * exponent)
    if len(flat_factors) == 1:
        return 1, number
    else:
        flat_factors = draw(permutations(flat_factors))
        a_length = draw(integers(min_value=1, max_value=len(flat_factors) - 1))
        a = int(np.prod(flat_factors[:a_length]))
        b = int(np.prod(flat_factors[a_length:]))
        return min(a, b), max(a, b)


T = TypeVar("T", bound=DTypeLike)


def repeated(
    child: SearchStrategy[SequencerInstruction[T]],
) -> SearchStrategy[SequencerInstruction[T]]:
    return builds(
        to_repeat,
        repetitions=integers(min_value=2, max_value=100),
        instruction=child,
    )


def to_repeat(
    repetitions: int, instruction: SequencerInstruction
) -> SequencerInstruction:
    return repetitions * instruction
