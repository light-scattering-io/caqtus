from typing import TypeVar

from hypothesis.strategies import SearchStrategy, recursive, one_of
from numpy.typing import DTypeLike

from caqtus.device.sequencer.instructions import SequencerInstruction
from .generate_concatenate import concatenation
from .generate_pattern import pattern
from .generate_repeat import repeated
from .ramp_strategy import ramp

T = TypeVar("T", bound=DTypeLike)


def instruction(dtype: T) -> SearchStrategy[SequencerInstruction[T]]:
    return recursive(
        one_of(pattern(dtype=dtype, min_length=1, max_length=100), ramp()),
        lambda s: one_of(concatenation(s), repeated(s)),
        max_leaves=10,
    )
