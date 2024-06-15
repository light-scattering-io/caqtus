from typing import TypeVar

from hypothesis.strategies import SearchStrategy, recursive, one_of
from numpy.typing import DTypeLike

from caqtus.device.sequencer.instructions import SequencerInstruction
from .generate_concatenate import concatenation
from .generate_pattern import pattern
from .generate_repeat import repeated

T = TypeVar("T", bound=DTypeLike)


def instruction(dtype: T) -> SearchStrategy[SequencerInstruction[T]]:
    return recursive(
        pattern(dtype=dtype, min_length=1, max_length=100),
        lambda s: one_of(concatenation(s), repeated(s)),
        max_leaves=10,
    )
