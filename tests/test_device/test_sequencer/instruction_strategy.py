from typing import TypeVar

import numpy as np
from hypothesis.strategies import SearchStrategy, recursive
from numpy.typing import DTypeLike

from caqtus.device.sequencer.instructions import SequencerInstruction
from .generate_concatenate import concatenation
from .generate_pattern import pattern
from .generate_repeat import repeated
from .ramp_strategy import ramp

T = TypeVar("T", bound=DTypeLike)


def instruction(
    leaf_strategy: SearchStrategy[SequencerInstruction[T]], max_leaves: int
) -> SearchStrategy[SequencerInstruction[T]]:
    return recursive(
        leaf_strategy,
        lambda s: concatenation(s) | repeated(s),
        max_leaves=max_leaves,
    )


def digital_instruction(
    max_leaves: int = 100,
) -> SearchStrategy[SequencerInstruction[np.bool_]]:
    return instruction(pattern(dtype=np.bool_, min_length=1, max_length=10), max_leaves)


def analog_instruction(
    max_leaves: int = 20,
) -> SearchStrategy[SequencerInstruction[np.floating]]:
    return instruction(
        pattern(dtype=np.float64, min_length=1, max_length=100) | ramp(), max_leaves
    )
