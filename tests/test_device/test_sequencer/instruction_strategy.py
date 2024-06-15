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

analog = pattern(dtype=np.float64, min_length=1, max_length=100) | ramp()
digital = pattern(dtype=np.bool_, min_length=1, max_length=10)


def instruction(
    leaf_strategy: SearchStrategy[SequencerInstruction[T]],
) -> SearchStrategy[SequencerInstruction[T]]:
    return recursive(
        leaf_strategy,
        lambda s: concatenation(s) | repeated(s),
    )


digital_instruction = instruction(digital)
analog_instruction = instruction(analog)
