from typing import TypeVar

import numpy as np
from hypothesis.extra.numpy import arrays, from_dtype
from hypothesis.strategies import integers, SearchStrategy
from numpy.typing import DTypeLike

from caqtus.device.sequencer.instructions import Pattern


def generate_pattern(length: int, offset: int = 0) -> Pattern:
    return Pattern(np.arange(offset, offset + length))


T = TypeVar("T", bound=DTypeLike)


def pattern(
    dtype: T, min_length: int = 0, max_length=1000
) -> SearchStrategy[Pattern[T]]:
    return arrays(
        dtype=dtype,
        shape=integers(min_value=min_length, max_value=max_length),
        elements=from_dtype(np.dtype(dtype), allow_infinity=False, allow_nan=False),
    ).map(Pattern)
