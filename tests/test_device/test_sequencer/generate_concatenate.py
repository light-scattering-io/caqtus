import numpy as np
from hypothesis.strategies import composite, integers

from caqtus.device.sequencer.instructions import Concatenated
from .generate_pattern import generate_pattern, pattern


@composite
def generate_concatenate(draw, length: int, offset: int = 0) -> Concatenated:
    if length <= 1:
        raise ValueError("Length must be strictly greater than 1.")
    else:
        left_length = draw(integers(min_value=1, max_value=length - 1))
        right_length = length - left_length

        left = generate_pattern(left_length, offset=offset)
        if right_length == 1:
            right = generate_pattern(right_length, offset=offset + left_length)
        else:
            right = draw(
                generate_concatenate(right_length, offset=offset + left_length)
            )
        return left + right


@composite
def bool_concatenation(
    draw,
    number_patterns=integers(min_value=2, max_value=10),
    bool_patterns=pattern(dtype=np.bool_, min_length=1),
) -> Concatenated[bool]:
    length = draw(number_patterns)
    patterns = [draw(bool_patterns) for _ in range(length)]
    return Concatenated(*patterns)
