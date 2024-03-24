import numpy as np

from caqtus.device.sequencer.instructions import Pattern


def generate_pattern(length: int, offset: int = 0) -> Pattern:
    return Pattern(np.arange(offset, offset + length))