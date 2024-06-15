"""This module contains the implementation of :class:`SequencerInstruction` and its
subclasses.
These classes are used to represent sequences of values to output on several channels
in a compact form.
"""

from ._instructions import (
    SequencerInstruction,
    Concatenated,
    Repeated,
    Pattern,
    join,
    concatenate,
)
from ._ramp import ramp, Ramp
from ._stack import stack_instructions
from ._to_time_array import convert_to_change_arrays
from ._with_name import with_name

__all__ = [
    "SequencerInstruction",
    "Concatenated",
    "Repeated",
    "Pattern",
    "convert_to_change_arrays",
    "with_name",
    "stack_instructions",
    "join",
    "concatenate",
    "ramp",
    "Ramp",
]
