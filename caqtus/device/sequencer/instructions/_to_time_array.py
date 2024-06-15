import functools

import numpy as np

from ._instructions import SequencerInstruction, Pattern, Concatenated, Repeated
from ._ramp import Ramp


def convert_to_change_arrays(
    sequence: SequencerInstruction,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a sequence to arrays of change times and values.

    Args:
        sequence: The sequence to convert.

    Returns:
        Two arrays, the first containing times indexes at which the values of the
        sequence change, and the second containing the corresponding values for each
        change.
        The time array will have dtype np.int64 and the value array will have
        the same dtype as the sequence passed in argument.
    """

    times, values = _convert_to_change_arrays(sequence)
    return np.concatenate([times, [len(sequence)]], axis=0), np.concatenate(
        [values, [values[-1]]], axis=0
    )


@functools.singledispatch
def _convert_to_change_arrays(
    sequence: SequencerInstruction,
) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError(f"Cannot convert {type(sequence)} to time arrays")


@_convert_to_change_arrays.register
def _(sequence: Pattern) -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(len(sequence), dtype=np.int64)
    values = sequence.array

    return times, values


@_convert_to_change_arrays.register
def _(sequence: Concatenated) -> tuple[np.ndarray, np.ndarray]:
    time_arrays = []
    value_arrays = []

    cumulated_start = 0

    for instruction in sequence.instructions:
        times, values = _convert_to_change_arrays(instruction)
        time_arrays.append(times + cumulated_start)
        value_arrays.append(values)
        cumulated_start += len(instruction)

    return np.concatenate(time_arrays), np.concatenate(value_arrays)


@_convert_to_change_arrays.register
def _(sequence: Repeated) -> tuple[np.ndarray, np.ndarray]:
    time_array, value_array = _convert_to_change_arrays(sequence.instruction)
    if len(time_array) == 1:
        return np.array([0], dtype=np.int64), value_array
    start_times = np.arange(sequence.repetitions, dtype=np.int64) * len(
        sequence.instruction
    )
    time_arrays = time_array + start_times[:, np.newaxis]
    return np.concatenate(time_arrays), np.tile(value_array, sequence.repetitions)


@_convert_to_change_arrays.register
def _(ramp: Ramp) -> tuple[np.ndarray, np.ndarray]:
    values = ramp.to_pattern().array
    times = np.arange(len(values), dtype=np.int64)
    return times, values
