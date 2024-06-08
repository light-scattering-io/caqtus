import functools

import numpy as np

from ..instructions import (
    SequencerInstruction,
    Pattern,
    Concatenated,
    concatenate,
    Repeated,
)


@functools.singledispatch
def broaden_left(instruction, width: int) -> tuple[SequencerInstruction[bool], int]:
    """Broaden the instruction to the left by n steps.

    Returns:
        - result: The expanded instruction, where the high values are expanded to the
            left, i.e. result[i] = any(instruction[i:i+width+1])
        - bleed: The number of steps before this instruction that must be set to True,
            i.e. bleed = max(0, width - first) where first is the index of the first
            high value in the instruction if it exists, otherwise bleed = 0.
    """

    raise NotImplementedError(
        f"Don't know how to broaden instruction of type {type(instruction)}"
    )


@broaden_left.register
def expand_pattern_left(instruction: Pattern, width: int):
    if not instruction.dtype == np.bool_:
        raise TypeError("Instruction must have dtype bool")
    pulse_length = min(len(instruction), width + 1)
    pulse = np.full(pulse_length, True)
    convolution = np.convolve(instruction.array, pulse)
    result = convolution[pulse_length - 1 :]
    high_indices = instruction.array.nonzero()[0]
    if len(high_indices) == 0:
        excess = 0
    else:
        first_high_index = int(high_indices[0])  # need to avoid numpy integers
        excess = max(0, width - first_high_index)
    return Pattern.create_without_copy(result), excess


@broaden_left.register
def expand_concatenated_left(instruction: Concatenated, width: int):
    new_instructions = []
    bleed = 0
    for sub_instruction in reversed(instruction.instructions):
        expanded, new_bleed = broaden_left(sub_instruction, width)
        overwritten_length = min(bleed, len(expanded))
        overwritten = Pattern([True]) * overwritten_length
        new_instructions.append(overwritten)
        kept = expanded[: len(expanded) - len(overwritten)]
        new_instructions.append(kept)
        bleed -= len(expanded)
        bleed = max(new_bleed, bleed)
    return concatenate(*reversed(new_instructions)), bleed


@broaden_left.register
def expand_repeated_left(repeated: Repeated, width: int):
    expanded, bleed = broaden_left(repeated.instruction, width)
    if bleed == 0:
        # This is a special were expanding the instruction has no effect on the previous
        # instructions.
        return expanded * repeated.repetitions, bleed
    if bleed >= len(expanded):
        # This is a special case where the previous instructions are completely
        # overwritten by the expanded instruction, so we can return a simplified
        # instruction.
        instr = Pattern([True]) * len(expanded) * (repeated.repetitions - 1) + expanded
        return instr, bleed
    overwritten_length = min(bleed, len(expanded))
    overwritten = Pattern([True]) * overwritten_length
    kept = expanded[: len(expanded) - len(overwritten)]
    left_instr = kept + overwritten
    if expanded == left_instr:
        return expanded * repeated.repetitions, bleed
    else:
        instr = (kept + overwritten) * (repeated.repetitions - 1) + expanded
        return instr, bleed
