from __future__ import annotations

import functools
from typing import Optional, Mapping, Any

import attrs
import cattrs
import numpy as np

from caqtus.shot_compilation import ShotContext
from caqtus.shot_compilation.timing import duration_to_ticks
from caqtus.types.expression import Expression
from caqtus.types.units import Unit, Quantity
from caqtus.types.variable_name import DottedVariableName
from caqtus.utils import serialization
from .._structure_hook import structure_channel_output
from ..channel_output import ChannelOutput
from ...instructions import (
    SequencerInstruction,
    Pattern,
    Concatenated,
    concatenate,
    Repeated,
)


@attrs.define
class BroadenLeft(ChannelOutput):
    """Indicates that output should go high before the input pulses go high.

    The output y(t) of this operation should be high when any of the input x(s) is high
    for s in [t, t + width].

    The operation is only valid for boolean inputs, and it will produce a boolean
    output.

    It is meant to be used to compensate for finite rise times in the hardware.
    For example, if a shutter takes 10 ms to open, and we want to open it at time t, we
    can use this operation to start opening the shutter at time t - 10 ms.
    """

    input_: ChannelOutput = attrs.field(
        validator=attrs.validators.instance_of(ChannelOutput),
        on_setattr=attrs.setters.validate,
    )
    width: Expression = attrs.field(
        validator=attrs.validators.instance_of(Expression),
        on_setattr=attrs.setters.validate,
    )

    def evaluate(
        self,
        required_time_step: int,
        required_unit: Optional[Unit],
        prepend: int,
        append: int,
        shot_context: ShotContext,
    ) -> SequencerInstruction:
        instruction = self.input_.evaluate(
            required_time_step, required_unit, prepend, append, shot_context
        )
        if not instruction.dtype == np.bool_:
            raise TypeError(
                f"Can't broaden {self.input_} because it is not a digital instruction"
            )
        width = self.width.evaluate(shot_context.get_variables())
        if not isinstance(width, Quantity):
            raise TypeError(
                f"Can't broaden {self.input_} by {width} because it is not "
                f"a quantity"
            )
        if not width.is_compatible_with(Unit("s")):
            raise TypeError(
                f"Can't broaden {self.input_} by {width} because it is not "
                f"a time quantity"
            )
        seconds = width.to(Unit("s")).magnitude
        if seconds < 0:
            raise ValueError(
                f"Can't broaden {self.input_} by {width} because it is negative"
            )
        ticks = duration_to_ticks(seconds, required_time_step)

        broadened, bleed = _broaden_left(instruction, ticks)

        return broadened

    def evaluate_max_advance_and_delay(
        self,
        time_step: int,
        variables: Mapping[DottedVariableName, Any],
    ) -> tuple[int, int]:
        return self.input_.evaluate_max_advance_and_delay(time_step, variables)


broaden_left_structure_hook = cattrs.gen.make_dict_structure_fn(
    BroadenLeft,
    serialization.converters["json"],
    input_=cattrs.override(struct_hook=structure_channel_output),
)

serialization.register_structure_hook(BroadenLeft, broaden_left_structure_hook)


@functools.singledispatch
def _broaden_left(instruction, width: int) -> tuple[SequencerInstruction[bool], int]:
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


@_broaden_left.register
def _expand_pattern_left(instruction: Pattern, width: int):
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


@_broaden_left.register
def _expand_concatenated_left(instruction: Concatenated, width: int):
    new_instructions = []
    bleed = 0
    for sub_instruction in reversed(instruction.instructions):
        expanded, new_bleed = _broaden_left(sub_instruction, width)
        overwritten_length = min(bleed, len(expanded))
        overwritten = Pattern([True]) * overwritten_length
        new_instructions.append(overwritten)
        kept = expanded[: len(expanded) - len(overwritten)]
        new_instructions.append(kept)
        bleed -= len(expanded)
        bleed = max(new_bleed, bleed)
    return concatenate(*reversed(new_instructions)), bleed


@_broaden_left.register
def _expand_repeated_left(repeated: Repeated, width: int):
    expanded, bleed = _broaden_left(repeated.instruction, width)
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
