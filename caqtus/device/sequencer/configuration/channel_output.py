"""This module defines the configuration used to compute the output of a sequencer
channel.

A channel can typically output a constant value, the values of a lane, a trigger for
another device, or a functional combination of these.

The union type `ChannelOutput` is used to represent the different possible outputs of a
channel.
Each possible type of output is represented by a different class.
An output class is a high-level description of what should be outputted by a channel.
The classes defined are only declarative and do not contain any logic to compute the
output.
For more information on how the output is evaluated, see
:mod:`core.compilation.sequencer_parameter_compiler`.
"""

from __future__ import annotations

import abc
from typing import Optional

import attrs

from caqtus.device.sequencer.instructions import SequencerInstruction
from caqtus.shot_compilation import ShotContext
from caqtus.types.units import Unit


@attrs.define
class ChannelOutput(abc.ABC):
    @abc.abstractmethod
    def evaluate(
        self,
        required_time_step: int,
        required_unit: Optional[Unit],
        prepend: int,
        append: int,
        shot_context: ShotContext,
    ) -> SequencerInstruction:
        raise NotImplementedError
