"""This module implement the various components for sequencer devices.
Sequencers are devices that output values at regular time intervals.
"""

from . import channel_commands
from . import instructions
from ._controller import SequencerController
from ._proxy import SequencerProxy
from .compilation import SequencerCompiler
from .configuration import (
    SequencerConfiguration,
    ChannelConfiguration,
    DigitalChannelConfiguration,
    AnalogChannelConfiguration,
)
from .runtime import Sequencer
from .trigger import (
    Trigger,
    SoftwareTrigger,
    ExternalTriggerStart,
    ExternalClock,
    ExternalClockOnChange,
    TriggerEdge,
)

__all__ = [
    "SequencerConfiguration",
    "Sequencer",
    "Trigger",
    "SoftwareTrigger",
    "ExternalClock",
    "ExternalTriggerStart",
    "ExternalClockOnChange",
    "TriggerEdge",
    "ChannelConfiguration",
    "DigitalChannelConfiguration",
    "AnalogChannelConfiguration",
    "SequencerCompiler",
    "SequencerProxy",
    "SequencerController",
    "channel_commands",
    "instructions",
]
