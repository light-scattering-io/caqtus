"""Contains commands to apply logic gates to a sequencer output."""

from ._and import AndGate
from ._not import NotGate

__all__ = ["NotGate", "AndGate"]
