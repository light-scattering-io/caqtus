from __future__ import annotations

import abc
from collections.abc import Mapping
from typing import Protocol, Any, runtime_checkable

from .compilation_contexts import ShotContext


@runtime_checkable
class DeviceCompiler(Protocol):
    """Defines the interface for a device compiler.

    Each device compiler is responsible to evaluate the parameters to apply to a device.

    Its role is to translate high-level and user-friendly parameters into concrete
    low-level device parameters that can be used to program the device.

    The compiler is initialized at the beginning of a sequence.
    Its method :meth:`compile_initialization_parameters` is then called once to
    obtain the parameters to pass to the device constructor.

    The for each shot, the method :meth:`compile_shot_parameters` is called to pass to
    the device controller for this shot.

    If it is necessary to generate a trigger for the device under consideration, the
    device compiler should inherit from
    :class:`caqtus.device.sequencer.compilation.TriggerableDeviceCompiler`.
    """

    @abc.abstractmethod
    def compile_initialization_parameters(self) -> Mapping[str, Any]:
        """Compile the parameters to pass to the device constructor."""

        return {}

    @abc.abstractmethod
    def compile_shot_parameters(self, shot_context: ShotContext) -> Mapping[str, Any]:
        """Compile the parameters to pass to the device controller for a shot.

        Args:
            shot_context: The context of the shot being compiled.
                It contains information about the shot that can be useful to evaluate
                device parameters.
        """

        return {}


class DeviceNotUsedException(Exception):  # noqa: N818
    """Raised when a device is not used in a sequence."""

    pass
