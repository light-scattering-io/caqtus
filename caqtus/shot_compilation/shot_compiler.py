from collections.abc import Mapping
from typing import Protocol, Any

from caqtus.device import DeviceName, DeviceConfiguration, DeviceParameter
from caqtus.session.shot import TimeLanes
from .variable_namespace import VariableNamespace


class ShotCompiler(Protocol):
    """Converts high level description of a shot into low level device parameters.

    Shot compilation is the process of evaluating expressions inside the shot
    representation and converting them into numerical values that can be understood by
    the devices.
    """

    def compile_shot(
        self, shot_parameters: VariableNamespace
    ) -> Mapping[DeviceName, Mapping[DeviceParameter, Any]]:
        ...


class ShotCompilerFactory(Protocol):
    """Creates shot compilers."""

    def __call__(
        self,
        shot_timelanes: TimeLanes,
        device_configurations: Mapping[DeviceName, DeviceConfiguration],
    ) -> ShotCompiler:
        """Create a shot compiler.

        The shot compiler returned by this function will be used to compile the shot
        represented by `shot_timelanes` using the device configurations in
        `device_configurations`.

        The keys of the mapping returned by the method :meth:`compile_shot` of the
        shot compiler created must be a subset of the keys of `device_configurations`.

        Args:
            shot_timelanes: The generic shot description to compile.
            device_configurations: The device configurations to use to compile the shot.
        """

        ...
