from abc import abstractmethod, ABC
from collections.abc import Mapping
from typing import Any, TypeVar, Optional, NewType

import attrs

from .name import DeviceName
from .parameter import DeviceParameter

DeviceServerName = NewType("DeviceServerName", str)


@attrs.define
class DeviceConfigurationAttrs(ABC):
    """Handle the static experiment wide configuration of a device.

    This class is used to store the persistent configuration of a device. The
    information it contains is constant within a sequence and is not expected to
    change often between sequences. Subclasses of this class are associated with a
    given subclass of RuntimeDevice. Subclasses of this class typically should hold
    information to connect to the device, such as the IP address, port, etc.

    Attributes:
        remote_server: The name of the computer on which the device should be
        instantiated.
        If None, the device should be instantiated in the local process controlling
        the experiment.
    """

    remote_server: Optional[DeviceServerName] = attrs.field(
        converter=attrs.converters.optional(str), on_setattr=attrs.setters.convert
    )

    @abstractmethod
    def get_device_type(self) -> str:
        """Return the runtime type of the device.

        This function must be implemented by all subclasses. It is used to determine
        what device this configuration is for.

        Return:
            A string with the name of the class that will be instantiated when actually
            creating the device. This class must be a subclass of RuntimeDevice. Note
            that this return a string and not the class itself because the runtime class
            may not be accessible on the computer that is handling the experiment
            configuration.
        """

        raise NotImplementedError(
            "get_device_type must be implemented by all subclasses"
        )

    @abstractmethod
    def get_device_init_args(self, *args, **kwargs) -> dict[DeviceParameter, Any]:
        return {}


DeviceConfigType = TypeVar("DeviceConfigType", bound=DeviceConfigurationAttrs)


def get_configurations_by_type(
    device_configurations: Mapping[DeviceName, DeviceConfigurationAttrs],
    device_type: type[DeviceConfigType],
) -> dict[DeviceName, DeviceConfigType]:
    return {
        name: configuration
        for name, configuration in device_configurations.items()
        if isinstance(configuration, device_type)
    }
