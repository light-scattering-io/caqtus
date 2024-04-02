from typing import runtime_checkable, Protocol, Self, ParamSpec

from ..name import DeviceName
from ...types.data import Data, DataLabel

P = ParamSpec("P")


@runtime_checkable
class Device(Protocol[P]):
    """Defines the interface that a device must satisfy.

    This is a runtime checkable protocol so that we can test at runtime is an object has
    all the required methods and implement this interface, even if it not a direct
    subclass of Device.
    """

    def get_name(self) -> DeviceName:
        """A unique name given to the device.

        It is used to identify the device in the experiment.
        This name must remain constant during the lifetime of the device.
        """

        ...

    def __str__(self) -> str:
        return self.get_name()

    def __enter__(self) -> Self:
        """Initiate the communication to the device.

        Starts the device and acquire the necessary resources.
        """

        ...

    def update_parameters(self, *args: P.args, **kwargs: P.kwargs) -> None:
        """Apply new values for some parameters of the device.

        This method is meant to be reimplemented for each specific device.
        It can be called as many times as needed.
        """

        ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Shutdown the device.

        Used to terminate communication to the device and free the associated resources.
        """

        ...


@runtime_checkable
class AcquisitionDevice(Protocol):
    """Defines the interface that a device must satisfy to provide data."""

    def get_data(self) -> dict[DataLabel, Data]:
        """Return the data produced by the device."""

        ...
