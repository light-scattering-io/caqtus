import abc
from typing import runtime_checkable, Protocol, Self, ParamSpec

from ..name import DeviceName

UpdateParams = ParamSpec("UpdateParams")
InitParams = ParamSpec("InitParams")


@runtime_checkable
class Device(Protocol[InitParams, UpdateParams]):
    """Wraps a low-level instrument that can be controlled during an experiment.

    This abstract class defines the necessary methods that a device must implement to be
    used in an experiment.
    """

    @abc.abstractmethod
    def __init__(self, *args: InitParams.args, **kwargs: InitParams.kwargs) -> None:
        """Device constructor.

        No communication to an instrument or initialization should be done in the
        constructor.
        Instead, use the :meth:`__enter__` method to acquire the necessary resources.
        """

        raise NotImplementedError

    def get_name(self) -> DeviceName:
        """A unique name given to the device.

        It is used to identify the device in the experiment.
        This name must remain constant during the lifetime of the device.
        """

        ...

    def __str__(self) -> str:
        return self.get_name()

    @abc.abstractmethod
    def __enter__(self) -> Self:
        """Initialize the device.

        Used to establish communication to the device and allocate the necessary
        resources.
        No initialization should be done in the constructor.
        """

        raise NotImplementedError

    def update_parameters(
        self, *args: UpdateParams.args, **kwargs: UpdateParams.kwargs
    ) -> None:
        """Apply new values for some parameters of the device."""

        ...

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Shutdown the device.

        Used to terminate communication to the device and free the associated resources.
        """

        raise NotImplementedError
