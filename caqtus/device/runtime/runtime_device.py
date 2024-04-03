import abc
from contextlib import ExitStack, AbstractContextManager
from typing import ClassVar, Optional, TypeVar, Self

import attrs

from .device import Device
from ..name import DeviceName

_T = TypeVar("_T")


@attrs.define
class RuntimeDevice(Device, abc.ABC):
    """An implementation of the Device class that provides some useful operations.

    Class inheriting from RuntimeDevice can use the methods `_add_closing_callback` and
    `_enter_context` to facilitate managing resources.

    Fields:
        name: A unique name given to the device. Cannot be changed during the lifetime
        of the device.

    Warnings:
        All device classes subclassed from this class should call parent_item class
        initialize and close methods when overwriting them.
    """

    name: DeviceName = attrs.field(on_setattr=attrs.setters.frozen)

    _close_stack: Optional[ExitStack] = attrs.field(init=False, default=None)
    __devices_already_in_use: ClassVar[dict[DeviceName, "RuntimeDevice"]] = {}

    @name.validator  # type: ignore
    def _validate_name(self, _, value):
        if not isinstance(value, str):
            raise TypeError(f"Expected DeviceName, got {type(value)}")

    @abc.abstractmethod
    def initialize(self) -> None:
        """Initiate the communication to the device.

        This method is meant to be reimplemented for each specific device.
        The base class implementation registers the device in the list of devices
        already in use.
        It must be called when subclassing this class.
        """

        self._close_stack = ExitStack()

        if self.name in self.__devices_already_in_use:
            raise ValueError(f"A device with name {self.name} is already in use")
        self.__devices_already_in_use[self.name] = self
        self._add_closing_callback(self.__devices_already_in_use.pop, self.name)

    def _add_closing_callback(self, callback, /, *args, **kwargs):
        """Add a callback function to be called when the device is closed.

        Callbacks will be called in the reverse order they were added.
        The callback is called with the same arguments as passed to this method.
        """

        if self._close_stack is None:
            raise UninitializedDeviceError(
                f"Method RuntimeDevice.initialize must be called on the instance "
                f"before adding shutdown callbacks."
            )
        self._close_stack.callback(callback, *args, **kwargs)

    def _enter_context(self, cm: AbstractContextManager[_T]) -> _T:
        """Enter a context manager to be closed when the device is closed."""

        if self._close_stack is None:
            raise UninitializedDeviceError(
                f"Method RuntimeDevice.initialize must be called on the instance "
                f"before entering context managers."
            )
        return self._close_stack.enter_context(cm)

    def close(self) -> None:
        """Close the communication to the device and free the resources used.

        This method must be called once when use of the device is finished.
        The base class implementation unwinds the stack of closing callbacks that where
        registered when it is called.
        If you only use `_enter_context` and `_add_closing_callback`, there is no need
        to reimplement this method in subclasses.
        """

        if self._close_stack is None:
            raise UninitializedDeviceError(
                f"method initialize of RuntimeDevice must be called before calling "
                f"close."
            )
        self._close_stack.close()
        self._close_stack = None

    def __enter__(self) -> Self:
        """Initialize the device.

        When entering the device as a context manager, it will try to acquire necessary
        resources by calling `initialize`.
        If an error occurs while calling initialization, the `close` method will be
        called.
        Typically, subclasses only need to reimplement `initialize` using only
        `_enter_context` and `_add_closing_callback`.
        This will ensure proper cleanup.
        """

        try:
            self.initialize()
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_name(self) -> DeviceName:
        return self.name


class UninitializedDeviceError(Exception):
    """Raised when a device is used before being initialized"""

    pass
