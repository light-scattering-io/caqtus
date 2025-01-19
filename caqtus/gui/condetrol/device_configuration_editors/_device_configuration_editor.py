import abc
import copy
from typing import Optional

from PySide6.QtWidgets import QWidget, QFormLayout

import caqtus.gui.qtutil.qabc as qabc
from caqtus.device import DeviceConfiguration


class DeviceConfigurationEditor[T: DeviceConfiguration](
    QWidget, metaclass=qabc.QABCMeta
):
    """A widget that allows to edit the configuration of a device.

    This class is generic in the type of the device configuration it allows to edit.
    """

    @abc.abstractmethod
    def get_configuration(self) -> T:
        """Return a new configuration that represents what is currently displayed."""

        raise NotImplementedError

    @abc.abstractmethod
    def set_configuration(self, configuration: T) -> None:
        """Set the configuration to be displayed."""

        raise NotImplementedError


class FormDeviceConfigurationEditor[T: DeviceConfiguration](
    DeviceConfigurationEditor[T]
):
    """Displays a list of fields to edit the configuration of a device.

    Widgets of this class initially only present a single field to edit the remote
    server name.

    Other device specific fields can be added by calling the :meth:`insert_row` method.
    """

    def __init__(self, device_configuration: T, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.form = QFormLayout()
        self.device_configuration = copy.deepcopy(device_configuration)
        self.setLayout(self.form)

    def append_row(self, label: str, widget: QWidget):
        """Append a widget field at the end of the form."""

        self.form.addRow(label, widget)

    def insert_row(self, label: str, widget: QWidget, row: int):
        """Insert a widget field at the specified row."""

        self.form.insertRow(row, label, widget)

    def get_configuration(self) -> T:
        """Return a new configuration with fields updated from the UI.

        Returns:
            A copy of the configuration that was passed to the constructor with the
            remote server field updated to the value set in the UI.

            Subclasses should override this method to update other fields as well.
        """

        configuration = copy.deepcopy(self.device_configuration)
        return configuration

    def set_configuration(self, configuration: T) -> None:
        """Set the configuration to be displayed."""

        self.device_configuration = copy.deepcopy(configuration)
