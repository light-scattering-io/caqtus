import copy
from collections.abc import Mapping, Iterable
from typing import Optional

from PySide6.QtCore import Qt, QAbstractListModel, QModelIndex
from PySide6.QtWidgets import (
    QDialog,
    QWidget,
    QVBoxLayout,
    QDialogButtonBox,
    QHBoxLayout,
    QListView,
    QStyledItemDelegate,
)

from caqtus.device import DeviceConfiguration, DeviceName
from .add_device_dialog_ui import Ui_AddDeviceDialog
from .device_configuration_editor import (
    DeviceConfigurationEditor,
)
from .device_configurations_dialog_ui import Ui_DeviceConfigurationsDialog
from .extension import CondetrolDeviceExtensionProtocol
from ..icons import get_icon

_CONFIG_ROLE = Qt.ItemDataRole.UserRole + 1


class DeviceConfigurationsDialog(QDialog, Ui_DeviceConfigurationsDialog):
    """A dialog for displaying and editing a collection of device configurations."""

    def __init__(
        self,
        extension: CondetrolDeviceExtensionProtocol,
        parent: Optional[QWidget] = None,
    ):
        """Initialize the dialog."""

        super().__init__(parent, Qt.WindowType.Window)
        self._configs_view = DeviceConfigurationsEditor(extension, self)
        self.add_device_dialog = AddDeviceDialog(extension, self)
        self._extension = extension
        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        self.setupUi(self)
        layout = self.layout()
        assert isinstance(layout, QVBoxLayout)
        layout.insertWidget(0, self._configs_view, 1)
        self.add_device_button.setIcon(get_icon("plus"))
        self.remove_device_button.setIcon(get_icon("minus"))

    def setup_connections(self):
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.add_device_button.clicked.connect(self._on_add_configuration)
        self.remove_device_button.clicked.connect(
            self._configs_view.delete_selected_configuration
        )

    def _on_add_configuration(self) -> None:
        result = self.add_device_dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            device_name, device_type = (
                self.add_device_dialog.device_name_line_edit.text(),
                self.add_device_dialog.device_type_combo_box.currentText(),
            )
            if not device_name:
                return
            device_configuration = self._extension.create_new_device_configuration(
                device_type
            )
            self._configs_view.add_configuration(device_name, device_configuration)

    def get_device_configurations(self) -> dict[DeviceName, DeviceConfiguration]:
        return self._configs_view.get_device_configurations()

    def set_device_configurations(
        self, device_configurations: Mapping[DeviceName, DeviceConfiguration]
    ) -> None:
        self._configs_view.set_device_configurations(device_configurations)


class DeviceConfigurationsEditor(QWidget):
    """A widget for displaying and editing a collection of device configurations.

    This widget displays a list of device configurations, each of which can be edited
    using a custom editor provided by the device plugin.
    """

    def __init__(
        self,
        extension: CondetrolDeviceExtensionProtocol,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self._list_view = QListView(self)
        self._model = DeviceConfigurationModel(self)
        self._list_view.setModel(self._model)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(self._list_view, 0)
        self._delegate = DeviceEditorDelegate(self._layout, extension, self)
        self._list_view.setItemDelegate(self._delegate)
        self._list_view.setEditTriggers(
            QListView.EditTrigger.DoubleClicked | QListView.EditTrigger.CurrentChanged
        )

    def set_device_configurations(
        self, device_configurations: Mapping[DeviceName, DeviceConfiguration]
    ) -> None:
        """Set the device configurations to display.

        The view will copy the configurations and store them internally, so external
        changes to the configurations will not affect the view.
        """

        self._model.set_device_configurations(device_configurations)

    def get_device_configurations(self) -> dict[DeviceName, DeviceConfiguration]:
        """Return a copy of the configurations currently displayed in the view."""

        # We need to ensure that the data in the current editor is committed before
        # we read the model data.
        w = self.editor()
        if w is not None:
            self._list_view.commitData(w)

        return self._model.get_configurations()

    def add_configuration(
        self, device_name: DeviceName, device_configuration: DeviceConfiguration
    ) -> None:
        """Add a device configuration to the view.

        The view will copy the configuration and store it internally, so external
        changes to the configuration will not affect the view.
        """

        if not isinstance(device_configuration, DeviceConfiguration):
            raise TypeError(
                f"Expected a {DeviceConfiguration}, got {type(device_configuration)}"
            )

        self._model.add_configuration(device_name, device_configuration)

    def delete_selected_configuration(self) -> None:
        """Delete the configuration that is currently selected in the view."""

        index = self._list_view.currentIndex()
        if index.isValid():
            self._model.removeRow(index.row())

    def edit(self, row: int) -> None:
        index = self._model.index(row, 0)
        self._list_view.setCurrentIndex(index)
        self._list_view.edit(index)

    def editor(self) -> Optional[DeviceConfigurationEditor]:
        w = self._list_view.indexWidget(self._list_view.currentIndex())
        assert isinstance(w, DeviceConfigurationEditor) or w is None
        return w


class DeviceEditorDelegate(QStyledItemDelegate):
    # This is a delegate that doesn't display the editor in the view cell, but instead
    # displays it in an external layout.
    def __init__(
        self,
        layout,
        extension: CondetrolDeviceExtensionProtocol,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._layout = layout
        self._extension = extension

    def createEditor(self, parent, option, index):
        config = index.data(_CONFIG_ROLE)
        editor = self._extension.get_device_configuration_editor(config)
        self._layout.addWidget(editor, 1)

        return editor

    def setEditorData(self, editor, index):
        pass

    def setModelData(self, editor, model, index):
        assert isinstance(editor, DeviceConfigurationEditor)
        config = copy.deepcopy(editor.get_configuration())
        model.setData(index, config, _CONFIG_ROLE)

    def updateEditorGeometry(self, editor, option, index):
        # This is necessary to ensure that the editor is resized to fit the layout.
        editor.updateGeometry()


class AddDeviceDialog(QDialog, Ui_AddDeviceDialog):
    def __init__(
        self,
        extension: CondetrolDeviceExtensionProtocol,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setup_ui(extension.available_new_configurations())

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.device_type_combo_box.currentTextChanged.connect(
            self._on_device_type_changed
        )
        self._on_device_type_changed(self.device_type_combo_box.currentText())

    def _on_device_type_changed(self, device_type: str):
        ok_button = self.buttonBox.button(QDialogButtonBox.StandardButton.Ok)
        assert ok_button is not None
        ok_button.setEnabled(bool(device_type))

    def setup_ui(self, device_types: Iterable[str]):
        self.setupUi(self)
        for device_type in device_types:
            self.device_type_combo_box.addItem(device_type)


class DeviceConfigurationModel(QAbstractListModel):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._device_configurations: list[tuple[DeviceName, DeviceConfiguration]] = []

    def rowCount(self, parent=QModelIndex()):
        return len(self._device_configurations)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            return self._device_configurations[index.row()][0]
        if role == _CONFIG_ROLE:
            return self._device_configurations[index.row()][1]
        return None

    def flags(self, index):
        return (
            Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsEditable
        )

    def set_device_configurations(
        self, device_configurations: Mapping[DeviceName, DeviceConfiguration]
    ) -> None:
        self.beginResetModel()
        self._device_configurations = copy.deepcopy(list(device_configurations.items()))
        self.endResetModel()

    def get_configurations(self) -> dict[DeviceName, DeviceConfiguration]:
        return dict(copy.deepcopy(self._device_configurations))

    def add_configuration(
        self, device_name: DeviceName, device_configuration: DeviceConfiguration
    ) -> None:
        self.beginInsertRows(
            QModelIndex(),
            len(self._device_configurations),
            len(self._device_configurations),
        )
        self._device_configurations.append((device_name, device_configuration))
        self.endInsertRows()

    def removeRow(self, row):
        self.beginRemoveRows(QModelIndex(), row, row)
        del self._device_configurations[row]
        self.endRemoveRows()
