import copy
import functools
from typing import Optional, Protocol

from PySide6.QtWidgets import QWidget, QLineEdit, QVBoxLayout

from caqtus.device import DeviceConfiguration
from caqtus.device.configuration import DeviceServerName
from caqtus.gui.condetrol.device_configuration_editors import DeviceConfigurationEditor
from ._editor_builder import EditorBuilder
from ._value_editor import ValueEditor


class GeneratedConfigEditor[C: DeviceConfiguration](DeviceConfigurationEditor[C]):
    def __init__(
        self,
        config: C,
        parent: Optional[QWidget] = None,
        *,
        config_editor_type: type[ValueEditor[C]],
    ) -> None:
        super().__init__(parent)
        self._editor = config_editor_type(config, self)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._editor.widget())
        self.setLayout(layout)

    # TODO: Understand why need to silence pyright
    def get_configuration(self) -> C:  # type: ignore[reportIncompatibleMethodOverride]
        return self._editor.read_value()


class DeviceConfigurationEditorType[C: DeviceConfiguration](Protocol):
    def __call__(
        self, config: C, parent: Optional[QWidget] = None
    ) -> GeneratedConfigEditor[C]: ...


def build_device_configuration_editor[
    C: DeviceConfiguration
](config_type: type[C], builder: EditorBuilder) -> DeviceConfigurationEditorType[C]:
    """Builds a device configuration editor for the given configuration type.

    Args:
        config_type: The type of configuration to construct the editor for.
        builder: Used to build editors for the fields of the configuration.

    Returns:
        An automatically generated class of type :class:`DeviceConfigurationEditor`
        that can be used to edit configurations with type `config_type`.
    """

    config_editor_type = builder.build_editor(config_type)
    return functools.partial(
        GeneratedConfigEditor, config_editor_type=config_editor_type
    )


_builder = EditorBuilder()


class DeviceServerNameEditor(ValueEditor[Optional[DeviceServerName]]):
    def __init__(
        self, value: Optional[DeviceServerName], parent: Optional[QWidget] = None
    ) -> None:
        self.line_edit = QLineEdit(parent)
        self.line_edit.setPlaceholderText("None")
        if value:
            self.line_edit.setText(value)
        else:
            self.line_edit.setText("")

    def read_value(self) -> Optional[DeviceServerName]:
        text = self.line_edit.text()
        if text:
            return DeviceServerName(text)
        else:
            return None

    def set_editable(self, editable: bool) -> None:
        self.line_edit.setReadOnly(not editable)

    def widget(self) -> QLineEdit:
        return self.line_edit


_builder.register_editor(Optional[DeviceServerName], DeviceServerNameEditor)


def get_editor_builder() -> EditorBuilder:
    """Return a new editor builder with basic types registered.

    The editor builder returned knows how to handle:

        - Device server name
    """

    return copy.deepcopy(_builder)
