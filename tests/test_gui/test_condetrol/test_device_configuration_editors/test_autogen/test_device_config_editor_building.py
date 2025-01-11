from pytestqt.qtbot import QtBot

from caqtus.device import DeviceConfiguration
from caqtus.device.configuration import DeviceServerName
from caqtus.gui.autogen import (
    build_device_configuration_editor,
    get_editor_builder,
    generate_device_configuration_editor,
)


def test_remote_server(qtbot: QtBot):
    class DeviceConfigA(DeviceConfiguration):
        pass

    value = DeviceConfigA(remote_server=DeviceServerName("James"))

    builder = get_editor_builder()
    device_config_editor_type = generate_device_configuration_editor(
        DeviceConfigA, builder
    )

    editor = device_config_editor_type(value)
    qtbot.add_widget(editor)
    editor.show()
    editor._editor.editor_remote_server.widget.setText("")  # type: ignore
    assert editor.get_configuration().remote_server is None
