from caqtus.device.camera import CameraConfiguration
from caqtus.device.configuration import DeviceServerName
from caqtus.gui.condetrol.device_configuration_editors.camera_configuration_editor import (  # noqa: E501
    RectangularROIEditor,
    CameraConfigurationEditor,
)
from caqtus.types.image import Width, Height
from caqtus.types.image.roi import RectangularROI


def test_1(qtbot):
    roi = RectangularROI((Width(100), Height(100)), 0, 100, 0, 100)
    editor = RectangularROIEditor(max_width=100, max_height=100)
    editor.set_roi(roi)
    editor.show()
    qtbot.addWidget(editor)

    editor._x_spinbox.setValue(1)
    new_roi = editor.get_roi()

    assert new_roi.x == 1
    assert new_roi.width == 99


def test_2(qtbot):
    camera_config = CameraConfiguration(
        remote_server=None,
        roi=RectangularROI((Width(100), Height(100)), 0, 100, 0, 100),
    )
    editor = CameraConfigurationEditor(camera_config)
    editor.show()
    qtbot.addWidget(editor)

    editor.set_remote_server(DeviceServerName("test"))

    new_config = editor.get_configuration()

    assert new_config.remote_server == "test"
    assert new_config.roi == camera_config.roi
