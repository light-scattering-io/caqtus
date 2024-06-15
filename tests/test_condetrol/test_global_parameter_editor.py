import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from pytestqt.qtbot import QtBot

from caqtus.gui.condetrol.parameter_tables_editor import ParameterNamespaceEditor


@pytest.mark.skip(reason="Qt test is not working for drag and drop")
def test_0(qtbot: QtBot):
    editor = ParameterNamespaceEditor()
    qtbot.addWidget(editor)
    editor.add_namespace_action.trigger()
    editor.add_parameter_action.trigger()
    editor.show()

    target_index = editor.view.model().index(0, 0)
    target_rect = editor.view.visualRect(target_index)
    source_index = editor.view.model().index(1, 0)
    source_rect = editor.view.visualRect(source_index)
    print(source_rect, target_rect)
    dropped = False

    def drop():
        qtbot.mouseMove(editor.view.viewport(), target_rect.center())
        QTest.qWait(50)
        qtbot.mouseRelease(editor.view.viewport(), Qt.LeftButton)
        nonlocal dropped
        dropped = True

    QTimer().singleShot(5000, drop)
    qtbot.mousePress(editor.view.viewport(), Qt.LeftButton, pos=source_rect.center())
    qtbot.wait_until(lambda: dropped)

    path = qtbot.screenshot(editor)
    assert False, path
