from pytestqt.modeltest import ModelTester
from pytestqt.qtbot import QtBot

from caqtus.gui.condetrol.sequence_iteration_editors import StepsIterationEditor
from caqtus.gui.condetrol.sequence_iteration_editors.steps_iteration_editor.steps_model import (
    VariableDeclarationData,
    LinspaceLoopData,
    ArrangeLoopData,
)
from caqtus.types.expression import Expression
from caqtus.types.iteration import (
    StepsConfiguration,
    VariableDeclaration,
    LinspaceLoop,
    ExecuteShot,
    ArangeLoop,
    Step,
)
from caqtus.types.variable_name import DottedVariableName
from caqtus.utils import serialization


def test_0(qtbot: QtBot, qtmodeltester: ModelTester):
    # Simple instantiation test that can also be used to screenshot the widget for the
    # documentation.
    editor = StepsIterationEditor()
    steps = StepsConfiguration(
        [
            VariableDeclaration(DottedVariableName("a"), Expression("1")),
            LinspaceLoop(
                variable=DottedVariableName("b"),
                start=Expression("0"),
                stop=Expression("1"),
                num=10,
                sub_steps=[ExecuteShot()],
            ),
            ArangeLoop(
                variable=DottedVariableName("c"),
                start=Expression("0"),
                stop=Expression("1"),
                step=Expression("0.1"),
                sub_steps=[ExecuteShot()],
            ),
        ]
    )
    with qtbot.assert_not_emitted(editor.iteration_edited):
        editor.set_iteration(steps)

    qtbot.addWidget(editor)

    editor.show()
    assert editor.get_iteration() == steps
    qtmodeltester.check(editor.model())


def test_1(qtbot: QtBot):
    editor = StepsIterationEditor()
    qtbot.addWidget(editor)
    editor.set_read_only(False)

    steps = StepsConfiguration(
        [VariableDeclaration(DottedVariableName("a"), Expression("1"))]
    )
    editor.set_iteration(steps)

    editor.show()

    editor.model().setData(
        editor.model().index(0, 0),
        VariableDeclarationData(DottedVariableName("b"), Expression("2")),
    )
    qtbot.wait_signal(editor.iteration_edited)
    assert editor.get_iteration() == StepsConfiguration(
        [VariableDeclaration(DottedVariableName("b"), Expression("2"))]
    )


def test_2(qtbot: QtBot):
    editor = StepsIterationEditor()
    qtbot.addWidget(editor)
    editor.set_read_only(False)

    steps = StepsConfiguration(
        [
            LinspaceLoop(
                variable=DottedVariableName("b"),
                start=Expression("0"),
                stop=Expression("1"),
                num=10,
                sub_steps=[ExecuteShot()],
            )
        ]
    )
    editor.set_iteration(steps)

    editor.show()

    editor.model().setData(
        editor.model().index(0, 0),
        LinspaceLoopData(
            variable=DottedVariableName("c"),
            start=Expression("2"),
            stop=Expression("3"),
            num=10,
        ),
    )
    qtbot.wait_signal(editor.iteration_edited)
    assert editor.get_iteration() == StepsConfiguration(
        [
            LinspaceLoop(
                variable=DottedVariableName("c"),
                start=Expression("2"),
                stop=Expression("3"),
                num=10,
                sub_steps=[ExecuteShot()],
            )
        ]
    )


def test_3(qtbot: QtBot):
    editor = StepsIterationEditor()
    qtbot.addWidget(editor)
    editor.set_read_only(False)

    steps = StepsConfiguration(
        [
            ArangeLoop(
                variable=DottedVariableName("b"),
                start=Expression("0"),
                stop=Expression("1"),
                step=Expression("0.1"),
                sub_steps=[ExecuteShot()],
            )
        ]
    )
    editor.set_iteration(steps)

    editor.show()

    editor.model().setData(
        editor.model().index(0, 0),
        ArrangeLoopData(
            variable=DottedVariableName("c"),
            start=Expression("2"),
            stop=Expression("3"),
            step=Expression("0.2"),
        ),
    )
    qtbot.wait_signal(editor.iteration_edited)
    assert editor.get_iteration() == StepsConfiguration(
        [
            ArangeLoop(
                variable=DottedVariableName("c"),
                start=Expression("2"),
                stop=Expression("3"),
                step=Expression("0.2"),
                sub_steps=[ExecuteShot()],
            )
        ]
    )


def test_4(qtbot: QtBot, qtmodeltester: ModelTester):
    # Test for issue #17.
    # Issue was fixed by replacing QStandardItem.appendRows with
    # QStandardItem.appendRow.
    # Seems like a bug in Qt/Pyside6?
    steps_data = [
        {
            "sub_steps": [
                {
                    "sub_steps": [
                        {
                            "sub_steps": [{"execute": "shot"}],
                            "variable": "pre_ramsey_duration",
                            "start": "0 us",
                            "stop": "3 us",
                            "num": 5,
                        }
                    ],
                    "variable": "probe.frequency",
                    "start": "-17.2 MHz",
                    "stop": "-19.2 MHz",
                    "num": 50,
                }
            ],
            "variable": "rep",
            "start": "0",
            "stop": "40",
            "step": "1",
        },
    ]
    steps = serialization.converters["json"].structure(steps_data, list[Step])
    editor = StepsIterationEditor()
    editor.set_read_only(False)
    editor.set_iteration(StepsConfiguration(steps))
    qtbot.addWidget(editor)
    editor.show()

    def remove():
        model = editor.model()
        first_loop_index = model.index(0, 0)
        second_loop_index = model.index(0, 0, first_loop_index)
        model.removeRow(0, second_loop_index)

    remove()
    qtbot.wait_for_window_shown(editor)
    qtmodeltester.check(editor.model())
