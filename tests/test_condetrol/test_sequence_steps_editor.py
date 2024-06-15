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
)
from caqtus.types.variable_name import DottedVariableName


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
