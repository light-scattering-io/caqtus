from PySide6.QtCore import Qt
from pytestqt.modeltest import ModelTester

from caqtus.gui.common.sequence_hierarchy import AsyncPathHierarchyModel
from caqtus.gui.qtutil import QtAsyncio
from caqtus.session import PureSequencePath
from caqtus.session.sequence import State
from tests.fixtures import steps_configuration, time_lanes
from .session_maker import session_maker


def test_0(session_maker, qtmodeltester: ModelTester):
    model = AsyncPathHierarchyModel(session_maker)
    with session_maker() as session:
        session.paths.create_path(PureSequencePath(r"\test"))
    qtmodeltester.check(model)


def test_1(session_maker, qtmodeltester: ModelTester):
    model = AsyncPathHierarchyModel(session_maker)
    with session_maker() as session:
        session.paths.create_path(PureSequencePath(r"\test\test2"))
    qtmodeltester.check(model)
    assert model.rowCount() == 1
    child = model.index(0, 0)
    assert model.rowCount(child) == 1
    assert model.data(child, Qt.ItemDataRole.DisplayRole) == "test"
    child = model.index(0, 0, child)
    assert model.data(child, Qt.ItemDataRole.DisplayRole) == "test2"


def test_2(session_maker, qtmodeltester: ModelTester, steps_configuration, time_lanes):
    model = AsyncPathHierarchyModel(session_maker)
    with session_maker() as session:
        sequence = session.sequences.create(
            PureSequencePath(r"\test"), steps_configuration, time_lanes
        )
    qtmodeltester.check(model)
    index = model.index(0, 1)
    assert index.data().state == State.DRAFT

    with session_maker() as session:
        session.sequences.set_state(sequence.path, State.PREPARING)

    QtAsyncio.run(model.update_stats(model.index(0, 0)), keep_running=False)
    assert index.data().state == State.PREPARING