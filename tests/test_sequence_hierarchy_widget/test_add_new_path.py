import logging

import pytest
from PySide6.QtCore import QModelIndex
from pytestqt.modeltest import ModelTester

from caqtus.gui.common.sequence_hierarchy import AsyncPathHierarchyModel
from caqtus.gui.qtutil import qt_trio
from caqtus.session import PureSequencePath
from .session_maker import session_maker


async def wrap(coro):
    try:
        await coro
    except Exception as e:
        logging.critical("Exception in async function", exc_info=e)


def test_0(session_maker, qtbot):
    model = AsyncPathHierarchyModel(session_maker)
    with session_maker() as session:
        path = PureSequencePath(r"\a")
        session.paths.create_path(path)
    model.fetchMore(QModelIndex())
    with session_maker() as session:
        session.paths.create_path(PureSequencePath(r"\b"))
    qt_trio.run(model.add_new_paths)
    assert model.rowCount() == 2


@pytest.mark.xfail
def test_1(session_maker, qtmodeltester: ModelTester, qtbot):
    model = AsyncPathHierarchyModel(session_maker)
    model.fetchMore(QModelIndex())
    qtmodeltester.check(model, force_py=True)
    with session_maker() as session:
        path = PureSequencePath(r"\a")
        session.paths.create_path(path)
    qt_trio.run(model.add_new_paths)
    assert model.rowCount() == 1
