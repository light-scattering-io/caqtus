from typing import Optional

from PyQt6.QtCore import pyqtSignal, QObject, Qt, QModelIndex
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QTableView, QMenu

from core.session.shot import TimeLanes
from .default_lane_model_factory import default_lane_model_factory
from .model import TimeLanesModel


class TimeLanesEditor(QTableView):
    time_lanes_changed = pyqtSignal()

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._read_only: bool = False
        self._model = TimeLanesModel(default_lane_model_factory, self)
        self.setModel(self._model)

        self.horizontalHeader().setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.verticalHeader().setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.setup_connections()

    def setup_connections(self):
        self.horizontalHeader().customContextMenuRequested.connect(
            self.show_steps_context_menu
        )
        self.verticalHeader().customContextMenuRequested.connect(
            self.show_lanes_context_menu
        )
        self.customContextMenuRequested.connect(self.show_cell_context_menu)
        self._model.modelReset.connect(self.update_spans)

        self._model.dataChanged.connect(self.time_lanes_changed)
        self._model.rowsInserted.connect(self.time_lanes_changed)
        self._model.rowsRemoved.connect(self.time_lanes_changed)
        self._model.columnsInserted.connect(self.time_lanes_changed)
        self._model.columnsRemoved.connect(self.time_lanes_changed)
        self._model.modelReset.connect(self.time_lanes_changed)

    def get_time_lanes(self) -> TimeLanes:
        return self._model.get_timelanes()

    def set_time_lanes(self, time_lanes: TimeLanes) -> None:
        self._model.set_timelanes(time_lanes)

    def update_spans(self):
        self.clearSpans()
        for row in range(self._model.rowCount()):
            for column in range(self._model.columnCount()):
                index = self._model.index(row, column, QModelIndex())
                span = self._model.span(index)
                if span.width() > 1 or span.height() > 1:
                    self.setSpan(row, column, span.height(), span.width())

    def set_read_only(self, read_only: bool) -> None:
        raise NotImplementedError

    def show_steps_context_menu(self, pos):
        menu = QMenu(self.horizontalHeader())

        index = self.horizontalHeader().logicalIndexAt(pos.x())
        if index == -1:
            add_step_action = QAction("Add step")
            menu.addAction(add_step_action)
            add_step_action.triggered.connect(
                lambda: self._model.insertColumn(
                    self._model.columnCount(), QModelIndex()
                )
            )
        elif 0 <= index < self.model().columnCount():
            add_step_before_action = QAction("Insert step before")
            menu.addAction(add_step_before_action)
            add_step_before_action.triggered.connect(
                lambda: self._model.insertColumn(index, QModelIndex())
            )

            add_step_after_action = QAction("Insert step after")
            menu.addAction(add_step_after_action)
            add_step_after_action.triggered.connect(
                lambda: self._model.insertColumn(index + 1, QModelIndex())
            )
            if self.model().columnCount() > 1:
                remove_step_action = QAction("Remove")
                menu.addAction(remove_step_action)
                remove_step_action.triggered.connect(
                    lambda: self._model.removeColumn(index, QModelIndex())
                )
        menu.exec(self.horizontalHeader().mapToGlobal(pos))

    def show_lanes_context_menu(self, pos):
        menu = QMenu(self.verticalHeader())

        index = self.verticalHeader().logicalIndexAt(pos.y())
        if 2 <= index < self.model().rowCount():
            remove_lane_action = QAction("Remove")
            menu.addAction(remove_lane_action)
            remove_lane_action.triggered.connect(
                lambda: self._model.removeRow(index, QModelIndex())
            )
        else:
            return
        menu.exec(self.verticalHeader().mapToGlobal(pos))

    def show_cell_context_menu(self, pos):
        index = self.indexAt(pos)
        cell_actions = self._model.get_cell_context_actions(index)
        if not cell_actions:
            return
        menu = QMenu(self)
        for action in cell_actions:
            if isinstance(action, QAction):
                menu.addAction(action)
            elif isinstance(action, QMenu):
                menu.addMenu(action)
        menu.exec(self.mapToGlobal(pos))
