import logging
from typing import Optional

from PyQt6 import QtCore, QtGui
from PyQt6.QtCore import Qt, QModelIndex
from PyQt6.QtGui import QColor, QKeySequence, QShortcut, QAction
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QMenu,
    QTreeView,
    QHeaderView,
)

from experiment.configuration import ExperimentConfig
from experiment.session import ExperimentSessionMaker
from sequence.runtime import Sequence, State
from yaml_clipboard_mixin import YAMLClipboardMixin
from .swim_lane_model import SwimLaneModel

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


class SpanColumnsDelegate(QStyledItemDelegate):
    def __init__(self, view: QTreeView, *args, **kwargs):
        self._view = view
        super().__init__(*args, **kwargs)

    def paint(
        self,
        painter: QtGui.QPainter,
        option: "QStyleOptionViewItem",
        index: QModelIndex,
    ) -> None:
        model: SwimLaneModel = index.model()
        span = model.span(index)
        if span.width() == 0:
            return
        else:
            option.rect = self._view.visualRect(index)
            if model.is_lane_cell(index) or model.is_step_cell(index):
                super().paint(painter, option, index)
                painter.save()
                painter.setPen(QColor.fromRgb(69, 83, 100))
                painter.drawRect(option.rect)
                painter.restore()
            elif model.is_lane_group_cell(index):
                painter.save()
                painter.fillRect(option.rect, QColor.fromRgb(69, 83, 100))
                # painter.setPen(QColor.fromRgb(25, 35, 45))
                # painter.drawLine(option.rect.topLeft(), option.rect.topRight())
                painter.restore()
                super().paint(painter, option, index)
            else:
                super().paint(painter, option, index)


class ShotWidget(QWidget, YAMLClipboardMixin):
    """A widget that shows the timeline of a shot

    Columns represent the different steps of the shot. Rows represent the different
    lanes (i.e. action channels).

    """

    def __init__(
        self,
        sequence: Sequence,
        experiment_config: ExperimentConfig,
        session_maker: ExperimentSessionMaker,
        *args,
    ):
        super().__init__(*args)
        self._sequence = sequence
        self._session_maker = session_maker

        self.experiment_config = experiment_config

        self.swim_lane_widget = SwimLaneView(session_maker, parent=self)

        self.model = SwimLaneModel(
            self._sequence, "shot", self.experiment_config, self._session_maker, parent=self.swim_lane_widget
        )
        size = QtCore.QSize(20, 40)
        index = self.model.index(0, 0)
        self.model.setData(index, size, Qt.ItemDataRole.SizeHintRole)

        self.layout = QVBoxLayout()
        self.swim_lane_widget.setModel(self.model)
        self.swim_lane_widget.expandAll()
        self.swim_lane_widget.resizeColumnToContents(0)
        self.swim_lane_widget.setItemDelegate(
            SpanColumnsDelegate(self.swim_lane_widget)
        )
        self.layout.addWidget(self.swim_lane_widget)
        self.setLayout(self.layout)

        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self, self.undo)
        self.redo_shortcut = QShortcut(QKeySequence("Ctrl+Y"), self, self.redo)

    def undo(self):
        self.model.undo()

    def redo(self):
        self.model.redo()

    def update_experiment_config(self, new_config: ExperimentConfig):
        self.model.update_experiment_config(new_config)

    def convert_to_external_use(self):
        # noinspection PyTypeChecker
        model: SwimLaneModel = self.swim_lane_widget.model()
        return model.shot_config

    def update_from_external_source(self, shot_config):
        self.swim_lane_widget.model().set_shot_config(shot_config)


class SwimLaneView(QTreeView):
    def __init__(self, session_maker: ExperimentSessionMaker, *args, **kwargs):
        self._session = session_maker()
        self._sequence: Optional[Sequence] = None

        super().__init__(*args, **kwargs)
        self.setUniformRowHeights(True)
        # self.setAlternatingRowColors(True)

        self.header().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        # noinspection PyUnresolvedReferences
        self.header().customContextMenuRequested.connect(self.show_steps_context_menu)
        self.header().setStretchLastSection(False)
        self.header().setSectionsMovable(False)
        self.header().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        # noinspection PyUnresolvedReferences
        self.customContextMenuRequested.connect(self.show_context_menu)

        self.setSelectionBehavior(QTreeView.SelectionBehavior.SelectItems)
        self.setSelectionMode(self.selectionMode().ContiguousSelection)

    def drawBranches(
        self, painter: QtGui.QPainter, rect: QtCore.QRect, index: QtCore.QModelIndex
    ) -> None:
        if index.isValid():
            if not index.parent().isValid() and index.row() < 2:
                return

        painter.save()
        painter.fillRect(rect, QColor.fromRgb(69, 83, 100))
        # painter.setPen(QColor.fromRgb(25, 35, 45))
        # painter.drawLine(rect.topLeft(), rect.topRight())
        painter.restore()
        super().drawBranches(painter, rect, index)

    def visualRect(self, index: QtCore.QModelIndex) -> QtCore.QRect:
        if not index.isValid():
            return super().visualRect(index)
        span = index.model().span(index)
        if span.width() == 0:
            return QtCore.QRect()
        rect = super().visualRect(index)

        width = 0
        for i in range(span.width()):
            column = index.column() + i
            new_index = index.model().index(index.row(), column, index.parent())
            width += super().visualRect(new_index).width()
        rect.setWidth(width)
        return rect

    def visualRegionForSelection(
        self, selection: QtCore.QItemSelection
    ) -> QtGui.QRegion:
        region = QtGui.QRegion()
        for index in self.selectedIndexes():
            if index.model().span(index).width() == 0:
                continue
            region += self.visualRect(index)
        return region

    def setModel(self, model: SwimLaneModel) -> None:
        self._sequence = model.sequence
        super().setModel(model)

    def get_sequence_state(self, session):
        return self._sequence.get_state(session)

    def show_steps_context_menu(self, position):
        """Show the context menu on the step header to remove or add a new time step"""

        with self._session as session:
            if self.get_sequence_state(session) != State.DRAFT:
                return

        menu = QMenu(self.header())

        index = self.header().logicalIndexAt(position)
        logger.debug(f"{index=}")
        if index == 0:
            return
        if index == -1:
            index = self.header().count() - 1
        else:
            add_step_before_action = QAction("Insert before")
            menu.addAction(add_step_before_action)
            # noinspection PyUnresolvedReferences
            add_step_before_action.triggered.connect(
                lambda: self.model().insertColumn(index, QModelIndex())
            )

        add_step_after_action = QAction("Insert after")
        menu.addAction(add_step_after_action)
        # noinspection PyUnresolvedReferences
        add_step_after_action.triggered.connect(
            lambda: self.model().insertColumn(index + 1, QModelIndex())
        )

        if index != -1:
            remove_step_action = QAction("Remove")
            menu.addAction(remove_step_action)
            # noinspection PyUnresolvedReferences
            remove_step_action.triggered.connect(
                lambda: self.model().removeColumn(index, QModelIndex())
            )

        menu.exec(self.header().mapToGlobal(position))

    def show_context_menu(self, position):
        with self._session as session:
            if self.get_sequence_state(session) != State.DRAFT:
                return

        menu = QMenu(self)

        index = self.indexAt(position)
        # noinspection PyTypeChecker
        model: SwimLaneModel = self.model()
        actions = model.get_context_actions(index)
        if actions:
            for action in actions:
                if isinstance(action, QMenu):
                    menu.addMenu(action)
                elif isinstance(action, QAction):
                    menu.addAction(action)

        selected_indices = self.selectionModel().selectedIndexes()
        logger.debug(len(selected_indices))
        if len(selected_indices) == 1:
            selected_indices = [
                index,
                self.model().index(index.row(), index.column() + 1, index.parent()),
            ]
        merge_action = QAction("Merge")
        menu.addAction(merge_action)
        # noinspection PyUnresolvedReferences
        merge_action.triggered.connect(lambda: model.merge(selected_indices))
        break_up_action = QAction("Break up")
        menu.addAction(break_up_action)
        # noinspection PyUnresolvedReferences
        break_up_action.triggered.connect(lambda: model.break_up(selected_indices))

        menu.exec(self.mapToGlobal(position))
