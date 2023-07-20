import logging
from itertools import groupby
from types import NotImplementedType
from typing import Type, Iterable, Any

from PyQt6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    Qt,
    QSize,
)
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QWidget, QMenu

from experiment.configuration import ExperimentConfig
from lane.configuration import Lane
from lane.model import LaneModel
from .lane_model import get_lane_model, create_new_lane

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


class LanesModel(QAbstractTableModel):
    def __init__(
        self, lanes: list[Lane], experiment_config: ExperimentConfig, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.experiment_config = experiment_config
        self.lanes = lanes

    @property
    def lanes(self):
        return self._lanes

    @lanes.setter
    def lanes(self, lanes: list[Lane]):
        self.beginResetModel()
        self._lanes = lanes
        self._lane_models: list[LaneModel] = [
            get_lane_model(lane, self.experiment_config) for lane in self._lanes
        ]
        self.endResetModel()

    @property
    def experiment_config(self):
        return self._experiment_config

    @experiment_config.setter
    def experiment_config(self, experiment_config: ExperimentConfig):
        self.beginResetModel()
        self._experiment_config = experiment_config
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._lanes)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if not self._lane_models:
            return 0
        length = self._lane_models[0].rowCount()
        assert all(lane_model.rowCount() == length for lane_model in self._lane_models)
        return length

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        return self._lane_models[index.row()].data(
            self.get_lane_model_index(index), role
        )

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return f"Step {section}"
        if orientation == Qt.Orientation.Vertical:
            return self._lane_models[section].headerData(
                0, Qt.Orientation.Horizontal, role
            )

    def setData(self, index: QModelIndex, value, role: int = Qt.ItemDataRole.EditRole) -> bool:
        return self._lane_models[index.row()].setData(
            self.get_lane_model_index(index), value, role
        )

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        return self._lane_models[index.row()].flags(self.get_lane_model_index(index))

    def span(self, index: QModelIndex) -> "QSize":
        return self._lane_models[index.row()].span(self.get_lane_model_index(index))

    def get_lane_model_index(self, index: QModelIndex) -> QModelIndex:
        # noinspection PyTypeChecker
        return self._lane_models[index.row()].index(index.column())

    def insertColumn(self, column: int, parent: QModelIndex = QModelIndex()) -> bool:
        self.beginInsertColumns(parent, column, column)
        for lane_model in self._lane_models:
            lane_model.insertRow(column)
        self.endInsertColumns()
        return True

    def removeColumn(self, column: int, parent: QModelIndex = QModelIndex()) -> bool:
        self.beginRemoveColumns(parent, column, column)
        for lane_model in self._lane_models:
            lane_model.removeRow(column)
        self.endRemoveColumns()
        return True

    def removeRow(self, row: int, parent: QModelIndex = QModelIndex()) -> bool:
        self.beginRemoveRows(parent, row, row)
        self._lanes.pop(row)
        self._lane_models.pop(row)
        self.endRemoveRows()
        return True

    def removeRows(
        self, row: int, count: int, parent: QModelIndex = QModelIndex()
    ) -> bool:
        self.beginRemoveRows(parent, row, row + count - 1)
        for _ in range(count):
            self._lanes.pop(row)
            self._lane_models.pop(row)
        self.endRemoveRows()
        return True

    def insert_lane(
        self, row: int, lane_type: Type[Lane], name: str, number_steps: int
    ) -> bool:
        new_lane = create_new_lane(
            number_steps, lane_type, name, self.experiment_config
        )
        self.beginInsertRows(QModelIndex(), row, row)
        self._lanes.insert(row, new_lane)
        self._lane_models.insert(row, get_lane_model(new_lane, self.experiment_config))
        self.endInsertRows()
        return True

    def merge(self, indexes: Iterable[QModelIndex]) -> bool:
        coordinates = [(index.row(), index.column()) for index in indexes]
        coordinates.sort()
        lanes = groupby(coordinates, key=lambda x: x[0])
        self.beginResetModel()
        for lane, group in lanes:
            l = list(group)
            start = l[0][1]
            stop = l[-1][1] + 1
            self._lane_models[lane].merge(start, stop)
        self.endResetModel()
        return True

    def break_up(self, indexes: Iterable[QModelIndex]):
        coordinates = [(index.row(), index.column()) for index in indexes]
        coordinates.sort()
        lanes = groupby(coordinates, key=lambda x: x[0])
        self.beginResetModel()
        for lane, group in lanes:
            l = list(group)
            start, stop = self._lanes[lane].span(l[0][1])
            self._lane_models[lane].break_up(start, stop)
        self.endResetModel()
        return True

    def map_name_to_row(self, name: str) -> int:
        return next(
            index for index, lane in enumerate(self._lanes) if lane.name == name
        )

    def create_editor(
        self, parent: QWidget, index: QModelIndex
    ) -> QWidget | NotImplementedType:
        if not index.isValid():
            return NotImplemented
        return self._lane_models[index.row()].create_editor(
            parent, self.get_lane_model_index(index)
        )

    def set_editor_data(
        self, editor: QWidget, index: QModelIndex
    ) -> None | NotImplementedType:
        if not index.isValid():
            return NotImplemented
        return self._lane_models[index.row()].set_editor_data(
            editor, self.get_lane_model_index(index)
        )

    def get_editor_data(
        self, editor: QWidget, index: QModelIndex
    ) -> Any | NotImplementedType:
        if not index.isValid():
            return NotImplemented
        return self._lane_models[index.row()].get_editor_data(
            editor, self.get_lane_model_index(index)
        )

    def get_cell_context_actions(self, index: QModelIndex) -> list[QAction | QMenu]:
        if not index.isValid():
            return []
        return self._lane_models[index.row()].get_cell_context_actions(
            self.get_lane_model_index(index)
        )
