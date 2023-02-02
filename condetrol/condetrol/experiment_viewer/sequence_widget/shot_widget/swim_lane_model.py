import logging
from itertools import groupby
from pathlib import Path
from typing import Type, Iterable

from PyQt6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    Qt,
    QSize,
    QMimeData,
    QByteArray,
)
from PyQt6.QtGui import QColor, QIcon

from condetrol.utils import UndoStack
from experiment_config import ExperimentConfig, ChannelSpecialPurpose
from expression import Expression
from sequence import SequenceStats, SequenceConfig, SequenceState
from sequence.shot import (
    DigitalLane,
    AnalogLane,
    Lane,
    CameraLane,
    TakePicture,
    CameraAction,
    Ramp,
    ShotConfiguration,
)
from settings_model import YAMLSerializable
from ..sequence_watcher import SequenceWatcher

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


class SwimLaneModel(QAbstractTableModel):
    """Model for a shot parallel time steps

    Each column corresponds to a time step.
    Row 0 corresponds to the name of the time steps and Row 1 corresponds to the
    duration of the steps. Other rows are device lanes.
    """

    def __init__(
        self, sequence_path: Path, shot_name: str, experiment_config: ExperimentConfig
    ):
        super().__init__()
        self.experiment_config = experiment_config

        self.shot_name = shot_name
        self.sequence_watcher = SequenceWatcher(sequence_path)
        self.sequence_config = self.sequence_watcher.read_config()
        self.sequence_state = self.sequence_watcher.read_stats().state
        self.shot_config = self.sequence_config.shot_configurations[self.shot_name]

        self.sequence_watcher.config_changed.connect(self.change_sequence_config)
        self.sequence_watcher.stats_changed.connect(self.change_sequence_state)

        self.undo_stack = UndoStack()
        self.undo_stack.push(self.shot_config.to_yaml())

    def update_experiment_config(self, new_config: ExperimentConfig):
        self.beginResetModel()
        self.experiment_config = new_config
        self.endResetModel()

    def change_sequence_state(self, stats: SequenceStats):
        self.beginResetModel()
        self.sequence_state = stats.state
        self.endResetModel()
        self.layoutChanged.emit()

    def change_sequence_config(self, sequence_config: SequenceConfig):
        self.beginResetModel()
        self.sequence_config = sequence_config
        self.shot_config = self.sequence_config.shot_configurations[self.shot_name]
        self.undo_stack.push(self.shot_config.to_yaml())
        self.endResetModel()
        self.layoutChanged.emit()

    def rowCount(self, parent: QModelIndex = ...) -> int:
        return 2 + len(self.shot_config.lanes)

    def columnCount(self, parent: QModelIndex = ...) -> int:
        return len(self.shot_config.step_names)

    def data(self, index: QModelIndex, role: int = ...):
        if index.row() == 0:
            if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
                return self.shot_config.step_names[index.column()]
        elif index.row() == 1:
            if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
                return self.shot_config.step_durations[index.column()].body
        else:
            lane = self.get_lane(index)
            if isinstance(lane, DigitalLane):
                if (
                    role == Qt.ItemDataRole.DisplayRole
                    or role == Qt.ItemDataRole.EditRole
                ):
                    return lane[index.column()]
            elif isinstance(lane, AnalogLane):
                if (
                    role == Qt.ItemDataRole.DisplayRole
                    or role == Qt.ItemDataRole.EditRole
                ):
                    value = lane[index.column()]
                    if isinstance(value, Expression):
                        return lane[index.column()].body
                    elif isinstance(value, Ramp):
                        return "\u279F"
                elif role == Qt.ItemDataRole.TextColorRole:
                    try:
                        color = self.experiment_config.get_color(lane.name)
                    except ValueError:
                        return QColor.fromRgb(0, 0, 0)
                    else:
                        if color is not None:
                            return QColor.fromRgb(*color.as_rgb_tuple())
                elif role == Qt.ItemDataRole.TextAlignmentRole:
                    if lane.spans[index.column()] > 1 or isinstance(
                        lane[index.column()], Ramp
                    ):
                        return Qt.AlignCenter
                    else:
                        return Qt.AlignLeft
            elif isinstance(lane, CameraLane):
                camera_action = lane[index.column()]
                if isinstance(camera_action, TakePicture):
                    if (
                        role == Qt.ItemDataRole.DisplayRole
                        or role == Qt.ItemDataRole.EditRole
                    ):
                        return camera_action.picture_name
                    elif role == Qt.ItemDataRole.DecorationRole:
                        return QIcon(":/icons/camera-icon")
                    elif role == Qt.ItemDataRole.TextColorRole:
                        try:
                            color = self.experiment_config.get_color(
                                ChannelSpecialPurpose(purpose=lane.name)
                            )
                        except ValueError:
                            return QColor.fromRgb(0, 0, 0)
                        else:
                            if color is not None:
                                return QColor.fromRgb(*color.as_rgb_tuple())

    def setData(self, index: QModelIndex, value, role: int = ...) -> bool:
        edit = False
        if role == Qt.ItemDataRole.EditRole:
            if index.row() == 0:
                self.shot_config.step_names[index.column()] = value
                edit = True
            elif index.row() == 1:
                previous = self.shot_config.step_durations[index.column()].body
                try:
                    self.shot_config.step_durations[index.column()].body = value
                    edit = True
                except SyntaxError as error:
                    logger.error(error.msg)
                    self.shot_config.step_durations[index.column()].body = previous
            else:
                lane = self.get_lane(index)
                if isinstance(lane, AnalogLane):
                    previous = lane[index.column()].body
                    try:
                        lane[index.column()].body = value
                        edit = True
                    except SyntaxError as error:
                        logger.error(error.msg)
                        lane[index.column()].body = previous

                elif isinstance(lane, DigitalLane):
                    lane[index.column()] = value
                    edit = True
                elif isinstance(lane, CameraLane):
                    if value is None or isinstance(value, CameraAction):
                        lane[index.column()] = value
                        edit = True
                    elif isinstance(value, str) and isinstance(
                        cell := lane[index.column()], TakePicture
                    ):
                        cell.picture_name = value
                        edit = True
        if edit:
            self.save_config()
        return edit

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if index.isValid():
            flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
            if index.row() > 1:
                flags |= Qt.ItemFlag.ItemIsDragEnabled
            if self.sequence_state == SequenceState.DRAFT:
                if self.is_editable(index):
                    flags |= Qt.ItemFlag.ItemIsEditable
                if index.row() > 1:
                    flags |= Qt.ItemFlag.ItemIsDropEnabled
        else:
            flags = Qt.ItemFlag.NoItemFlags
        return flags

    def is_editable(self, index: QModelIndex) -> bool:
        editable = True
        if index.row() > 1:
            lane = self.get_lane(index)
            if isinstance(lane, CameraLane):
                if self.data(index, Qt.ItemDataRole.EditRole) is None:
                    editable = False
        return editable

    # noinspection PyTypeChecker
    def supportedDropActions(self) -> Qt.DropAction:
        return Qt.DropAction.MoveAction | Qt.DropAction.CopyAction

    def supportedDragActions(self) -> Qt.DropAction:
        if self.sequence_state == SequenceState.DRAFT:
            return Qt.DropAction.MoveAction
        else:
            return Qt.DropAction.CopyAction

    def mimeTypes(self) -> list[str]:
        return ["application/x-shot_lanes"]

    def mimeData(self, indexes: Iterable[QModelIndex]) -> QMimeData:
        rows = set(index.row() for index in indexes)
        data = [self.shot_config.lanes[row - 2] for row in rows]
        serialized = YAMLSerializable.dump(data).encode("utf-8")
        mime_data = QMimeData()
        mime_data.setData("application/x-shot_lanes", QByteArray(serialized))
        logger.debug(data)
        return mime_data

    def dropMimeData(
        self,
        data: QMimeData,
        action: Qt.DropAction,
        row: int,
        column: int,
        parent: QModelIndex,
    ) -> bool:
        yaml_string = data.data("application/x-shot_lanes").data().decode("utf-8")
        lanes: list[Lane] = YAMLSerializable.load(yaml_string)
        return False

        correct_length = all(
            len(lane) == len(self.shot_config.step_names) for lane in lanes
        )
        if correct_length:
            self.beginInsertRows(parent, row, row + len(lanes) - 1)
            for lane in lanes:
                logger.debug(row)
                self.shot_config.lanes.insert(row - 2, lane)
            self.endInsertRows()
            self.save_config()
            return True
        else:
            return False

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = ...):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return f"Step {section}"
            elif orientation == Qt.Orientation.Vertical:
                if section == 0:
                    return "Name"
                elif section == 1:
                    return "Duration"
                else:
                    return self.shot_config.lanes[section - 2].name

    def span(self, index: QModelIndex) -> QSize:
        if index.row() <= 1:
            return QSize(1, 1)
        else:
            return QSize(
                self.shot_config.lanes[index.row() - 2].spans[index.column()], 1
            )

    def get_lane(self, index: QModelIndex):
        if index.row() >= 1:
            return self.shot_config.lanes[index.row() - 2]

    def insertColumn(self, column: int, parent: QModelIndex = ...) -> bool:
        self.beginInsertColumns(parent, column, column)
        self.shot_config.step_names.insert(column, f"...")
        self.shot_config.step_durations.insert(column, Expression("..."))
        for lane in self.shot_config.lanes:
            if isinstance(lane, DigitalLane):
                lane.insert(column, False)
            elif isinstance(lane, AnalogLane):
                lane.insert(column, Expression("..."))
            elif isinstance(lane, CameraLane):
                lane.insert(column, None)

        self.endInsertColumns()
        self.save_config()
        self.layoutChanged.emit()
        return True

    def removeColumn(self, column: int, parent: QModelIndex = ...) -> bool:
        self.beginRemoveColumns(parent, column, column)
        self.shot_config.step_names.pop(column)
        self.shot_config.step_durations.pop(column)
        for lane in self.shot_config.lanes:
            lane.remove(column)
        self.endRemoveColumns()
        self.save_config()
        self.layoutChanged.emit()
        return True

    def removeRow(self, row: int, parent: QModelIndex = ...) -> bool:
        self.beginRemoveRows(parent, row, row)
        self.shot_config.lanes.pop(row - 2)
        self.endRemoveRows()
        self.save_config()
        return True

    def removeRows(self, row: int, count: int, parent: QModelIndex = ...) -> bool:
        self.beginRemoveRows(parent, row, row + count - 1)
        for _ in range(count):
            self.shot_config.lanes.pop(row - 2)
        self.endRemoveRows()
        self.save_config()
        return True

    def insert_lane(self, row: int, lane_type: Type[Lane], name: str):
        new_lane = None
        if lane_type == DigitalLane:
            new_lane = DigitalLane(
                name=name,
                values=tuple(False for _ in range(self.columnCount())),
                spans=tuple(1 for _ in range(self.columnCount())),
            )
        elif lane_type == AnalogLane:
            new_lane = AnalogLane(
                name=name,
                values=tuple(Expression("...") for _ in range(self.columnCount())),
                spans=tuple(1 for _ in range(self.columnCount())),
                units=self.experiment_config.get_input_units(name),
            )
        elif lane_type == CameraLane:
            new_lane = CameraLane(
                name=name,
                values=(None,) * self.columnCount(),
                spans=(1,) * self.columnCount(),
            )
        if new_lane:
            self.beginInsertRows(QModelIndex(), row, row)
            self.shot_config.lanes.insert(row - 2, new_lane)
            self.endInsertRows()
            self.save_config()

    def save_config(self, save_undo: bool = True) -> bool:
        with self.sequence_watcher.block_signals():
            YAMLSerializable.dump(
                self.sequence_config, self.sequence_watcher.config_path
            )
            if save_undo:
                self.undo_stack.push(
                    YAMLSerializable.dump(
                        self.sequence_config.shot_configurations[self.shot_name]
                    )
                )
            return True

    def merge(self, indexes: Iterable[QModelIndex]):
        coordinates = [(index.row() - 2, index.column()) for index in indexes]
        coordinates.sort()
        lanes = groupby(coordinates, key=lambda x: x[0])
        for lane, group in lanes:
            l = list(group)
            start = l[0][1]
            stop = l[-1][1] + 1
            self.shot_config.lanes[lane].merge(start, stop)
        self.save_config()
        self.layoutChanged.emit()

    def break_(self, indexes: Iterable[QModelIndex]):
        coordinates = [(index.row() - 2, index.column()) for index in indexes]
        coordinates.sort()
        lanes = groupby(coordinates, key=lambda x: x[0])
        for lane, group in lanes:
            l = list(group)
            start = l[0][1]
            stop = l[-1][1] + 1
            self.shot_config.lanes[lane].break_(start, stop)
        self.save_config()
        self.layoutChanged.emit()

    def undo(self):
        new_yaml = self.undo_stack.undo()
        new_config = ShotConfiguration.from_yaml(new_yaml)
        self.beginResetModel()
        self.sequence_config.shot_configurations[self.shot_name] = new_config
        self.shot_config = new_config
        self.save_config(save_undo=False)
        self.endResetModel()
        self.layoutChanged.emit()

    def redo(self):
        new_yaml = self.undo_stack.redo()
        new_config = ShotConfiguration.from_yaml(new_yaml)
        self.beginResetModel()
        self.sequence_config.shot_configurations[self.shot_name] = new_config
        self.shot_config = new_config
        self.save_config(save_undo=False)
        self.endResetModel()
        self.layoutChanged.emit()
