from collections.abc import Sequence
from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex
from PySide6.QtCore import Qt
from caqtus.device.sequencer.configuration import (
    ChannelConfiguration,
)

delay_multiplier = 1e-6


class SequencerChannelsModel(QAbstractTableModel):
    """A model to display and edit the channels of a sequencer device."""

    def __init__(self, channels: Sequence[ChannelConfiguration], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channels = tuple(channels)

    @property
    def channels(self) -> tuple[ChannelConfiguration, ...]:
        return self._channels

    @channels.setter
    def channels(self, channels: tuple[ChannelConfiguration, ...]):
        self.beginResetModel()
        self._channels = channels
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._channels)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 2

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            if index.column() == 0:
                return self._channels[index.row()].description
            elif index.column() == 1:
                if role == Qt.ItemDataRole.DisplayRole:
                    return str(self._channels[index.row()].output)
                elif role == Qt.ItemDataRole.EditRole:
                    return self._channels[index.row()]

    def setData(
        self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole
    ) -> bool:
        channel = index.row()
        if role == Qt.ItemDataRole.EditRole:
            if index.column() == 0:
                self._channels[channel].description = str(value)
                self.dataChanged.emit(index, index)
                return True
            elif index.column() == 1:
                if isinstance(value, ChannelConfiguration):
                    self._channels[channel].output = value.output
                    self.dataChanged.emit(index, index)
                    return True
        return super().setData(index, value, role)

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> str:
        if orientation == Qt.Orientation.Horizontal:
            if role == Qt.ItemDataRole.DisplayRole:
                if section == 0:
                    return "Description"
                elif section == 1:
                    return "Output"
        elif orientation == Qt.Orientation.Vertical:
            if role == Qt.ItemDataRole.DisplayRole:
                return str(section)
        return super().headerData(section, orientation, role)

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        return (
            Qt.ItemFlag.ItemIsEditable
            | Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
        )