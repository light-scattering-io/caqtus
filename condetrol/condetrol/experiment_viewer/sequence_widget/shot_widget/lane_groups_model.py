from PyQt6.QtCore import QAbstractItemModel, QModelIndex, Qt

from sequence.configuration import LaneGroup, LaneReference
from sequence.configuration.shot import LaneGroupRoot


class LaneGroupModel(QAbstractItemModel):
    def __init__(self, root: LaneGroupRoot, parent=None):
        super().__init__(parent)
        self.root = root

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, root: LaneGroupRoot):
        self.beginResetModel()
        self._root = root
        self.endResetModel()

    def columnCount(self, parent: QModelIndex = ...) -> int:
        return 1

    def rowCount(self, parent: QModelIndex = ...) -> int:
        if not parent.isValid():
            return len(self.root.children)
        else:
            return len(parent.internalPointer().children)

    def index(self, row: int, column: int, parent: QModelIndex = ...) -> QModelIndex:
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
        elif not parent.isValid():
            parent_item = self.root
            return self.createIndex(row, column, parent_item.children[row])
        else:
            parent_item = parent.internalPointer()
            return self.createIndex(row, column, parent_item.children[row])

    def parent(self, child: QModelIndex) -> QModelIndex:
        if not child.isValid():
            return QModelIndex()
        child_item = child.internalPointer()
        if child_item is self.root:
            return QModelIndex()
        else:
            parent_item = child_item.parent
            return self.createIndex(parent_item.row, 0, parent_item)

    def data(self, index: QModelIndex, role: int = ...):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            item = index.internalPointer()
            if isinstance(item, LaneGroup):
                return item.name
            elif isinstance(item, LaneReference):
                return item.lane_name
            else:
                raise NotImplementedError(f"Unknown item type: {type(item)}")

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        base_flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        item = index.internalPointer()
        if isinstance(item, LaneGroup):
            return base_flags | Qt.ItemFlag.ItemIsEditable
        elif isinstance(item, LaneReference):
            return base_flags

    def setData(self, index: QModelIndex, value: str, role: int = ...) -> bool:
        if not index.isValid():
            return False
        item = index.internalPointer()
        if isinstance(item, LaneGroup):
            item.name = value
            self.dataChanged.emit(index, index)
            return True
        else:
            return False

    def insert_lane(self, parent: QModelIndex, row: int, name: str):
        if not parent.isValid():
            parent_item = self.root
        else:
            raise NotImplementedError
        self.beginInsertRows(parent, row, row)
        parent_item.insert_lane(row, name)
        self.endInsertRows()

    def is_lane(self, index: QModelIndex) -> bool:
        if index.isValid():
            return isinstance(index.internalPointer(), LaneReference)
        else:
            return False

    def removeRow(self, row: int, parent: QModelIndex = ...) -> bool:
        if not parent.isValid():
            parent_item = self.root
        else:
            parent_item = parent.internalPointer()
        self.beginRemoveRows(parent, row, row)
        new_children = list(parent_item.children)
        new_children.pop(row)
        parent_item.children = new_children
        self.endRemoveRows()
        return True
