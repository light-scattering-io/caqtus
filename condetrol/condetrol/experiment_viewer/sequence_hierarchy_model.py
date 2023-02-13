import logging
from typing import TypedDict, Optional

from PyQt6.QtCore import QAbstractItemModel, QModelIndex, Qt
from anytree import NodeMixin
from sqlalchemy import func
from sqlalchemy.orm import sessionmaker, Session

from sequence.runtime import SequencePath
from sequence.runtime.model import SequencePathModel
from sequence.runtime.state import State

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SequenceHierarchyModel(QAbstractItemModel):
    """Tree model for sequence hierarchy.

    This model stores an in-memory representation of the database sequence structure.
    """

    def __init__(self, session_maker: sessionmaker):
        self._session_maker = session_maker

        with self._session_maker.begin() as session:
            self._root = SequenceHierarchyItem(
                SequencePath.root(),
                children=_build_children(SequencePath.root(), session),
                row=0,
                is_sequence=False,
            )

        super().__init__()

    def index(self, row: int, column: int, parent: QModelIndex = ...) -> QModelIndex:
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
        elif not parent.isValid():
            return self.createIndex(row, column, self._root.children[row])
        else:
            parent: SequenceHierarchyItem = parent.internalPointer()
            if row < len(parent.children):
                return self.createIndex(row, column, parent.children[row])
            else:
                return QModelIndex()

    def parent(self, child: QModelIndex) -> QModelIndex:
        if not child.isValid():
            return QModelIndex()

        child: SequenceHierarchyItem = child.internalPointer()
        if child.is_root:
            return QModelIndex()
        else:
            return self.createIndex(child.parent.row, 0, child.parent)

    def rowCount(self, parent: QModelIndex = ...) -> int:
        if not parent.isValid():
            return len(self._root.children)
        else:
            return len(parent.internalPointer().children)

    def columnCount(self, parent: QModelIndex = ...) -> int:
        return 2

    def data(self, index: QModelIndex, role: int = ...):
        if not index.isValid():
            return

        if role == Qt.ItemDataRole.DisplayRole:
            if index.column() == 0:
                return self.get_sequence_name(index.internalPointer())
            else:
                stats = self.get_sequence_stats(index)
                return stats

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = ...):
        if (
            role == Qt.ItemDataRole.DisplayRole
            and orientation == Qt.Orientation.Horizontal
        ):
            if section == 0:
                return "Name"
            elif section == 1:
                return "Status"

    @staticmethod
    def get_sequence_name(item: "SequenceHierarchyItem"):
        return item.sequence_path.name

    def get_sequence_stats(self, index: QModelIndex) -> Optional["SequenceStats"]:
        item: "SequenceHierarchyItem" = index.internalPointer()
        if item.is_sequence:
            with self._session_maker.begin() as session:
                sequence = item.sequence_path.query_model(session).get_sequence()
                return SequenceStats(
                    state=sequence.get_state(),
                    total_number_shots=sequence.total_number_shots,
                    number_completed_shots=sequence.get_number_completed_shots(),
                )
        else:
            return None

    def get_sequence_state(self, item: "SequenceHierarchyItem"):
        if item.is_sequence:
            with self._session_maker.begin() as session:
                sequence = item.sequence_path.query_model(session).get_sequence()
                return str(sequence.state)
        else:
            return ""

    def get_sequence_progress(self, item: "SequenceHierarchyItem"):
        if item.is_sequence:
            with self._session_maker.begin() as session:
                sequence = item.sequence_path.query_model(session).get_sequence()
                return (
                    f"{sequence.number_completed_shots}/{sequence.total_number_shots}"
                )
        else:
            return ""

    def hasChildren(self, parent: QModelIndex = ...) -> bool:
        if not parent.isValid():
            return True
        else:
            return parent.internalPointer().is_folder()

    def canFetchMore(self, parent: QModelIndex) -> bool:
        if not parent.isValid():
            return False
        else:
            parent_item: SequenceHierarchyItem = parent.internalPointer()
            return parent_item.is_folder() and len(parent_item.children) == 0

    def fetchMore(self, parent: QModelIndex) -> None:
        if not parent.isValid():
            return

        parent_item: SequenceHierarchyItem = parent.internalPointer()

        if parent_item.is_sequence:
            return

        with self._session_maker.begin() as session:
            children = _build_children(
                parent_item.sequence_path,
                session,
            )

        self.beginInsertRows(parent, 0, len(children) - 1)
        parent_item.children = children
        self.endInsertRows()

    def is_sequence(self, index: QModelIndex) -> bool:
        item: "SequenceHierarchyItem" = index.internalPointer()
        with self._session_maker.begin() as session:
            return item.sequence_path.query_model(session).is_sequence()

    def create_new_folder(self, index: QModelIndex, name: str):
        if index.isValid():
            item: "SequenceHierarchyItem" = index.internalPointer()
        else:
            item = self._root
        new_path = item.sequence_path / name

        children = list(item.children)
        new_row = len(children)
        children.append(
            SequenceHierarchyItem(path=new_path, is_sequence=False, row=new_row)
        )
        with self._session_maker.begin() as session:
            number_created_paths = len(new_path.create(session))
            if number_created_paths == 1:
                self.beginInsertRows(index, new_row, new_row)
                item.children = children
                self.endInsertRows()
            elif number_created_paths == 0:
                logger.warning(f"Path \"{str(new_path)}\" already exists and was not created")
            elif number_created_paths > 1:
                raise RuntimeError(
                    "Created more than one path and couldn't update the views"
                )


class SequenceStats(TypedDict):
    state: State
    total_number_shots: Optional[int]
    number_completed_shots: int


class SequenceHierarchyItem(NodeMixin):
    """Item in the sequence hierarchy model.

    This class represents a single item in the sequence hierarchy model.
    """

    def __init__(
        self,
        path: SequencePath,
        is_sequence: bool,
        row: int,
        parent=None,
        children=None,
    ):
        super().__init__()
        self.sequence_path = path
        self.parent = parent
        self.is_sequence = is_sequence
        self.row = row
        if children:
            self.children = children

    def __repr__(self):
        return f"{self.__class__.__name__}({self.sequence_path})"

    def is_folder(self):
        return not self.is_sequence


def _build_children(
    parent: SequencePath, session: Session
) -> list[SequenceHierarchyItem]:

    children_items = []

    if parent.is_root():
        query_children = (
            session.query(SequencePathModel)
            .filter(func.nlevel(SequencePathModel.path) == 1)
            .order_by(SequencePathModel.creation_date)
        )
        children = session.scalars(query_children)
    else:
        path = parent.query_model(session)
        children = path.children
        children.sort(key=lambda x: x.creation_date)

    for child in children:
        child: SequencePathModel
        children_items.append(
            SequenceHierarchyItem(
                SequencePath(str(child.path)),
                row=len(children_items),
                is_sequence=child.is_sequence(),
            )
        )
    return children_items
