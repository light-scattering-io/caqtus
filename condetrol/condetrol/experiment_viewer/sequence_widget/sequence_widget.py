"""This module implements an editor for the configuration of a sequence.

It provides a pseudo-code editor for the different steps of the sequence. The only role
of this module is to generate and edit a yaml file that is then consumed by other parts.
"""

import logging
from abc import abstractmethod, ABCMeta
from functools import singledispatch
from pathlib import Path
from typing import Optional

import yaml
from PyQt5.QtCore import (
    QFileSystemWatcher,
    QAbstractItemModel,
    QModelIndex,
    Qt,
    QSize,
    QObject,
    pyqtSignal,
    QThread,
)
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtWidgets import (
    QDockWidget,
    QTreeView,
    QStyledItemDelegate,
    QWidget,
    QStyleOptionViewItem,
    QStyle,
    QAbstractItemView,
    QTabWidget,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsGridLayout,
    QGraphicsWidget,
    QLabel,
    QMenu,
    QAction,
)

from sequence import (
    SequenceConfig,
    Step,
    VariableDeclaration,
    SequenceStats,
    SequenceState,
    LinspaceLoop,
)
from sequence.sequence_config import ArangeLoop
from settings_model.settings_model import YAMLSerializable
from .sequence_arange_iteration_ui import Ui_ArangeDeclaration
from .sequence_linspace_iteration_ui import Ui_LinspaceDeclaration
from .sequence_variable_declaration_ui import Ui_VariableDeclaration

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


class QABCMeta(type(QObject), ABCMeta):
    pass


class StepWidget(QWidget, metaclass=QABCMeta):
    """Abstract class for a widget used to display/edit a sequence step"""

    @abstractmethod
    def set_step_data(self, data: Step):
        raise NotImplementedError()

    @abstractmethod
    def get_step_data(self) -> dict[str]:
        raise NotImplementedError()


class VariableDeclarationWidget(Ui_VariableDeclaration, StepWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.setAutoFillBackground(True)

    def set_step_data(self, declaration: VariableDeclaration):
        self.name_edit.setText(declaration.name)
        self.expression_edit.setText(declaration.expression)

    def get_step_data(self) -> dict[str]:
        return dict(name=self.name_edit.text(), expression=self.expression_edit.text())


class LinspaceIterationWidget(Ui_LinspaceDeclaration, StepWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.setAutoFillBackground(True)

    def set_step_data(self, data: LinspaceLoop):
        self.name_edit.setText(data.name)
        self.start_edit.setText(data.start)
        self.stop_edit.setText(data.stop)
        self.num_edit.setValue(data.num)

    def get_step_data(self):
        return dict(
            name=self.name_edit.text(),
            start=self.start_edit.text(),
            stop=self.stop_edit.text(),
            num=self.num_edit.value(),
        )


class ArangeIterationWidget(Ui_ArangeDeclaration, StepWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.setAutoFillBackground(True)

    def set_step_data(self, data: ArangeLoop):
        self.name_edit.setText(data.name)
        self.start_edit.setText(data.start)
        self.stop_edit.setText(data.stop)
        self.step_edit.setText(data.step)

    def get_step_data(self):
        return dict(
            name=self.name_edit.text(),
            start=self.start_edit.text(),
            stop=self.stop_edit.text(),
            step=self.step_edit.text(),
        )


# class ExecuteShotWidget(Ui_ExecuteShot, StepWidget):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.setupUi(self)
#         self.setAutoFillBackground(True)
#
#     def set_step_data(self, shot: ExecuteShot):
#         self.name_edit.setText(shot.name)
#         self.shot = shot
#
#     def get_step_data(self) -> ExecuteShot:
#         self.shot.name = self.name_edit.text()
#         return self.shot


@singledispatch
def create_editor(step: Step, _: QWidget) -> StepWidget:
    """Create an editor for a step depending on its type"""
    raise NotImplementedError(f"Not implemented for {type(step)}")


@create_editor.register
def _(_: VariableDeclaration, parent: QWidget):
    return VariableDeclarationWidget(parent)


@create_editor.register
def _(_: LinspaceLoop, parent: QWidget):
    return LinspaceIterationWidget(parent)


@create_editor.register
def _(_: ArangeLoop, parent: QWidget):
    return ArangeIterationWidget(parent)


class StepDelegate(QStyledItemDelegate):
    """Delegate for a sequence step (see PyQt Model/View/Delegate)

    This delegate creates a widget editor to edit the data of a step. It also paints the
    widget on the view when editing if done.
    """

    def createEditor(
        self, parent: QWidget, option: QStyleOptionViewItem, index: QModelIndex
    ) -> StepWidget:
        data: Step = index.data(role=Qt.ItemDataRole.EditRole)
        # noinspection PyTypeChecker
        editor = create_editor(data, None)
        editor.setParent(parent)
        return editor

    def setEditorData(self, editor: StepWidget, index: QModelIndex):
        editor.set_step_data(index.data(role=Qt.ItemDataRole.EditRole))

    def setModelData(
        self, editor: StepWidget, model: QAbstractItemModel, index: QModelIndex
    ) -> None:
        model.setData(index, editor.get_step_data(), Qt.ItemDataRole.EditRole)

    def updateEditorGeometry(
        self, editor: StepWidget, option: QStyleOptionViewItem, index: QModelIndex
    ):
        editor.setGeometry(option.rect)

    def sizeHint(self, option: "QStyleOptionViewItem", index: QModelIndex) -> QSize:
        step = index.data(role=Qt.ItemDataRole.DisplayRole)
        # noinspection PyTypeChecker
        w = create_editor(step, None)
        self.setEditorData(w, index)
        w.resize(option.rect.size())
        return w.sizeHint()

    def paint(
        self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex
    ) -> None:
        data = index.data(Qt.ItemDataRole.DisplayRole)
        if isinstance(data, Step):
            if option.state & QStyle.StateFlag.State_Selected:
                painter.fillRect(option.rect, option.palette.highlight())
            w = create_editor(data, option.widget)
            w.set_step_data(data)
            self.updateEditorGeometry(w, option, index)
            if not (option.state & QStyle.StateFlag.State_Enabled):
                w.setEnabled(False)
            pixmap = QPixmap(option.rect.size())
            w.render(pixmap)
            painter.drawPixmap(option.rect, pixmap)
        else:
            super().paint(painter, option, index)


class FinishedSignal(QObject):
    finished = pyqtSignal()


class SaveThread(QThread):
    finished = pyqtSignal()

    def __init__(self, data: str, file: Path):
        super().__init__()
        self._data = data
        self._file = file

    def run(self):
        with open(self._file, "w") as file:
            file.write(self._data)
        self.finished.emit()


class StepsModel(QAbstractItemModel):
    """Qt Model for sequence steps (see PyQt Model/View)

    This model provides data for a view to display the different steps of a sequence. It
    watches and saves the sequence steps on the sequence_path/sequence_config.yaml file.
    It also watches the sequence_path/sequence_state.yaml file, and if the sequence
    state is not 'DRAFT', it sets the data to not editable.
    """

    def __init__(self, sequence_path: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = sequence_path / "sequence_config.yaml"
        self.state_path = sequence_path / "sequence_state.yaml"

        with open(self.config_path) as file:
            self.config: SequenceConfig = yaml.safe_load(file)
        self.root = self.config.program
        self.layoutChanged.emit()

        self.sequence_config_watcher = QFileSystemWatcher()
        self.sequence_config_watcher.addPath(str(self.config_path))
        self.sequence_config_watcher.fileChanged.connect(self.change_sequence_config)

        self.sequence_state_watcher = QFileSystemWatcher()
        self.sequence_state_watcher.addPath(str(self.state_path))
        self.sequence_state_watcher.fileChanged.connect(self.change_sequence_state)
        self.sequence_state = SequenceState.DRAFT

        self.save_thread: Optional[SaveThread] = None

    def change_sequence_state(self, path):
        with open(path, "r") as file:
            stats: SequenceStats = yaml.safe_load(file)
        self.sequence_state = stats.state
        self.layoutChanged.emit()

    def change_sequence_config(self, sequence_config):
        with open(sequence_config) as file:
            self.config: SequenceConfig = yaml.safe_load(file)
        self.root = self.config.program
        self.layoutChanged.emit()

    def index(self, row: int, column: int, parent: QModelIndex = ...) -> QModelIndex:
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
        if not parent.isValid():
            current_node = self.root.children[row]
        else:
            parent_node: Step = parent.internalPointer()
            if row < len(parent_node.children):
                current_node = parent_node.children[row]
            else:
                return QModelIndex()

        return self.createIndex(row, column, current_node)

    def parent(self, child: QModelIndex) -> QModelIndex:
        if not child.isValid():
            return QModelIndex()

        child_node: Step = child.internalPointer()
        if not isinstance(child_node, Step):
            logger.debug("danger")
            return QModelIndex()
        if not child_node.is_root:
            parent_node = child_node.parent
            return self.createIndex(parent_node.row(), 0, child_node.parent)
        else:
            return QModelIndex()

    def rowCount(self, parent: QModelIndex = ...) -> int:
        if not parent.isValid():
            return len(self.root.children)
        else:
            parent_node: Step = parent.internalPointer()
            return len(parent_node.children)

    def columnCount(self, parent: QModelIndex = ...) -> int:
        return 1

    def data(self, index: QModelIndex, role: int = ...):
        if index.isValid():
            if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
                return index.internalPointer()

    def setData(self, index: QModelIndex, values: dict[str], role: int = ...) -> bool:
        edit = False
        if (
            index.isValid()
            and role == Qt.ItemDataRole.EditRole
            and self.sequence_state == SequenceState.DRAFT
        ):
            node: Step = index.internalPointer()
            for attr, value in values.items():
                setattr(node, attr, value)
            edit = True

        if edit:
            self.save_config(self.config)
        return edit

    def save_config(self, config):
        # save config to file in a new thread to avoid blocking the GUI
        # TODO: need to fix crashing when letting the thread finish before
        self.sequence_config_watcher.removePath(str(self.config_path))
        serialized_config = yaml.dump(
            config, Dumper=YAMLSerializable.get_dumper(), sort_keys=False
        )
        save = SaveThread(serialized_config, self.config_path)
        # noinspection PyTypeChecker
        save.finished.connect(
            lambda: self.sequence_config_watcher.addPath(str(self.config_path))
        )
        save.start()
        save.wait()

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if index.isValid():
            if index.column() == 0:
                flags = Qt.ItemFlag.ItemIsSelectable
                if self.sequence_state == SequenceState.DRAFT:
                    return (
                        flags | Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsEnabled
                    )
                else:
                    return flags
        return Qt.ItemFlag.NoItemFlags

    def insert_step(self, new_step: Step, index: QModelIndex):
        # insert at the end of all steps if clicked at invalid index
        if not index.isValid():
            position = len(self.root.children)
            self.beginInsertRows(QModelIndex(), position, position)
            self.root.children += (new_step,)
            self.endInsertRows()
        else:
            node: Step = index.internalPointer()
            # if the selected step can't have children, the new step is added below it
            if isinstance(node, VariableDeclaration):
                position = index.row() + 1
                self.beginInsertRows(QModelIndex(), position, position)
                new_children = list(node.parent.children)
                new_children.insert(position, new_step)
                node.parent.children = new_children
                self.endInsertRows()
            # otherwise it's added as the last child of the selected step
            else:
                position = len(node.children)
                self.beginInsertRows(index, position, position)
                new_children = list(node.children)
                new_children.insert(position, new_step)
                node.children = new_children
                self.endInsertRows()

        self.save_config(self.config)

    def removeRow(self, row: int, parent: QModelIndex = ...) -> bool:
        self.beginRemoveRows(parent, row, row)
        if not parent.isValid():
            parent = self.root
        else:
            parent: Step = parent.internalPointer()
        new_children = list(parent.children)
        new_children.pop(row)
        parent.children = new_children
        self.endRemoveRows()
        self.save_config(self.config)
        return True


class SequenceWidget(QDockWidget):
    """Dockable widget that shows the sequence steps and shot"""

    def __init__(self, sequence_path: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._path = sequence_path
        self.setWindowTitle(f"{self._path}")

        self.tab_widget = QTabWidget()
        self.setWidget(self.tab_widget)

        self.program_tree = self.create_sequence_tree()
        self.program_tree.customContextMenuRequested.connect(self.show_context_menu)
        self.tab_widget.addTab(self.program_tree, "Sequence steps")

        self.shot_widget = self.create_shot_widget()
        self.tab_widget.addTab(self.shot_widget, "Shot")

    def create_sequence_tree(self):
        tree = QTreeView()
        tree.setHeaderHidden(True)
        tree.setAnimated(True)
        tree.setContentsMargins(0, 0, 0, 0)
        program_model = StepsModel(self._path)
        tree.setModel(program_model)
        program_model.layoutChanged.connect(lambda: self.program_tree.expandAll())
        tree.expandAll()
        delegate = StepDelegate()
        tree.setItemDelegate(delegate)
        tree.setEditTriggers(QAbstractItemView.EditTrigger.AllEditTriggers)

        tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        return tree

    def show_context_menu(self, position):
        index = self.program_tree.indexAt(position)
        # noinspection PyTypeChecker
        model: StepsModel = self.program_tree.model()

        menu = QMenu(self.program_tree)

        add_menu = QMenu()
        add_menu.setTitle("Add..")
        menu.addMenu(add_menu)

        create_variable_action = QAction("variable declaration")
        add_menu.addAction(create_variable_action)
        create_variable_action.triggered.connect(
            lambda: model.insert_step(
                VariableDeclaration(name="", expression=""), index
            )
        )

        create_linspace_action = QAction("linspace loop")
        add_menu.addAction(create_linspace_action)
        create_linspace_action.triggered.connect(
            lambda: model.insert_step(
                LinspaceLoop(name="", start="", stop="", num=10), index
            )
        )

        create_arange_action = QAction("arange loop")
        add_menu.addAction(create_arange_action)
        create_arange_action.triggered.connect(
            lambda: model.insert_step(
                ArangeLoop(name="", start="", stop="", step=""), index
            )
        )

        if index.isValid():
            delete_action = QAction("Delete")
            menu.addAction(delete_action)
            delete_action.triggered.connect(
                lambda: model.removeRow(index.row(), index.parent())
            )

        menu.exec(self.program_tree.mapToGlobal(position))

    def create_shot_widget(self):
        scene = QGraphicsScene()
        textEdit = scene.addWidget(QLabel("Step 0"))
        pushButton = scene.addWidget(QLabel("Step 1"))

        layout = QGraphicsGridLayout()
        layout.addItem(textEdit, 0, 0)
        layout.addItem(pushButton, 0, 1)

        form = QGraphicsWidget()
        form.setLayout(layout)
        scene.addItem(form)

        view = QGraphicsView(scene)
        return view
