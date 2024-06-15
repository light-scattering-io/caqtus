import functools
from collections.abc import Set, Iterable
from typing import Optional

from PySide6 import QtCore
from PySide6.QtCore import Qt, QPersistentModelIndex, QModelIndex
from PySide6.QtGui import QKeySequence, QShortcut, QAction, QFont, QGuiApplication
from PySide6.QtWidgets import QWidget, QTreeView, QAbstractItemView, QMenu

from caqtus.gui.qtutil import block_signals
from caqtus.types.expression import Expression
from caqtus.types.iteration import (
    StepsConfiguration,
    VariableDeclaration,
    ExecuteShot,
    LinspaceLoop,
    ArangeLoop,
)
from caqtus.types.variable_name import DottedVariableName
from .delegate import StepDelegate
from .steps_model import StepsModel
from ..sequence_iteration_editor import SequenceIterationEditor


def create_variable_declaration():
    return VariableDeclaration(
        variable=DottedVariableName("new_variable"), value=Expression("...")
    )


def create_linspace_loop():
    return LinspaceLoop(
        variable=DottedVariableName("new_variable"),
        start=Expression("..."),
        stop=Expression("..."),
        num=10,
        sub_steps=[],
    )


def create_arange_loop():
    return ArangeLoop(
        variable=DottedVariableName("new_variable"),
        start=Expression("..."),
        stop=Expression("..."),
        step=Expression("..."),
        sub_steps=[],
    )


class StepsIterationEditor(QTreeView, SequenceIterationEditor[StepsConfiguration]):
    """Editor for the steps of a sequence.

    .. figure:: screenshot_StepsIterationEditor.png
       :align: center

       Example of a StepsIterationEditor.

    Args:
        parent: The parent widget of the editor.

    Signals:
        iteration_edited: Emitted when the iteration displayed by the editor is changed
            by the user.
            This signal is not emitted when the iteration is set programmatically.
    """

    iteration_edited = QtCore.Signal(StepsConfiguration)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._model = StepsModel(StepsConfiguration([]))
        self.setModel(self._model)
        self.expandAll()
        self.header().hide()
        self._delegate = StepDelegate(self)
        self.setItemDelegate(self._delegate)

        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)

        self.delete_shortcut = QShortcut(QKeySequence("Delete"), self)
        self.delete_shortcut.activated.connect(self.delete_selected)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

        self._model.dataChanged.connect(self._emit_iteration_edited)
        self._model.rowsInserted.connect(self._emit_iteration_edited)
        self._model.rowsRemoved.connect(self._emit_iteration_edited)
        self._model.modelReset.connect(self._emit_iteration_edited)

        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        self.select_all_shortcut = QShortcut(QKeySequence("Ctrl+A"), self)
        self.select_all_shortcut.activated.connect(self.selectAll)
        self.copy_shortcut = QShortcut(QKeySequence("Ctrl+C"), self)
        self.copy_shortcut.activated.connect(self.copy_selection)
        self.cut_shortcut = QShortcut(QKeySequence("Ctrl+X"), self)
        self.cut_shortcut.activated.connect(self.cut_selected)
        self.paste_shortcut = QShortcut(QKeySequence("Ctrl+V"), self)
        self.paste_shortcut.activated.connect(self.paste)

        font = QFont("JetBrains Mono")
        font.setPixelSize(13)
        self.setFont(font)

    def _emit_iteration_edited(self, *args, **kwargs):
        self.iteration_edited.emit(self.get_iteration())
        self.expandAll()

    def get_iteration(self) -> StepsConfiguration:
        """Returns the iteration currently displayed by the editor.

        This method returns a copy of the iteration, so the caller can modify it
        without affecting the editor.
        """

        return self._model.get_steps()

    def set_iteration(self, iteration: StepsConfiguration):
        """Updates the iteration displayed by the editor.

        This method does not emit the signal iteration_edited.
        """

        with block_signals(self):
            self._model.set_steps(iteration)
        self.expandAll()

    def set_available_parameter_names(self, parameter_names: Set[DottedVariableName]):
        """Sets the names that can be used in the iteration.

        The names are used to populate the auto-completion of variable names when the
        user try to edit the name of a variable or a loop.
        """

        self._delegate.set_available_names(parameter_names)

    def set_read_only(self, read_only: bool) -> None:
        """Sets the editor in read-only mode.

        When the editor is in read-only mode, the user cannot edit the steps.
        Even if the editor is in read-only mode, the iteration can still be set
        programmatically with :meth:`set_iteration`.
        """

        self._model.set_read_only(read_only)

    def delete_selected(self) -> None:
        self.remove_indices(self.selectedIndexes())

    def remove_indices(self, indices: Iterable[QModelIndex]) -> None:
        # Need to be careful that the indexes are not invalidated by the removal of
        # previous rows, that's why we convert them to QPersistentModelIndex.
        persistent_indices = [QPersistentModelIndex(index) for index in indices]
        for index in persistent_indices:
            if index.isValid():
                self._model.removeRow(index.row(), index.parent())

    def copy_selection(self) -> None:
        selected_indexes = self.selectedIndexes()
        data = self._model.mimeData(selected_indexes)
        clipboard = QGuiApplication.clipboard()
        clipboard.setMimeData(data)

    def cut_selected(self) -> None:
        self.copy_selection()
        self.remove_indices(self.selectedIndexes())

    def paste(self) -> None:
        clipboard = QGuiApplication.clipboard()
        data = clipboard.mimeData()

        if self.selectedIndexes():
            first = QPersistentModelIndex(self.selectedIndexes()[0])
            parent = first.parent()
            row = first.row()
            if self._model.dropMimeData(data, Qt.DropAction.MoveAction, row, 0, parent):
                self._model.removeRow(first.row(), first.parent())
        else:
            parent = QModelIndex()
            row = self._model.rowCount()
            self._model.dropMimeData(data, Qt.DropAction.MoveAction, row, 0, parent)

    def show_context_menu(self, position):
        index = self.indexAt(position)
        if not index.isValid():
            return
        menu = QMenu(self)

        add_menu = QMenu()
        add_menu.setTitle("Insert above...")
        menu.addMenu(add_menu)

        create_variable_action = QAction("variable")
        add_menu.addAction(create_variable_action)
        new_variable = create_variable_declaration()
        create_variable_action.triggered.connect(
            functools.partial(self._model.insert_above, new_variable, index)
        )
        create_shot_action = QAction("Shot")
        add_menu.addAction(create_shot_action)
        new_shot = ExecuteShot()
        create_shot_action.triggered.connect(
            functools.partial(self._model.insert_above, new_shot, index)
        )
        create_linspace_action = QAction("Linspace loop")
        add_menu.addAction(create_linspace_action)
        new_linspace = create_linspace_loop()
        create_linspace_action.triggered.connect(
            functools.partial(self._model.insert_above, new_linspace, index)
        )
        create_arange_action = QAction("Arange loop")
        add_menu.addAction(create_arange_action)
        new_arange = create_arange_loop()
        create_arange_action.triggered.connect(
            functools.partial(self._model.insert_above, new_arange, index)
        )
        menu.exec(self.mapToGlobal(position))
