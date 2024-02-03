import copy
import functools

from PyQt6 import QtCore
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QMenu, QMessageBox, QInputDialog, QLineEdit, QApplication
from core.session import ExperimentSessionMaker, PureSequencePath
from core.session.path import InvalidPathFormatError
from core.session.result import unwrap
from core.session.sequence import State
from core.session.sequence.iteration_configuration import (
    StepsConfiguration,
    ArangeLoop,
    ExecuteShot,
)
from core.session.sequence_collection import PathIsSequenceError
from core.session.shot import TimeLanes
from core.types.expression import Expression
from core.types.variable_name import DottedVariableName
from sequence_hierarchy import PathHierarchyView
from waiting_widget import run_with_wip_widget

DEFAULT_ITERATION_CONFIG = StepsConfiguration(
    steps=[
        ArangeLoop(
            variable=DottedVariableName("rep"),
            start=Expression("0"),
            stop=Expression("10"),
            step=Expression("1"),
            sub_steps=[ExecuteShot()],
        ),
    ]
)

DEFAULT_TIME_LANES = TimeLanes(
    step_names=["step 0"],
    step_durations=[Expression("...")],
    lanes={},
)


class EditablePathHierarchyView(PathHierarchyView):
    def __init__(
        self,
        session_maker: ExperimentSessionMaker,
        parent=None,
    ):
        super().__init__(session_maker, parent)
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)  # type: ignore

    def show_context_menu(self, pos):
        index = self.indexAt(pos)

        path = self._model.get_path(self._proxy_model.mapToSource(index))

        menu = QMenu(self)

        with self.session_maker() as session:
            is_sequence = unwrap(session.sequences.is_sequence(path))
        if not is_sequence:
            new_menu = QMenu("New...")
            menu.addMenu(new_menu)

            create_folder_action = QAction("folder")
            new_menu.addAction(create_folder_action)
            create_folder_action.triggered.connect(
                functools.partial(self.create_new_folder, path)
            )

            create_sequence_action = QAction("sequence")
            new_menu.addAction(create_sequence_action)
            create_sequence_action.triggered.connect(
                functools.partial(self.create_new_sequence, path)
            )
        if is_sequence:
            dupplicate_action = QAction("Duplicate")
            menu.addAction(dupplicate_action)
            dupplicate_action.triggered.connect(
                functools.partial(self.on_sequence_duplication_requested, path)
            )
            clear_action = QAction("Clear")
            menu.addAction(clear_action)
            clear_action.triggered.connect(
                functools.partial(self.on_clear_sequence_requested, path)
            )

        if not path.is_root():
            delete_action = QAction("Delete")
            menu.addAction(delete_action)
            delete_action.triggered.connect(functools.partial(self.delete, path))

        menu.exec(self.mapToGlobal(pos))

    def on_clear_sequence_requested(self, path: PureSequencePath) -> None:
        """Clear the sequence at the given path.

        This will revert the sequence to the draft state, effectively clearing all
        the data in it.
        During the process, a waiting widget will be shown to the user to prevent them
        from interacting with the sequence while it is being cleared.
        """

        def clear():
            with self.session_maker() as session:
                session.sequences.set_state(path, State.DRAFT)

        run_with_wip_widget(self, "Clearing sequence", clear)

    def on_sequence_duplication_requested(self, path: PureSequencePath):
        """Ask the user for a new sequence name and duplicate the sequence."""

        text, ok = QInputDialog().getText(
            self,
            f"Duplicate {path}...",
            "New sequence name:",
            QLineEdit.EchoMode.Normal,
            path.name,
        )
        if ok and text:
            try:
                if text.startswith(PureSequencePath.separator()):
                    new_path = PureSequencePath(text)
                else:
                    assert path.parent is not None
                    new_path = path.parent / text
            except InvalidPathFormatError:
                if (application := QApplication.instance()) is None:
                    raise RuntimeError("No QApplication instance")
                QMessageBox.critical(
                    self,
                    application.applicationName(),
                    f"The path '{text}' is not a valid path.",
                )
                return
            with self.session_maker() as session:
                iterations = session.sequences.get_iteration_configuration(path)
                timelanes = session.sequences.get_time_lanes(path)
                session.sequences.create(
                    new_path,
                    iterations,
                    timelanes,
                )

    def create_new_folder(self, path: PureSequencePath):
        text, ok = QInputDialog().getText(
            self,
            f"New folder in {path}...",
            "Folder name:",
            QLineEdit.EchoMode.Normal,
            "new folder",
        )
        if ok and text:
            new_path = path / text
            with self.session_maker() as session:
                session.paths.create_path(new_path)

    def create_new_sequence(self, path: PureSequencePath):
        text, ok = QInputDialog().getText(
            self,
            f"New sequence in {path}...",
            "Sequence name:",
            QLineEdit.EchoMode.Normal,
            "new sequence",
        )
        if ok and text:
            new_path = path / text
            with self.session_maker() as session:
                session.sequences.create(
                    new_path,
                    copy.deepcopy(DEFAULT_ITERATION_CONFIG),
                    copy.deepcopy(DEFAULT_TIME_LANES),
                )

    def delete(self, path: PureSequencePath):
        message = (
            f'You are about to delete the path "{path}".\n'
            "All data inside will be irremediably lost."
        )
        if self.exec_confirmation_message_box(message):
            with self.session_maker() as session:
                if unwrap(session.sequences.is_sequence(path)):
                    session.paths.delete_path(path, delete_sequences=True)
                else:
                    # An error will be raised if someone tries to delete a folder that
                    # contains sequences.
                    try:
                        session.paths.delete_path(path, delete_sequences=False)
                    except PathIsSequenceError:
                        QMessageBox.critical(
                            self,
                            QApplication.instance().applicationName(),
                            f"The path '{path}' contains sequences and therefore "
                            f"cannot be deleted",
                        )

    def exec_confirmation_message_box(self, message: str) -> bool:
        """Show a popup box to ask  a question."""

        message_box = QMessageBox(self)
        app = QApplication.instance()
        message_box.setWindowTitle(app.applicationName())
        message_box.setText(message)
        message_box.setInformativeText("Are you really sure you want to continue?")
        message_box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
        )
        message_box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        message_box.setIcon(QMessageBox.Icon.Warning)
        result = message_box.exec()
        if result == QMessageBox.StandardButton.Cancel:
            return False
        return True
