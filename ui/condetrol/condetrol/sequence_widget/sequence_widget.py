from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import pyqtSignal, QThread, QTimer, Qt
from PyQt6.QtWidgets import QWidget

from core.session import ExperimentSessionMaker, PureSequencePath, BoundSequencePath
from core.session._return_or_raise import unwrap
from core.session.path_hierarchy import PathNotFoundError
from core.session.sequence import State, Sequence
from core.session.sequence_collection import PathIsNotSequenceError, SequenceStats

from .sequence_widget_ui import Ui_SequenceWidget
from ..sequence_iteration_editors import create_default_editor


class SequenceWidget(QWidget, Ui_SequenceWidget):
    sequence_start_requested = pyqtSignal(PureSequencePath)

    def __init__(
        self,
        sequence: PureSequencePath,
        session_maker: ExperimentSessionMaker,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setupUi(self)
        self.session_maker = session_maker
        self.sequence_path = sequence
        self.apply_state(State.DRAFT)

        with self.session_maker() as session:
            sequence = Sequence(BoundSequencePath(self.sequence_path, session))
            iteration_config = sequence.get_iteration_configuration()
        self.iteration_editor = create_default_editor(iteration_config)
        self.iteration_editor.iteration_changed.connect(
            self.on_sequence_iteration_changed
        )
        self.tabWidget.clear()
        self.tabWidget.addTab(self.iteration_editor, "Iteration")
        self.tabWidget.addTab(QWidget(), "Shot")

        self.state_watcher_thread = self.StateWatcherThread(self)

        self.setup_connections()
        self.state_watcher_thread.start()

    def setup_connections(self):
        self.start_button.clicked.connect(
            lambda _: self.sequence_start_requested.emit(self.sequence_path)
        )
        self.state_watcher_thread.sequence_not_found.connect(self.deleteLater)
        self.state_watcher_thread.stats_changed.connect(self.apply_stats)

    def on_sequence_iteration_changed(self):
        iterations = self.iteration_editor.get_iteration()
        with self.session_maker() as session:
            session.sequence_collection.set_iteration_configuration(
                Sequence(BoundSequencePath(self.sequence_path, session)), iterations
            )

    def apply_stats(self, stats: SequenceStats):
        self.apply_state(stats.state)

    def closeEvent(self, event):
        self.state_watcher_thread.quit()
        self.state_watcher_thread.wait()
        super().closeEvent(event)

    def apply_state(self, state: State):
        if state == State.DRAFT:
            self.start_button.setEnabled(True)
        else:
            self.start_button.setEnabled(False)
        if state in {State.RUNNING}:
            self.interrupt_button.setEnabled(True)
        else:
            self.interrupt_button.setEnabled(False)
        if state in {State.FINISHED, State.INTERRUPTED, State.CRASHED}:
            self.clear_button.setEnabled(True)
        else:
            self.clear_button.setEnabled(False)

    class StateWatcherThread(QThread):
        stats_changed = pyqtSignal(SequenceStats)
        sequence_not_found = pyqtSignal()

        def __init__(self, sequence_widget: SequenceWidget):
            super().__init__(sequence_widget)
            self.sequence_widget = sequence_widget
            with self.sequence_widget.session_maker() as session:
                self.stats = unwrap(
                    session.sequence_collection.get_stats(
                        self.sequence_widget.sequence_path
                    )
                )

        def run(self) -> None:
            def watch():
                with self.sequence_widget.session_maker() as session:
                    try:
                        stats = unwrap(
                            session.sequence_collection.get_stats(
                                self.sequence_widget.sequence_path
                            )
                        )
                        if stats != self.stats:
                            self.stats = stats
                            self.stats_changed.emit(stats)
                    except (PathNotFoundError, PathIsNotSequenceError):
                        self.sequence_not_found.emit()
                        self.quit()

            timer = QTimer()
            timer.timeout.connect(watch)
            timer.start(50)
            self.exec()
            timer.stop()
