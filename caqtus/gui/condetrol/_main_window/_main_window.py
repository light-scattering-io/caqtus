from __future__ import annotations

import functools
from collections.abc import Callable

import anyio
import anyio.to_thread
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtWidgets import (
    QMainWindow,
    QDockWidget,
    QDialog,
)

from caqtus.experiment_control.manager import ExperimentManager, Procedure
from caqtus.gui.common.exception_tree import ExceptionDialog
from caqtus.gui.common.waiting_widget import run_with_wip_widget
from caqtus.gui.condetrol._parameter_tables_editor import ParameterNamespaceEditor
from caqtus.session import ExperimentSessionMaker, PureSequencePath
from caqtus.types.parameter import ParameterNamespace
from caqtus.types.recoverable_exceptions import (
    split_recoverable,
    SequenceInterruptedException,
)
from ._main_window_ui import Ui_CondetrolMainWindow
from .._extension import CondetrolExtensionProtocol
from .._logger import logger
from .._path_view import EditablePathHierarchyView
from .._sequence_widget import SequenceWidget
from ..device_configuration_editors._configurations_editor import (
    DeviceConfigurationsDialog,
)


class CondetrolWindowHandler:
    def __init__(
        self, main_window: CondetrolMainWindow, session_maker: ExperimentSessionMaker
    ):
        self.main_window = main_window
        self.session_maker = session_maker
        self.task_group = anyio.create_task_group()
        self.is_running_sequence = False

        self.main_window.path_view.sequence_start_requested.connect(self.start_sequence)
        self.main_window.sequence_widget.sequence_start_requested.connect(
            self.start_sequence
        )

    async def run_async(self) -> None:
        """Run the main window asynchronously."""

        async with self.task_group:
            self.task_group.start_soon(self.main_window.path_view.run_async)
            self.task_group.start_soon(self._monitor_global_parameters)
            self.task_group.start_soon(self.main_window.sequence_widget.exec_async)

    async def _monitor_global_parameters(self) -> None:
        while True:
            async with self.session_maker.async_session() as session:
                parameters = await session.get_global_parameters()
            if parameters != self.main_window.global_parameters_editor.get_parameters():
                self.main_window.global_parameters_editor.set_parameters(parameters)
                self.main_window.sequence_widget.set_available_parameter_names(
                    parameters.names()
                )
            await anyio.sleep(0.2)

    def start_sequence(self, path: PureSequencePath):
        try:
            experiment_manager = run_with_wip_widget(
                self.main_window,
                "Connecting to experiment manager...",
                self.main_window.connect_to_experiment_manager,
            )
        except Exception as e:
            logger.error("Failed to connect to experiment manager.", exc_info=e)
            self.main_window.display_error(
                "Failed to connect to experiment manager.", e
            )
            return

        if self.is_running_sequence:
            self.main_window.display_error(
                "A sequence is already running.",
                RuntimeError("A sequence is already running."),
            )
            return
        procedure = experiment_manager.create_procedure(
            "sequence launched from GUI", acquisition_timeout=1
        )
        self.task_group.start_soon(self._run_sequence, procedure, path)
        self.is_running_sequence = True

    async def _run_sequence(self, procedure: Procedure, sequence):
        with procedure:
            try:
                await anyio.to_thread.run_sync(procedure.start_sequence, sequence)
            except Exception as e:
                exception = RuntimeError(
                    f"An error occurred while starting the sequence {sequence}."
                )
                exception.__cause__ = e
                self.main_window.signal_exception_while_running_sequence(exception)
                return

            while await anyio.to_thread.run_sync(procedure.is_running_sequence):
                await anyio.sleep(50e-3)

            if (exc := procedure.exception()) is not None:
                # Here we ignore the SequenceInterruptedException because it is
                # expected to happen when the sequence is interrupted and we don't
                # want to display it to the user as an actual error.
                if isinstance(exc, SequenceInterruptedException):
                    exc = None
                elif isinstance(exc, ExceptionGroup):
                    _, exc = exc.split(SequenceInterruptedException)
                if exc is not None:
                    self.main_window.signal_exception_while_running_sequence(exc)
        self.is_running_sequence = False


class CondetrolMainWindow(QMainWindow, Ui_CondetrolMainWindow):
    """The main window of the Condetrol GUI.

    Parameters
    ----------
    session_maker
        A callable that returns an ExperimentSession.
        This is used to access the storage in which to look for sequences to display
        and edit.
    connect_to_experiment_manager
        A callable that is called to connect to an experiment manager in charge of
        running sequences.
        This is used to submit sequences to the manager when the user starts them
        in the GUI.
    extension
        The extension that provides the GUI with the necessary tools to edit sequences
        and device configurations.
    """

    def __init__(
        self,
        session_maker: ExperimentSessionMaker,
        connect_to_experiment_manager: Callable[[], ExperimentManager],
        extension: CondetrolExtensionProtocol,
    ):
        super().__init__()
        self.path_view = EditablePathHierarchyView(session_maker, self)
        self.global_parameters_editor = ParameterNamespaceEditor()
        self.connect_to_experiment_manager = connect_to_experiment_manager
        self.session_maker = session_maker
        self.sequence_widget = SequenceWidget(
            self.session_maker, extension.lane_extension, parent=self
        )
        self.device_configurations_dialog = DeviceConfigurationsDialog(
            extension.device_extension, parent=self
        )
        self.setup_ui()
        self.restore_window()
        self.setup_connections()
        self.timer = QTimer(self)

    def setup_ui(self):
        self.setupUi(self)
        self.setCentralWidget(self.sequence_widget)
        paths_dock = QDockWidget("Sequences", self)
        paths_dock.setObjectName("SequencesDock")
        paths_dock.setWidget(self.path_view)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, paths_dock)
        self.dock_menu.addAction(paths_dock.toggleViewAction())
        global_parameters_dock = QDockWidget("Global parameters", self)
        global_parameters_dock.setWidget(self.global_parameters_editor)
        global_parameters_dock.setObjectName("GlobalParametersDock")
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, global_parameters_dock
        )
        self.dock_menu.addAction(global_parameters_dock.toggleViewAction())
        # We hide the global parameters dock by default to reduce clutter when
        # launching the app the first time.
        global_parameters_dock.hide()

    def setup_connections(self):
        self.action_edit_device_configurations.triggered.connect(
            self.open_device_configurations_editor
        )
        self.path_view.sequence_double_clicked.connect(self.set_edited_sequence)
        self.path_view.sequence_interrupt_requested.connect(self.interrupt_sequence)
        self.global_parameters_editor.parameters_edited.connect(
            self._on_global_parameters_edited
        )

    def set_edited_sequence(self, path: PureSequencePath):
        self.sequence_widget.set_sequence(path)

    def on_procedure_exception(self, exception: Exception):
        recoverable, non_recoverable = split_recoverable(exception)
        if recoverable:
            logger.warning(
                f"Recoverable exception occurred while running a sequence",
                exc_info=recoverable,
            )
        if non_recoverable:
            # The exception will be logged anyway when condetrol crashes, so we don't
            # need to log it here.
            raise non_recoverable

        assert recoverable is not None

        self.display_error(
            f"An error occurred while running a sequence.",
            recoverable,
        )

    def open_device_configurations_editor(self) -> None:
        with self.session_maker() as session:
            previous_device_configurations = dict(session.default_device_configurations)
        self.device_configurations_dialog.set_device_configurations(
            previous_device_configurations
        )
        if self.device_configurations_dialog.exec() == QDialog.DialogCode.Accepted:
            new_device_configurations = (
                self.device_configurations_dialog.get_device_configurations()
            )
            with self.session_maker() as session:
                for device_name in session.default_device_configurations:
                    if device_name not in new_device_configurations:
                        del session.default_device_configurations[device_name]
                for (
                    device_name,
                    device_configuration,
                ) in new_device_configurations.items():
                    session.default_device_configurations[device_name] = (
                        device_configuration
                    )

    def closeEvent(self, a0):
        self.save_window()
        super().closeEvent(a0)

    def restore_window(self) -> None:
        ui_settings = QSettings()
        state = ui_settings.value(f"{__name__}/state", defaultValue=None)
        if state is not None:
            self.restoreState(state)
        geometry = ui_settings.value(f"{__name__}/geometry", defaultValue=None)
        if geometry is not None:
            self.restoreGeometry(geometry)

    def save_window(self) -> None:
        ui_settings = QSettings()
        ui_settings.setValue(f"{__name__}/state", self.saveState())
        ui_settings.setValue(f"{__name__}/geometry", self.saveGeometry())

    def display_error(self, message: str, exception: BaseException):
        exception_dialog = ExceptionDialog(self)
        exception_dialog.set_exception(exception)
        exception_dialog.set_message(message)
        exception_dialog.exec()

    def interrupt_sequence(self, path: PureSequencePath):
        experiment_manager = run_with_wip_widget(
            self,
            "Connecting to experiment manager...",
            self.connect_to_experiment_manager,
        )
        # we're actually lying here because we interrupt the running procedure, which
        # may be different from the one passed in argument.
        experiment_manager.interrupt_running_procedure()

    def _on_global_parameters_edited(self, parameters: ParameterNamespace) -> None:
        with self.session_maker() as session:
            session.set_global_parameters(parameters)
            logger.info(f"Global parameters written to storage: {parameters}")
        self.sequence_widget.set_available_parameter_names(parameters.names())

    def signal_exception_while_running_sequence(self, exception: Exception):
        # This is a bit ugly because on_procedure_exception runs a dialog, which
        # messes up the event loop, so instead we schedule the exception handling
        # to be done in the next event loop iteration.
        self.timer.singleShot(
            0, functools.partial(self.on_procedure_exception, exception)
        )