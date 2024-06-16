import asyncio
import functools
from collections.abc import Callable
from typing import Optional

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtWidgets import (
    QMainWindow,
    QDockWidget,
    QDialog,
)

from caqtus.experiment_control import SequenceInterruptedException
from caqtus.experiment_control.manager import ExperimentManager, Procedure
from caqtus.gui.common.exception_tree import ExceptionDialog
from caqtus.gui.common.waiting_widget import run_with_wip_widget
from caqtus.gui.condetrol.parameter_tables_editor import ParameterNamespaceEditor
from caqtus.session import ExperimentSessionMaker, PureSequencePath
from caqtus.types.parameter import ParameterNamespace
from ._main_window_ui import Ui_CondetrolMainWindow
from ..device_configuration_editors import DeviceConfigurationsDialog
from ..extension import CondetrolExtensionProtocol
from ..logger import logger
from ..path_view import EditablePathHierarchyView
from ..sequence_widget import SequenceWidget


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
        self._path_view = EditablePathHierarchyView(session_maker, self)
        self._global_parameters_editor = ParameterNamespaceEditor()
        self._connect_to_experiment_manager = connect_to_experiment_manager
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
        self._task_group = asyncio.TaskGroup()
        self._run_sequence_task: Optional[asyncio.Task] = None
        self.timer = QTimer(self)

    async def run_async(self) -> None:
        """Run the main window asynchronously."""

        async with self._task_group:
            self._task_group.create_task(self._path_view.run_async())
            self._task_group.create_task(self._monitor_global_parameters())
            self._task_group.create_task(self.sequence_widget.exec_async())

    def setup_ui(self):
        self.setupUi(self)
        self.setCentralWidget(self.sequence_widget)
        paths_dock = QDockWidget("Sequences", self)
        paths_dock.setObjectName("SequencesDock")
        paths_dock.setWidget(self._path_view)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, paths_dock)
        self.dock_menu.addAction(paths_dock.toggleViewAction())
        global_parameters_dock = QDockWidget("Global parameters", self)
        global_parameters_dock.setWidget(self._global_parameters_editor)
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
        self._path_view.sequence_double_clicked.connect(self.set_edited_sequence)
        self._path_view.sequence_start_requested.connect(self.start_sequence)
        self._path_view.sequence_interrupt_requested.connect(self.interrupt_sequence)
        self.sequence_widget.sequence_start_requested.connect(self.start_sequence)
        self._global_parameters_editor.parameters_edited.connect(
            self._on_global_parameters_edited
        )

    def set_edited_sequence(self, path: PureSequencePath):
        self.sequence_widget.set_sequence(path)

    def start_sequence(self, path: PureSequencePath):
        try:
            experiment_manager = run_with_wip_widget(
                self,
                "Connecting to experiment manager...",
                self._connect_to_experiment_manager,
            )
        except Exception as e:
            self.display_error("Failed to connect to experiment manager.", e)
            return
        sequence_already_running = (
            self._run_sequence_task is not None and not self._run_sequence_task.done()
        )
        if sequence_already_running:
            self.display_error(
                "A sequence is already running.",
                RuntimeError("A sequence is already running."),
            )
            return
        procedure = experiment_manager.create_procedure(
            "sequence launched from GUI", acquisition_timeout=1
        )
        self._run_sequence_task = self._task_group.create_task(
            self._run_sequence(procedure, path)
        )

    def on_procedure_exception(self, exception: Exception):
        self.display_error(
            f"An error occurred while running a sequence.",
            exception,
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

    def display_error(self, message: str, exception: Exception):
        logger.error(message, exc_info=exception)
        exception_dialog = ExceptionDialog(self)
        exception_dialog.set_exception(exception)
        exception_dialog.set_message(message)
        exception_dialog.exec()

    def interrupt_sequence(self, path: PureSequencePath):
        experiment_manager = run_with_wip_widget(
            self,
            "Connecting to experiment manager...",
            self._connect_to_experiment_manager,
        )
        # we're actually lying here because we interrupt the running procedure, which
        # may be different from the one passed in argument.
        experiment_manager.interrupt_running_procedure()

    def _on_global_parameters_edited(self, parameters: ParameterNamespace) -> None:
        with self.session_maker() as session:
            session.set_global_parameters(parameters)
            logger.info(f"Global parameters written to storage: {parameters}")
        self.sequence_widget.set_available_parameter_names(parameters.names())

    async def _monitor_global_parameters(self) -> None:
        while True:
            async with self.session_maker.async_session() as session:
                parameters = await session.get_global_parameters()
            if parameters != self._global_parameters_editor.get_parameters():
                self._global_parameters_editor.set_parameters(parameters)
                self.sequence_widget.set_available_parameter_names(parameters.names())
            await asyncio.sleep(0.2)

    async def _run_sequence(self, procedure: Procedure, sequence):
        with procedure:
            try:
                await asyncio.to_thread(procedure.start_sequence, sequence)
            except Exception as e:
                exception = RuntimeError(
                    f"An error occurred while starting the sequence {sequence}."
                )
                exception.__cause__ = e
                self._signal_exception_while_running_sequence(exception)
                return

            while await asyncio.to_thread(procedure.is_running_sequence):
                await asyncio.sleep(50e-3)

            if (exc := procedure.exception()) is not None:
                # Here we ignore the SequenceInterruptedException because it is
                # expected to happen when the sequence is interrupted and we don't
                # want to display it to the user as an actual error.
                if isinstance(exc, SequenceInterruptedException):
                    exc = None
                elif isinstance(exc, ExceptionGroup):
                    _, exc = exc.split(SequenceInterruptedException)
                if exc is not None:
                    self._signal_exception_while_running_sequence(exc)

    def _signal_exception_while_running_sequence(self, exception: Exception):
        # This is a bit ugly because on_procedure_exception runs a dialog, which
        # messes up the event loop, so instead we schedule the exception handling
        # to be done in the next event loop iteration.
        self.timer.singleShot(
            0, functools.partial(self.on_procedure_exception, exception)
        )
