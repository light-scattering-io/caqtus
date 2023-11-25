import contextlib
import threading
from collections.abc import Mapping
from typing import Self, Optional

import polars
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMainWindow, QWidget, QProgressBar

from analyza.loading.importers import ShotImporter
from experiment.session import ExperimentSessionMaker, ExperimentSession
from sequence.runtime import Sequence, Shot
from util.concurrent import BackgroundScheduler
from ._main_window_ui import Ui_MainWindow
from .._sequence_hierarchy_widget import SequenceHierarchyWidget
from ..data_loading import DataLoaderSelector, DataImporter, ShotData
from ..sequence_analyzer import SequenceAnalyzer
from ..visualization import VisualizerCreator, Visualizer, VisualizerCreatorSelector
from ..watchlist import WatchlistWidget


def import_nothing(shot: Shot, session: ExperimentSession) -> ShotData:
    return polars.DataFrame()


class GraphPlotMainWindow(QMainWindow, Ui_MainWindow):
    """Main window of GraphPlot.

    This is a main window widget used to plot graphs about a sequence (or collection of sequences). This widget has a
    dock on the left showing the sequence hierarchy from which the user can choose which sequences to watch. It also
    has a dock on the right with a tab for the sequence watchlist, another tab to choose how to import the data from a
    shot and a last tab to choose how to visualize the data. The central widget contains the actual visualization for
    the data of the sequences.
    """

    def __init__(
        self,
        session_maker: ExperimentSessionMaker,
        data_loaders: Mapping[str, ShotImporter],
        visualizer_creators: Mapping[str, VisualizerCreator],
    ) -> None:
        """Initialize a new GraphPlotMainWindow.

        Args:
            session_maker: An object that can create sessions containing access to the permanent storage of the
                experiment. The sequence hierarchy and the data from the shots are pulled from sessions created by this
                object.
            data_loaders: A mapping of shot importers that can be chosen from to load data from the shots.
            visualizer_creators: A mapping of objects that can create the central widget. User can choose from an
                element of this mapping to display information about sequences.
        """

        super().__init__()

        self._session_maker = session_maker

        self._exit_stack = contextlib.ExitStack()
        self._sequences_analyzer = SequenceAnalyzer(session_maker)
        self._background_scheduler = BackgroundScheduler(max_workers=1)
        self._watchlist_widget = WatchlistWidget(self._sequences_analyzer)
        self._data_loader_selector = DataLoaderSelector(data_loaders)
        self._visualizer_selector = VisualizerCreatorSelector(visualizer_creators)
        self._data_loader: DataImporter = import_nothing
        self._visualizer: Optional[Visualizer] = None

        self._sequence_hierarchy_widget = SequenceHierarchyWidget(self._session_maker)
        self._current_visualizer_lock = threading.Lock()
        self._loading_bar = QProgressBar()
        self._timer = QTimer(self)
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setupUi(self)
        self._sequences_dock.setWidget(self._sequence_hierarchy_widget)
        self._sequence_hierarchy_widget.sequence_double_clicked.connect(
            self._on_sequence_double_clicked
        )
        self._toolbox_dock.setTitleBarWidget(QWidget())
        while self._tool_box.count() > 0:
            self._tool_box.removeItem(0)
        self._tool_box.addItem(self._watchlist_widget, "Watchlist")
        self._tool_box.addItem(self._data_loader_selector, "Data loading")
        self._data_loader_selector.data_loader_selected.connect(
            self._on_data_loader_selected
        )
        self._tool_box.addItem(self._visualizer_selector, "Visualization")
        self._visualizer_selector.visualizer_selected.connect(
            self.change_current_visualizer
        )
        self._status_bar.addPermanentWidget(self._loading_bar)
        self._timer.timeout.connect(self._update_loading_bar)
        self._timer.start(50)

    def _update_loading_bar(self) -> None:
        current, maximum = self._sequences_analyzer.get_progress()
        if maximum == 0:
            current = 0
            maximum = 1
        self._loading_bar.setValue(current)
        self._loading_bar.setMaximum(maximum)

    def _on_data_loader_selected(self, data_loader: DataImporter) -> None:
        self._data_loader = data_loader
        for sequence in self._sequences_analyzer.sequences:
            self._sequences_analyzer.monitor_sequence(sequence, self._data_loader)

    def _on_sequence_double_clicked(self, sequence: Sequence) -> None:
        self._watchlist_widget.add_sequence(sequence, self._data_loader)

    def change_current_visualizer(self, visualizer: Visualizer) -> None:
        """Sets a new visualizer for the central widget."""

        # Here we loose all references to the old visualizer, so it will be freed. To avoid having functions running on
        # the old visualizer while it's being freed, we put all accesses to the current visualizer behind a lock.
        with self._current_visualizer_lock:
            self._visualizer = visualizer
            while self._central_widget.layout().count() > 0:
                self._central_widget.layout().itemAt(0).widget().setParent(None)
            self._central_widget.layout().addWidget(self._visualizer)

    def __enter__(self) -> Self:
        self._exit_stack.__enter__()
        self._exit_stack.enter_context(self._sequences_analyzer)
        self._exit_stack.enter_context(self._background_scheduler)
        self._background_scheduler.schedule_task(self._update_visualizer, 1)
        return self

    def _update_visualizer(self) -> None:
        """Feed new data to the current visualizer."""

        if self._visualizer is not None:
            dataframe = self._sequences_analyzer.get_dataframe()
            # Here we put a lock to ensure that sel._visualize won't be reassigned while it is processing some
            # data. The issue is that if we loose all references on the visualizer while it is processing data in
            # another thread, the C++ parts of a widget will be freed while they are still being used.
            with self._current_visualizer_lock:
                self._visualizer.update_data(dataframe)

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return self._exit_stack.__exit__(exc_type, exc_val, exc_tb)
