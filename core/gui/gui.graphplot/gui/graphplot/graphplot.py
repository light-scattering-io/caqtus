import asyncio
import logging
from typing import Self, Optional

import PySide6.QtAsyncio as QtAsyncio
import polars
import qtawesome
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QDockWidget

from core.data_analysis.loading import DataImporter
from core.session import ExperimentSessionMaker
from graphplot.data_loading import DataLoader
from graphplot.views.error_bar_view import ErrorBarView
from sequence_hierarchy import PathHierarchyView
from .graphplot_main_window_ui import Ui_GraphPlotMainWindow

logger = logging.getLogger(__name__)


async def wrap(coro):
    # noinspection PyBroadException
    try:
        return await coro
    except Exception:
        logger.critical("Unhandled exception", exc_info=True)
        QApplication.quit()


class GraphPlot:
    def __init__(
        self, data_importer: DataImporter, session_maker: ExperimentSessionMaker, *args
    ) -> None:
        """
        Args:
            data_importer: A callable used to import data from shots.
            session_maker: A callable used to create sessions from which the application can retrieve data.
        """

        self.app = QApplication(*args)
        self.app.setApplicationName("GraphPlot")
        self.app.setStyle("Fusion")
        self.app.setWindowIcon(qtawesome.icon("mdi6.chart-line", size=64))
        self.main_window = GraphPlotMainWindow(data_importer, session_maker)

    def run(self) -> None:
        with self.main_window:
            self.main_window.show()
            QtAsyncio.run(self.main_window.start())


class GraphPlotMainWindow(QMainWindow, Ui_GraphPlotMainWindow):
    """The main window for the GraphPlot application.

    On the left, it displays a tree view of the experiment session's sequences.
    On the right, there is a widget to define how to import data from the sequences.
    In the middle, there is a view of the data loaded from the sequences.
    """

    def __init__(
        self,
        data_loader: DataImporter,
        session_maker: ExperimentSessionMaker,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.setupUi(self)

        self.session_maker = session_maker
        self.path_view = PathHierarchyView(self.session_maker, self)
        paths_dock = QDockWidget("Sequences", self)
        paths_dock.setWidget(self.path_view)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, paths_dock)
        self.docks_menu.addAction(paths_dock.toggleViewAction())
        self.loader = DataLoader(data_loader, session_maker, self)
        loader_dock = QDockWidget("Watchlist", self)
        loader_dock.setWidget(self.loader)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, loader_dock)
        self.docks_menu.addAction(loader_dock.toggleViewAction())

        self.path_view.sequence_double_clicked.connect(
            self.loader.add_sequence_to_watchlist
        )
        self.view = ErrorBarView(self)
        self.setCentralWidget(self.view)
        self.task: Optional[asyncio.Task] = None

    async def start(self):
        # TODO: Remove this when QtAsyncio has proper exception handling in tasks
        async with asyncio.TaskGroup() as tg:
            tg.create_task(wrap(self.loader.process()))
            tg.create_task(wrap(self.update_view()))

    async def update_view(self):
        while True:
            sequences_data = self.loader.get_sequences_data()
            non_empty_dataframes = [
                d for d in sequences_data.values() if not d.is_empty()
            ]
            if non_empty_dataframes:
                data = await asyncio.to_thread(polars.concat, non_empty_dataframes)
            else:
                data = polars.DataFrame()
            await self.view.update_data(data)
            await asyncio.sleep(400e-3)

    def __enter__(self) -> Self:
        self.path_view.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return self.path_view.__exit__(exc_type, exc_val, exc_tb)