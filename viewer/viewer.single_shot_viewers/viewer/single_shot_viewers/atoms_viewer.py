import threading
from typing import Optional

from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from analyza.loading.importers import ShotImporter
from experiment.session import ExperimentSession, get_standard_experiment_session
from sequence.runtime import Shot
from .single_shot_viewer import SingleShotViewer


class AtomsViewer(SingleShotViewer):
    def __init__(
        self,
        *,
        importer: ShotImporter[dict[tuple[float, float], bool]],
        session: Optional[ExperimentSession] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent=parent)

        if session is None:
            session = get_standard_experiment_session()

        self._importer = importer
        self._session = session

        self._lock = threading.Lock()

        self._setup_ui()

    def _setup_ui(self) -> None:
        self._figure = Figure()
        self._axes = self._figure.add_subplot()
        self._axes.set_aspect("equal")
        self._canvas = FigureCanvasQTAgg(self._figure)

        self.setLayout(QVBoxLayout())
        navigation_toolbar = NavigationToolbar2QT(self._canvas, self)
        self.layout().addWidget(navigation_toolbar)
        self.layout().addWidget(self._canvas)

    def set_shot(self, shot: Shot) -> None:
        with self._lock, self._session.activate():
            try:
                atoms = self._importer(shot, self._session)
            except Exception as e:
                self._set_exception(e)
            else:
                self._paint_atoms(atoms)

    def update_view(self) -> None:
        self._canvas.draw()

    def _paint_atoms(self, atoms: dict[tuple[float, float], bool]) -> None:
        self._axes.clear()
        for (x, y), atom in atoms.items():
            if atom:
                self._axes.plot(x, y, "o", color="black")
            else:
                self._axes.plot(x, y, "o", color="black", alpha=0.1)

    def _set_exception(self, error: Exception):
        self._axes.clear()
        self._axes.text(
            0.5,
            0.5,
            f"{error!r}",
            horizontalalignment="center",
            verticalalignment="center",
            color="red",
        )
        self._canvas.draw()
