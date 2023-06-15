import threading
from collections.abc import Mapping
from typing import Optional

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt6.QtWidgets import QWidget, QTableView, QVBoxLayout

from analyza.loading.importers import ShotImporter
from experiment.session import ExperimentSession, get_standard_experiment_session
from parameter_types import ParameterType
from sequence.runtime import Shot
from .single_shot_viewer import SingleShotViewer

ParameterName = str


class ParamsViewer(SingleShotViewer):
    def __init__(
        self,
        *,
        importer: ShotImporter[Mapping[ParameterName, ParameterType]],
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
        self._model = ParamsModel()
        self._view = QTableView()
        self._view.setModel(self._model)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self._view)

    def set_shot(self, shot: Shot) -> None:
        with self._lock, self._session.activate():
            params = self._importer(shot, self._session)
        self._model.set_params(params)


class ParamsModel(QAbstractTableModel):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent=parent)

        self._params: dict[ParameterName, ParameterType] = {}

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._params)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 2

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if index.column() == 0:
                return str(list(self._params.keys())[index.row()])
            elif index.column() == 1:
                return str(list(self._params.values())[index.row()])
        return None

    def set_params(self, params: Mapping[ParameterName, ParameterType]) -> None:
        keys_values = zip(params.keys(), params.values())
        params = dict(sorted(keys_values, key=lambda x: x[0]))
        self.beginResetModel()
        self._params = dict(params)
        self.endResetModel()

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = ...):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                if section == 0:
                    return "Name"
                elif section == 1:
                    return "Value"
        return super().headerData(section, orientation, role)
