from collections.abc import Mapping, Iterable
from datetime import datetime
from threading import Lock
from typing import Optional, Any

import sqlalchemy.orm
from attrs import define
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy_utils import Ltree

from data_types import Data, DataLabel, is_data_label, is_data
from device.name import DeviceName, is_device_name
from experiment.configuration import ExperimentConfig
from parameter_types import Parameter, is_parameter
from sequence.configuration import SequenceConfig
from sequence.runtime import Sequence, SequenceNotFoundError, SequencePath, Shot
from sequence.runtime.path import PathNotFoundError
from sequence.runtime.sequence import SequenceNotEditableError, SequenceStats
from sequence.runtime.shot import ShotNotFoundError
from sql_model import SequenceModel, SequencePathModel, State
from sql_model.model import (
    ExperimentConfigModel,
    ShotModel,
    DataType,
    CurrentExperimentConfigModel,
)
from sql_model.sequence_state import InvalidSequenceStateError
from variable.name import DottedVariableName
from .experiment_session import (
    ExperimentSession,
    ExperimentSessionNotActiveError,
    ShotCollection,
)
from .sequence_file_system import PathIsSequenceError


@define
class SQLShotCollection(ShotCollection):
    parent_session: "SQLExperimentSession"

    def get_shot_data(self, shot: Shot, data_type: DataType) -> dict[str, Any]:
        shot_sql = self._query_shot_model(shot)
        return shot_sql.get_data(data_type, self._get_sql_session())

    def add_shot_data(
        self, shot: Shot, data: dict[str, Any], data_type: DataType
    ) -> None:
        session = self._get_sql_session()
        shot_sql = self._query_shot_model(shot)
        shot_sql.add_data(data, DataType.SCORE, session)
        session.flush()

    def get_shot_start_time(self, shot: Shot) -> datetime:
        shot_sql = self._query_shot_model(shot)
        return shot_sql.start_time

    def get_shot_end_time(self, shot: Shot) -> datetime:
        shot_sql = self._query_shot_model(shot)
        return shot_sql.end_time

    def _query_shot_model(self, shot: Shot) -> ShotModel:
        query_shot = select(ShotModel).where(
            ShotModel.sequence == self._query_sequence_model(shot.sequence),
            ShotModel.name == shot.name,
            ShotModel.index == shot.index,
        )
        result = self._get_sql_session().execute(query_shot)
        if shot := result.scalar():
            return shot
        else:
            raise ShotNotFoundError(f"Could not find shot {shot} in database")

    def _query_sequence_model(self, sequence: Sequence) -> SequenceModel:
        # noinspection PyProtectedMember
        return self.parent_session._query_sequence_model(sequence)

    def _get_sql_session(self) -> sqlalchemy.orm.Session:
        # noinspection PyProtectedMember
        return self.parent_session._get_sql_session()


@define(init=False)
class SQLExperimentSession(ExperimentSession):
    shot_collection: SQLShotCollection

    _sql_session: sqlalchemy.orm.Session
    _is_active: bool
    _lock: Lock

    def __init__(self, session: sqlalchemy.orm.Session, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sql_session = session
        self._is_active = False
        self._lock = Lock()
        self.shot_collection = SQLShotCollection(parent_session=self)

    def __enter__(self):
        with self._lock:
            if self._is_active:
                raise RuntimeError("Session is already active")
            self._transaction = self._sql_session.begin().__enter__()
            self._is_active = True
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with self._lock:
            self._transaction.__exit__(exc_type, exc_val, exc_tb)
            self._transaction = None
            self._is_active = False

    # Path methods
    def does_path_exists(self, path: SequencePath) -> bool:
        session = self._get_sql_session()
        return (
            session.scalar(
                select(SequencePathModel).filter(
                    SequencePathModel.path == Ltree(str(path))
                )
            )
            is not None
        )

    def is_sequence_path(self, path: SequencePath) -> bool:
        path = self._query_path_model(path)
        return bool(path.sequence)

    def create_path(self, path: SequencePath) -> list[SequencePath]:
        session = self._get_sql_session()

        created_paths: list[SequencePath] = []

        paths_to_create: list[SequencePath] = []

        for ancestor in path.get_ancestors(strict=False):
            if self.does_path_exists(ancestor):
                if self.is_sequence_path(ancestor):
                    raise PathIsSequenceError(
                        f"Cannot create path {path} because {ancestor} is already a sequence"
                    )
            else:
                paths_to_create.append(ancestor)
                SequencePathModel.create_path(str(ancestor), session)
                created_paths.append(ancestor)
        return created_paths

    def delete_path(self, path: SequencePath, delete_sequences: bool = False):
        session = self._get_sql_session()

        if not delete_sequences:
            if sub_sequences := path.get_contained_sequences(self):
                raise RuntimeError(
                    f"Cannot delete a path that contains sequences: {sub_sequences}"
                )

        session.delete(self._query_path_model(path))
        session.flush()

    def get_path_children(self, path: SequencePath) -> set[SequencePath]:
        session = self._get_sql_session()

        if path.is_root():
            query_children = (
                session.query(SequencePathModel)
                .filter(func.nlevel(SequencePathModel.path) == 1)
                .order_by(SequencePathModel.creation_date)
            )
            children = session.scalars(query_children)
        else:
            path = self._query_path_model(path)
            if path.sequence:
                raise RuntimeError("Cannot check children of a sequence")
            # noinspection PyUnresolvedReferences
            children = path.children
        return set(SequencePath(str(child.path)) for child in children)

    def get_path_creation_date(self, path: SequencePath) -> datetime:
        return self._query_path_model(path).creation_date

    # Sequence methods
    def does_sequence_exist(self, sequence: Sequence) -> bool:
        try:
            self._query_sequence_model(sequence)
            return True
        except SequenceNotFoundError:
            return False

    def get_sequence_state(self, sequence: Sequence) -> State:
        state = self._query_sequence_model(sequence).state
        return state

    def get_sequence_shots(self, sequence: Sequence) -> list[Shot]:
        sequence_model = self._query_sequence_model(sequence)
        return [Shot(sequence, shot.name, shot.index) for shot in sequence_model.shots][
            : sequence_model.number_completed_shots
        ]

    def set_sequence_state(self, sequence: Sequence, state: State):
        session = self._get_sql_session()
        sequence_model = self._query_sequence_model(sequence)
        sequence_model.set_state(state)
        session.flush()

    def get_sequence_creation_date(self, sequence: Sequence) -> datetime:
        sequence_sql = self._query_sequence_model(sequence)
        return sequence_sql.creation_date

    def get_sequence_config_yaml(self, sequence: Sequence) -> str:
        return self._query_sequence_model(sequence).config.sequence_config_yaml

    def set_sequence_config_yaml(
        self, sequence: Sequence, config_yaml: str, total_number_shots: Optional[int]
    ):
        sql_session = self._get_sql_session()
        session = sql_session
        sequence_model = self._query_sequence_model(sequence)
        if not sequence_model.state.is_editable():
            raise SequenceNotEditableError(
                f"Sequence is in state {sequence_model.state}"
            )
        sequence_model.total_number_shots = total_number_shots

        sequence_model.config.sequence_config_yaml = config_yaml
        sequence.modification_date = datetime.now()
        session.flush()

    def get_sequence_experiment_config(
        self, sequence: Sequence
    ) -> Optional[ExperimentConfig]:
        sequence_model = self._query_sequence_model(sequence)

        experience_config_model = sequence_model.get_experiment_config()
        if experience_config_model is None:
            return None
        return ExperimentConfig.from_yaml(
            experience_config_model.experiment_config_yaml
        )

    def set_sequence_experiment_config(
        self, sequence: Sequence, experiment_config: str
    ):
        if not self.get_sequence_state(sequence).is_editable():
            raise RuntimeError(
                "Cannot set experiment config for a sequence that is not in an editable"
                " state"
            )
        session = self._get_sql_session()
        sequence_model = self._query_sequence_model(sequence)
        sequence_model.set_experiment_config(experiment_config)
        session.flush()

    def get_sequence_stats(self, sequence: Sequence) -> SequenceStats:
        sequence_model = self._query_path_model(sequence.path).get_sequence()
        return SequenceStats(
            state=sequence_model.get_state(),
            total_number_shots=sequence_model.total_number_shots,
            number_completed_shots=sequence_model.get_number_completed_shots(),
            start_date=sequence_model.start_date,
            stop_date=sequence_model.stop_date,
        )

    def get_all_sequence_names(self) -> set[str]:
        query = select(SequenceModel)
        result = self._get_sql_session().execute(query)
        return {str(sequence.path) for sequence in result.scalars()}

    def query_sequence_stats(
        self, sequences: Iterable[Sequence]
    ) -> dict[SequencePath, SequenceStats]:
        paths = self._query_path_models([sequence.path for sequence in sequences])
        query = select(SequenceModel).where(
            SequenceModel.path_id.in_(path.id_ for path in paths)
        )
        result = self._get_sql_session().execute(query)
        return {
            SequencePath(str(sequence.path)): SequenceStats(
                state=sequence.get_state(),
                total_number_shots=sequence.total_number_shots,
                number_completed_shots=sequence.get_number_completed_shots(),
                start_date=sequence.start_date,
                stop_date=sequence.stop_date,
            )
            for sequence in result.scalars()
        }

    def create_sequence(
        self,
        path: SequencePath,
        sequence_config: SequenceConfig,
        experiment_config_name: Optional[str],
    ) -> Sequence:
        if not isinstance(sequence_config, SequenceConfig):
            raise TypeError(
                f"Type of sequence_config {type(sequence_config)} is not SequenceConfig"
            )

        path.create(self)
        if path.has_children(self):
            raise RuntimeError(
                f"Cannot create a sequence at {path} because it is a folder with"
                " children"
            )

        SequenceModel.create_sequence(
            str(path),
            sequence_config,
            experiment_config_name,
            self._get_sql_session(),
        )
        sequence = Sequence(path)
        return sequence

    def create_sequence_shot(
        self,
        sequence: Sequence,
        name: str,
        start_time: datetime,
        end_time: datetime,
        parameters: Mapping[DottedVariableName, Parameter],
        measures: Mapping[DeviceName, Mapping[DataLabel, Data]],
    ):
        if self.get_sequence_state(sequence) != State.RUNNING:
            raise InvalidSequenceStateError(
                f"Can't create a shot unless the sequence is running"
            )
        session = self._get_sql_session()
        sequence_model = self._query_sequence_model(sequence)
        shot = ShotModel.create_shot(
            sequence_model, name, start_time, end_time, session
        )
        shot.add_data(transform_parameters(parameters), DataType.PARAMETER, session)
        shot.add_data(transform_measures(measures), DataType.MEASURE, session)
        sequence_model.increment_number_completed_shots()
        session.flush()
        return Shot(sequence, shot.name, shot.index)

    def _query_sequence_model(self, sequence: Sequence) -> SequenceModel:
        try:
            path = self._query_path_model(sequence.path)
        except PathNotFoundError:
            raise SequenceNotFoundError(
                f"Could not find sequence '{sequence}' in database"
            )
        query_sequence = select(SequenceModel).where(SequenceModel.path == path)
        result = self._get_sql_session().execute(query_sequence)
        if sequence := result.scalar():
            return sequence
        else:
            raise SequenceNotFoundError(
                f"Could not find sequence '{sequence}' in database"
            )

    def _query_path_model(self, path: SequencePath) -> SequencePathModel:
        stmt = select(SequencePathModel).where(
            SequencePathModel.path == Ltree(str(path))
        )
        result = self._get_sql_session().execute(stmt)
        if path := result.scalar():
            return path
        else:
            raise PathNotFoundError(f"Could not find path '{path}' in database")

    def _query_path_models(
        self, paths: Iterable[SequencePath]
    ) -> list[SequencePathModel]:
        stmt = select(SequencePathModel).where(
            SequencePathModel.path.in_([Ltree(str(path)) for path in paths])
        )
        result = self._get_sql_session().execute(stmt)
        return result.scalars().all()

    # Experiment config methods
    def get_experiment_config(self, name: str) -> ExperimentConfig:
        return ExperimentConfig.from_yaml(
            ExperimentConfigModel.get_config(name, self._get_sql_session())
        )

    def get_all_experiment_config_names(self) -> set[str]:
        session = self._get_sql_session()
        query_names = session.query(ExperimentConfigModel.name)
        names = {name for name in session.scalars(query_names)}
        return names

    def get_experiment_config_yamls(
        self, from_date: Optional[datetime] = None, to_date: Optional[datetime] = None
    ) -> dict[str, str]:
        results = ExperimentConfigModel.get_configs(
            from_date,
            to_date,
            self._get_sql_session(),
        )
        return {name: yaml_ for name, yaml_ in results.items()}

    def add_experiment_config(
        self,
        experiment_config: ExperimentConfig,
        name: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> str:
        if name is None:
            name = self._get_new_experiment_config_name()
        yaml_ = experiment_config.to_yaml()
        assert ExperimentConfig.from_yaml(yaml_) == experiment_config
        ExperimentConfigModel.add_config(
            name=name,
            yaml=yaml_,
            comment=comment,
            session=self._get_sql_session(),
        )
        return name

    def set_current_experiment_config(self, name: str):
        if not isinstance(name, str):
            raise TypeError(f"Expected <str> for name, got {type(name)}")
        CurrentExperimentConfigModel.set_current_experiment_config(
            name=name, session=self._get_sql_session()
        )

    def get_current_experiment_config_name(self) -> Optional[str]:
        return CurrentExperimentConfigModel.get_current_experiment_config_name(
            session=self._get_sql_session()
        )

    def _get_sql_session(self) -> sqlalchemy.orm.Session:
        if not self._is_active:
            raise ExperimentSessionNotActiveError(
                "Experiment session was not activated"
            )
        return self._sql_session


def transform_parameters(
    parameters: Mapping[DottedVariableName, Parameter]
) -> dict[str, Parameter]:
    result = {}
    for dotted_name, parameter in parameters.items():
        if not isinstance(dotted_name, DottedVariableName):
            raise TypeError(
                "Expected instance of <DottedVariableName> for parameter name, got"
                f" {type(dotted_name)}"
            )
        if not is_parameter(parameter):
            raise TypeError(
                f"Expected instance of <Parameter> for parameter '{dotted_name}', got"
                f" {type(parameter)}"
            )
        result[str(dotted_name)] = parameter
    return result


def transform_measures(
    measures: Mapping[DeviceName, Mapping[DataLabel, Data]]
) -> dict[DeviceName, dict[DataLabel, Data]]:
    result = {}
    for device_name, data_group in measures.items():
        if not is_device_name(device_name):
            raise TypeError(
                "Expected instance of <DeviceName> for device name, got"
                f" {type(device_name)}"
            )
        new_group = {}
        for label, data in data_group.items():
            if not is_data_label(label):
                raise TypeError(
                    "Expected instance of <DataLabel> for data label, got"
                    f" {type(label)}"
                )
            if not is_data(data):
                raise TypeError(
                    f"Expected instance of <Data> for device '{device_name}', got"
                    f" {type(data)}"
                )
            new_group[DataLabel(label)] = data
        result[DeviceName(device_name)] = new_group
    return result
