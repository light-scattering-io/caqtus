from datetime import datetime
from typing import Optional, TypedDict, Self, Iterable, Mapping, TYPE_CHECKING

from data_types import Data, DataLabel
from device.name import DeviceName
from experiment.configuration import ExperimentConfig
from parameter_types import Parameter
from sequence.configuration import SequenceConfig, ShotConfiguration, SequenceSteps
from sql_model import State
from variable.name import DottedVariableName
from .path import SequencePath
from .shot import Shot

if TYPE_CHECKING:
    from experiment.session import ExperimentSession
else:
    ExperimentSession = "ExperimentSession"


class SequenceNotEditableError(Exception):
    pass


class Sequence:
    """Contains the runtime information and data of a sequence.

    Only methods that take an ExperimentSession argument actually connect to the
    permanent storage of the experiment. Such methods can raise SequenceNotFoundError if
    the sequence does not exist in the session. They are also expected to be
    comparatively slow since they require a file system access, possibly over the
    network.
    """

    def __init__(self, path: SequencePath | str):
        if isinstance(path, SequencePath):
            self._path = path
        elif isinstance(path, str):
            self._path = SequencePath(path)
        else:
            raise TypeError(
                f"Expected instance of <SequencePath> or <str>, got {type(path)}"
            )

    def __str__(self):
        return f'Sequence("{str(self._path)}")'

    @property
    def path(self) -> SequencePath:
        return self._path

    def exists(self, experiment_session: ExperimentSession) -> bool:
        """Check if the sequence exists.

        Args:
            experiment_session: The active experiment session in which to check for the
            sequence.
        """

        return experiment_session.does_sequence_exist(self)

    def get_creation_date(self, experiment_session: ExperimentSession) -> datetime:
        """Get the creation date of the sequence.

        Args:
            experiment_session: The active experiment session from which to get the
            creation date.
        Raises:
            SequenceNotFoundError: If the sequence does not exist in the session.
        """

        return experiment_session.get_sequence_creation_date(self)

    def get_config(self, experiment_session: ExperimentSession) -> SequenceConfig:
        """Return the configuration of the sequence.

        Args:
            experiment_session: The active experiment session from which to get the
            configuration.
        Raises:
            SequenceNotFoundError: If the sequence does not exist in the session.
        """

        yaml = experiment_session.get_sequence_config_yaml(self)
        return SequenceConfig.from_yaml(yaml)

    def set_config(self, config: SequenceConfig, experiment_session: ExperimentSession):
        if not isinstance(config, SequenceConfig):
            raise TypeError(
                f"Expected instance of <SequenceConfig>, got {type(config)}"
            )

        yaml = config.to_yaml()
        assert (
            SequenceConfig.from_yaml(yaml) == config
        )  # will trigger validation before saving

        experiment_session.set_sequence_config_yaml(
            self, yaml, config.compute_total_number_of_shots()
        )

    def set_experiment_config(
        self, experiment_config: str, experiment_session: ExperimentSession
    ):
        experiment_session.set_sequence_experiment_config(self, experiment_config)

    def get_experiment_config(
        self, experiment_session: ExperimentSession
    ) -> Optional[ExperimentConfig]:
        return experiment_session.get_sequence_experiment_config(self)

    def set_shot_config(
        self,
        shot_name: str,
        shot_config: ShotConfiguration,
        experiment_session: ExperimentSession,
    ):
        if not isinstance(shot_config, ShotConfiguration):
            raise TypeError(
                f"Expected instance of <ShotConfiguration>, got {type(shot_config)}"
            )
        sequence_config = self.get_config(experiment_session)
        sequence_config.shot_configurations[shot_name] = shot_config
        self.set_config(sequence_config, experiment_session)

    def set_steps_program(
        self, steps: SequenceSteps, experiment_session: ExperimentSession
    ):
        """Set the steps of the sequence."""

        if not isinstance(steps, SequenceSteps):
            raise TypeError(f"Expected instance of <SequenceSteps>, got {type(steps)}")
        sequence_config = self.get_config(experiment_session)
        sequence_config.program = steps
        self.set_config(sequence_config, experiment_session)

    def get_state(self, experiment_session: ExperimentSession) -> State:
        return experiment_session.get_sequence_state(self)

    def set_state(self, new_state: State, experiment_session: ExperimentSession):
        experiment_session.set_sequence_state(self, new_state)

    def get_shots(self, experiment_session: ExperimentSession) -> list[Shot]:
        return experiment_session.get_sequence_shots(self)

    def create_shot(
        self,
        name: str,
        start_time: datetime,
        end_time: datetime,
        parameters: Mapping[DottedVariableName, Parameter],
        measures: Mapping[DeviceName, Mapping[DataLabel, Data]],
        experiment_session: ExperimentSession,
    ) -> Shot:
        return experiment_session.create_sequence_shot(
            self, name, start_time, end_time, parameters, measures
        )

    @classmethod
    def create_sequence(
        cls,
        path: SequencePath,
        sequence_config: SequenceConfig,
        experiment_config_name: Optional[str],
        experiment_session: ExperimentSession,
    ) -> Self:
        return experiment_session.create_sequence(
            path, sequence_config, experiment_config_name
        )

    def get_stats(self, experiment_session: ExperimentSession) -> "SequenceStats":
        return experiment_session.get_sequence_stats(self)

    @classmethod
    def query_sequence_stats(
        cls, sequences: Iterable["Sequence"], experiment_session: ExperimentSession
    ) -> dict[SequencePath, "SequenceStats"]:
        return experiment_session.query_sequence_stats(sequences)

    @classmethod
    def get_all_sequence_names(cls, experiment_session: ExperimentSession) -> set[str]:
        """Get all the sequence names within a given session.

        Args:
            experiment_session: The activated experiment session in which to query the
            sequences.
        """

    def __eq__(self, other):
        if isinstance(other, Sequence):
            return self.path == other.path
        return False

    def __hash__(self):
        return hash(self.path)


class SequenceStats(TypedDict):
    state: State
    total_number_shots: Optional[int]
    number_completed_shots: int
    start_date: Optional[datetime]
    stop_date: Optional[datetime]


class SequenceNotFoundError(Exception):
    pass
