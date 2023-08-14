from abc import abstractmethod
from collections.abc import Mapping
from datetime import datetime
from typing import Optional, Iterable, Protocol

from data_types import DataLabel, Data
from device.name import DeviceName
from experiment.configuration import ExperimentConfig
from parameter_types import Parameter
from sequence.configuration import SequenceConfig
from sequence.runtime import Sequence, Shot, SequencePath, SequenceStats
from sql_model import State
from variable.name import DottedVariableName


class SequenceHierarchy(Protocol):
    @abstractmethod
    def does_path_exists(self, path: SequencePath) -> bool:
        """Check if the path exists in the session.

        Args:
            path: the path to check for existence.

        Returns:
            True if the path exists in the session. False otherwise.
        """

        raise NotImplementedError()

    @abstractmethod
    def is_sequence_path(self, path: SequencePath) -> bool:
        """Check if the path is a sequence.

        Args:
            path: the path to check.

        Returns:
            True if the path is a sequence path. False otherwise.
        """

        raise NotImplementedError()

    @abstractmethod
    def create_path(self, path: SequencePath) -> list[SequencePath]:
        """Create the path in the session and its parent paths if they do not exist.

        Args:
            path: the path to create.

        Returns:
            A list of the paths that were created.

        Raises:
            PathIsSequenceError: If the path or one of its ancestors is a sequence.
        """

        raise NotImplementedError()

    @abstractmethod
    def delete_path(self, path: SequencePath, delete_sequences: bool = False):
        """
        Delete the path and all its children if they exist

        Warnings:
            If delete_sequences is True, all sequences in the path will be deleted.

        Args:
            path: The path to delete. Children will be deleted recursively.
            delete_sequences: If False, raise an error if the path or one of its
            children is a sequence.

        Raises:
            RuntimeError: If the path or one of its children is a sequence and
            delete_sequence is False
        """

        raise NotImplementedError()

    @abstractmethod
    def get_path_children(self, path: SequencePath) -> set[SequencePath]:
        """Get the children of the path.

        Args:
            path: the path to get the children for.

        Returns:
            The children of the path.
        """

        raise NotImplementedError()

    @abstractmethod
    def get_path_creation_date(self, path: SequencePath) -> datetime:
        """Get the creation date of the path.

        Args:
            path: the path to get the creation date for.

        Returns:
            The creation date of the path.
        """

        raise NotImplementedError()

    # Sequence methods
    @abstractmethod
    def does_sequence_exist(self, sequence: Sequence) -> bool:
        """Check if the sequence exists in the session.

        Args:
            sequence: the sequence to check for existence.

        Returns:
            True if the sequence exists in the session. False otherwise.
        """

        raise NotImplementedError()

    @abstractmethod
    def get_sequence_state(self, sequence: Sequence) -> State:
        """Get the state of the sequence.

        Args:
            sequence: the sequence to get the state for.

        Returns:
            The state of the sequence.
        """

        raise NotImplementedError()

    @abstractmethod
    def set_sequence_state(self, sequence: Sequence, state: State):
        """Set the state of the sequence.

        This should mostly be used by the program running the experiment.

        Args:
            sequence: the sequence to set the state for.
            state: the state to set.
        """

        raise NotImplementedError()

    @abstractmethod
    def get_sequence_shots(self, sequence: Sequence) -> list[Shot]:
        """Get the shots that have been run for the sequence.

        Args:
            sequence: the sequence to get the shots for.

        Returns:
            The shots of the sequence.
        """

        raise NotImplementedError()

    @abstractmethod
    def get_sequence_creation_date(self, sequence: Sequence) -> datetime:
        """Get the creation date of the sequence.

        Args:
            sequence: the sequence to get the creation date for.

        Returns:
            The creation date of the sequence.
        """

        raise NotImplementedError()

    @abstractmethod
    def get_sequence_config_yaml(self, sequence: Sequence) -> str:
        """Get the configuration of the sequence.

        Args:
            sequence: the sequence to get the configuration for.

        Returns:
            The configuration of the sequence as a yaml string.
        """

        raise NotImplementedError()

    @abstractmethod
    def set_sequence_config_yaml(
        self, sequence: Sequence, yaml_config: str, total_number_shots: Optional[int]
    ):
        """Set the configuration of the sequence.

        Args:
            sequence: the sequence to set the configuration for.
            yaml_config: the configuration of the sequence as a yaml string.
            total_number_shots: the total number of shots that will be run for this
                sequence. It is stored alongside the sequence configuration.
        """

        raise NotImplementedError()

    @abstractmethod
    def get_sequence_experiment_config(
        self, sequence: Sequence
    ) -> Optional[ExperimentConfig]:
        """Get the experiment config of the sequence.

        Args:
            sequence: the sequence to get the experiment config for.

        Returns:
            The experiment config of the sequence. If no experiment config is set for
            the sequence, None is returned.
        """

        raise NotImplementedError()

    @abstractmethod
    def set_sequence_experiment_config(
        self, sequence: Sequence, experiment_config: str
    ):
        """Set the experiment config of the sequence.

        Args:
            sequence: the sequence to set the experiment config for.
            experiment_config: the name of the experiment config to set. The referred
                experiment config must exist in the session.
        """

        raise NotImplementedError()

    @abstractmethod
    def get_sequence_stats(self, sequence: Sequence) -> SequenceStats:
        """Get the stats of the sequence.

        Args:
            sequence: the sequence to get the stats for.

        Returns:
            The stats of the sequence.
        """

        raise NotImplementedError()

    @abstractmethod
    def get_all_sequence_names(self) -> set[str]:
        """Get the names of all sequences in the session.

        Returns:
            The names of all sequences in the session.
        """

        raise NotImplementedError()

    @abstractmethod
    def query_sequence_stats(
        self, sequences: Iterable[Sequence]
    ) -> dict[SequencePath, SequenceStats]:
        """Get the stats of the sequences.

        Args:
            sequences: the sequences to get the stats for.

        Returns:
            The stats of the sequences.
        """

        raise NotImplementedError()

    @abstractmethod
    def create_sequence(
        self,
        path: SequencePath,
        sequence_config: SequenceConfig,
        experiment_config_name: Optional[str],
    ) -> Sequence:
        """Create a new sequence in the session."""

        raise NotImplementedError()

    @abstractmethod
    def create_sequence_shot(
        self,
        sequence: Sequence,
        name: str,
        start_time: datetime,
        end_time: datetime,
        parameters: Mapping[DottedVariableName, Parameter],
        measures: Mapping[DeviceName, Mapping[DataLabel, Data]],
    ):
        """Create a new shot for the sequence."""

        raise NotImplementedError()


class PathIsSequenceError(Exception):
    pass
