import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias, TypeVar, Generic

import attrs
import sqlalchemy.orm
from returns.result import Result
from returns.result import Success, Failure
from sqlalchemy import select

from util import serialization
from ._path_table import SQLSequencePath
from ._sequence_table import SQLSequence, SQLIterationConfiguration
from .._return_or_raise import unwrap
from ..path import PureSequencePath, BoundSequencePath
from ..path_hierarchy import PathNotFoundError, PathHasChildrenError
from ..sequence import Sequence
from ..sequence_collection import PathIsSequenceError, PathIsNotSequenceError
from ..sequence_collection import SequenceCollection
from ..sequence.iteration_configuration import (
    IterationConfiguration,
    StepsConfiguration,
)

if TYPE_CHECKING:
    from ._experiment_session import SQLExperimentSession

T = TypeVar("T", bound=IterationConfiguration)

IterationConfigurationJSONSerializer: TypeAlias = Callable[
    [IterationConfiguration], tuple[str, serialization.JSON]
]
IterationConfigurationJSONConstructor: TypeAlias = Callable[
    [str, serialization.JSON], IterationConfiguration
]


@functools.singledispatch
def default_iteration_configuration_serializer(
    iteration_configuration: IterationConfiguration,
) -> tuple[str, serialization.JSON]:
    raise TypeError(
        f"Cannot serialize iteration configuration of type "
        f"{type(iteration_configuration)}"
    )


@default_iteration_configuration_serializer.register
def _(
    iteration_configuration: StepsConfiguration,
):
    return (
        "steps",
        serialization.to_json(iteration_configuration, StepsConfiguration),
    )


def default_iteration_configuration_constructor(
    iteration_type: str, iteration_content: serialization.JSON
) -> IterationConfiguration:
    if iteration_type == "steps":
        return serialization.from_json(iteration_content, StepsConfiguration)
    else:
        raise ValueError(f"Unknown iteration type {iteration_type}")


@attrs.frozen
class SQLSequenceCollection(SequenceCollection):
    parent_session: "SQLExperimentSession"
    iteration_config_serializer: IterationConfigurationJSONSerializer
    iteration_config_constructor: IterationConfigurationJSONConstructor

    def is_sequence(self, path: PureSequencePath) -> Result[bool, PathNotFoundError]:
        if path.is_root():
            return Success(False)
        return self._query_path_model(path).map(
            lambda path_model: bool(path_model.sequence)
        )

    def get_contained_sequences(self, path: PureSequencePath) -> list[PureSequencePath]:
        if unwrap(self.is_sequence(path)):
            return [path]

        path_hierarchy = self.parent_session.paths
        result = []
        for child in unwrap(path_hierarchy.get_children(path)):
            result += self.get_contained_sequences(child)
        return result

    def get_iteration_configuration(self, sequence: Sequence) -> IterationConfiguration:
        sequence_model = unwrap(self._query_sequence_model(sequence))
        return self.iteration_config_constructor(
            sequence_model.iteration_config.iteration_type,
            sequence_model.iteration_config.content,
        )

    def set_iteration_configuration(
        self, sequence: Sequence, iteration_configuration: IterationConfiguration
    ) -> None:
        sequence_model = unwrap(self._query_sequence_model(sequence))
        iteration_type, iteration_content = self.iteration_config_serializer(
            iteration_configuration
        )
        sequence_model.iteration_config.iteration_type = iteration_type
        sequence_model.iteration_config.content = iteration_content

    def create(
        self, path: PureSequencePath, iteration_configuration: IterationConfiguration
    ) -> Sequence:
        self.parent_session.paths.create_path(path)
        if unwrap(self.is_sequence(path)):
            raise PathIsSequenceError(path)
        if unwrap(self.parent_session.paths.get_children(path)):
            raise PathHasChildrenError(path)
        iteration_type, iteration_content = self.iteration_config_serializer(
            iteration_configuration
        )
        new_sequence = SQLSequence(
            path=unwrap(self._query_path_model(path)),
            iteration_config=SQLIterationConfiguration(
                iteration_type=iteration_type, content=iteration_content
            ),
        )
        self._get_sql_session().add(new_sequence)
        return Sequence(BoundSequencePath(path, self.parent_session))

    def _query_path_model(
        self, path: PureSequencePath
    ) -> Result[SQLSequencePath, PathNotFoundError]:
        stmt = select(SQLSequencePath).where(SQLSequencePath.path == str(path))
        result = self._get_sql_session().execute(stmt)
        if found := result.scalar():
            return Success(found)
        else:
            return Failure(PathNotFoundError(path))

    def _query_sequence_model(
        self, sequence: Sequence
    ) -> Result[SQLSequence, PathNotFoundError | PathIsNotSequenceError]:
        path_result = self._query_path_model(sequence.path)
        match path_result:
            case Success(path_model):
                stmt = select(SQLSequence).where(SQLSequence.path == path_model)
                result = self._get_sql_session().execute(stmt)
                if found := result.scalar():
                    return Success(found)
                else:
                    return Failure(PathIsNotSequenceError(sequence.path))
            case Failure() as failure:
                return failure

    def _get_sql_session(self) -> sqlalchemy.orm.Session:
        # noinspection PyProtectedMember
        return self.parent_session._get_sql_session()
