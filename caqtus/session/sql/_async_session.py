import contextlib
import functools
from datetime import datetime
from typing import Callable, Concatenate, TypeVar, ParamSpec, Mapping, Optional, Self

import anyio.to_thread
import attrs
from returns.result import Result
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from caqtus.types.data import DataLabel, Data
from caqtus.types.iteration import IterationConfiguration
from caqtus.types.parameter import Parameter, ParameterNamespace
from caqtus.types.timelane import TimeLanes
from caqtus.types.variable_name import DottedVariableName
from ._experiment_session import _get_global_parameters, _set_global_parameters
from ._path_hierarchy import _does_path_exists, _get_children, _get_path_creation_date
from ._sequence_collection import (
    _get_stats,
    _is_sequence,
    _get_sequence_global_parameters,
    _get_time_lanes,
    _get_iteration_configuration,
    _get_shots,
    _get_shot_parameters,
    _get_shot_end_time,
    _get_shot_start_time,
    _get_shot_data_by_label,
    _get_all_shot_data,
)
from ._serializer import SerializerProtocol
from .. import PureSequencePath
from ..async_session import (
    AsyncExperimentSession,
    AsyncPathHierarchy,
    AsyncSequenceCollection,
)
from .._experiment_session import ExperimentSessionNotActiveError
from .._path_hierarchy import PathNotFoundError, PathIsRootError
from .._sequence_collection import (
    PathIsSequenceError,
    SequenceStats,
    PathIsNotSequenceError,
    PureShot,
)

_T = TypeVar("_T")
_P = ParamSpec("_P")


class AsyncSQLExperimentSession(AsyncExperimentSession):
    def __init__(
        self,
        async_session_context: contextlib.AbstractAsyncContextManager[AsyncSession],
        serializer: SerializerProtocol,
    ):
        self._async_session_context = async_session_context
        self._async_session: Optional[AsyncSession] = None
        self.paths = AsyncSQLPathHierarchy(parent_session=self)
        self.sequences = AsyncSQLSequenceCollection(
            parent_session=self, serializer=serializer
        )

    async def __aenter__(self) -> Self:
        if self._async_session is not None:
            error = RuntimeError("Session has already been activated")
            error.add_note(
                "You cannot reactivate a session, you must create a new one."
            )
        self._async_session = await self._async_session_context.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._async_session_context.__aexit__(exc_type, exc_val, exc_tb)

    async def get_global_parameters(self) -> ParameterNamespace:
        return await self._run_sync(_get_global_parameters)

    async def set_global_parameters(self, parameters: ParameterNamespace) -> None:
        return await self._run_sync(_set_global_parameters, parameters)

    async def _run_sync(
        self,
        fun: Callable[Concatenate[Session, _P], _T],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _T:
        return await self._session().run_sync(fun, *args, **kwargs)

    def _session(self) -> AsyncSession:
        if self._async_session is None:
            raise ExperimentSessionNotActiveError(
                "Experiment session was not activated"
            )
        return self._async_session


class ThreadedAsyncSQLExperimentSession(AsyncSQLExperimentSession):
    def __init__(
        self,
        session_context: contextlib.AbstractContextManager[Session],
        serializer: SerializerProtocol,
    ):
        self._session_context = session_context
        self._session: Optional[Session] = None

        self.paths = AsyncSQLPathHierarchy(parent_session=self)
        self.sequences = AsyncSQLSequenceCollection(
            parent_session=self, serializer=serializer
        )

    async def __aenter__(self):
        if self._session is not None:
            raise RuntimeError("Session is already active")
        self._session = await anyio.to_thread.run_sync(self._session_context.__enter__)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await anyio.to_thread.run_sync(
            self._session_context.__exit__, exc_type, exc_val, exc_tb
        )

    async def _run_sync(
        self,
        fun: Callable[Concatenate[Session, _P], _T],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _T:
        if self._session is None:
            raise ExperimentSessionNotActiveError(
                "Experiment session was not activated"
            )
        wrapped = functools.partial(fun, self._session, *args, **kwargs)
        return await anyio.to_thread.run_sync(wrapped)


@attrs.frozen
class AsyncSQLPathHierarchy(AsyncPathHierarchy):
    parent_session: AsyncSQLExperimentSession

    async def does_path_exists(self, path: PureSequencePath) -> bool:
        return await self._run_sync(_does_path_exists, path)

    async def get_children(
        self, path: PureSequencePath
    ) -> Result[set[PureSequencePath], PathNotFoundError | PathIsSequenceError]:
        return await self._run_sync(_get_children, path)

    async def get_path_creation_date(
        self, path: PureSequencePath
    ) -> Result[datetime, PathNotFoundError | PathIsRootError]:
        return await self._run_sync(_get_path_creation_date, path)

    async def _run_sync(
        self,
        fun: Callable[Concatenate[Session, _P], _T],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _T:
        return await self.parent_session._run_sync(fun, *args, **kwargs)


@attrs.frozen
class AsyncSQLSequenceCollection(AsyncSequenceCollection):
    parent_session: AsyncSQLExperimentSession
    serializer: SerializerProtocol

    async def is_sequence(
        self, path: PureSequencePath
    ) -> Result[bool, PathNotFoundError]:
        return await self._run_sync(_is_sequence, path)

    async def get_stats(
        self, path: PureSequencePath
    ) -> Result[SequenceStats, PathNotFoundError | PathIsNotSequenceError]:
        return await self._run_sync(_get_stats, path)

    async def get_time_lanes(self, path: PureSequencePath) -> TimeLanes:
        return await self._run_sync(_get_time_lanes, path, self.serializer)

    async def get_global_parameters(self, path: PureSequencePath) -> ParameterNamespace:
        return await self._run_sync(_get_sequence_global_parameters, path)

    async def get_iteration_configuration(
        self, path: PureSequencePath
    ) -> IterationConfiguration:
        return await self._run_sync(_get_iteration_configuration, path, self.serializer)

    async def get_shots(
        self, path: PureSequencePath
    ) -> Result[list[PureShot], PathNotFoundError | PathIsNotSequenceError]:
        return await self._run_sync(_get_shots, path)

    async def get_shot_parameters(
        self, path: PureSequencePath, shot_index: int
    ) -> Mapping[DottedVariableName, Parameter]:
        return await self._run_sync(_get_shot_parameters, path, shot_index)

    async def get_all_shot_data(
        self, path: PureSequencePath, shot_index: int
    ) -> Mapping[DataLabel, Data]:
        return await self._run_sync(_get_all_shot_data, path, shot_index)

    async def get_shot_data_by_label(
        self, path: PureSequencePath, shot_index: int, data_label: DataLabel
    ) -> Data:
        return await self._run_sync(
            _get_shot_data_by_label, path, shot_index, data_label
        )

    async def get_shot_start_time(
        self, path: PureSequencePath, shot_index: int
    ) -> datetime:
        return await self._run_sync(_get_shot_start_time, path, shot_index)

    async def get_shot_end_time(
        self, path: PureSequencePath, shot_index: int
    ) -> datetime:
        return await self._run_sync(_get_shot_end_time, path, shot_index)

    async def _run_sync(
        self,
        fun: Callable[Concatenate[Session, _P], _T],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _T:
        return await self.parent_session._run_sync(fun, *args, **kwargs)
