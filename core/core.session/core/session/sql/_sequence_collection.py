from __future__ import annotations

import datetime
import functools
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Generic, TypeVar

import attrs
import numpy as np
import sqlalchemy.orm
from returns.result import Result
from returns.result import Success, Failure
from sqlalchemy import func
from sqlalchemy import select

from core.device import DeviceConfigurationAttrs, DeviceName
from core.session.shot.timelane import AnalogTimeLane
from core.types.data import DataLabel, Data, is_data
from core.types.expression import Expression
from core.types.parameter import Parameter
from core.types.units import Quantity
from core.types.variable_name import DottedVariableName
from util import serialization
from util.serialization import JSON
from ._path_table import SQLSequencePath
from ._sequence_table import (
    SQLSequence,
    SQLIterationConfiguration,
    SQLTimelanes,
    SQLDeviceConfiguration,
    SQLSequenceParameters,
)
from ._shot_tables import SQLShot, SQLShotParameter, SQLShotArray, SQLStructuredShotData
from .._return_or_raise import unwrap
from ..parameter_namespace import ParameterNamespace
from ..path import PureSequencePath, BoundSequencePath
from ..path_hierarchy import PathNotFoundError, PathHasChildrenError
from ..sequence import Sequence, Shot
from ..sequence.iteration_configuration import (
    IterationConfiguration,
    StepsConfiguration,
)
from ..sequence.state import State
from ..sequence_collection import (
    PathIsSequenceError,
    PathIsNotSequenceError,
    InvalidStateTransitionError,
    SequenceNotEditableError,
    SequenceStats,
    ShotNotFoundError,
)
from ..sequence_collection import SequenceCollection
from ..shot import TimeLane, DigitalTimeLane, TimeLanes, CameraTimeLane

if TYPE_CHECKING:
    from ._experiment_session import SQLExperimentSession


@attrs.define
class SequenceSerializer:
    """Indicates how to serialize and deserialize sequence configurations."""

    iteration_serializer: Callable[[IterationConfiguration], serialization.JSON]
    iteration_constructor: Callable[[serialization.JSON], IterationConfiguration]
    time_lane_serializer: Callable[[TimeLane], serialization.JSON]
    time_lane_constructor: Callable[[serialization.JSON], TimeLane]


@functools.singledispatch
def default_iteration_configuration_serializer(
    iteration_configuration: IterationConfiguration,
) -> serialization.JSON:
    raise TypeError(
        f"Cannot serialize iteration configuration of type "
        f"{type(iteration_configuration)}"
    )


@default_iteration_configuration_serializer.register
def _(
    iteration_configuration: StepsConfiguration,
):
    content = serialization.converters["json"].unstructure(iteration_configuration)
    content["type"] = "steps"
    return content


def default_iteration_configuration_constructor(
    iteration_content: serialization.JSON,
) -> IterationConfiguration:
    iteration_type = iteration_content.pop("type")
    if iteration_type == "steps":
        return serialization.converters["json"].structure(
            iteration_content, StepsConfiguration
        )
    else:
        raise ValueError(f"Unknown iteration type {iteration_type}")


@functools.singledispatch
def default_time_lane_serializer(time_lane: TimeLane) -> serialization.JSON:
    error = TypeError(f"Cannot serialize time lane of type {type(time_lane)}")

    error.add_note(
        f"{default_time_lane_serializer} doesn't support saving this lane type."
    )
    error.add_note(
        "You need to provide a custom lane serializer to the experiment session maker."
    )
    raise error


@default_time_lane_serializer.register
def _(time_lane: DigitalTimeLane):
    content = serialization.converters["json"].unstructure(time_lane, DigitalTimeLane)
    content["type"] = "digital"
    return content


@default_time_lane_serializer.register
def _(time_lane: AnalogTimeLane):
    content = serialization.converters["json"].unstructure(time_lane, AnalogTimeLane)
    content["type"] = "analog"
    return content


@default_time_lane_serializer.register
def _(time_lane: CameraTimeLane):
    content = serialization.converters["json"].unstructure(time_lane, CameraTimeLane)
    content["type"] = "camera"
    return content


def default_time_lane_constructor(
    time_lane_content: serialization.JSON,
) -> TimeLane:
    time_lane_type = time_lane_content.pop("type")
    if time_lane_type == "digital":
        return serialization.converters["json"].structure(
            time_lane_content, DigitalTimeLane
        )
    elif time_lane_type == "analog":
        return serialization.converters["json"].structure(
            time_lane_content, AnalogTimeLane
        )
    elif time_lane_type == "camera":
        return serialization.converters["json"].structure(
            time_lane_content, CameraTimeLane
        )
    else:
        raise ValueError(f"Unknown time lane type {time_lane_type}")


default_sequence_serializer = SequenceSerializer(
    iteration_serializer=default_iteration_configuration_serializer,
    iteration_constructor=default_iteration_configuration_constructor,
    time_lane_serializer=default_time_lane_serializer,
    time_lane_constructor=default_time_lane_constructor,
)


@attrs.frozen
class SQLSequenceCollection(SequenceCollection):
    parent_session: "SQLExperimentSession"
    serializer: SequenceSerializer
    device_configuration_serializers: Mapping[str, DeviceConfigurationSerializer]

    def __getitem__(self, item: str) -> Sequence:
        return Sequence(BoundSequencePath(item, self.parent_session))

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

    def set_parameters(
        self, path: PureSequencePath, parameters: ParameterNamespace
    ) -> None:
        sequence = unwrap(self._query_sequence_model(path))
        if not sequence.state.is_editable():
            raise SequenceNotEditableError(path)

        if not isinstance(parameters, ParameterNamespace):
            raise TypeError(
                f"Invalid parameters type {type(parameters)}, "
                f"expected ParameterNamespace"
            )

        parameters_content = serialization.converters["json"].unstructure(
            parameters, ParameterNamespace
        )

        sequence.parameters.content = parameters_content

    def get_parameters(self, path: PureSequencePath) -> ParameterNamespace:
        sequence = unwrap(self._query_sequence_model(path))

        parameters_content = sequence.parameters.content

        return serialization.converters["json"].structure(
            parameters_content, ParameterNamespace
        )

    def get_iteration_configuration(
        self, sequence: PureSequencePath
    ) -> IterationConfiguration:
        sequence_model = unwrap(self._query_sequence_model(sequence))
        return self.serializer.iteration_constructor(
            sequence_model.iteration.content,
        )

    def set_iteration_configuration(
        self, sequence: Sequence, iteration_configuration: IterationConfiguration
    ) -> None:
        sequence_model = unwrap(self._query_sequence_model(sequence.path))
        if not sequence_model.state.is_editable():
            raise SequenceNotEditableError(sequence.path)
        iteration_content = self.serializer.iteration_serializer(
            iteration_configuration
        )
        sequence_model.iteration.content = iteration_content
        sequence_model.expected_number_of_shots = (
            iteration_configuration.expected_number_shots()
        )

    def create(
        self,
        path: PureSequencePath,
        parameters: ParameterNamespace,
        iteration_configuration: IterationConfiguration,
        time_lanes: TimeLanes,
    ) -> Sequence:
        self.parent_session.paths.create_path(path)
        if unwrap(self.is_sequence(path)):
            raise PathIsSequenceError(path)
        if unwrap(self.parent_session.paths.get_children(path)):
            raise PathHasChildrenError(path)

        iteration_content = self.serializer.iteration_serializer(
            iteration_configuration
        )
        parameters_content = serialization.converters["json"].unstructure(
            parameters, ParameterNamespace
        )

        new_sequence = SQLSequence(
            path=unwrap(self._query_path_model(path)),
            parameters=SQLSequenceParameters(content=parameters_content),
            iteration=SQLIterationConfiguration(content=iteration_content),
            time_lanes=SQLTimelanes(content=self.serialize_time_lanes(time_lanes)),
            state=State.DRAFT,
            device_configurations=[],
            start_time=None,
            stop_time=None,
            expected_number_of_shots=iteration_configuration.expected_number_shots(),
        )
        self._get_sql_session().add(new_sequence)
        return Sequence(path)

    def serialize_time_lanes(self, time_lanes: TimeLanes) -> serialization.JSON:
        return dict(
            step_names=serialization.converters["json"].unstructure(
                time_lanes.step_names, list[str]
            ),
            step_durations=serialization.converters["json"].unstructure(
                time_lanes.step_durations, list[Expression]
            ),
            lanes={
                lane: self.serializer.time_lane_serializer(time_lane)
                for lane, time_lane in time_lanes.lanes.items()
            },
        )

    def construct_time_lanes(self, time_lanes_content: serialization.JSON) -> TimeLanes:
        return TimeLanes(
            step_names=serialization.converters["json"].structure(
                time_lanes_content["step_names"], list[str]
            ),
            step_durations=serialization.converters["json"].structure(
                time_lanes_content["step_durations"], list[Expression]
            ),
            lanes={
                lane: self.serializer.time_lane_constructor(time_lane_content)
                for lane, time_lane_content in time_lanes_content["lanes"].items()
            },
        )

    def get_time_lanes(self, sequence_path: PureSequencePath) -> TimeLanes:
        sequence_model = unwrap(self._query_sequence_model(sequence_path))
        return self.construct_time_lanes(sequence_model.time_lanes.content)

    def set_time_lanes(
        self, sequence_path: PureSequencePath, time_lanes: TimeLanes
    ) -> None:
        sequence_model = unwrap(self._query_sequence_model(sequence_path))
        if not sequence_model.state.is_editable():
            raise SequenceNotEditableError(sequence_path)
        sequence_model.time_lanes.content = self.serialize_time_lanes(time_lanes)

    def get_state(
        self, path: PureSequencePath
    ) -> Result[State, PathNotFoundError | PathIsNotSequenceError]:
        result = self._query_sequence_model(path)
        return result.map(lambda sequence: sequence.state)

    def set_state(self, path: PureSequencePath, state: State) -> None:
        sequence = unwrap(self._query_sequence_model(path))
        if not State.is_transition_allowed(sequence.state, state):
            raise InvalidStateTransitionError(
                f"Sequence at {path} can't transition from {sequence.state} to {state}"
            )
        sequence.state = state
        if state == State.DRAFT:
            sequence.start_time = None
            sequence.stop_time = None
            delete_device_configurations = sqlalchemy.delete(
                SQLDeviceConfiguration
            ).where(SQLDeviceConfiguration.sequence == sequence)
            self._get_sql_session().execute(delete_device_configurations)

            delete_shots = sqlalchemy.delete(SQLShot).where(
                SQLShot.sequence == sequence
            )
            self._get_sql_session().execute(delete_shots)
        elif state == State.RUNNING:
            sequence.start_time = datetime.datetime.now(
                tz=datetime.timezone.utc
            ).replace(tzinfo=None)
        elif state in (State.INTERRUPTED, State.CRASHED, State.FINISHED):
            sequence.stop_time = datetime.datetime.now(
                tz=datetime.timezone.utc
            ).replace(tzinfo=None)

    def set_device_configurations(
        self,
        path: PureSequencePath,
        device_configurations: Mapping[DeviceName, DeviceConfigurationAttrs],
    ) -> None:
        sequence = unwrap(self._query_sequence_model(path))
        if sequence.state != State.PREPARING:
            raise SequenceNotEditableError(path)
        sql_device_configs = []
        for order, (name, device_configuration) in enumerate(
            device_configurations.items()
        ):
            type_name = type(device_configuration).__qualname__
            serializer = self.device_configuration_serializers[type_name]
            sql_device_configs.append(
                SQLDeviceConfiguration(
                    name=name,
                    order=order,
                    device_type=type_name,
                    content=serializer.dumper(device_configuration),
                )
            )
        sequence.device_configurations = sql_device_configs

    def get_device_configurations(
        self, path: PureSequencePath
    ) -> dict[DeviceName, DeviceConfigurationAttrs]:
        device_configurations = {}
        sequence = unwrap(self._query_sequence_model(path))
        for device_configuration in sequence.device_configurations:
            serializer = self.device_configuration_serializers[
                device_configuration.device_type
            ]
            device_configurations[device_configuration.name] = serializer.loader(
                device_configuration.content
            )
        return device_configurations

    def get_stats(
        self, path: PureSequencePath
    ) -> Result[SequenceStats, PathNotFoundError | PathIsNotSequenceError]:
        result = self._query_sequence_model(path)

        def extract_stats(sequence: SQLSequence) -> SequenceStats:
            number_shot_query = select(func.count()).select_from(
                select(SQLShot).where(SQLShot.sequence == sequence).subquery()
            )
            number_shot_run = (
                self._get_sql_session().execute(number_shot_query).scalar_one()
            )
            return SequenceStats(
                state=sequence.state,
                start_time=(
                    sequence.start_time.replace(tzinfo=datetime.timezone.utc)
                    if sequence.start_time is not None
                    else None
                ),
                stop_time=(
                    sequence.stop_time.replace(tzinfo=datetime.timezone.utc)
                    if sequence.stop_time is not None
                    else None
                ),
                number_completed_shots=number_shot_run,
                expected_number_shots=sequence.expected_number_of_shots,
            )

        return result.map(extract_stats)

    def create_shot(
        self,
        path: PureSequencePath,
        shot_index: int,
        shot_parameters: Mapping[DottedVariableName, Parameter],
        shot_data: Mapping[DataLabel, Data],
        shot_start_time: datetime.datetime,
        shot_end_time: datetime.datetime,
    ) -> None:
        sequence = unwrap(self._query_sequence_model(path))
        if sequence.state != State.RUNNING:
            raise RuntimeError("Can't create shot in sequence that is not running")
        if shot_index < 0:
            raise ValueError("Shot index must be non-negative")
        if sequence.expected_number_of_shots is not None:
            if shot_index >= sequence.expected_number_of_shots:
                raise ValueError(
                    f"Shot index must be less than the expected number of shots "
                    f"({sequence.expected_number_of_shots})"
                )

        parameters = self.serialize_shot_parameters(shot_parameters)

        array_data, structured_data = self.serialize_data(shot_data)

        shot = SQLShot(
            sequence=sequence,
            index=shot_index,
            parameters=SQLShotParameter(content=parameters),
            array_data=array_data,
            structured_data=structured_data,
            start_time=shot_start_time.astimezone(datetime.timezone.utc).replace(
                tzinfo=None
            ),
            end_time=shot_end_time.astimezone(datetime.timezone.utc).replace(
                tzinfo=None
            ),
        )
        self._get_sql_session().add(shot)

    @staticmethod
    def serialize_data(
        data: Mapping[DataLabel, Data]
    ) -> tuple[list[SQLShotArray], list[SQLStructuredShotData]]:
        arrays = []
        structured_data = []
        for label, value in data.items():
            if not is_data(value):
                raise TypeError(f"Invalid data type for {label}: {type(value)}")
            if isinstance(value, np.ndarray):
                arrays.append(
                    SQLShotArray(
                        label=label,
                        dtype=str(value.dtype),
                        shape=value.shape,
                        bytes_=value.tobytes(),
                    )
                )
            else:
                structured_data.append(
                    SQLStructuredShotData(label=label, content=value)
                )
        return arrays, structured_data

    @staticmethod
    def serialize_shot_parameters(
        shot_parameters: Mapping[DottedVariableName, Parameter]
    ) -> dict[str, serialization.JSON]:
        return {
            str(variable_name): serialization.converters["json"].unstructure(
                parameter, Parameter
            )
            for variable_name, parameter in shot_parameters.items()
        }

    def get_shots(self, path: PureSequencePath) -> list[Shot]:
        sql_sequence = unwrap(self._query_sequence_model(path))
        sequence = Sequence(BoundSequencePath(path, self.parent_session))

        return [Shot(sequence, shot.index) for shot in sql_sequence.shots]

    def get_shot_parameters(
        self, path: PureSequencePath, shot_index: int
    ) -> Mapping[DottedVariableName, Parameter]:
        shot_model = unwrap(self._query_shot_model(path, shot_index))
        values = shot_model.parameters.content
        parameters = serialization.converters["json"].structure(
            values, dict[DottedVariableName, bool | int | float | Quantity]
        )
        return parameters

    def get_all_shot_data(
        self, path: PureSequencePath, shot_index: int
    ) -> dict[DataLabel, Data]:
        shot_model = unwrap(self._query_shot_model(path, shot_index))
        arrays = shot_model.array_data
        structured_data = shot_model.structured_data
        result = {}
        for array in arrays:
            result[array.label] = np.frombuffer(
                array.bytes_, dtype=array.dtype
            ).reshape(array.shape)
        for data in structured_data:
            result[data.label] = data.content
        return result

    def get_shot_data_by_label(
        self, path: PureSequencePath, shot_index: int, data_label: DataLabel
    ) -> Data:
        shot_model = unwrap(self._query_shot_model(path, shot_index))
        structure_query = select(SQLStructuredShotData).where(
            (SQLStructuredShotData.shot == shot_model)
            & (SQLStructuredShotData.label == data_label)
        )
        result = self._get_sql_session().execute(structure_query)
        if found := result.scalar():
            return found.content
        array_query = select(SQLShotArray).where(
            (SQLShotArray.shot == shot_model) & (SQLShotArray.label == data_label)
        )
        result = self._get_sql_session().execute(array_query)
        if found := result.scalar():
            return np.frombuffer(found.bytes_, dtype=found.dtype).reshape(found.shape)
        raise KeyError(f"Data <{data_label}> not found in shot {shot_index}")

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
        self, path: PureSequencePath
    ) -> Result[SQLSequence, PathNotFoundError | PathIsNotSequenceError]:
        path_result = self._query_path_model(path)
        match path_result:
            case Success(path_model):
                stmt = select(SQLSequence).where(SQLSequence.path == path_model)
                result = self._get_sql_session().execute(stmt)
                if found := result.scalar():
                    return Success(found)
                else:
                    return Failure(PathIsNotSequenceError(path))
            case Failure() as failure:
                return failure

    def _query_shot_model(
        self, path: PureSequencePath, shot_index: int
    ) -> Result[
        SQLShot, PathNotFoundError | PathIsNotSequenceError | ShotNotFoundError
    ]:
        sequence_model_result = self._query_sequence_model(path)
        match sequence_model_result:
            case Success(sequence_model):
                stmt = (
                    select(SQLShot)
                    .where(SQLShot.sequence == sequence_model)
                    .where(SQLShot.index == shot_index)
                )
                result = self._get_sql_session().execute(stmt)
                if found := result.scalar():
                    return Success(found)
                else:
                    return Failure(PathIsNotSequenceError(path))
            case Failure() as failure:
                return failure

    def _get_sql_session(self) -> sqlalchemy.orm.Session:
        # noinspection PyProtectedMember
        return self.parent_session._get_sql_session()


T = TypeVar("T", bound=DeviceConfigurationAttrs)


@attrs.define
class DeviceConfigurationSerializer(Generic[T]):
    """Indicates how to serialize and deserialize device configurations."""

    dumper: Callable[[T], JSON]
    loader: Callable[[JSON], T]
