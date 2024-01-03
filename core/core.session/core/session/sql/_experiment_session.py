from typing import Mapping

import sqlalchemy.orm
from attrs import define

from ._device_configuration_collection import (
    SQLDeviceConfigurationCollection,
    DeviceConfigurationSerializer,
)
from ._path_hierarchy import SQLPathHierarchy
from ._sequence_collection import (
    SQLSequenceCollection,
    IterationConfigurationJSONSerializer,
)
from ..experiment_session import (
    ExperimentSession,
    ExperimentSessionNotActiveError,
)


@define(init=False)
class SQLExperimentSession(ExperimentSession):
    paths: SQLPathHierarchy
    sequence_collection: SQLSequenceCollection
    device_configurations: SQLDeviceConfigurationCollection

    _sql_session: sqlalchemy.orm.Session
    _is_active: bool

    def __init__(
        self,
        session: sqlalchemy.orm.Session,
        device_configuration_serializers: Mapping[str, DeviceConfigurationSerializer],
        iteration_configuration_serializer: IterationConfigurationJSONSerializer,
        *args,
        **kwargs,
    ):
        """Create a new experiment session.

        This constructor is not meant to be called directly.
        Instead, use a :py:class:`SQLExperimentSessionMaker` to create a new session.
        """

        super().__init__(*args, **kwargs)
        self._sql_session = session
        self._is_active = False
        self.paths = SQLPathHierarchy(parent_session=self)
        self.sequence_collection = SQLSequenceCollection(
            parent_session=self,
            iteration_configuration_serializer=iteration_configuration_serializer,
        )
        self.device_configurations = SQLDeviceConfigurationCollection(
            parent_session=self,
            device_configuration_serializers=device_configuration_serializers,
        )

    def __enter__(self):
        if self._is_active:
            raise RuntimeError("Session is already active")
        self._transaction = self._sql_session.begin().__enter__()
        self._is_active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._transaction.__exit__(exc_type, exc_val, exc_tb)
        self._transaction = None
        self._is_active = False

    def _get_sql_session(self) -> sqlalchemy.orm.Session:
        if not self._is_active:
            raise ExperimentSessionNotActiveError(
                "Experiment session was not activated"
            )
        return self._sql_session
