import contextlib
from typing import Optional

import attrs
import sqlalchemy.orm

from caqtus.session import ParameterNamespace
from caqtus.session.sql.parameters_table import SQLParameters
from caqtus.utils import serialization
from ._device_configuration_collection import SQLDeviceConfigurationCollection
from ._path_hierarchy import SQLPathHierarchy
from ._sequence_collection import (
    SQLSequenceCollection,
)
from ._serializer import Serializer
from ..experiment_session import (
    ExperimentSession,
    ExperimentSessionNotActiveError,
)


class SQLExperimentSession(ExperimentSession):
    """Used to store experiment data in a SQL database.

    This class implements the :class:`ExperimentSession` interface and the documentation
    of the related methods can be found in the :class:`ExperimentSession`documentation.
    """

    def __init__(
        self,
        session: sqlalchemy.orm.Session,
        serializer: Serializer,
        *args,
        **kwargs,
    ):
        """Create a new experiment session.

        This constructor is not meant to be called directly.
        Instead, use a :py:class:`SQLExperimentSessionMaker` to create a new session.
        """

        super().__init__(*args, **kwargs)
        self._sql_session = session
        self.paths = SQLPathHierarchy(parent_session=self)
        self.sequences = SQLSequenceCollection(
            parent_session=self, serializer=serializer
        )
        self.default_device_configurations = SQLDeviceConfigurationCollection(
            parent_session=self, serializer=serializer
        )
        self._transaction: Optional[sqlalchemy.orm.SessionTransaction] = None
        self._exit_stack = contextlib.ExitStack()

    def get_global_parameters(self) -> ParameterNamespace:
        return _get_global_parameters(self._get_sql_session())

    def set_global_parameters(self, parameters: ParameterNamespace) -> None:
        return _set_global_parameters(self._get_sql_session(), parameters)

    def __str__(self):
        return f"<{self.__class__.__name__} @ {self._sql_session.get_bind()}>"

    def __enter__(self):
        if self._transaction is not None:
            error = RuntimeError("Session has already been activated")
            error.add_note(
                "You cannot reactivate a session, you must create a new one."
            )
            raise error
        self._exit_stack.__enter__()
        self._exit_stack.enter_context(self._sql_session)
        try:
            self._transaction = self._exit_stack.enter_context(
                self._sql_session.begin()
            )
        except Exception:
            self._exit_stack.close()
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._exit_stack.__exit__(exc_type, exc_val, exc_tb)

    def _get_sql_session(self) -> sqlalchemy.orm.Session:
        if self._transaction is None:
            raise ExperimentSessionNotActiveError(
                "Experiment session was not activated"
            )
        return self._sql_session


def _get_global_parameters(session: sqlalchemy.orm.Session) -> ParameterNamespace:
    stmt = sqlalchemy.select(SQLParameters).where(SQLParameters.name == "global")
    result = session.execute(stmt)
    if found := result.scalar():
        return serialization.converters["json"].structure(
            found.content, ParameterNamespace
        )
    else:
        # It could be that the table is empty if set_global_parameters was never
        # called before, in which case we return an empty ParameterNamespace.
        return ParameterNamespace.empty()


def _set_global_parameters(
    session: sqlalchemy.orm.Session, parameters: ParameterNamespace
) -> None:
    if not isinstance(parameters, ParameterNamespace):
        raise TypeError(
            f"Expected a ParameterNamespace, got {type(parameters).__name__}"
        )
    query = sqlalchemy.select(SQLParameters).where(SQLParameters.name == "global")
    result = session.execute(query)
    content = serialization.converters["json"].unstructure(
        parameters, ParameterNamespace
    )
    if found := result.scalar():
        found.content = content
    else:
        new_parameters = SQLParameters(name="global", content=content)
        session.add(new_parameters)
