import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime
from threading import Lock
from typing import Optional, overload, Literal

import sqlalchemy
import sqlalchemy.orm
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from experiment.configuration import ExperimentConfig
from sql_model.model import ExperimentConfigModel, CurrentExperimentConfigModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ExperimentSessionNotActiveError(RuntimeError):
    pass


class _ExperimentSession(ABC):
    """Manage the experiment session

    Instances of this class manage access to the permanent storage of the experiment.
    A session contains the history of the experiment configuration and the current configuration.
    It also contains the sequence tree of the experiment, with the sequence states and data.

    Some objects in the sequence.runtime package (Sequence, Shot) that can read and write to the experiment data storage
    have methods that require an activated ExperimentSession.

    If an error occurs within an activated session block, the session is automatically rolled back to the beginning of
    the activation block. This prevents leaving some data in an inconsistent state.
    """

    def __init__(self):
        self._is_active = False
        self._lock = Lock()

    def add_experiment_config(
        self,
        experiment_config: ExperimentConfig,
        name: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> str:
        if name is None:
            name = self._get_new_experiment_config_name()
        yaml = experiment_config.to_yaml()
        assert ExperimentConfig.from_yaml(yaml) == experiment_config
        ExperimentConfigModel.add_config(
            name=name,
            yaml=yaml,
            comment=comment,
            session=self.get_sql_session(),
        )
        return name

    def _get_new_experiment_config_name(self) -> str:
        session = self.get_sql_session()
        query_names = session.query(ExperimentConfigModel.name)
        numbers = []
        pattern = re.compile("config_(\\d+)")
        for name in session.scalars(query_names):
            if match := pattern.match(name):
                numbers.append(int(match.group(1)))
        return f"config_{_find_first_unused_number(numbers)}"

    def get_experiment_config(self, name: str) -> ExperimentConfig:
        return ExperimentConfig.from_yaml(
            ExperimentConfigModel.get_config(name, self.get_sql_session())
        )

    def get_experiment_configs(
        self, from_date: Optional[datetime] = None, to_date: Optional[datetime] = None
    ) -> dict[str, ExperimentConfig]:
        results = ExperimentConfigModel.get_configs(
            from_date,
            to_date,
            self.get_sql_session(),
        )

        return {
            name: ExperimentConfig.from_yaml(yaml) for name, yaml in results.items()
        }

    def set_current_experiment_config(self, name: str):
        CurrentExperimentConfigModel.set_current_experiment_config(
            name=name, session=self.get_sql_session()
        )

    def get_current_experiment_config_name(self) -> Optional[str]:
        return CurrentExperimentConfigModel.get_current_experiment_config_name(
            session=self.get_sql_session()
        )

    def get_current_experiment_config(self) -> Optional[ExperimentConfig]:
        name = self.get_current_experiment_config_name()
        if name is None:
            return None
        return self.get_experiment_configs()[name]

    def activate(self):
        """Activate the session

        This method is meant to be used in a with statement.

        Example:
            # Ok
            with session.activate():
                config = session.get_current_experiment_config()

            # Not ok
            config = session.get_current_experiment_config()

            # Not ok
            session.activate()
            config = session.get_current_experiment_config()
        """

        return self

    @abstractmethod
    def get_sql_session(self) -> sqlalchemy.orm.Session:
        ...


class ExperimentSession(_ExperimentSession):
    """A synchronous version of the ExperimentSession"""

    def __init__(self, _session: sqlalchemy.orm.Session):
        super().__init__()
        self._sql_session = _session

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

    def get_sql_session(self) -> sqlalchemy.orm.Session:
        if not self._is_active:
            raise ExperimentSessionNotActiveError(
                "Experiment session was not activated"
            )
        return self._sql_session


class AsyncExperimentSession(_ExperimentSession):
    def __init__(self, _session: AsyncSession):
        super().__init__()
        self._sql_session = _session

    def __aenter__(self):
        with self._lock:
            if self._is_active:
                raise RuntimeError("Session is already active")
            self._transaction = self._sql_session.bein().__aenter__()
            self._is_active = True
            return self

    def __aexit__(self, exc_type, exc_val, exc_tb):
        with self._lock:
            self._transaction.__aexit__(exc_type, exc_val, exc_tb)
            self._transaction = None
            self._is_active = False

    def get_sql_session(self) -> sqlalchemy.orm.Session:
        if not self._is_active:
            raise ExperimentSessionNotActiveError(
                "Experiment session was not activated"
            )
        # noinspection PyTypeChecker
        return self._sql_session


def _find_first_unused_number(numbers: list[int]) -> int:
    for index, value in enumerate(sorted(numbers)):
        if index != value:
            return index
    return len(numbers)


class ExperimentSessionMaker:
    def __init__(
        self,
        user: str,
        ip: str,
        password: str,
        database: str,
    ):
        self._kwargs = {
            "user": user,
            "ip": ip,
            "password": password,
            "database": database,
        }
        sync_database_url = f"postgresql+psycopg2://{user}:{password}@{ip}/{database}"
        async_database_url = f"postgresql+asyncpg://{user}:{password}@{ip}/{database}"
        self._engine = sqlalchemy.create_engine(sync_database_url)
        self._session_maker = sqlalchemy.orm.sessionmaker(self._engine)
        self._async_engine = create_async_engine(async_database_url)
        self._async_session_maker = async_sessionmaker(self._async_engine)

    @overload
    def __call__(self) -> ExperimentSession:
        ...

    @overload
    def __call__(self, async_session: Literal[False]) -> ExperimentSession:
        ...

    @overload
    def __call__(self, async_session: Literal[True]) -> AsyncExperimentSession:
        ...

    def __call__(
        self, async_session: bool = False
    ) -> ExperimentSession | AsyncExperimentSession:
        """Create a new ExperimentSession"""

        if not async_session:
            return ExperimentSession(self._session_maker())
        else:
            return AsyncExperimentSession(self._async_session_maker())

    # The following methods are required to make ExperimentSessionMaker pickleable to pass it to other processes.
    # sqlalchemy engine is not pickleable, so we just pickle the database url and create a new engine upon unpickling.
    def __getstate__(self) -> dict:
        return self._kwargs

    def __setstate__(self, state: dict):
        self.__init__(**state)


def get_standard_experiment_session_maker() -> ExperimentSessionMaker:
    return ExperimentSessionMaker(
        user="caqtus",
        ip="192.168.137.4",
        password="Deardear",
        database="test_database",
    )


def get_standard_experiment_session() -> ExperimentSession:
    return get_standard_experiment_session_maker()()
