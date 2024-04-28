from sqlite3 import Connection as SQLite3Connection

import sqlalchemy
import sqlalchemy.orm
from sqlalchemy import event, Engine, create_engine, URL
from sqlalchemy.ext.asyncio import create_async_engine

from ._async_session import AsyncSQLExperimentSession, ThreadedAsyncSQLExperimentSession
from ._experiment_session import SQLExperimentSession
from ._serializer import Serializer
from ._table_base import create_tables
from ..async_session import AsyncExperimentSession
from ..experiment_session import ExperimentSession
from ..session_maker import ExperimentSessionMaker


# We need to enable foreign key constraints for sqlite databases and not for other
# types of databases.
@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    if isinstance(dbapi_connection, SQLite3Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()


class SQLExperimentSessionMaker(ExperimentSessionMaker):
    """Used to access experiment storage were the data are stored in a SQL database.

    This session maker can create session that connects to a database using sqlalchemy.

    This object is pickleable and can be passed to other processes, assuming that the
    database referenced by the engine is accessible from the other processes.
    In particular, in-memory sqlite databases are not accessible from other processes.

    Args:
        engine: This is used by the sessions to connect to the database.
            See sqlalchemy documentation for more information on how to create an
            engine.
        serializer: This is used to convert user defined objects to a JSON format that
            can be stored in the database.

    """

    def __init__(
        self,
        engine: sqlalchemy.Engine,
        async_engine: sqlalchemy.ext.asyncio.AsyncEngine,
        serializer: Serializer,
    ) -> None:
        self._engine = engine
        self._async_engine = async_engine
        self._session_maker = sqlalchemy.orm.sessionmaker(self._engine)
        self._async_session_maker = sqlalchemy.ext.asyncio.async_sessionmaker(
            self._async_engine
        )
        self._serializer = serializer

    def create_tables(self) -> None:
        """Create the tables in the database.

        This method is useful the first time the database is created.
        It will create missing tables and ignore existing ones.
        """

        create_tables(self._engine)

    def __call__(self) -> ExperimentSession:
        """Create a new ExperimentSession with the engine used at initialization."""

        return SQLExperimentSession(
            self._session_maker(),
            self._serializer,
        )

    def async_session(self) -> AsyncExperimentSession:
        return AsyncSQLExperimentSession(
            self._async_session_maker(),
            self._serializer,
        )

    # The following methods are required to make ExperimentSessionMaker pickleable since
    # sqlalchemy engine is not pickleable.
    # Only the engine url is pickled so the engine created upon unpickling might not be
    # exactly the same as the original one.
    def __getstate__(self):
        return {
            "url": self._engine.url,
            "async_url": self._async_engine.url,
            "serializer": self._serializer,
        }

    def __setstate__(self, state):
        engine = sqlalchemy.create_engine(state.pop("url"))
        async_engine = create_async_engine(state.pop("async_url"))
        self.__init__(engine, async_engine, **state)

    def upgrade_tables(self) -> None:
        """Updates the database schema to the latest version.

        When called on a database already setup, this method will upgrade the database
        tables to the latest version.
        When called on an empty database, this method will create the necessary tables.
        """

        # TODO: Handle upgrading the database schema if the tables already exist.
        create_tables(self._engine)


class SQLiteExperimentSessionMaker(SQLExperimentSessionMaker):
    def __init__(
        self,
        path: str,
        serializer: Serializer,
    ):
        engine = create_engine(f"sqlite:///{path}?check_same_thread=False")
        async_engine = create_async_engine(
            f"sqlite+aiosqlite:///{path}?check_same_thread=False"
        )
        super().__init__(engine, async_engine, serializer)

    def __getstate__(self):
        return {
            "path": self._engine.url.database,
            "serializer": self._serializer,
        }

    def __setstate__(self, state):
        path = state.pop("path")
        serializer = state.pop("serializer")
        self.__init__(path, serializer)


class PostgreSQLExperimentSessionMaker(SQLExperimentSessionMaker):
    """Used to access experiment data stored in a PostgreSQL database.

    Args:
        username: The username to use to connect to the database.
        password: The password to use to connect to the database.
        host: The host of the database.
        port: The port of the database.
        database: The name of the database.
        serializer: This is used to convert user defined sequence objects to a JSON
            format that can be stored in the database.
    """

    def __init__(
        self,
        username: str,
        password: str,
        host: str,
        port: int,
        database: str,
        serializer: Serializer,
    ):
        sync_url = URL.create(
            "postgresql+psycopg",
            username=username,
            password=password,
            host=host,
            port=port,
            database=database,
        )
        engine = create_engine(sync_url, isolation_level="REPEATABLE READ")
        async_url = URL.create(
            "postgresql+psycopg",
            username=username,
            password=password,
            host=host,
            port=port,
            database=database,
        )
        async_engine = create_async_engine(async_url, isolation_level="REPEATABLE READ")

        super().__init__(engine, async_engine, serializer=serializer)

    def async_session(self) -> AsyncExperimentSession:
        return ThreadedAsyncSQLExperimentSession(
            self._session_maker(),
            self._serializer,
        )

    def __getstate__(self):
        return {
            "username": self._engine.url.username,
            "password": self._engine.url.password,
            "host": self._engine.url.host,
            "port": self._engine.url.port,
            "database": self._engine.url.database,
            "serializer": self._serializer,
        }

    def __setstate__(self, state):
        username = state.pop("username")
        password = state.pop("password")
        host = state.pop("host")
        port = state.pop("port")
        database = state.pop("database")
        serializer = state.pop("serializer")
        self.__init__(username, password, host, port, database, serializer)
