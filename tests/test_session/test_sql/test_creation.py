import pytest

from caqtus.extension import Experiment, upgrade_database
from caqtus.session.sql import PostgreSQLExperimentSessionMaker
from caqtus.session.sql._session_maker import InvalidDatabaseSchemaError


def test_initialization_postgres(empty_database_config):

    exp = Experiment(empty_database_config)
    with pytest.raises(InvalidDatabaseSchemaError):
        exp.get_storage_manager()

    upgrade_database(exp)

    session_maker = exp.get_storage_manager()
    assert isinstance(session_maker, PostgreSQLExperimentSessionMaker)
    session_maker.check()
