from caqtus.session import ParameterNamespace
from caqtus.types.expression import Expression

from .session_maker import session_maker


def test_0(session_maker):
    with session_maker() as session:
        assert session.get_global_parameters() == ParameterNamespace.empty()
        params_1 = ParameterNamespace.from_mapping(
            {"a": Expression("1"), "b": Expression("test")}
        )
        session.set_global_parameters(params_1)
        assert session.get_global_parameters() == params_1
        params_2 = ParameterNamespace.from_mapping(
            {"a": Expression("2"), "b": Expression("test")}
        )
        session.set_global_parameters(params_2)
        assert session.get_global_parameters() == params_2
