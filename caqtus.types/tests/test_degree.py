from caqtus.types.expression import Expression
from caqtus.types.units import ureg
from caqtus.types.variable_name import DottedVariableName


def test_degree_compilation():
    expr = Expression("140°")
    value = expr.evaluate({DottedVariableName("deg"): ureg.deg})
    assert value == 140 * ureg.deg
