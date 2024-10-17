from __future__ import annotations

import abc
from collections.abc import Mapping
from typing import Any, TypeAlias, Union, assert_never

import attrs

import caqtus.formatter as fmt
from caqtus.types.expression import Expression
from caqtus.types.recoverable_exceptions import InvalidTypeError
from caqtus.types.units import is_scalar_quantity, Quantity
from caqtus.types.units.base import BaseUnit
from caqtus.types.variable_name import DottedVariableName


@attrs.define
class Transformation(abc.ABC):
    """Defines a transformation that can be applied to produce an output value."""

    @abc.abstractmethod
    def evaluate(self, variables: Mapping[DottedVariableName, Any]) -> OutputValue:
        """Evaluates the transformation using the given variables.

        If the value returned is a quantity, it must be expressed in base units.
        """

        raise NotImplementedError


type OutputValue = float | int | bool | Quantity[float, BaseUnit]
"""A value that can be used to compute the output of a device.

If the value is a quantity, it must be a scalar and expressed in base units.
"""

EvaluableOutput: TypeAlias = Union[Expression, Transformation]
"""Defines an operation that can be evaluated to an output value.

Evaluable object can be used in the :func:`evaluate` function.
"""


def evaluate(
    input_: EvaluableOutput, variables: Mapping[DottedVariableName, Any]
) -> OutputValue:
    """Evaluates the input and returns the result as a parameter.

    If the evaluated input is a quantity, it is converted to its base units.
    """

    if isinstance(input_, Transformation):
        return input_.evaluate(variables)
    elif isinstance(input_, Expression):
        evaluated = input_.evaluate(variables)

        if isinstance(evaluated, (float, int, bool)):
            return evaluated
        elif is_scalar_quantity(evaluated):
            return evaluated.to_base_units()
        else:
            raise InvalidTypeError(
                f"{fmt.expression(input_)} does not evaluate to a parameter, "
                f"got {fmt.type_(type(evaluated))}.",
            )
    assert_never(input_)


evaluable_output_validator = attrs.validators.instance_of((Expression, Transformation))