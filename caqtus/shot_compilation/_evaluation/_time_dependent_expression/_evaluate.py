from typing import Mapping, assert_never

import numpy as np

import caqtus.formatter as fmt
import caqtus_parsing.nodes as nodes
from caqtus.shot_compilation.timing import Time, number_ticks
from caqtus.types.expression import Expression
from caqtus.types.parameter import Parameter
from caqtus.types.recoverable_exceptions import EvaluationError, InvalidTypeError
from caqtus.types.units import Quantity
from caqtus.types.variable_name import DottedVariableName
from caqtus_parsing import parse, InvalidSyntaxError
from ._is_time_dependent import is_time_dependent
from ._result import (
    EvaluationResult,
    IntResult,
    FloatResult,
    BoolResult,
    QuantityResult,
)
from .._evaluate_scalar_expression import _evaluate_scalar_ast
from .._scalar import Scalar
from ...timed_instructions import Pattern, TimedInstruction


def evaluate_time_dependent_digital_expression(
    expression: Expression,
    parameters: Mapping[DottedVariableName, Parameter],
    initial_time: Time,
    final_time: Time,
    time_step: Time,
) -> TimedInstruction[np.bool]:
    """Evaluate an expression that depends on time and returns a digital value.

    Args:
        expression: The expression to evaluate.
        parameters: The parameters to use in the evaluation.
        initial_time: The initial time of the evaluation.
        final_time: The final time of the evaluation.
        time_step: The time step for discretizing the time, in seconds.

    Returns:
        A timed instruction representing the digital value of the expression.
        The length of the instructions corresponds to the number of time steps between
        the initial and final time.

    Raises:
        InvalidTypeError: if the expression does not evaluate to a digital value.
    """

    result = evaluate_time_dependent_expression(
        expression, parameters, initial_time, final_time, time_step
    )
    if not isinstance(result, BoolResult):
        raise InvalidTypeError(
            f"{fmt.expression(expression)} does not evaluate to a digital value.",
        )
    return result.values


def evaluate_time_dependent_expression(
    expression: Expression,
    parameters: Mapping[DottedVariableName, Parameter],
    initial_time: Time,
    final_time: Time,
    time_step: Time,
) -> EvaluationResult:
    try:
        ast = parse(str(expression))
        return _evaluate_ast(ast, parameters, initial_time, final_time, time_step)
    except (EvaluationError, InvalidSyntaxError) as error:
        raise EvaluationError(
            f"Could not evaluate {fmt.expression(expression)}."
        ) from error


def _evaluate_ast(
    expression: nodes.Expression,
    parameters: Mapping[DottedVariableName, Parameter],
    initial_time: Time,
    final_time: Time,
    time_step: Time,
) -> EvaluationResult:
    if is_time_dependent(expression):
        raise NotImplementedError("Can't evaluate expressions that depend on t")
    else:
        value = _evaluate_scalar_ast(expression, parameters)
        return embed_scalar_as_time_dependent(
            value, initial_time, final_time, time_step
        )


def embed_scalar_as_time_dependent(
    value: Scalar, initial_time: Time, final_time: Time, time_step: Time
) -> EvaluationResult:
    length = number_ticks(initial_time, final_time, time_step)
    match value:
        case bool():
            return BoolResult(
                values=Pattern([value], dtype=np.dtype(np.bool)) * length,
                initial_value=value,
                final_value=value,
            )
        case int():
            return IntResult(
                values=Pattern([value], dtype=np.dtype(np.int64)) * length,
                initial_value=value,
                final_value=value,
            )
        case float():
            return FloatResult(
                values=Pattern([value], dtype=np.dtype(np.float64)) * length,
                initial_value=value,
                final_value=value,
            )
        case Quantity():
            in_base_units = value.to_base_units()
            return QuantityResult(
                values=Pattern([in_base_units.magnitude], dtype=np.dtype(np.float64))
                * length,
                initial_value=value.magnitude,
                final_value=value.magnitude,
                unit=value.unit,
            )
        case _:
            assert_never(value)
