import functools
from collections.abc import Mapping
from typing import assert_never

import numpy as np

import caqtus.formatter as fmt
import caqtus_parsing.nodes as nodes
from caqtus.types.expression import Expression
from caqtus.types.parameter import Parameter
from caqtus.types.recoverable_exceptions import EvaluationError
from caqtus.types.variable_name import DottedVariableName
from caqtus_parsing import parse, InvalidSyntaxError
from .._evaluate_scalar_expression import evaluate_bool_expression
from ...timed_instructions import TimedInstruction, Pattern
from ...timing import Time, number_ticks

type DigitalInstruction = TimedInstruction[np.bool]
type Parameters = Mapping[DottedVariableName, Parameter]


def evaluate_time_dependent_digital_expression(
    expression: Expression, parameters: Parameters, t1: Time, t2: Time, timestep: Time
) -> DigitalInstruction:
    """Evaluate a time-dependent digital expression.

    Args:
        expression: The expression to evaluate.
        parameters: The parameters to use in the evaluation.
        t1: The start time of the evaluation.
        t2: The end time of the evaluation.
        timestep: The time step of the evaluation.

    Returns:
        The result of the evaluation.

    Raises:
        EvaluationError: if an error occurred during evaluation, with the reason for the
            error as the exception cause.
    """

    try:
        ast = parse(str(expression))
        return evaluate_digital_expression(ast, parameters, t1, t2, timestep)
    except (EvaluationError, InvalidSyntaxError) as error:
        raise EvaluationError(
            f"Could not evaluate {fmt.expression(expression)}."
        ) from error


def evaluate_digital_expression(
    expression: nodes.Expression,
    parameters: Parameters,
    t1: Time,
    t2: Time,
    timestep: Time,
) -> DigitalInstruction:
    if not is_time_dependent(expression):
        value = evaluate_bool_expression(expression, parameters)
        length = number_ticks(t1, t2, timestep)
        return Pattern([value]) * length

    match expression:
        case int() | float() | nodes.Quantity():
            raise AssertionError(
                "This should never happen, because at this point, the expression "
                "is known to be time-dependent."
            )
        case nodes.Variable(name=name):
            assert name == "t"
            raise InvalidOperationError(
                f"{fmt.expression(expression)} is not a valid digital expression."
            )
        case (
            nodes.Add()
            | nodes.Subtract()
            | nodes.Multiply()
            | nodes.Divide()
            | nodes.Power()
            | nodes.Plus()
            | nodes.Minus()
        ):
            raise InvalidOperationError(
                f"{fmt.expression(expression)} is not a valid digital expression."
            )
        case nodes.Call():
            return evaluate_call(expression, parameters, t1, t2, timestep)
        case _:
            assert_never(expression)


@functools.lru_cache
def is_time_dependent(expression: nodes.Expression) -> bool:
    match expression:
        case int() | float() | nodes.Quantity():
            return False
        case nodes.Variable(name=name):
            return name == "t"
        case (
            nodes.Add()
            | nodes.Subtract()
            | nodes.Multiply()
            | nodes.Divide()
            | nodes.Power() as binary_operator
        ):
            return is_time_dependent(binary_operator.left) or is_time_dependent(
                binary_operator.right
            )
        case nodes.Plus() | nodes.Minus() as unary_operator:
            return is_time_dependent(unary_operator.operand)
        case nodes.Call():
            return any(is_time_dependent(arg) for arg in expression.args)
        case _:
            assert_never(expression)


def evaluate_call(
    call: nodes.Call,
    parameters: Parameters,
    t1: Time,
    t2: Time,
    timestep: Time,
) -> DigitalInstruction:
    raise NotImplementedError


class InvalidOperationError(EvaluationError):
    """Raised when an invalid operation is attempted."""

    pass