import functools
from typing import assert_type, assert_never

import attrs
import numpy as np

import caqtus_parsing.nodes as nodes
from caqtus.shot_compilation.timed_instructions import (
    TimedInstruction,
    Pattern,
    create_ramp,
    Concatenated,
    Repeated,
    Ramp,
    merge_instructions,
    concatenate,
)
from caqtus.shot_compilation.timing import Time, number_ticks, start_tick, stop_tick
from caqtus.types.parameter import Parameters
from caqtus.types.units import BaseUnit, dimensionless, Quantity, SECOND, Unit
from ._is_time_dependent import is_time_dependent
from .._evaluate_scalar_expression import evaluate_expression


@attrs.define
class AnalogInstruction:
    magnitudes: TimedInstruction[np.float64]
    units: BaseUnit


def evaluate_analog_ast(
    ast: nodes.Expression, parameters: Parameters, t1: Time, t2: Time, timestep: Time
) -> AnalogInstruction:
    if not is_time_dependent(ast):
        value = evaluate_expression(ast, parameters)
        length = number_ticks(t1, t2, timestep)
        if isinstance(value, (bool, int, float)):
            return AnalogInstruction(
                magnitudes=Pattern([float(value)]) * length,
                units=dimensionless,
            )
        else:
            assert_type(value, Quantity[float])
            in_base_units = value.to_base_units()
            return AnalogInstruction(
                magnitudes=Pattern([in_base_units.magnitude]) * length,
                units=in_base_units.units,
            )
    else:
        match ast:
            case int() | float() | nodes.Quantity():
                raise AssertionError("Unreachable")
            case nodes.Variable(name=name):
                if name != "t":
                    raise AssertionError("Unreachable")
                tick_start = start_tick(t1, timestep)
                tick_stop = stop_tick(t2, timestep)
                value_start = tick_start * timestep - t1
                value_stop = tick_stop * timestep - t1
                length = tick_stop - tick_start
                return AnalogInstruction(
                    magnitudes=create_ramp(value_start, value_stop, length),
                    units=SECOND,
                )
            case nodes.Call():
                raise NotImplementedError(
                    "Time dependent function calls are not yet supported."
                )
            case nodes.Plus() | nodes.Minus() as unary_operator:
                return evaluate_unary_operator(
                    unary_operator, parameters, t1, t2, timestep
                )
            case nodes.Multiply():
                left = evaluate_analog_ast(ast.left, parameters, t1, t2, timestep)
                right = evaluate_analog_ast(ast.right, parameters, t1, t2, timestep)
                units = left.units * right.units
                assert isinstance(units, Unit)
                # Can multiply magnitudes because the result and operand are expressed
                # in base units.
                return AnalogInstruction(
                    magnitudes=multiply(left.magnitudes, right.magnitudes),
                    units=units.to_base(),
                )
            case _:
                assert_never(ast)


def evaluate_unary_operator(
    unary_operator: nodes.UnaryOperator,
    parameters: Parameters,
    t1: Time,
    t2: Time,
    timestep: Time,
) -> AnalogInstruction:
    operand = evaluate_analog_ast(unary_operator.operand, parameters, t1, t2, timestep)
    match unary_operator:
        case nodes.Plus():
            return operand
        case nodes.Minus():
            return AnalogInstruction(
                magnitudes=negate(operand.magnitudes),
                units=operand.units,
            )
        case _:
            assert_never(unary_operator)


@functools.singledispatch
def negate(instruction) -> TimedInstruction[np.float64]:
    raise NotImplementedError(f"Negation of {type(instruction)} is not supported.")


@negate.register(Pattern)
def negate_pattern(pattern: Pattern[np.float64]) -> Pattern[np.float64]:
    return Pattern.create_without_copy(-pattern.array)


@negate.register(Concatenated)
def negate_concatenated(
    concatenated: Concatenated[np.float64],
) -> Concatenated[np.float64]:
    return Concatenated(*(negate(instr) for instr in concatenated.instructions))


@negate.register(Repeated)
def negate_repeated(repeated: Repeated[np.float64]) -> Repeated[np.float64]:
    return Repeated(repeated.repetitions, negate(repeated.instruction))


@negate.register(Ramp)
def negate_ramp(ramp: Ramp[np.float64]) -> TimedInstruction[np.float64]:
    return create_ramp(-ramp.start, -ramp.stop, len(ramp))


def multiply(
    a: TimedInstruction[np.float64], b: TimedInstruction[np.float64]
) -> TimedInstruction[np.float64]:
    merged = merge_instructions(left=a, right=b)
    return _multiply(merged)


@functools.singledispatch
def _multiply(instruction) -> TimedInstruction[np.float64]:
    raise NotImplementedError(
        f"Multiplication of {type(instruction)} is not supported."
    )


@_multiply.register(Pattern)
def _multiply_pattern(pattern: Pattern[np.void]) -> Pattern[np.float64]:
    left_array = pattern["left"].array
    right_array = pattern["right"].array
    return Pattern.create_without_copy(left_array * right_array)


@_multiply.register(Concatenated)
def _multiply_concatenated(
    concatenated: Concatenated[np.void],
) -> TimedInstruction[np.float64]:
    return concatenate(*(_multiply(instr) for instr in concatenated.instructions))


@_multiply.register(Repeated)
def _multiply_repeated(repeated: Repeated[np.void]) -> TimedInstruction[np.float64]:
    return repeated.repetitions * _multiply(repeated.instruction)


@_multiply.register(Ramp)
def _multiply_ramp(ramp: Ramp[np.void]) -> TimedInstruction[np.float64]:
    left = ramp["left"]
    right = ramp["right"]

    a2 = left.slope * right.slope
    if a2 == 0:
        a1 = left.slope * right.intercept + right.slope * left.intercept
        a0 = left.intercept * right.intercept
        stop = a0 + a1 * len(ramp)
        return create_ramp(a0, stop, len(ramp))
    else:
        return _multiply(ramp.to_pattern())
