import contextlib
import difflib
from collections.abc import Mapping
from typing import Self, assert_never, assert_type

import attrs
from caqtus_parsing import AST, ParseNode, parse

from caqtus.types.parameter import ParameterSchema

from ...types.recoverable_exceptions import RecoverableException
from ...types.units import Quantity, Unit
from ...types.variable_name import DottedVariableName
from ._compiled_expression import (
    BooleanLiteral,
    CompiledExpression,
    FloatLiteral,
    IntegerLiteral,
    QuantityLiteral,
    _CompiledExpression,
)


@attrs.frozen
class CompilationContext:
    parameter_schema: ParameterSchema
    units: Mapping[str, Unit]


def compile_expression(
    expression: str,
    compilation_context: CompilationContext,
    time_dependent: bool,
) -> CompiledExpression:
    ast = parse(str(expression))

    compiled = _compile_ast(expression, ast, compilation_context, time_dependent)
    if isinstance(compiled, CompiledExpression):
        return compiled
    else:
        assert_type(compiled, Unit)
        raise TypeError(f"Expression {expression} evaluates to a unit, not a value.")


def _compile_ast(
    expression: str,
    ast: AST,
    ctx: CompilationContext,
    time_dependent: bool,
) -> _CompiledExpression | Unit:
    match ast:
        case ParseNode.Integer(x):
            return IntegerLiteral(x)
        case ParseNode.Float(x):
            return FloatLiteral(x)
        case ParseNode.Quantity(magnitude, unit_name):
            try:
                unit = ctx.units[unit_name]
            except KeyError:
                with error_context(expression, ast):
                    raise UndefinedUnitError(
                        f"Unit {unit_name} is not defined."
                    ) from None
            return QuantityLiteral(magnitude, unit)
        case ParseNode.Identifier(name):
            with error_context(expression, ast):
                return compile_identifier(DottedVariableName(name), ctx)
        case ParseNode.UnaryOperation() as unary_op:
            return compile_unary_operation(expression, unary_op, ctx, time_dependent)
        case _:
            assert_never(ast)


def compile_identifier(
    name: DottedVariableName, ctx: CompilationContext
) -> _CompiledExpression | Unit:
    if str(name) in ctx.units:
        return ctx.units[str(name)]
    elif name in ctx.parameter_schema.constant_schema:
        parameter = ctx.parameter_schema.constant_schema[name]
        match parameter:
            case bool():
                return BooleanLiteral(parameter)
            case int():
                return IntegerLiteral(parameter)
            case float():
                return FloatLiteral(parameter)
            case Quantity():
                return QuantityLiteral(parameter.magnitude, parameter.units)
            case _:
                assert_never(parameter)
    elif name in ctx.parameter_schema.variable_schema:
        raise NotImplementedError
    else:
        matches = difflib.get_close_matches(
            str(name), [str(n) for n in ctx.parameter_schema.names()], n=1
        )
        error = UndefinedParameterError(f'Parameter "{name}" is not defined.')
        if matches:
            error.add_note(f'Did you mean "{matches[0]}"?')
        raise error


def compile_unary_operation(
    expression: str,
    unary_op: ParseNode.UnaryOperation,
    ctx: CompilationContext,
    time_dependent: bool,
) -> _CompiledExpression | Unit:
    operand = _compile_ast(expression, unary_op.operand, ctx, time_dependent)
    match operand:
        case Unit():
            with error_context(expression, unary_op):
                raise ValueError(f"Cannot apply {unary_op.operator} to {operand:~}")
        case _:
            raise NotImplementedError(f"Cannot apply {unary_op.operator} to {operand}")


class UndefinedParameterError(ValueError):
    pass


class UndefinedUnitError(ValueError):
    pass


class CompilationError(RecoverableException):
    @classmethod
    def new(cls, expr: str, node: AST) -> Self:
        underlined = (
            expr[: node.span[0]]
            + "\033[4m"
            + expr[node.span[0] : node.span[1]]
            + "\033[0m"
            + expr[node.span[1] :]
        )
        return cls(f'An error occurred while compiling "{underlined}".')


@contextlib.contextmanager
def error_context(
    expression: str,
    ast: AST,
):
    try:
        yield
    except Exception as e:
        raise CompilationError.new(expression, ast) from e
