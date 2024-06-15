from __future__ import annotations

import functools
from collections.abc import Mapping, Iterable
from typing import TypeAlias, TypeGuard, assert_never, Any

import attrs
import numpy

from caqtus.types.expression import Expression
from caqtus.types.expression.expression import EvaluationError
from caqtus.types.parameter import (
    AnalogValue,
    is_analog_value,
    NotAnalogValueError,
    get_unit,
    add_unit,
    magnitude_in_unit,
)
from caqtus.utils import serialization
from .iteration_configuration import IterationConfiguration, Unknown
from ..units import DimensionalityError
from ..variable_name import DottedVariableName


def validate_step(instance, attribute, step):
    if is_step(step):
        return
    else:
        raise TypeError(f"Invalid step: {step}")


@attrs.define
class ContainsSubSteps:
    sub_steps: list[Step] = attrs.field(
        validator=attrs.validators.deep_iterable(
            iterable_validator=attrs.validators.instance_of(list),
            member_validator=validate_step,
        ),
        on_setattr=attrs.setters.validate,
    )


@attrs.define
class VariableDeclaration:
    """Represents the declaration of a variable.

    Attributes:
        variable: The name of the variable.
        value: The unevaluated to assign to the variable.
    """

    __match_args__ = ("variable", "value")

    variable: DottedVariableName = attrs.field(
        validator=attrs.validators.instance_of(DottedVariableName),
        on_setattr=attrs.setters.validate,
    )
    value: Expression = attrs.field(
        validator=attrs.validators.instance_of(Expression),
        on_setattr=attrs.setters.validate,
    )

    def __str__(self):
        return f"{self.variable} = {self.value}"


@attrs.define
class LinspaceLoop(ContainsSubSteps):
    """Represents a loop that iterates between two values with a fixed number of steps.

    Attributes:
        variable: The name of the variable that is being iterated over.
        start: The start value of the variable.
        stop: The stop value of the variable.
        num: The number of steps to take between the start and stop values.
    """

    __match_args__ = ("variable", "start", "stop", "num", "sub_steps")
    variable: DottedVariableName = attrs.field(
        validator=attrs.validators.instance_of(DottedVariableName),
        on_setattr=attrs.setters.validate,
    )
    start: Expression = attrs.field(
        validator=attrs.validators.instance_of(Expression),
        on_setattr=attrs.setters.validate,
    )
    stop: Expression = attrs.field(
        validator=attrs.validators.instance_of(Expression),
        on_setattr=attrs.setters.validate,
    )
    num: int = attrs.field(
        converter=int,
        validator=attrs.validators.ge(0),
        on_setattr=attrs.setters.pipe(attrs.setters.convert, attrs.setters.validate),
    )

    def __str__(self):
        return (
            f"for {self.variable} = {self.start} to {self.stop} with {self.num} steps"
        )

    def loop_values(
        self, evaluation_context: Mapping[DottedVariableName, Any]
    ) -> Iterable[AnalogValue]:
        """Returns the values that the variable represented by this loop takes.

        Args:
            evaluation_context: Contains the value of the variables with which to
                evaluate the start and stop expressions of the loop.

        Raises:
            EvaluationError: if the start or stop expressions could not be evaluated.
            NotAnalogValueError: if the start or stop expressions don't evaluate to an
                analog value.
            DimensionalityError: if the start or stop values are not commensurate.
        """

        start = self.start.evaluate(evaluation_context)
        if not is_analog_value(start):
            raise NotAnalogValueError(f"Start of '{self}' is not an analog value.")
        stop = self.stop.evaluate(evaluation_context)
        if not is_analog_value(stop):
            raise NotAnalogValueError(f"Stop of '{self}' is not an analog value.")

        unit = get_unit(start)
        start_magnitude = magnitude_in_unit(start, unit)
        stop_magnitude = magnitude_in_unit(stop, unit)

        for value in numpy.linspace(start_magnitude, stop_magnitude, self.num):
            # val.item() is used to convert numpy scalar to python scalar
            value_with_unit = add_unit(value.item(), unit)
            yield value_with_unit


@attrs.define
class ArangeLoop(ContainsSubSteps):
    """Represents a loop that iterates between two values with a fixed step size.

    Attributes:
        variable: The name of the variable that is being iterated over.
        start: The start value of the variable.
        stop: The stop value of the variable.
        step: The step size between each value.
    """

    __match_args__ = ("variable", "start", "stop", "step", "sub_steps")
    variable: DottedVariableName = attrs.field(
        validator=attrs.validators.instance_of(DottedVariableName),
        on_setattr=attrs.setters.validate,
    )
    start: Expression = attrs.field(
        validator=attrs.validators.instance_of(Expression),
        on_setattr=attrs.setters.validate,
    )
    stop: Expression = attrs.field(
        validator=attrs.validators.instance_of(Expression),
        on_setattr=attrs.setters.validate,
    )
    step: Expression = attrs.field(
        validator=attrs.validators.instance_of(Expression),
        on_setattr=attrs.setters.validate,
    )

    def __str__(self):
        return (
            f"for {self.variable} = {self.start} to {self.stop} with {self.step} "
            f"spacing"
        )

    def loop_values(
        self, evaluation_context: Mapping[DottedVariableName, Any]
    ) -> Iterable[AnalogValue]:
        """Returns the values that the variable represented by this loop takes.

        Args:
            evaluation_context: Contains the value of the variables with which to
                evaluate the start, stop and step expressions of the loop.

        Raises:
            EvaluationError: if the start, stop or step expressions could not be
                evaluated.
            NotAnalogValueError: if the start, stop or step expressions don't evaluate
                to an analog value.
            DimensionalityError: if the start, stop and step values are not
                commensurate.
        """

        start = self.start.evaluate(evaluation_context)
        if not is_analog_value(start):
            raise NotAnalogValueError(f"Start of '{self}' is not an analog value.")
        stop = self.stop.evaluate(evaluation_context)
        if not is_analog_value(stop):
            raise NotAnalogValueError(f"Stop of '{self}' is not an analog value.")
        step = self.step.evaluate(evaluation_context)
        if not is_analog_value(step):
            raise NotAnalogValueError(f"Step of '{self}' is not an analog value.")

        unit = get_unit(start)
        start_magnitude = magnitude_in_unit(start, unit)
        stop_magnitude = magnitude_in_unit(stop, unit)
        step_magnitude = magnitude_in_unit(step, unit)

        for value in numpy.arange(start_magnitude, stop_magnitude, step_magnitude):
            # val.item() is used to convert numpy scalar to python scalar
            value_with_unit = add_unit(value.item(), unit)
            yield value_with_unit


@attrs.define
class ExecuteShot:
    """Step that represents the execution of a shot."""

    pass


def unstructure_hook(execute_shot: ExecuteShot) -> str:
    return {"execute": "shot"}


def structure_hook(data: str, cls: type[ExecuteShot]) -> ExecuteShot:
    return ExecuteShot()


serialization.register_unstructure_hook(ExecuteShot, unstructure_hook)

serialization.register_structure_hook(ExecuteShot, structure_hook)

"""TypeAlias for the different types of steps."""
Step: TypeAlias = ExecuteShot | VariableDeclaration | LinspaceLoop | ArangeLoop


def is_step(step) -> TypeGuard[Step]:
    return isinstance(
        step,
        (
            ExecuteShot,
            VariableDeclaration,
            LinspaceLoop,
            ArangeLoop,
        ),
    )


@attrs.define
class StepsConfiguration(IterationConfiguration):
    """Define the parameter iteration of a sequence as a list of steps.

    Attributes:
        steps: The steps of the iteration.
    """

    steps: list[Step] = attrs.field(
        validator=attrs.validators.deep_iterable(
            iterable_validator=attrs.validators.instance_of(list),
            member_validator=validate_step,
        ),
        on_setattr=attrs.setters.validate,
    )

    def expected_number_shots(self) -> int | Unknown:
        return sum(expected_number_shots(step) for step in self.steps)

    def get_parameter_names(self) -> set[DottedVariableName]:
        return set().union(*[get_parameter_names(step) for step in self.steps])

    @classmethod
    def dump(cls, steps_configuration: StepsConfiguration) -> serialization.JSON:
        return serialization.unstructure(steps_configuration, StepsConfiguration)

    @classmethod
    def load(cls, data: serialization.JSON) -> StepsConfiguration:
        return serialization.structure(data, StepsConfiguration)


@functools.singledispatch
def expected_number_shots(step: Step) -> int | Unknown:  # type: ignore
    assert_never(step)


@expected_number_shots.register
def _(step: VariableDeclaration):
    return 0


@expected_number_shots.register
def _(step: ExecuteShot):
    return 1


@expected_number_shots.register
def _(step: LinspaceLoop):
    sub_steps_number = sum(
        expected_number_shots(sub_step) for sub_step in step.sub_steps
    )
    return sub_steps_number * step.num


@expected_number_shots.register
def _(step: ArangeLoop):
    try:
        length = len(list(step.loop_values({})))
    except (EvaluationError, NotAnalogValueError, DimensionalityError):
        # The errors above can occur if the steps are still being edited or if the
        # expressions depend on other variables that are not defined here.
        # These can be errors on the user side, so we don't want to crash on them, and
        # we just indicate that we don't know the number of shots.
        return Unknown()

    sub_steps_number = sum(
        expected_number_shots(sub_step) for sub_step in step.sub_steps
    )
    return sub_steps_number * length


def get_parameter_names(step: Step) -> set[DottedVariableName]:
    match step:
        case VariableDeclaration(variable=variable, value=_):
            return {variable}
        case ExecuteShot():
            return set()
        case LinspaceLoop(variable=variable, sub_steps=sub_steps) | ArangeLoop(
            variable=variable, sub_steps=sub_steps
        ):
            return {variable}.union(
                *[get_parameter_names(sub_step) for sub_step in sub_steps]
            )
        case _:
            assert_never(step)
