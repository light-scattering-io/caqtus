from collections.abc import Sequence, Mapping, Iterable
from dataclasses import dataclass
from itertools import accumulate
from numbers import Real

import numpy as np

from expression import Expression
from parameter_types import is_analog_value, Parameter
from parameter_types.analog_value import magnitude_in_unit
from sequence.configuration import DigitalLane, StepName, AnalogLane, Ramp
from sequencer.channel import ChannelInstruction, ChannelPattern
from units import Quantity, ureg, units
from variable.name import DottedVariableName
from variable.namespace import VariableNamespace
from .evaluation_error import ShotEvaluationError


def compile_digital_lane(
    step_durations: Sequence[float],
    lane: DigitalLane,
    time_step: float,
) -> ChannelInstruction[bool]:
    step_bounds = get_step_bounds(step_durations)
    instructions = []
    for cell_value, start, stop in lane.get_value_spans():
        length = number_ticks(step_bounds[start], step_bounds[stop], time_step)
        instructions.append(ChannelPattern([cell_value]) * length)
    return ChannelInstruction.join(instructions, dtype=bool)


def number_ticks(start_time: Real, stop_time: Real, time_step: Real) -> int:
    return int(stop_time / time_step) - int(start_time / time_step)


def get_step_bounds(step_durations: Iterable[float]) -> Sequence[float]:
    return [0.0] + list((accumulate(step_durations)))


def compile_analog_lane(
    step_names: Sequence[StepName],
    step_durations: Sequence[float],
    lane: AnalogLane,
    variables: VariableNamespace,
    time_step: float,
) -> ChannelInstruction[float]:
    return CompileAnalogLane(
        step_names, step_durations, lane, variables, time_step
    ).compile()


@dataclass(slots=True)
class CompileAnalogLane:
    step_names: Sequence[StepName]
    step_durations: Sequence[float]
    lane: AnalogLane
    variables: VariableNamespace
    time_step: float

    def compile(self) -> ChannelInstruction[float]:
        step_bounds = get_step_bounds(self.step_durations)
        instructions = []
        for cell, start, stop in self.lane.get_value_spans():
            length = number_ticks(step_bounds[start], step_bounds[stop], self.time_step)
            if isinstance(cell, Expression):
                instructions.append(self._compile_expression_cell(cell, length))
            elif isinstance(cell, Ramp):
                instructions.append(self._compile_ramp_cell(start - 1, stop, length))
        return ChannelInstruction.join(instructions, dtype=float)

    def _compile_expression_cell(
        self, expression: Expression, length: int
    ) -> ChannelInstruction[float]:
        variables = self.variables | units
        if _is_constant(expression):
            result = (
                ChannelPattern(
                    [float(self._evaluate_expression(expression, variables))]
                )
                * length
            )
        else:
            variables = variables | {
                DottedVariableName("t"): _compute_time_array(length, self.time_step)
            }
            result = ChannelPattern(self._evaluate_expression(expression, variables))
        if not len(result) == length:
            raise ShotEvaluationError(
                f"Expression '{expression}' evaluates to an array of length"
                f" {len(result)} while the expected length is {length}"
            )
        return result

    def _compile_ramp_cell(
        self, previous_index: int, next_index: int, length: int
    ) -> ChannelInstruction[float]:
        previous_step_duration = sum(
            self.step_durations[
                self.lane.start_index(previous_index) : self.lane.end_index(
                    previous_index
                )
            ]
        )
        variables = (
            self.variables
            | units
            | {DottedVariableName("t"): previous_step_duration * ureg.s}
        )
        previous_value = self._evaluate_expression(
            self.lane.get_effective_value(previous_index), variables
        )

        variables = self.variables | units | {DottedVariableName("t"): 0.0 * ureg.s}
        next_value = self._evaluate_expression(
            self.lane.get_effective_value(next_index), variables
        )
        return ChannelPattern(
            np.linspace(previous_value, next_value, length), dtype=float
        )

    def _evaluate_expression(
        self, expression: Expression, variables: Mapping[DottedVariableName, Parameter]
    ) -> Real | np.ndarray:
        try:
            value = expression.evaluate(variables | units)
        except Exception as e:
            raise ShotEvaluationError(
                f"Could not evaluate expression '{expression.body}'"
            ) from e
        if not is_analog_value(value):
            raise ShotEvaluationError(
                f"Expression '{expression.body}' evaluates to a non-analog value"
                f" ({value})"
            )
        return self._convert_to_lane_units(value)

    def _convert_to_lane_units(self, value: Quantity) -> Real | np.ndarray:
        return magnitude_in_unit(value, self.lane.units)


def _is_constant(expression: Expression) -> bool:
    return "t" not in expression.upstream_variables


def _compute_time_array(length: int, time_step: float) -> Quantity:
    return (np.arange(length) * time_step) * ureg.s
