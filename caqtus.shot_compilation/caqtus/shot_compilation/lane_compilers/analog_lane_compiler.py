from collections.abc import Sequence
from typing import assert_never, Optional

import numpy as np

from caqtus.device.sequencer.instructions import SequencerInstruction, Pattern, join
from caqtus.session.shot.timelane import AnalogTimeLane, Ramp
from caqtus.types.expression import Expression
from caqtus.types.parameter import (
    AnalogValue,
    is_analog_value,
    is_quantity,
    magnitude_in_unit,
)
from caqtus.types.units import ureg
from caqtus.types.variable_name import VariableName
from .evaluate_step_durations import evaluate_step_durations
from .timing import get_step_bounds, start_tick, stop_tick, number_ticks, ns
from ..unit_namespace import units
from ..variable_namespace import VariableNamespace

TIME_VARIABLE = VariableName("t")


class AnalogLaneCompiler:
    """Evaluates an analog time lane to a sequencer instruction."""

    def __init__(
        self,
        lane: AnalogTimeLane,
        step_names: Sequence[str],
        step_durations: Sequence[Expression],
        unit: Optional[str],
    ):
        """

        Args:
            lane: The lane to compile by replacing the expressions inside it when given
            the variable values.
            step_names: The names of the steps in the lane. Must be the same length as
            the lane.
            step_durations: The durations to be evaluated of the steps in the lane.
            Must be the same length as the lane.
            unit: The unit in which the sequencer instruction should be returned.
            Can be None if the lane is dimensionless.
        """
        if len(lane) != len(step_names):
            raise ValueError(
                f"Number of steps in lane ({len(lane)}) does not match number of"
                f" step names ({len(step_names)})"
            )
        if len(lane) != len(step_durations):
            raise ValueError(
                f"Number of steps in lane ({len(lane)}) does not match number of"
                f" step durations ({len(step_durations)})"
            )
        self.lane = lane
        self.steps = list(zip(step_names, step_durations))
        self.unit = unit

    def compile(
        self, variables: VariableNamespace, time_step: int
    ) -> SequencerInstruction[np.float64]:
        """Compile the lane to a sequencer instruction.

        This function discretizes the lane time and replaces the expressions in the
        lane with the given variable values.
        It also evaluates the ramps in the lane.
        The sequencer instruction returned is the magnitude of the lane in the given
        unit.
        """

        step_durations = evaluate_step_durations(self.steps, variables)
        step_bounds = get_step_bounds(step_durations)
        instructions = []
        for cell_value, (cell_start_index, cell_stop_index) in zip(
            self.lane.values(), self.lane.bounds()
        ):
            cell_start_time = step_bounds[cell_start_index]
            cell_stop_time = step_bounds[cell_stop_index]
            if isinstance(cell_value, Expression):
                instruction = self._compile_expression_cell(
                    variables | units,
                    cell_value,
                    cell_start_time,
                    cell_stop_time,
                    time_step,
                )
            elif isinstance(cell_value, Ramp):
                instruction = self._compile_ramp_cell(
                    cell_start_index, cell_stop_index, step_bounds, variables, time_step
                )
            else:
                assert_never(cell_value)
            instructions.append(instruction)
        return join(*instructions)

    def _compile_expression_cell(
        self,
        variables,
        expression: Expression,
        start: float,
        stop: float,
        time_step: int,
    ) -> SequencerInstruction[np.float64]:
        length = number_ticks(start, stop, time_step * ns)
        if is_constant(expression):
            evaluated = self._evaluate_expression(expression, variables)
            value = magnitude_in_unit(evaluated, self.unit)
            result = Pattern([float(value)], dtype=np.float64) * length
        else:
            variables = variables | {
                TIME_VARIABLE: (get_time_array(start, stop, time_step) - start) * ureg.s
            }
            evaluated = self._evaluate_expression(expression, variables)
            result = Pattern(magnitude_in_unit(evaluated, self.unit), dtype=np.float64)
        if not len(result) == length:
            raise ValueError(
                f"Expression <{expression}> evaluates to an array of length"
                f" {len(result)} while the expected length is {length}"
            )
        return result

    def _compile_ramp_cell(
        self,
        start_index: int,
        stop_index: int,
        step_bounds: Sequence[float],
        variables,
        time_step: int,
    ) -> SequencerInstruction[np.float64]:
        t0 = step_bounds[start_index]
        t1 = step_bounds[stop_index]
        previous_step_duration = (
            step_bounds[self.lane.get_bounds(start_index - 1)[1]]
            - step_bounds[self.lane.get_bounds(start_index - 1)[0]]
        )
        v = variables | units | {TIME_VARIABLE: previous_step_duration * ureg.s}
        ramp_start = self._evaluate_expression(self.lane[start_index - 1], v)
        if is_quantity(ramp_start):
            ramp_start = ramp_start.to_base_units()

        v = variables | units | {TIME_VARIABLE: 0.0 * ureg.s}
        ramp_end = self._evaluate_expression(self.lane[stop_index], v)
        if is_quantity(ramp_end):
            ramp_end = ramp_end.to_base_units()

        # Don't need to give units to t, because we'll be dividing by t1 - t0 anyway
        t = get_time_array(t0, t1, time_step)
        result = (t - t0) / (t1 - t0) * (ramp_end - ramp_start) + ramp_start

        return Pattern(magnitude_in_unit(result, self.unit), dtype=np.float64)

    @staticmethod
    def _evaluate_expression(expression: Expression, variables) -> AnalogValue:
        try:
            value = expression.evaluate(variables | units)
        except Exception as e:
            raise ValueError(f"Could not evaluate expression <{expression}>") from e
        if not is_analog_value(value):
            raise ValueError(
                f"Expression <{expression}> evaluates to a non-analog value ({value})"
            )
        return value


def is_constant(expression: Expression) -> bool:
    return TIME_VARIABLE not in expression.upstream_variables


def get_time_array(start: float, stop: float, time_step: int) -> np.ndarray:
    times = (
        np.arange(start_tick(start, time_step * ns), stop_tick(stop, time_step * ns))
        * time_step
        * ns
    )
    return times