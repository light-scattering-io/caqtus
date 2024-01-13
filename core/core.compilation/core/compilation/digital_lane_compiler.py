from collections.abc import Sequence

from core.device.sequencer.instructions import SequencerInstruction, Pattern, join
from core.session.shot.timelane import DigitalTimeLane
from core.types.expression import Expression
from .timing import number_ticks, ns
from .unit_namespace import units
from .variable_namespace import VariableNamespace


class DigitalLaneCompiler:
    def __init__(self, lane: DigitalTimeLane):
        self.lane = lane

    def compile(
        self, step_bounds: Sequence[float], variables: VariableNamespace, time_step: int
    ) -> SequencerInstruction[bool]:
        """Compiles the lane into a sequencer instruction.

        Args:
            step_bounds: Contains the start time and stop time of each step.
            It must have one more element than the number of steps in the lane.
            step_bounds[i] is the start time of step i and step_bounds[i+1] is the stop
            time of step i.
            variables: The variables to use for evaluating expressions.
            time_step: The time step in nanoseconds to discretize the lane.

        Returns:

        """

        instructions = []
        for cell_value, (start, stop) in zip(self.lane.values(), self.lane.bounds()):
            length = number_ticks(step_bounds[start], step_bounds[stop], time_step * ns)
            if isinstance(cell_value, bool):
                instructions.append(self.get_constant_instruction(cell_value, length))
            elif isinstance(cell_value, Expression):
                value = cell_value.evaluate(variables | units)
                if not isinstance(value, bool):
                    raise TypeError(
                        f"Expression {cell_value} does not evaluate to bool, but to"
                        f" {value}"
                    )
                instructions.append(self.get_constant_instruction(value, length))

            else:
                raise NotImplementedError(
                    f"Unexpected value {cell_value} in digital lane"
                )
        return join(*instructions)

    @staticmethod
    def get_constant_instruction(
        value: bool, length: int
    ) -> SequencerInstruction[bool]:
        return Pattern([value]) * length


#
# elif isinstance(cell_value, Blink):
# period = (
#     cell_value.period.evaluate(variables | units).to("ns").magnitude
# )
# duty_cycle = (
#     Quantity(cell_value.duty_cycle.evaluate(variables | units))
#     .to(dimensionless)
#     .magnitude
# )
# if not 0 <= duty_cycle <= 1:
#     raise ShotEvaluationError(
#         f"Duty cycle '{cell_value.duty_cycle.body}' must be between 0 and"
#         f" 1, not {duty_cycle}"
#     )
# num_ticks_per_period, _ = divmod(period, time_step)
# num_ticks_high = math.ceil(num_ticks_per_period * duty_cycle)
# num_ticks_low = num_ticks_per_period - num_ticks_high
# num_clock_pulses, remainder = divmod(length, num_ticks_per_period)
# phase = (
#     Quantity(cell_value.phase.evaluate(variables | units))
#     .to(dimensionless)
#     .magnitude
# )
# if not 0 <= phase <= 2 * math.pi:
#     raise ShotEvaluationError(
#         f"Phase '{cell_value.phase.body}' must be between 0 and 2*pi, not"
#         f" {phase}"
#     )
# split_position = round(phase / (2 * math.pi) * num_ticks_per_period)
# clock_pattern = (
#         Pattern([True]) * num_ticks_high + Pattern([False]) * num_ticks_low
# )
# a, b = clock_pattern[:split_position], clock_pattern[split_position:]
# clock_pattern = b + a
# pattern = (
#         clock_pattern * num_clock_pulses + Pattern([False]) * remainder
# )
# if not len(pattern) == length:
#     raise RuntimeError(
#         f"Pattern length {len(pattern)} does not match expected length"
#         f" {length}"
#     )
# print(f"{pattern=}")
