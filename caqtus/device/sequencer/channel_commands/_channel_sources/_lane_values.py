from __future__ import annotations

from typing import Optional, Mapping, Any

import attrs
from cattrs.gen import make_dict_structure_fn, override

import caqtus.formatter as fmt
from caqtus.shot_compilation import ShotContext
from caqtus.types.expression import Expression
from caqtus.types.recoverable_exceptions import InvalidValueError, InvalidTypeError
from caqtus.types.timelane import DigitalTimeLane, AnalogTimeLane
from caqtus.types.variable_name import DottedVariableName
from caqtus.utils import serialization
from ._compile_digital_lane import compile_digital_lane
from ._constant import Constant
from .compile_analog_lane import compile_analog_lane
from ..channel_output import ChannelOutput, DimensionedSeries
from ...instructions import Pattern


@attrs.define
class LaneValues(ChannelOutput):
    """Indicates that the output should be the values taken by a given lane.

    Attributes:
        lane: The name of the lane from which to take the values.
        default: The default value to take if the lane is absent from the shot
            time lanes.
    """

    lane: str = attrs.field(
        converter=str,
        on_setattr=attrs.setters.convert,
    )
    default: Optional[ChannelOutput] = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            attrs.validators.instance_of(ChannelOutput)
        ),
        on_setattr=attrs.setters.validate,
    )

    def __str__(self) -> str:
        if self.default is not None:
            return f"{self.lane} | {self.default}"
        return self.lane

    def evaluate(
        self,
        required_time_step: int,
        prepend: int,
        append: int,
        shot_context: ShotContext,
    ):
        """Evaluate the output of a channel as the values of a lane.

        This function will look in the shot time lanes to find the lane referenced by
        the output and evaluate the values of this lane.
        If the lane cannot be found, and the output has a default value, this default
        value will be used.
        If the lane cannot be found and there is no default value, a ValueError will be
        raised.
        """

        lane_name = self.lane
        try:
            lane = shot_context.get_lane(lane_name)
        except KeyError:
            if self.default is not None:
                return self.default.evaluate(
                    required_time_step,
                    prepend,
                    append,
                    shot_context,
                )
            else:
                raise InvalidValueError(
                    f"Could not find {fmt.lane(lane_name)}"
                ) from None
        if isinstance(lane, DigitalTimeLane):
            lane_values = compile_digital_lane(lane, required_time_step, shot_context)
            result = DimensionedSeries(lane_values, units=None)
        elif isinstance(lane, AnalogTimeLane):
            result = compile_analog_lane(
                lane,
                shot_context.get_variables(),
                shot_context.get_step_start_times(),
                required_time_step,
            )
        else:
            raise InvalidTypeError(
                f"Don't know how to compile lane {fmt.lane(lane_name)} with type "
                f"{fmt.type_(type(lane))}"
            )

        prepend_value = result.values[0]
        prepend_pattern = prepend * Pattern([prepend_value])
        append_value = result.values[-1]
        append_pattern = append * Pattern([append_value])
        return DimensionedSeries(
            values=prepend_pattern + result.values + append_pattern,
            units=result.units,
        )

    def evaluate_max_advance_and_delay(
        self,
        time_step: int,
        variables: Mapping[DottedVariableName, Any],
    ) -> tuple[int, int]:
        return 0, 0


def structure_lane_default(default_data, _):
    # We need this custom structure hook, because in the past the default value of a
    # LaneValues was a Constant and not any ChannelOutput.
    # In that case, the type of the default value was not serialized, so we need to
    # deal with this special case.
    if default_data is None:
        return None
    elif isinstance(default_data, str):
        default_expression = serialization.structure(default_data, Expression)
        return Constant(value=default_expression)
    elif "type" in default_data:
        return serialization.structure(default_data, ChannelOutput)
    else:
        return serialization.structure(default_data, Constant)


structure_lane_values = make_dict_structure_fn(
    LaneValues,
    serialization.converters["json"],
    default=override(struct_hook=structure_lane_default),
)


def unstructure_lane_values(lane_values):
    return {
        "lane": lane_values.lane,
        "default": serialization.unstructure(
            lane_values.default, Optional[ChannelOutput]
        ),
    }


serialization.register_structure_hook(LaneValues, structure_lane_values)
serialization.register_unstructure_hook(LaneValues, unstructure_lane_values)
