from __future__ import annotations

import abc
import functools
import math
from collections.abc import Iterable, Sequence
from typing import Optional, Mapping, Any, TypeVar

import attrs
import cattrs
import numpy as np
from numpy.typing import ArrayLike

from caqtus.shot_compilation import ShotContext
from caqtus.types.parameter import add_unit, magnitude_in_unit
from caqtus.types.units import Unit
from caqtus.types.variable_name import DottedVariableName
from caqtus.utils import serialization
from caqtus.utils.itertools import pairwise
from ._structure_hook import structure_channel_output
from .channel_output import ChannelOutput
from ..instructions import (
    SequencerInstruction,
    Pattern,
    Concatenated,
    Repeated,
    Ramp,
    ramp,
    concatenate,
)


class TimeIndependentMapping(ChannelOutput, abc.ABC):
    """A functional mapping of input values to output values independent of time.

    This represents channel transformations of the form:

    .. math::
        y(t) = f(x_0(t), x_1(t), ..., x_n(t))

    where x_0, x_1, ..., x_n are the input and y is the output.
    """

    @abc.abstractmethod
    def inputs(self) -> tuple[ChannelOutput, ...]:
        """Returns the input values of the mapping."""

        raise NotImplementedError

    def evaluate_max_advance_and_delay(
        self,
        time_step: int,
        variables: Mapping[DottedVariableName, Any],
    ) -> tuple[int, int]:
        advances_and_delays = [
            input_.evaluate_max_advance_and_delay(time_step, variables)
            for input_ in self.inputs()
        ]
        advances, delays = zip(*advances_and_delays)
        return max(advances), max(delays)


def data_points_converter(data_points: Iterable[tuple[float, float]]):
    point_to_tuple = [(x, y) for x, y in data_points]
    return tuple(sorted(point_to_tuple))


@attrs.define
class CalibratedAnalogMapping(TimeIndependentMapping):
    """Maps its input to an output quantity by interpolating a set of points.

    This mapping is useful for example when one needs to convert an experimentally
    measurable quantity (e.g. the frequency sent to an AOM) as a function of a control
    parameter (e.g. the voltage sent to the AOM driver).
    In this example, we need to know which voltage to apply to the AOM driver to obtain
    a given frequency.
    This conversion is defined by a set of points (x, y) where x is the input quantity
    and y is the output quantity.
    In the example above, x would be the frequency and y would be the voltage, because
    for a given frequency, we need to know which voltage to apply to the AOM driver.

    Attributes:
        input_units: The units of the input quantity
        input_: Describe the input argument of the mapping.
        output_units: The units of the output quantity
        measured_data_points: tuple of (input, output) tuples.
        The points will be rearranged to have the inputs sorted.
    """

    input_: ChannelOutput = attrs.field(
        validator=attrs.validators.instance_of(ChannelOutput),
        on_setattr=attrs.setters.validate,
    )
    input_units: Optional[str] = attrs.field(
        converter=attrs.converters.optional(str),
        on_setattr=attrs.setters.convert,
    )
    output_units: Optional[str] = attrs.field(
        converter=attrs.converters.optional(str),
        on_setattr=attrs.setters.convert,
    )
    measured_data_points: tuple[tuple[float, float], ...] = attrs.field(
        converter=data_points_converter, on_setattr=attrs.setters.convert
    )

    @property
    def input_values(self) -> tuple[float, ...]:
        return tuple(x[0] for x in self.measured_data_points)

    @property
    def output_values(self) -> tuple[float, ...]:
        return tuple(x[1] for x in self.measured_data_points)

    def inputs(self) -> tuple[ChannelOutput]:
        return (self.input_,)

    def __getitem__(self, index: int) -> tuple[float, float]:
        return self.measured_data_points[index]

    def __setitem__(self, index: int, values: tuple[float, float]):
        new_data_points = list(self.measured_data_points)
        new_data_points[index] = values
        self.measured_data_points = tuple(new_data_points)

    def set_input(self, index: int, value: float):
        self[index] = (value, self[index][1])

    def set_output(self, index: int, value: float):
        self[index] = (self[index][0], value)

    def pop(self, index: int):
        """Remove a data point from the mapping."""

        new_data_points = list(self.measured_data_points)
        new_data_points.pop(index)
        self.measured_data_points = tuple(new_data_points)

    def insert(self, index: int, input_: float, output: float):
        """Insert a data point into the mapping."""

        new_data_points = list(self.measured_data_points)
        new_data_points.insert(index, (input_, output))
        self.measured_data_points = tuple(new_data_points)

    def __str__(self):
        return f"{self.input_} [{self.input_units}] -> [{self.output_units}]"

    def evaluate(
        self,
        required_time_step: int,
        required_unit: Optional[Unit],
        prepend: int,
        append: int,
        shot_context: ShotContext,
    ) -> SequencerInstruction:
        input_values = self.input_.evaluate(
            required_time_step,
            self.input_units,
            prepend,
            append,
            shot_context,
        )
        calibration = Calibration(self.measured_data_points)
        output_values = calibration.apply(input_values)
        if required_unit != self.output_units:
            output_values = output_values.apply(
                functools.partial(
                    _convert_units,
                    input_unit=self.output_units,
                    output_unit=required_unit,
                )
            )
        return output_values


T = TypeVar("T", bound=ArrayLike)


class Calibration:
    def __init__(self, calibration_points: Sequence[tuple[float, float]]):
        if len(calibration_points) < 2:
            raise NotEnoughCalibrationPointsError(
                "Calibration must have at least 2 data points"
            )
        sorted_points = sorted(calibration_points, key=lambda x: x[0])
        self.input_values = (
            (-math.inf,) + tuple(x for x, _ in sorted_points) + (math.inf,)
        )
        self.output_values = (
            (sorted_points[0][1],)
            + tuple(y for _, y in sorted_points)
            + (sorted_points[-1][1],)
        )

    @functools.singledispatchmethod
    def apply(
        self, instruction: SequencerInstruction
    ) -> SequencerInstruction[np.floating]:
        raise NotImplementedError(
            f"Don't know how to apply calibration to instruction of type "
            f"{type(instruction)}"
        )

    @apply.register
    def _apply_calibration_pattern(self, pattern: Pattern) -> Pattern[np.floating]:
        if not issubclass(pattern.dtype.type, np.floating):
            raise TypeError(
                f"Can't apply calibration to pattern with dtype {pattern.dtype}"
            )
        result = self(pattern.array)
        return Pattern.create_without_copy(result)

    def __call__(self, value: T) -> T:
        interp = np.interp(
            x=value,
            xp=self.input_values,
            fp=self.output_values,
        )
        # Warning !!!
        # We want to make absolutely sure that the output is within the range of data
        # points that are measured, to avoid values that could be dangerous for the
        # hardware.
        # To ensure this, we clip the output to the range of the measured data points.
        min_ = np.min(self.output_values)
        max_ = np.max(self.output_values)
        clipped = np.clip(interp, min_, max_)
        return clipped

    @apply.register
    def _apply_calibration_concatenation(
        self, concatenation: Concatenated
    ) -> Concatenated[np.floating]:
        return Concatenated(
            *(self.apply(instruction) for instruction in concatenation.instructions)
        )

    @apply.register
    def _apply_calibration_repetition(
        self, repetition: Repeated
    ) -> Repeated[np.floating]:
        return Repeated(
            repetition.repetitions,
            self.apply(repetition.instruction),
        )

    @apply.register
    def _apply_calibration_ramp(self, r: Ramp) -> SequencerInstruction[np.floating]:
        results = []
        l = len(r)
        a = r.start
        b = r.stop
        if a == b:
            value = self(a)
            return ramp(value, value, l)
        elif a < b:
            for (x0, y0), (x1, y1) in pairwise(
                zip(self.input_values, self.output_values)
            ):
                i_min = math.ceil(max(0, l * (x0 - a) / (b - a)))
                i_max = math.floor(min(l, l * (x1 - a) / (b - a)))
                if i_min == l:
                    v_min = self(b)
                else:
                    v_min = self(r[i_min])
                if i_max == l:
                    v_max = self(b)
                else:
                    v_max = self(r[i_max])
                results.append(ramp(v_min, v_max, i_max - i_min))
            return concatenate(*results)
        else:
            return reversed(self.apply(reversed(r)))


# Workaround for https://github.com/python-attrs/cattrs/issues/430
structure_hook = cattrs.gen.make_dict_structure_fn(
    CalibratedAnalogMapping,
    serialization.converters["json"],
    input_=cattrs.override(struct_hook=structure_channel_output),
)

serialization.register_structure_hook(CalibratedAnalogMapping, structure_hook)


def _convert_units(
    array: np.ndarray, input_unit: Optional[str], output_unit: Optional[str]
) -> np.ndarray:
    if input_unit == output_unit:
        return array
    return magnitude_in_unit(add_unit(array, input_unit), output_unit)  # type: ignore


class NotEnoughCalibrationPointsError(ValueError):
    pass
