"""Contains functions for computing timing of a sequencer."""

import decimal
import math
from collections.abc import Iterable, Sequence
from itertools import accumulate
from typing import NewType

from caqtus.device.sequencer import TimeStep

Time = NewType("Time", decimal.Decimal)
"""A type for representing time in seconds.

It uses a decimal.Decimal to represent time in seconds to avoid floating point errors.
"""

ns = Time(decimal.Decimal("1e-9"))


def start_tick(start_time: Time, time_step: Time) -> int:
    """Returns the included first tick index of the step starting at start_time."""

    return math.ceil(start_time / time_step)


def stop_tick(stop_time: Time, time_step: Time) -> int:
    """Returns the excluded last tick index of the step ending at stop_time."""

    return math.ceil(stop_time / time_step)


def number_ticks(start_time: Time, stop_time: Time, time_step: Time) -> int:
    """Returns the number of ticks between start_time and stop_time.

    Args:
        start_time: The start time in seconds.
        stop_time: The stop time in seconds.
        time_step: The time step in seconds.
    """

    return stop_tick(stop_time, time_step) - start_tick(start_time, time_step)


def start_time_step(start_time: Time, time_step: TimeStep) -> int:
    """Returns the time of the step starting at start_time."""

    return start_tick(start_time, Time(time_step * ns))


def stop_time_step(stop_time: Time, time_step: TimeStep) -> int:
    """Returns the time of the step ending at stop_time."""

    return stop_tick(stop_time, Time(time_step * ns))


def number_time_steps(duration: Time, time_step: TimeStep) -> int:
    """Returns the number of ticks covering the given duration.

    Args:
        duration: The duration in seconds.
        time_step: The time step in seconds.
    """

    return number_ticks(Time(decimal.Decimal(0)), duration, Time(time_step * ns))


def number_time_steps_between(
    start_time: Time, stop_time: Time, time_step: TimeStep
) -> int:
    """Returns the number of ticks covering the given duration.

    Args:
        start_time: The start time in seconds.
        stop_time: The stop time in seconds.
        time_step: The time step in seconds.
    """

    return number_ticks(start_time, stop_time, Time(time_step * ns))


def get_step_bounds(step_durations: Iterable[Time]) -> Sequence[Time]:
    """Returns the time at which each step starts from their durations.

    For an iterable of step durations [d_0, d_1, ..., d_n], the step starts are
    [0, d_0, d_0 + d_1, ..., d_0 + ... + d_n]. It has one more element than the
    iterable of step durations, with the last element being the total duration.
    """

    zero = Time(decimal.Decimal(0))
    return [zero] + list((accumulate(step_durations)))
