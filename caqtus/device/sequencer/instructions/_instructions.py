from __future__ import annotations

import abc
import bisect
import collections
import itertools
import math
from collections.abc import Sequence
from heapq import merge
from typing import (
    NewType,
    TypeVar,
    Generic,
    overload,
    Optional,
    assert_never,
    Callable,
    TypeAlias,
    Any,
)

import numpy
import numpy as np

from caqtus.utils.itertools import pairwise

Length = NewType("Length", int)
Width = NewType("Width", int)
Depth = NewType("Depth", int)

_S = TypeVar("_S", covariant=True, bound=numpy.generic)
_T = TypeVar("_T", covariant=True, bound=numpy.generic)
_U = TypeVar("_U", covariant=True, bound=numpy.generic)

Array1D = numpy.ndarray[Any, numpy.dtype[_T]]


class _BaseInstruction(abc.ABC, Generic[_T]):
    """An immutable representation of instructions to output on a sequencer.

    This represents a high-level series of instructions to output on a sequencer.
    Each instruction is a compact representation of values to output at integer time
    steps.
    The length of the instruction is the number of time steps it takes to output all
    the values.
    The width of the instruction is the number of channels that are output at each time
    step.
    """

    @abc.abstractmethod
    def __len__(self) -> Length:
        """Returns the length of the instruction in clock cycles."""

        raise NotImplementedError

    @overload
    def __getitem__(self, item: int) -> _T:
        ...

    @overload
    def __getitem__(self, item: slice) -> SequencerInstruction[_T]:
        ...

    @overload
    def __getitem__(self, item: str) -> SequencerInstruction[_S]:
        ...

    @abc.abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dtype(self) -> numpy.dtype[_T]:
        """Returns the dtype of the instruction."""

        raise NotImplementedError

    @abc.abstractmethod
    def as_type(self, dtype: numpy.dtype[_S]) -> SequencerInstruction[_S]:
        """Returns a new instruction with the given dtype."""

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def width(self) -> Width:
        """Returns the number of parallel channels that are output at each time step."""

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def depth(self) -> Depth:
        """Returns the number of nested instructions.

        The invariant `instruction.depth <= len(instruction)` always holds.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def to_pattern(self) -> Pattern[_T]:
        """Returns a flattened pattern of the instruction."""

        raise NotImplementedError

    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    def __add__(self, other) -> SequencerInstruction[_T]:
        if isinstance(other, _BaseInstruction):
            if len(self) == 0:
                return other
            elif len(other) == 0:
                return self
            else:
                return join(self, other)
        else:
            return NotImplemented

    def __mul__(self, other) -> SequencerInstruction[_T]:
        if isinstance(other, int):
            if other < 0:
                raise ValueError("Repetitions must be a positive integer")
            elif other == 0:
                return empty_like(self)
            elif other == 1:
                return self
            else:
                if isinstance(self, Repeat):
                    return Repeat(self._repetitions * other, self._instruction)
                else:
                    return Repeat(other, self)
        else:
            return NotImplemented

    def __rmul__(self, other) -> SequencerInstruction[_T]:
        return self.__mul__(other)

    @classmethod
    def _create_pattern_without_copy(cls, array: Array1D[_S]) -> Pattern[_S]:
        array.setflags(write=False)
        pattern = Pattern.__new__(Pattern)
        pattern._pattern = array
        pattern._length = Length(len(array))
        return pattern

    @abc.abstractmethod
    def merge_channels(self, other: SequencerInstruction[_S]) -> Pattern[_U]:
        raise NotImplementedError

    @abc.abstractmethod
    def apply(
        self, func: Callable[[Array1D[_T]], Array1D[_S]]
    ) -> SequencerInstruction[_S]:
        raise NotImplementedError


class Pattern(_BaseInstruction[_T]):
    __slots__ = ("_pattern", "_length")
    """An instruction to output a pattern on a sequencer."""

    def __init__(self, pattern, dtype: Optional[np.dtype[_T]] = None):
        self._pattern = numpy.array(pattern, dtype=dtype)
        self._pattern.setflags(write=False)
        self._length = Length(len(self._pattern))

    def __repr__(self):
        return f"Pattern({self._pattern.tolist()!r})"

    def __str__(self):
        return str(self._pattern.tolist())

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._pattern[item]
        elif isinstance(item, slice):
            return self.create_without_copy(self._pattern[item])
        elif isinstance(item, str):
            return self.create_without_copy(self._pattern[item])
        else:
            assert_never(item)

    @classmethod
    def create_without_copy(cls, array: Array1D[_S]) -> Pattern[_S]:
        array.setflags(write=False)
        pattern = cls.__new__(cls)
        pattern._pattern = array
        pattern._length = Length(len(array))
        return pattern

    @property
    def dtype(self) -> numpy.dtype[_T]:
        return self._pattern.dtype

    def as_type(self, dtype: numpy.dtype[_S]) -> Pattern[_S]:
        return self.create_without_copy(self._pattern.astype(dtype, copy=False))

    def __len__(self) -> Length:
        return self._length

    @property
    def width(self) -> Width:
        fields = self.dtype.fields
        if fields is None:
            return Width(1)
        else:
            return Width(len(fields))

    @property
    def depth(self) -> Depth:
        return Depth(0)

    def to_pattern(self) -> Pattern[_T]:
        return self

    def __eq__(self, other):
        if isinstance(other, Pattern):
            return numpy.array_equal(self._pattern, other._pattern)
        else:
            return NotImplemented

    def merge_channels(self, other: SequencerInstruction[_S]) -> Pattern[_U]:
        if len(self) != len(other):
            raise ValueError("Instructions must have the same length")
        if self.dtype.fields is None:
            raise ValueError("Pattern must have at least one channel")
        if other.dtype.fields is None:
            raise ValueError("Pattern must have at least one channel")
        other_pattern = other.to_pattern()
        merged_dtype = numpy.dtype(
            [(name, self.dtype[name]) for name in self.dtype.names]
            + [(name, other_pattern.dtype[name]) for name in other_pattern.dtype.names]
        )
        merged = numpy.empty(len(self), dtype=merged_dtype)
        for name in self.dtype.names:
            merged[name] = self._pattern[name]
        for name in other_pattern.dtype.names:
            merged[name] = other_pattern._pattern[name]
        return self._create_pattern_without_copy(merged)

    def apply(self, func: Callable[[Array1D[_T]], Array1D[_S]]) -> Pattern[_S]:
        result = func(self._pattern)
        if len(result) != len(self):
            raise ValueError("Function must return an array of the same length")
        return self._create_pattern_without_copy(result)

    @property
    def array(self) -> Array1D[_T]:
        return self._pattern


class Concatenate(_BaseInstruction[_T]):
    """Represents an immutable concatenation of instructions.

    Attributes:
        instructions: The instructions concatenated together.
    """

    __slots__ = ("_instructions", "_instruction_bounds", "_length")
    __match_args__ = ("instructions",)

    def __init__(self, *instructions: SequencerInstruction[_T]):
        """
        Do not use this constructor in user code.
        Instead, use the `+` operator or the `join` function.
        """

        # The following assertions define a "pure" concatenation.
        # (i.e. no empty instructions, no nested concatenations, and at least two
        # instructions).
        assert all(len(instruction) >= 1 for instruction in instructions)
        assert len(instructions) >= 2
        assert all(
            not isinstance(instruction, Concatenate) for instruction in instructions
        )

        assert all(
            instruction.dtype == instructions[0].dtype for instruction in instructions
        )

        self._instructions = instructions

        # self._instruction_bounds[i] is the first element index (included) the i-th
        # instruction
        #
        # self._instruction_bounds[i+1] is the last element index (excluded) of the
        # i-th instruction
        self._instruction_bounds = (0,) + tuple(
            itertools.accumulate(len(instruction) for instruction in self._instructions)
        )
        self._length = Length(self._instruction_bounds[-1])

    @property
    def instructions(self) -> tuple[SequencerInstruction[_T], ...]:
        return self._instructions

    def __repr__(self):
        inner = ", ".join(repr(instruction) for instruction in self._instructions)
        return f"Concatenate({inner})"

    def __str__(self):
        sub_strings = [str(instruction) for instruction in self._instructions]
        return " + ".join(sub_strings)

    def __getitem__(self, item):
        match item:
            case int() as index:
                return self._get_index(index)
            case slice() as slice_:
                return self._get_slice(slice_)
            case str() as field:
                return self._get_field(field)
            case _:
                assert_never(item)

    def _get_index(self, index: int) -> _T:
        index = _normalize_index(index, len(self))
        instruction_index = bisect.bisect_right(self._instruction_bounds, index) - 1
        instruction = self._instructions[instruction_index]
        instruction_start_index = self._instruction_bounds[instruction_index]
        return instruction[index - instruction_start_index]

    def _get_slice(self, slice_: slice) -> SequencerInstruction[_T]:
        start, stop, step = _normalize_slice(slice_, len(self))
        if step != 1:
            raise NotImplementedError
        start_step_index = bisect.bisect_right(self._instruction_bounds, start) - 1
        stop_step_index = bisect.bisect_left(self._instruction_bounds, stop) - 1

        results = [empty_like(self)]
        for instruction_index in range(start_step_index, stop_step_index + 1):
            instruction_start_index = self._instruction_bounds[instruction_index]
            instruction_slice_start = max(start, instruction_start_index)
            instruction_stop_index = self._instruction_bounds[instruction_index + 1]
            instruction_slice_stop = min(stop, instruction_stop_index)
            instruction_slice = slice(
                instruction_slice_start - instruction_start_index,
                instruction_slice_stop - instruction_start_index,
                step,
            )
            results.append(self._instructions[instruction_index][instruction_slice])
        return join(*results)

    def _get_field(self, field: str) -> SequencerInstruction:
        return Concatenate(*(instruction[field] for instruction in self._instructions))

    @property
    def dtype(self) -> numpy.dtype[_T]:
        return self._instructions[0].dtype

    def as_type(self, dtype: numpy.dtype[_S]) -> Concatenate[_S]:
        return type(self)(
            *(instruction.as_type(dtype) for instruction in self._instructions)
        )

    def __len__(self) -> Length:
        return self._length

    @property
    def width(self) -> Width:
        return self._instructions[0].width

    @property
    def depth(self) -> Depth:
        return Depth(max(instruction.depth for instruction in self._instructions) + 1)

    def to_pattern(self) -> Pattern[_T]:
        # noinspection PyProtectedMember
        new_array = numpy.concatenate(
            [instruction.to_pattern()._pattern for instruction in self._instructions],
            casting="safe",
        )
        return self._create_pattern_without_copy(new_array)

    def __eq__(self, other):
        if isinstance(other, Concatenate):
            return self._instructions == other._instructions
        else:
            return NotImplemented

    # noinspection PyProtectedMember
    def merge_channels(
        self, other: SequencerInstruction[_S]
    ) -> SequencerInstruction[_U]:
        if len(self) != len(other):
            raise ValueError("Instructions must have the same length")
        match other:
            case Pattern() as pattern:
                return self.to_pattern().merge_channels(pattern)
            case Concatenate() as concatenate:
                new_bounds = merge(
                    self._instruction_bounds, concatenate._instruction_bounds
                )
                results = [empty_like(self).merge_channels(empty_like(concatenate))]
                for start, stop in pairwise(new_bounds):
                    results.append(
                        self[start:stop].merge_channels(concatenate[start:stop])
                    )
                return join(*results)
            case Repeat() as repeat:
                results = [empty_like(self).merge_channels(empty_like(repeat))]
                for (start, stop), instruction in zip(
                    pairwise(self._instruction_bounds), self._instructions
                ):
                    results.append(instruction.merge_channels(repeat[start:stop]))
                return join(*results)
            case _:
                assert_never(other)

    def apply(self, func: Callable[[Array1D[_T]], Array1D[_S]]) -> Concatenate[_S]:
        return Concatenate(
            *(instruction.apply(func) for instruction in self._instructions)
        )


class Repeat(_BaseInstruction[_T]):
    """Represents a repetition of an instruction.

    Attributes:
        instruction: The instruction to repeat.
        repetitions: The number of times to repeat the instruction.
    """

    __slots__ = ("_repetitions", "_instruction", "_length")

    @property
    def repetitions(self) -> int:
        return self._repetitions

    @property
    def instruction(self) -> SequencerInstruction[_T]:
        return self._instruction

    def __init__(self, repetitions: int, instruction: SequencerInstruction[_T]):
        """
        Do not use this constructor in user code.
        Instead, use the `*` operator.
        """

        assert isinstance(repetitions, int)
        assert isinstance(instruction, (Pattern, Concatenate, Repeat))
        assert repetitions >= 2
        assert len(instruction) >= 1

        self._repetitions = repetitions
        self._instruction = instruction
        self._length = Length(len(self._instruction) * self._repetitions)

    def __repr__(self):
        return (
            f"Repeat(repetitions={self._repetitions!r},"
            f" instruction={self._instruction!r})"
        )

    def __str__(self):
        if isinstance(self._instruction, Concatenate):
            return f"{self._repetitions} * ({self._instruction!s})"
        else:
            return f"{self._repetitions} * {self._instruction!s}"

    def __len__(self) -> Length:
        return self._length

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._get_index(item)
        elif isinstance(item, slice):
            return self._get_slice(item)
        elif isinstance(item, str):
            return self._get_field(item)
        else:
            assert_never(item)

    def _get_index(self, index: int) -> _T:
        index = _normalize_index(index, len(self))
        _, r = divmod(index, len(self._instruction))
        return self._instruction[r]

    def _get_slice(self, slice_: slice) -> SequencerInstruction[_T]:
        start, stop, step = _normalize_slice(slice_, len(self))
        if step != 1:
            raise NotImplementedError
        length = len(self._instruction)
        first_repetition = math.ceil(start / length)
        last_repetition = math.floor(stop / length)
        if first_repetition > last_repetition:
            return self._instruction[
                start - first_repetition * length : stop - first_repetition * length
            ]
        else:
            previous_repetition = math.floor(start / length)
            prepend = self._instruction[
                start
                - previous_repetition
                * length : (first_repetition - previous_repetition)
                * length
            ]
            middle = self._instruction * (last_repetition - first_repetition)
            append = self._instruction[: stop - last_repetition * length]
            return prepend + middle + append

    def _get_field(self, field: str) -> SequencerInstruction:
        return Repeat(self._repetitions, self._instruction[field])

    @property
    def dtype(self) -> numpy.dtype[_T]:
        return self._instruction.dtype

    def as_type(self, dtype: numpy.dtype[_S]) -> Repeat[_S]:
        return type(self)(self._repetitions, self._instruction.as_type(dtype))

    @property
    def width(self) -> Width:
        return self._instruction.width

    @property
    def depth(self) -> Depth:
        return Depth(self._instruction.depth + 1)

    def to_pattern(self) -> Pattern[_T]:
        inner_pattern = self._instruction.to_pattern()
        # noinspection PyProtectedMember
        new_array = numpy.tile(inner_pattern._pattern, self._repetitions)
        return self._create_pattern_without_copy(new_array)

    def __eq__(self, other):
        if isinstance(other, Repeat):
            return (
                self._repetitions == other._repetitions
                and self._instruction == other._instruction
            )
        else:
            return NotImplemented

    def merge_channels(
        self, other: SequencerInstruction[_S]
    ) -> SequencerInstruction[_U]:
        if len(self) != len(other):
            raise ValueError("Instructions must have the same length")
        match other:
            case Pattern() as pattern:
                return self.to_pattern().merge_channels(pattern)
            case Concatenate() as concatenate:
                results = [empty_like(self).merge_channels(empty_like(concatenate))]
                for (start, stop), instruction in zip(
                    pairwise(concatenate._instruction_bounds), concatenate._instructions
                ):
                    results.append(self[start:stop].merge_channels(instruction))
                return join(*results)
            case Repeat() as repeat:
                lcm = math.lcm(len(self._instruction), len(repeat._instruction))
                r_a = lcm // len(self._instruction)
                b_a = tile(self._instruction, r_a)
                r_b = lcm // len(repeat._instruction)
                b_b = tile(repeat._instruction, r_b)
                block = b_a.merge_channels(b_b)
                return block * (len(self) // len(block))
            case _:
                assert_never(other)

    def apply(self, func: Callable[[Array1D[_T]], Array1D[_S]]) -> Repeat[_S]:
        return Repeat(self._repetitions, self._instruction.apply(func))


def _normalize_index(index: int, length: int) -> int:
    normalized = index if index >= 0 else length + index
    if not 0 <= normalized < length:
        raise IndexError(f"Index {index} is out of bounds for length {length}")
    return normalized


def _normalize_slice_index(index: int, length: int) -> int:
    normalized = index if index >= 0 else length + index
    if not 0 <= normalized <= length:
        raise IndexError(f"Slice index {index} is out of bounds for length {length}")
    return normalized


def _normalize_slice(slice_: slice, length: int) -> tuple[int, int, int]:
    step = slice_.step or 1
    if step == 0:
        raise ValueError("Slice step cannot be zero")
    if slice_.start is None:
        start = 0 if step > 0 else length - 1
    else:
        start = _normalize_slice_index(slice_.start, length)
    if slice_.stop is None:
        stop = length if step > 0 else -1
    else:
        stop = _normalize_slice_index(slice_.stop, length)

    return start, stop, step


def empty_like(instruction: SequencerInstruction[_T]) -> Pattern[_T]:
    return Pattern([], dtype=instruction.dtype)


def to_flat_dict(instruction: SequencerInstruction[_T]) -> dict[str, np.ndarray]:
    array = instruction.to_pattern()._pattern
    fields = array.dtype.fields
    return {name: array[name] for name in fields}


def tile(
    instruction: SequencerInstruction[_T], repetitions: int
) -> SequencerInstruction[_T]:
    return join(*([instruction] * repetitions))


def join(*instructions: SequencerInstruction[_T]) -> SequencerInstruction[_T]:
    """Joins the given instructions into a single instruction.

    Raises:
        ValueError: If no instructions are provided.
        TypeError: If the instructions have different dtypes.
    """

    if len(instructions) == 0:
        raise ValueError("Must provide at least one instruction")
    dtype = instructions[0].dtype
    if not all(instruction.dtype == dtype for instruction in instructions):
        raise TypeError("All instructions must have the same dtype")
    return _join(*instructions)


def _join(*instructions: SequencerInstruction[_T]) -> SequencerInstruction[_T]:
    assert len(instructions) >= 1
    assert all(
        instruction.dtype == instructions[0].dtype for instruction in instructions
    )

    instruction_deque = collections.deque(_break_concatenations(instructions))

    useful_instructions = []
    while instruction_deque:
        instruction = instruction_deque.popleft()
        if len(instruction) == 0:
            continue
        if isinstance(instruction, Pattern):
            concatenated_patterns = [instruction]
            while instruction_deque and isinstance(instruction_deque[0], Pattern):
                concatenated_patterns.append(instruction_deque.popleft())
            if len(concatenated_patterns) == 1:
                useful_instructions.append(concatenated_patterns[0])
            else:
                useful_instructions.append(
                    Pattern(
                        numpy.concatenate(
                            [pattern.array for pattern in concatenated_patterns],
                            casting="safe",
                        )
                    )
                )
        else:
            useful_instructions.append(instruction)

    match useful_instructions:
        case []:
            return empty_like(instructions[0])
        case [instruction]:
            return instruction
        case [*instructions]:
            return Concatenate(*instructions)
        case _:
            assert_never(useful_instructions)


def _break_concatenations(
    instructions: Sequence[SequencerInstruction[_T]],
) -> list[SequencerInstruction[_T]]:
    flat = []
    for instruction in instructions:
        if isinstance(instruction, Concatenate):
            flat.extend(instruction.instructions)
        else:
            flat.append(instruction)
    return flat


SequencerInstruction: TypeAlias = Pattern[_T] | Concatenate[_T] | Repeat[_T]