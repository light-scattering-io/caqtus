from __future__ import annotations

import abc
import bisect
import collections
import itertools
import math
from collections.abc import Sequence
from heapq import merge
from types import NotImplementedType
from typing import (
    NewType,
    TypeVar,
    Generic,
    overload,
    Optional,
    assert_never,
    Callable,
    Any,
)

import numpy
import numpy as np
from numpy.typing import DTypeLike
from typing_extensions import deprecated

from caqtus.utils.itertools import pairwise

Length = NewType("Length", int)
Width = NewType("Width", int)
Depth = NewType("Depth", int)

_S = TypeVar("_S", covariant=True, bound=DTypeLike)
_T = TypeVar("_T", covariant=True, bound=DTypeLike)
_U = TypeVar("_U", covariant=True, bound=DTypeLike)

Array1D = numpy.ndarray[Any, numpy.dtype[_T]]


class SequencerInstruction(abc.ABC, Generic[_T]):
    """An immutable representation of instructions to output on a sequencer.

    This represents a high-level series of instructions to output on a sequencer.
    Each instruction is a compact representation of values to output at integer time
    steps.
    The length of the instruction is the number of time steps it takes to output all
    the values.
    The width of the instruction is the number of channels that are output at each time
    step.

    Instructions can be concatenated in time using the `+` operator or the
    :func:`concatenate`.
    An instruction can be repeated using the `*` operator with an integer.
    """

    @abc.abstractmethod
    def __len__(self) -> Length:
        """Returns the length of the instruction in clock cycles."""

        raise NotImplementedError

    @overload
    def __getitem__(self, item: int) -> _T: ...

    @overload
    def __getitem__(self, item: slice) -> SequencerInstruction[_T]: ...

    @overload
    def __getitem__(self, item: str) -> SequencerInstruction[_S]: ...

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
    def width(self) -> Width:
        """Returns the number of parallel channels that are output at each time step."""

        fields = self.dtype.fields
        if fields is None:
            return Width(1)
        else:
            return Width(len(fields))

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
        if isinstance(other, SequencerInstruction):
            if len(self) == 0:
                return other
            elif len(other) == 0:
                return self
            else:
                return concatenate(self, other)
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
                if isinstance(self, Repeated):
                    return Repeated(self._repetitions * other, self._instruction)
                else:
                    return Repeated(other, self)
        else:
            # We specifically raise an error here and not return NotImplemented to avoid
            # multiplication by a numpy integer taking over and returning a numpy
            # array instead of a SequencerInstruction.
            raise TypeError(f"Cannot multiply instruction by {other!r}")

    def __rmul__(self, other) -> SequencerInstruction[_T]:
        return self.__mul__(other)

    @abc.abstractmethod
    def __or__(
        self, other: SequencerInstruction[_S]
    ) -> SequencerInstruction[_U] | NotImplementedType:
        """Merge the channels of this instruction with another instruction.

        other:
            The other instruction to merge with this instruction.
            It must have the same length as the current instruction.

        Returns:
            A new instruction with the channels of this instruction and the other
            instruction merged to be output in parallel if this is supported.
            NotImplemented if the instructions cannot be merged. In this case, the
            caller should try the reverse order of the merge.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def __ror__(self, other):
        raise NotImplementedError

    @abc.abstractmethod
    def apply(
        self, func: Callable[[Array1D[_T]], Array1D[_S]]
    ) -> SequencerInstruction[_S]:
        raise NotImplementedError

    def _repr_mimebundle_(self, include=None, exclude=None):
        from ._to_graph import to_graph

        graph = to_graph(self)
        return graph._repr_mimebundle_(include, exclude)


class Pattern(SequencerInstruction[_T]):
    """An instruction representing a sequence of values.

    This is a fully explicit instruction for which each sample point must be given.
    """

    __slots__ = ("_pattern", "_length")

    def __init__(self, pattern, dtype: Optional[DTypeLike[_T]] = None):
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
            return Pattern.create_without_copy(self._pattern[item])
        elif isinstance(item, str):
            return Pattern.create_without_copy(self._pattern[item])
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
        return Pattern.create_without_copy(self._pattern.astype(dtype, copy=False))

    def __len__(self) -> Length:
        return self._length

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

    def __or__(self, other: SequencerInstruction[_S]) -> Pattern[_U]:
        if len(self) != len(other):
            raise ValueError("Instructions must have the same length")
        if self.dtype.fields is None:
            raise ValueError("Pattern must have at least one channel")
        if other.dtype.fields is None:
            raise ValueError("Pattern must have at least one channel")
        other_pattern = other.to_pattern()
        merged_dtype = merge_dtypes(self.dtype, other_pattern.dtype)
        merged = numpy.empty(len(self), dtype=merged_dtype)
        for name in self.dtype.names:
            merged[name] = self._pattern[name]
        for name in other_pattern.dtype.names:
            merged[name] = other_pattern._pattern[name]
        return Pattern.create_without_copy(merged)

    def __ror__(self, other):
        if len(self) != len(other):
            raise ValueError("Instructions must have the same length")
        if self.dtype.fields is None:
            raise ValueError("Pattern must have at least one channel")
        if other.dtype.fields is None:
            raise ValueError("Pattern must have at least one channel")
        other_pattern = other.to_pattern()
        merged_dtype = merge_dtypes(other_pattern.dtype, self.dtype)
        merged = numpy.empty(len(self), dtype=merged_dtype)
        for name in other_pattern.dtype.names:
            merged[name] = other_pattern._pattern[name]
        for name in self.dtype.names:
            merged[name] = self._pattern[name]
        return Pattern.create_without_copy(merged)

    def apply(self, func: Callable[[Array1D[_T]], Array1D[_S]]) -> Pattern[_S]:
        result = func(self._pattern)
        if len(result) != len(self):
            raise ValueError("Function must return an array of the same length")
        return Pattern.create_without_copy(result)

    @property
    def array(self) -> Array1D[_T]:
        return self._pattern


class Concatenated(SequencerInstruction[_T]):
    """Represents an immutable concatenation of instructions.

    Use the `+` operator or the function :func:`concatenate` to concatenate instructions.
    Do not use the class constructor directly.

    Attributes:
        instructions: The instructions concatenated by this instruction.
            This instruction is equivalent to chaining the instructions in this list
            one after the other.
    """

    __slots__ = ("_instructions", "_instruction_bounds", "_length")
    __match_args__ = ("instructions",)

    @property
    def instructions(self) -> tuple[SequencerInstruction[_T], ...]:
        return self._instructions

    def __init__(self, *instructions: SequencerInstruction[_T]):
        assert all(
            isinstance(instruction, SequencerInstruction)
            for instruction in instructions
        )
        # The following assertions define a "pure" concatenation.
        # (i.e. no empty instructions, no nested concatenations, and at least two
        # instructions).
        assert all(len(instruction) >= 1 for instruction in instructions)
        assert len(instructions) >= 2
        assert all(
            not isinstance(instruction, Concatenated) for instruction in instructions
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

    def __repr__(self):
        inner = ", ".join(repr(instruction) for instruction in self._instructions)
        return f"Concatenated({inner})"

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
        return concatenate(*results)

    def _get_field(self, field: str) -> SequencerInstruction:
        return Concatenated(*(instruction[field] for instruction in self._instructions))

    @property
    def dtype(self) -> numpy.dtype[_T]:
        return self._instructions[0].dtype

    def as_type(self, dtype: numpy.dtype[_S]) -> Concatenated[_S]:
        return type(self)(
            *(instruction.as_type(dtype) for instruction in self._instructions)
        )

    def __len__(self) -> Length:
        return self._length

    @property
    def depth(self) -> Depth:
        return Depth(max(instruction.depth for instruction in self._instructions) + 1)

    def to_pattern(self) -> Pattern[_T]:
        # noinspection PyProtectedMember
        new_array = numpy.concatenate(
            [instruction.to_pattern()._pattern for instruction in self._instructions],
            casting="safe",
        )
        return Pattern.create_without_copy(new_array)

    def __eq__(self, other):
        if isinstance(other, Concatenated):
            return self._instructions == other._instructions
        else:
            return NotImplemented

    # noinspection PyProtectedMember
    def __or__(self, other) -> SequencerInstruction[_U] | NotImplemented:
        if not isinstance(other, SequencerInstruction):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError("Instructions must have the same length")
        results = []
        if isinstance(other, Concatenated):
            new_bounds = merge(self._instruction_bounds, other._instruction_bounds)
            for start, stop in pairwise(new_bounds):
                results.append(self[start:stop] | other[start:stop])
        else:
            for (start, stop), instruction in zip(
                pairwise(self._instruction_bounds), self._instructions
            ):
                results.append(instruction | other[start:stop])
        if not results:
            return empty_like(self) | empty_like(other)
        return concatenate(*results)

    def __ror__(self, other):
        if not isinstance(other, SequencerInstruction):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError("Instructions must have the same length")
        results = []
        if isinstance(other, Concatenated):
            new_bounds = merge(self._instruction_bounds, other._instruction_bounds)
            for start, stop in pairwise(new_bounds):
                results.append(other[start:stop] | self[start:stop])
        else:
            for (start, stop), instruction in zip(
                pairwise(self._instruction_bounds), self._instructions
            ):
                results.append(other[start:stop] | instruction)
        if not results:
            return empty_like(other) | empty_like(self)
        return concatenate(*results)

    def apply(self, func: Callable[[Array1D[_T]], Array1D[_S]]) -> Concatenated[_S]:
        return Concatenated(
            *(instruction.apply(func) for instruction in self._instructions)
        )


class Repeated(SequencerInstruction[_T]):
    """Represents a repetition of an instruction.

    Use the `*` operator with an integer to repeat an instruction.
    Do not use the class constructor directly.

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
        assert isinstance(repetitions, int)
        assert isinstance(instruction, (Pattern, Concatenated, Repeated))
        assert repetitions >= 2
        assert len(instruction) >= 1

        self._repetitions = repetitions
        self._instruction = instruction
        self._length = Length(len(self._instruction) * self._repetitions)

    def __repr__(self):
        return (
            f"Repeated(repetitions={self._repetitions!r},"
            f" instruction={self._instruction!r})"
        )

    def __str__(self):
        if isinstance(self._instruction, Concatenated):
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
        return Repeated(self._repetitions, self._instruction[field])

    @property
    def dtype(self) -> numpy.dtype[_T]:
        return self._instruction.dtype

    def as_type(self, dtype: numpy.dtype[_S]) -> Repeated[_S]:
        return type(self)(self._repetitions, self._instruction.as_type(dtype))

    @property
    def depth(self) -> Depth:
        return Depth(self._instruction.depth + 1)

    def to_pattern(self) -> Pattern[_T]:
        inner_pattern = self._instruction.to_pattern()
        # noinspection PyProtectedMember
        new_array = numpy.tile(inner_pattern._pattern, self._repetitions)
        return Pattern.create_without_copy(new_array)

    def __eq__(self, other):
        if isinstance(other, Repeated):
            return (
                self._repetitions == other._repetitions
                and self._instruction == other._instruction
            )
        else:
            return NotImplemented

    def __or__(self, other: SequencerInstruction[_S]) -> SequencerInstruction[_U]:
        if not isinstance(other, SequencerInstruction):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError("Instructions must have the same length")
        if isinstance(other, Repeated):
            lcm = math.lcm(len(self._instruction), len(other.instruction))
            if lcm == len(self):
                b_a = tile(self.instruction, self.repetitions)
                b_b = tile(other.instruction, other.repetitions)
            else:
                r_a = lcm // len(self._instruction)
                b_a = self._instruction * r_a
                r_b = lcm // len(other.instruction)
                b_b = other.instruction * r_b
            block = b_a | b_b
            return block * (len(self) // len(block))
        else:
            return NotImplemented

    def __ror__(self, other):
        if not isinstance(other, SequencerInstruction):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError("Instructions must have the same length")
        if isinstance(other, Repeated):
            lcm = math.lcm(len(self._instruction), len(other.instruction))
            if lcm == len(self):
                b_a = tile(self.instruction, self.repetitions)
                b_b = tile(other.instruction, other.repetitions)
            else:
                r_a = lcm // len(self._instruction)
                b_a = self._instruction * r_a
                r_b = lcm // len(other.instruction)
                b_b = other.instruction * r_b
            block = b_b | b_a
            return block * (len(self) // len(block))
        else:
            return NotImplemented

    def apply(self, func: Callable[[Array1D[_T]], Array1D[_S]]) -> Repeated[_S]:
        return Repeated(self._repetitions, self._instruction.apply(func))


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
    return empty_with_dtype(instruction.dtype)


def empty_with_dtype(dtype: numpy.dtype[_T]) -> Pattern[_T]:
    return Pattern([], dtype=dtype)


def to_flat_dict(instruction: SequencerInstruction[_T]) -> dict[str, np.ndarray]:
    array = instruction.to_pattern()._pattern
    fields = array.dtype.fields
    return {name: array[name] for name in fields}


def tile(
    instruction: SequencerInstruction[_T], repetitions: int
) -> SequencerInstruction[_T]:
    return concatenate(*([instruction] * repetitions))


def concatenate(*instructions: SequencerInstruction[_T]) -> SequencerInstruction[_T]:
    """Concatenates the given instructions into a single instruction.

    Raises:
        ValueError: If no instructions are provided.
        TypeError: If the instructions have different dtypes.
    """

    if len(instructions) == 0:
        raise ValueError("Must provide at least one instruction")
    if not all(
        isinstance(instruction, SequencerInstruction) for instruction in instructions
    ):
        raise TypeError("All instructions must be instances of SequencerInstruction")
    dtype = instructions[0].dtype
    if not all(instruction.dtype == dtype for instruction in instructions):
        result_dtype = np.result_type(
            *[instruction.dtype for instruction in instructions]
        )
        instructions = [
            instruction.as_type(result_dtype) for instruction in instructions
        ]
    return _concatenate(*instructions)


@deprecated("Use concatenate instead")
def join(*instructions: SequencerInstruction[_T]) -> SequencerInstruction[_T]:
    return concatenate(*instructions)


def _concatenate(*instructions: SequencerInstruction[_T]) -> SequencerInstruction[_T]:
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
            return Concatenated(*instructions)
        case _:
            assert_never(useful_instructions)


def _break_concatenations(
    instructions: Sequence[SequencerInstruction[_T]],
) -> list[SequencerInstruction[_T]]:
    flat = []
    for instruction in instructions:
        if isinstance(instruction, Concatenated):
            flat.extend(instruction.instructions)
        else:
            flat.append(instruction)
    return flat


def merge_dtypes(a: numpy.dtype[_T], b: numpy.dtype[_S]) -> numpy.dtype[_U]:
    merged_dtype = numpy.dtype(
        [(name, a[name]) for name in a.names] + [(name, b[name]) for name in b.names]
    )
    return merged_dtype
