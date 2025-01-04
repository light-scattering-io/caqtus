from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence, Callable
from functools import cached_property
from typing import Any, Protocol, TypeVar, Generic, runtime_checkable

import attrs
import msgpack
import numpy as np
import polars as pl

ARRAY_TYPE = 1


class DataType[S, U](Protocol):
    def dumps(self, value: S) -> bytes:
        return msgpack.dumps(self.unstructure_hook(value))  # type: ignore[reportReturnType]

    def loads(self, data: bytes) -> S:
        return self.structure_hook(msgpack.loads(data))  # type: ignore[reportArgumentType]

    @property
    def unstructure_hook(self) -> Callable[[S], U]: ...

    @property
    def structure_hook(self) -> Callable[[U], S]: ...

    @abc.abstractmethod
    def to_polars_dtype(self) -> pl.DataType: ...


@attrs.frozen
class Float(DataType[float, float]):
    @property
    def unstructure_hook(self) -> Callable[[Any], float]:
        return float

    @property
    def structure_hook(self) -> Callable[[Any], float]:
        return self._hook

    @staticmethod
    def _hook(value):
        if not isinstance(value, float):
            raise ValueError(f"expected float, not {value}")
        return value

    def to_polars_dtype(self) -> pl.Float64:
        return pl.Float64()


@attrs.frozen
class Int(DataType[int, int]):
    @property
    def unstructure_hook(self) -> Callable[[Any], int]:
        return int

    @property
    def structure_hook(self) -> Callable[[Any], int]:
        return self._hook

    @staticmethod
    def _hook(value):
        if not isinstance(value, int):
            raise ValueError(f"expected int, not {value}")
        return value

    def to_polars_dtype(self) -> pl.Int64:
        return pl.Int64()


@attrs.frozen
class Boolean(DataType[bool, bool]):
    @property
    def unstructure_hook(self) -> Callable[[Any], bool]:
        return bool

    @property
    def structure_hook(self) -> Callable[[Any], bool]:
        return self._hook

    @staticmethod
    def _hook(value):
        if not isinstance(value, bool):
            raise ValueError(f"expected bool, not {value}")
        return value

    def to_polars_dtype(self) -> pl.Boolean:
        return pl.Boolean()

    def to_numpy_dtype(self) -> np.dtype:
        return np.dtype(np.bool)


@runtime_checkable
class ArrayInnerType(Protocol):
    def to_numpy_dtype(self) -> np.dtype: ...

    def to_polars_dtype(self) -> pl.DataType: ...


@attrs.frozen
class Float32(ArrayInnerType):
    def to_numpy_dtype(self) -> np.dtype:
        return np.dtype(np.float32)

    def to_polars_dtype(self) -> pl.Float32:
        return pl.Float32()


@attrs.frozen
class Float64(ArrayInnerType):
    def to_numpy_dtype(self) -> np.dtype:
        return np.dtype(np.float64)

    def to_polars_dtype(self) -> pl.Float64:
        return pl.Float64()


@attrs.frozen
class Int8(ArrayInnerType):
    def to_numpy_dtype(self) -> np.dtype:
        return np.dtype(np.int8)

    def to_polars_dtype(self) -> pl.Int8:
        return pl.Int8()


@attrs.frozen
class Int16(ArrayInnerType):
    def to_numpy_dtype(self) -> np.dtype:
        return np.dtype(np.int16)

    def to_polars_dtype(self) -> pl.Int16:
        return pl.Int16()


@attrs.frozen
class Int32(ArrayInnerType):
    def to_numpy_dtype(self) -> np.dtype:
        return np.dtype(np.int32)

    def to_polars_dtype(self) -> pl.Int32:
        return pl.Int32()


@attrs.frozen
class Int64(ArrayInnerType):
    def to_numpy_dtype(self) -> np.dtype:
        return np.dtype(np.int64)

    def to_polars_dtype(self) -> pl.Int64:
        return pl.Int64()


@attrs.frozen
class UInt8(ArrayInnerType):
    def to_numpy_dtype(self) -> np.dtype:
        return np.dtype(np.uint8)

    def to_polars_dtype(self) -> pl.UInt8:
        return pl.UInt8()


@attrs.frozen
class UInt16(ArrayInnerType):
    def to_numpy_dtype(self) -> np.dtype:
        return np.dtype(np.uint16)

    def to_polars_dtype(self) -> pl.UInt16:
        return pl.UInt16()


@attrs.frozen
class UInt32(ArrayInnerType):
    def to_numpy_dtype(self) -> np.dtype:
        return np.dtype(np.uint32)

    def to_polars_dtype(self) -> pl.UInt32:
        return pl.UInt32()


@attrs.frozen
class UInt64(ArrayInnerType):
    def to_numpy_dtype(self) -> np.dtype:
        return np.dtype(np.uint64)

    def to_polars_dtype(self) -> pl.UInt64:
        return pl.UInt64()


@attrs.frozen
class ArrayDataType(DataType):
    """Fixed shape array type.

    Attributes:
        inner: The type of the array elements.
        shape: The shape of the array.
            It must contain at least one element.
            Each element of the tuple is the size of the corresponding dimension.
            Each element must be a strictly positive integer.
    """

    inner: ArrayInnerType = attrs.field()
    shape: Sequence[int] = attrs.field()

    @shape.validator  # type: ignore
    def _shape_validator(self, attribute, value):
        if len(value) == 0:
            raise ValueError(f"shape must have at least one element, not {value}")
        if not all(isinstance(i, int) for i in value):
            raise ValueError(f"shape must be a tuple of integers, not {value}")
        if not all(i > 0 for i in value):
            raise ValueError(f"shape must be a tuple of positive integers, not {value}")

    @inner.validator  # type: ignore
    def _inner_validator(self, attribute, value):
        if not isinstance(value, ArrayInnerType):
            raise ValueError(f"inner must be an ArrayInnerType, not {value}")

    @cached_property
    def unstructure_hook(self) -> Callable[[Any], msgpack.ExtType]:  # type: ignore[reportIncompatibleMethodOverride]
        numpy_dtype = self.inner.to_numpy_dtype()
        shape = tuple(self.shape)

        def hook(value):
            if value.shape != shape:
                raise ValueError(f"expected shape {shape}, not {value.shape}")
            data = value.astype(numpy_dtype).tobytes()
            return msgpack.ExtType(ARRAY_TYPE, data)

        return hook

    @cached_property
    def structure_hook(self) -> Callable[[msgpack.ExtType], np.ndarray]:  # type: ignore[reportIncompatibleMethodOverride]
        numpy_dtype = self.inner.to_numpy_dtype()
        shape = tuple(self.shape)

        def hook(ext: msgpack.ExtType):
            if ext.code != ARRAY_TYPE:
                raise ValueError(f"expected code {ARRAY_TYPE}, not {ext.code}")
            data = np.frombuffer(ext.data, dtype=numpy_dtype)
            return data.reshape(shape)

        return hook

    def to_polars_dtype(self) -> pl.DataType:
        return pl.Array(inner=self.inner.to_polars_dtype(), shape=tuple(self.shape))


@attrs.frozen
class List(DataType):
    """Variable length list type."""

    inner: DataType

    def dumps(self, value) -> bytes:
        return msgpack.dumps(self.unstructure_hook(value))  # type: ignore[reportReturnType]

    def loads(self, data: bytes) -> list:
        return self.structure_hook(msgpack.loads(data))  # type: ignore[reportArgumentType]

    @cached_property
    def unstructure_hook(self) -> Callable[[Any], tuple]:  # type: ignore[reportIncompatibleMethodOverride]
        inner_hook = self.inner.unstructure_hook

        def hook(value):
            return tuple(inner_hook(x) for x in value)

        return hook

    @cached_property
    def structure_hook(self) -> Callable[[tuple], list]:  # type: ignore[reportIncompatibleMethodOverride]
        inner_hook = self.inner.structure_hook

        def hook(value):
            return [inner_hook(x) for x in value]

        return hook

    def to_polars_dtype(self) -> pl.DataType:
        return pl.List(self.inner.to_polars_dtype())


T = TypeVar("T", covariant=True)


class ConvertibleToPolarsDType(Protocol):
    def to_polars_dtype(self) -> pl.DataType: ...


@attrs.define(init=False)
class Struct(Generic[T]):
    """Composite data type.

    Args:
        fields: The name and type of each field.
            It must contain at least one element.

    """

    fields: dict[str, T] = attrs.field()

    def __init__(self, **fields: T):
        sorted_names = sorted(fields.keys())
        self.fields = {name: fields[name] for name in sorted_names}

    def dumps(self, value) -> bytes:
        return msgpack.dumps(self.unstructure_hook(value))  # type: ignore[reportReturnType]

    def loads(self, data: bytes) -> dict:
        return self.structure_hook(msgpack.loads(data))  # type: ignore[reportArgumentType]

    @fields.validator  # type: ignore
    def _fields_validator(self, attribute, value):
        if len(value) == 0:
            raise ValueError(f"fields must have at least one element, not {value}")

    @cached_property
    def unstructure_hook(self: Struct[DataType]) -> Callable[[Any], tuple]:
        field_hooks = {
            name: dtype.unstructure_hook for name, dtype in self.fields.items()
        }

        def hook(value):
            return tuple(hook(value[name]) for name, hook in field_hooks.items())

        return hook

    @cached_property
    def structure_hook(self: Struct[DataType]) -> Callable[[dict], dict]:
        field_hooks = {
            name: dtype.structure_hook for name, dtype in self.fields.items()
        }

        def hook(values):
            return {
                name: hook(value)
                for (name, hook), value in zip(field_hooks.items(), values, strict=True)
            }

        return hook

    def to_polars_dtype(self: Struct[ConvertibleToPolarsDType]) -> pl.Struct:
        return pl.Struct(
            fields={
                name: dtype.to_polars_dtype() for name, dtype in self.fields.items()
            }
        )

    def to_numpy_dtype(self: Struct[ArrayInnerType]) -> np.dtype:
        return np.dtype(
            [(name, dtype.to_numpy_dtype()) for name, dtype in self.fields.items()]
        )


type DataSchema = Mapping[str, DataType]
"""Contains the name and type of each data field."""
