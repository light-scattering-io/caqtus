import itertools
from collections.abc import Mapping
from typing import assert_never

import attrs

from caqtus.types.parameter import Parameter
from caqtus.types.units import Unit, Quantity, dimensionless
from caqtus.types.variable_name import DottedVariableName

type ConstantSchema = Mapping[DottedVariableName, Parameter]
type VariableSchema = Mapping[DottedVariableName, ParameterType]
type ParameterType = (Boolean | Integer | Float | QuantityType)


class ParameterSchema(Mapping[DottedVariableName | str, ParameterType]):
    """Contains the type of each parameter in a sequence.

    More explicitly, it contains the value of the parameters that are constant during
    the sequence and the types of the parameters that can change during the sequence.
    The constant and variable parameters have no overlap.

    This object behaves like an immutable dictionary with the keys being the parameter
    names and the values being the types of the parameters.
    """

    def __init__(
        self,
        *,
        _constant_values: ConstantSchema,
        _variable_types: VariableSchema,
    ) -> None:
        if set(_constant_values) & set(_variable_types):
            raise ValueError(
                "The constant and variable schemas must not have any parameters in "
                "common."
            )
        self._constant_values = _constant_values
        self._constant_types = {
            name: self.type_from_value(value)
            for name, value in _constant_values.items()
        }
        self._variable_types = _variable_types

    def __len__(self):
        return len(self._constant_types) + len(self._variable_types)

    def __iter__(self):
        return itertools.chain(self._constant_types, self._variable_types)

    def __contains__(self, item) -> bool:
        return item in self._constant_types or item in self._variable_types

    def __getitem__(self, key: DottedVariableName | str) -> ParameterType:
        if isinstance(key, str):
            key = DottedVariableName(key)
        if key in self._constant_types:
            return self._constant_types[key]
        elif key in self._variable_types:
            return self._variable_types[key]
        else:
            raise KeyError(key)

    @property
    def constant_values(self) -> ConstantSchema:
        """Values of the parameters that are constant during the sequence."""

        return self._constant_values

    @property
    def variable_types(self) -> VariableSchema:
        """Types of the parameters that can change during the sequence."""

        return self._variable_types

    def __repr__(self) -> str:
        return (
            f"ParameterSchema("
            f"_constant_schema={self._constant_values}, "
            f"_variable_schema={self._variable_types})"
        )

    def __str__(self) -> str:
        constants = (
            f'"{key}": {value}' for key, value in self._constant_values.items()
        )
        variables = (f'"{key}": {value}' for key, value in self._variable_types.items())
        joined = itertools.chain(constants, variables)
        return "{" + ", ".join(joined) + "}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParameterSchema):
            return NotImplemented
        return (
            self._constant_values == other._constant_values
            and self._variable_types == other._variable_types
        )

    @classmethod
    def type_from_value(cls, value: Parameter) -> ParameterType:
        if isinstance(value, bool):
            return Boolean()
        elif isinstance(value, int):
            return Integer()
        elif isinstance(value, float):
            return Float()
        elif isinstance(value, Quantity):
            return QuantityType(units=value.units)
        else:
            assert_never(value)

    def enforce(
        self, values: Mapping[DottedVariableName, Parameter]
    ) -> Mapping[DottedVariableName, Parameter]:
        """Enforce the schema on a set of values.

        This method checks that the values are compatible with the schema and returns a
        dictionary with the same keys and the values coerced to the correct types.

        Raises:
            KeyError: If a parameter is missing from the values.
            ValueError: If the values are not compatible with the schema, or if extra
                parameters are present in the values.
        """

        values = dict(values)

        for constant_name, constant_value in self._constant_values.items():
            if values.pop(constant_name) != constant_value:
                raise ValueError(
                    f"Constant parameter {constant_name} must be {constant_value}."
                )

        result = dict(self._constant_values)
        for variable_name, variable_type in self._variable_types.items():
            value = values.pop(variable_name)
            try:
                converted = variable_type.convert(value)
            except ValueError as e:
                raise ValueError(
                    f"Parameter {variable_name} cannot be converted to the expected "
                    f"type: {variable_type}"
                ) from e
            result[variable_name] = converted

        if values:
            raise ValueError(f"Unexpected parameters: {values.keys()}")

        return result


@attrs.frozen
class QuantityType:
    units: Unit

    def convert(self, value: Parameter) -> Quantity[float]:
        """Convert a value to a quantity with the correct units.

        Raises:
            ValueError: If the value is not compatible with the units.
        """

        match value:
            case int() | float() | bool() as v:
                if self.units.is_compatible_with(dimensionless):
                    q = Quantity(v, dimensionless)
                    return q.to_unit(self.units)
                else:
                    raise ValueError(f"Can't coerce value {value} to quantity.")
            case Quantity() as q:
                if q.unit.is_compatible_with(self.units):
                    return q.to_unit(self.units)
                else:
                    raise ValueError(f"Can't coerce value {value} to quantity.")
            case _:
                assert_never(value)

    def __str__(self):
        return f"quantity<{self.units}>"


@attrs.frozen
class Float:
    @property
    def units(self) -> None:
        return None

    @staticmethod
    def convert(value: Parameter) -> float:
        """Convert a value to a float.

        Raises:
            ValueError: If the value can't be converted to a float without loss of
                information.
        """

        match value:
            case bool(v):
                return float(v)
            case int(v):
                return float(v)
            case float(v):
                return v
            case Quantity() as q:
                if q.unit.is_compatible_with(dimensionless):
                    return q.to(dimensionless).magnitude
                else:
                    raise ValueError(f"Can't coerce quantity {value} to float.")
            case _:
                assert_never(value)

    def __str__(self):
        return "float"


@attrs.frozen
class Boolean:
    @property
    def units(self) -> None:
        return None

    @staticmethod
    def convert(value: Parameter) -> bool:
        match value:
            case bool(v):
                return v
            case int(v):
                if v == 0:
                    return False
                elif v == 1:
                    return True
                else:
                    raise ValueError(f"Can't coerce integer {value} to boolean.")
            case float():
                raise ValueError(f"Can't coerce float {value} to boolean.")
            case Quantity():
                raise ValueError(f"Can't coerce quantity {value} to boolean.")
            case _:
                assert_never(value)

    def __str__(self):
        return "bool"


@attrs.frozen
class Integer:
    @property
    def units(self) -> None:
        return None

    @staticmethod
    def convert(value: Parameter) -> int:
        match value:
            case bool(v):
                return int(v)
            case int(v):
                return v
            case float():
                raise ValueError(f"Can't coerce float {value} to integer.")
            case Quantity():
                raise ValueError(f"Can't coerce quantity {value} to integer.")
            case _:
                assert_never(value)

    def __str__(self):
        return "int"
