from typing import TypeGuard, Any, TypeAlias

from caqtus.utils import serialization
from ._analog_value import AnalogValue, is_analog_value, Quantity

Parameter: TypeAlias = AnalogValue | int | bool


def unstructure_quantity(value: Quantity):
    return float(value.magnitude), f"{value.units:~}"


def structure_quantity(value: Any, _) -> Quantity:
    try:
        return Quantity(*value)  # pyright: ignore[reportReturnType]
    except TypeError:
        raise ValueError(f"Cannot structure {value!r} as a Quantity.") from None


serialization.register_unstructure_hook(Quantity, unstructure_quantity)

serialization.register_structure_hook(Quantity, structure_quantity)


def is_parameter(parameter: Any) -> TypeGuard[Parameter]:
    """Returns True if the value is a valid parameter type, False otherwise."""

    return is_analog_value(parameter) or isinstance(parameter, (int, bool))