from pathlib import Path

import pint

from variable.name import VariableName

ureg = pint.UnitRegistry(
    Path(__file__).parent / "units_definition.txt",
    autoconvert_offset_to_baseunit=True,
    cache_folder=":auto:",
)
unit_registry = ureg
pint.set_application_registry(unit_registry)
Quantity = pint.quantity.Quantity
Unit = pint.unit.Unit
UndefinedUnitError = pint.UndefinedUnitError

DimensionalityError = pint.DimensionalityError
dimensionless = Quantity(1, "").units

TIME_UNITS = {"s", "ms", "µs", "us", "ns"}

FREQUENCY_UNITS = {
    "Hz",
    "kHz",
    "MHz",
    "GHz",
    "THz",
}

POWER_UNITS = {
    "W",
    "mW",
    "dBm",
}

DIMENSIONLESS_UNITS = {"dB"}

CURRENT_UNITS = {"A", "mA"}

VOLTAGE_UNITS = {"V", "mV"}

DISTANCE_UNITS = {"m", "mm", "µm", "um", "nm"}

UNITS = (
    TIME_UNITS
    | FREQUENCY_UNITS
    | POWER_UNITS
    | DIMENSIONLESS_UNITS
    | CURRENT_UNITS
    | VOLTAGE_UNITS
    | DISTANCE_UNITS
)


units: dict[VariableName, Unit] = {
    VariableName(unit): getattr(ureg, unit) for unit in UNITS
}
