from collections.abc import Mapping
from typing import Optional, Self, Generic, TypeVar

from benedict import benedict  # type: ignore

from variable.name import DottedVariableName, VariableName

T = TypeVar("T")


class VariableNamespace(Generic[T]):
    def __init__(self, initial_variables: Optional[Self] = None):
        self._dict: benedict = benedict()
        if initial_variables is not None:
            # noinspection PyProtectedMember
            self._dict = initial_variables._dict.clone()

    def update(self, values: dict[DottedVariableName, T]):
        for key, value in values.items():
            self._dict[str(key)] = value

    def to_dict(self) -> dict[VariableName, T]:
        return {VariableName(name): value for name, value in self._dict.items()}

    def __or__(
        self, other: Mapping[DottedVariableName, T]
    ) -> Mapping[DottedVariableName, T]:
        if isinstance(other, Mapping):
            new = self._dict.clone()
            for key, value in other.items():
                new[key] = value
            return new
        else:
            raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}({self._dict})"
