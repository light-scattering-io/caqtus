from __future__ import annotations

from abc import abstractmethod
from typing import Optional, TypeVar, Generic, Callable, NewType

import attrs
import qabc
from PySide6.QtWidgets import QWidget
from core.session.sequence import Shot
from util.serialization import JSON


class ShotView(QWidget, metaclass=qabc.QABCMeta):
    @abstractmethod
    def display_shot(self, shot: Shot) -> None:
        raise NotImplementedError


S = TypeVar("S", bound=JSON)
V = TypeVar("V", bound=ShotView)

ManagerName = NewType("ManagerName", str)


@attrs.define
class ViewManager(Generic[V, S]):
    constructor: Callable[[S], V]
    dumper: Callable[[V], S]
    state_generator: Callable[[QWidget], Optional[tuple[str, S]]]
