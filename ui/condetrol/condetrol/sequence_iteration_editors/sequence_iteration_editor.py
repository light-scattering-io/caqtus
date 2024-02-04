import abc
from collections.abc import Callable
from typing import TypeVar, Generic, TypeAlias

import qabc
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget
from core.session.sequence.iteration_configuration import IterationConfiguration

T = TypeVar("T", bound=IterationConfiguration)


class SequenceIterationEditor(Generic[T], metaclass=qabc.QABCMeta):
    iteration_changed = Signal()

    @abc.abstractmethod
    def get_iteration(self) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    def set_iteration(self, iteration: T):
        raise NotImplementedError

    @abc.abstractmethod
    def set_read_only(self, read_only: bool):
        raise NotImplementedError


IterationEditorCreator: TypeAlias = Callable[[T], SequenceIterationEditor[T]]
