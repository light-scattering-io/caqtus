from abc import abstractmethod

from PyQt6.QtWidgets import QWidget
from attrs import define

from qabc import QABC
from sequence.runtime import Shot


@define(slots=False)
class SingleShotViewer(QWidget, QABC):
    def __attrs_pre_init__(self) -> None:
        super().__init__()

    @abstractmethod
    def set_shot(self, shot: Shot) -> None:
        raise NotImplementedError()

    @abstractmethod
    def update_view(self) -> None:
        raise NotImplementedError()
