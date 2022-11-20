import logging
from abc import abstractmethod
from functools import singledispatch

from PyQt5.QtCore import QModelIndex, Qt, QAbstractItemModel, QSize
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtWidgets import QWidget, QStyledItemDelegate, QStyleOptionViewItem, QStyle

from expression import Expression
from sequence import Step, VariableDeclaration, ExecuteShot, LinspaceLoop
from sequence.sequence_config import ArangeLoop
from .step_uis import (
    Ui_ArangeDeclaration,
    Ui_VariableDeclaration,
    Ui_LinspaceDeclaration,
    Ui_ExecuteShot,
)
from .steps_model import QABCMeta

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


class StepWidget(QWidget, metaclass=QABCMeta):
    """Abstract class for a widget used to display/edit a sequence step"""

    @abstractmethod
    def set_step_data(self, data: Step):
        raise NotImplementedError()

    @abstractmethod
    def get_step_data(self) -> dict[str]:
        raise NotImplementedError()


class StepDelegate(QStyledItemDelegate):
    """Delegate for a sequence step (see PyQt Model/View/Delegate)

    This delegate creates a widget editor to edit the data of a step. It also paints the
    widget on the view when editing if done.
    """

    def createEditor(
        self, parent: QWidget, option: QStyleOptionViewItem, index: QModelIndex
    ) -> StepWidget:
        data: Step = index.data(role=Qt.ItemDataRole.EditRole)
        # noinspection PyTypeChecker
        editor = create_editor(data, None)
        editor.setParent(parent)
        editor.setAutoFillBackground(True)
        return editor

    def setEditorData(self, editor: StepWidget, index: QModelIndex):
        editor.set_step_data(index.data(role=Qt.ItemDataRole.EditRole))

    def setModelData(
        self, editor: StepWidget, model: QAbstractItemModel, index: QModelIndex
    ) -> None:
        model.setData(index, editor.get_step_data(), Qt.ItemDataRole.EditRole)

    def updateEditorGeometry(
        self, editor: StepWidget, option: QStyleOptionViewItem, index: QModelIndex
    ):
        geometry = option.rect
        editor.setGeometry(geometry)

    def sizeHint(self, option: "QStyleOptionViewItem", index: QModelIndex) -> QSize:
        step = index.data(role=Qt.ItemDataRole.DisplayRole)
        # noinspection PyTypeChecker
        w = create_editor(step, None)
        self.setEditorData(w, index)
        w.resize(option.rect.size())
        return w.sizeHint()

    def paint(
        self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex
    ) -> None:
        data = index.data(Qt.ItemDataRole.DisplayRole)
        if isinstance(data, Step):
            pixmap = QPixmap(option.rect.size())
            pixmap.fill(option.palette.base().color())
            w = create_editor(data, option.widget)
            w.set_step_data(data)
            self.updateEditorGeometry(w, option, index)
            if not (option.state & QStyle.StateFlag.State_Enabled):
                w.setEnabled(False)
            w.render(pixmap, flags=QWidget.RenderFlag.DrawChildren)
            painter.drawPixmap(option.rect, pixmap)
        else:
            super().paint(painter, option, index)


class VariableDeclarationWidget(Ui_VariableDeclaration, StepWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

    def set_step_data(self, declaration: VariableDeclaration):
        self.name_edit.setText(declaration.name)
        self.expression_edit.setText(declaration.expression.body)

    def get_step_data(self) -> dict[str]:
        return dict(
            name=self.name_edit.text(),
            expression=Expression(self.expression_edit.text()),
        )


class ExecuteShotWidget(Ui_ExecuteShot, StepWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

    def set_step_data(self, shot: ExecuteShot):
        self.name_edit.setText(shot.name)

    def get_step_data(self) -> dict[str]:
        return dict(
            name=self.name_edit.text(),
        )


class LinspaceIterationWidget(Ui_LinspaceDeclaration, StepWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

    def set_step_data(self, data: LinspaceLoop):
        self.name_edit.setText(data.name)
        self.start_edit.setText(data.start.body)
        self.stop_edit.setText(data.stop.body)
        self.num_edit.setValue(data.num)

    def get_step_data(self):
        return dict(
            name=self.name_edit.text(),
            start=Expression(self.start_edit.text()),
            stop=Expression(self.stop_edit.text()),
            num=self.num_edit.value(),
        )


class ArangeIterationWidget(Ui_ArangeDeclaration, StepWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

    def set_step_data(self, data: ArangeLoop):
        self.name_edit.setText(data.name)
        self.start_edit.setText(data.start.body)
        self.stop_edit.setText(data.stop.body)
        self.step_edit.setText(data.step.body)

    def get_step_data(self):
        return dict(
            name=self.name_edit.text(),
            start=Expression(self.start_edit.text()),
            stop=Expression(self.stop_edit.text()),
            step=Expression(self.step_edit.text()),
        )


@singledispatch
def create_editor(step: Step, _: QWidget) -> StepWidget:
    """Create an editor for a step depending on its type"""
    raise NotImplementedError(f"Not implemented for {type(step)}")


@create_editor.register
def _(_: VariableDeclaration, parent: QWidget):
    return VariableDeclarationWidget(parent)


@create_editor.register
def _(_: ExecuteShot, parent: QWidget):
    return ExecuteShotWidget(parent)


@create_editor.register
def _(_: LinspaceLoop, parent: QWidget):
    return LinspaceIterationWidget(parent)


@create_editor.register
def _(_: ArangeLoop, parent: QWidget):
    return ArangeIterationWidget(parent)
