from collections.abc import Set
from typing import Optional, assert_never

from PySide6.QtCore import (
    QModelIndex,
    Qt,
    QAbstractItemModel,
    QObject,
    QEvent,
    QStringListModel,
)
from PySide6.QtGui import (
    QTextDocument,
    QPalette,
    QFocusEvent,
)
from PySide6.QtWidgets import (
    QWidget,
    QStyleOptionViewItem,
    QHBoxLayout,
    QLabel,
    QApplication,
    QSpinBox,
    QCompleter,
)

from caqtus.types.expression import Expression
from caqtus.types.variable_name import (
    DottedVariableName,
    InvalidVariableNameError,
)
from .steps_model import (
    StepData,
    VariableDeclarationData,
    LinspaceLoopData,
    ArrangeLoopData,
)
from ...qt_util import AutoResizeLineEdit, HTMLItemDelegate

NAME_COLOR = "#AA4926"
VALUE_COLOR = "#6897BB"
HIGHLIGHT_COLOR = "#cc7832"


class StepDelegate(HTMLItemDelegate):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.doc = QTextDocument(self)
        self._completer = QCompleter(self)

    def createEditor(
        self, parent: QWidget, option: QStyleOptionViewItem, index: QModelIndex
    ) -> QWidget:
        value = index.data(role=Qt.ItemDataRole.EditRole)
        assert isinstance(value, StepData)
        if isinstance(value, VariableDeclarationData):
            editor = VariableDeclarationEditor(parent, option.font)
        elif isinstance(value, LinspaceLoopData):
            editor = LinspaceLoopEditor(parent, option.font)
        elif isinstance(value, ArrangeLoopData):
            editor = ArrangeLoopEditor(parent, option.font)
        else:
            assert_never(value)
        editor.set_name_completer(self._completer)
        return editor

    def setEditorData(self, editor: QWidget, index: QModelIndex):
        data = index.data(role=Qt.ItemDataRole.EditRole)
        assert isinstance(data, StepData)
        match data:
            case VariableDeclarationData() as declaration:
                assert isinstance(editor, VariableDeclarationEditor)
                editor.set_step_data(declaration)
            case LinspaceLoopData() as loop:
                assert isinstance(editor, LinspaceLoopEditor)
                editor.set_step_data(loop)
            case ArrangeLoopData():
                assert isinstance(editor, ArrangeLoopEditor)
                editor.set_step_data(data)
            case _:
                raise ValueError(f"Can't set editor data for {data}")

    def updateEditorGeometry(
        self, editor: QWidget, option: QStyleOptionViewItem, index: QModelIndex
    ):
        geometry = option.rect
        editor.setGeometry(geometry)

    def setModelData(
        self, editor: QWidget, model: QAbstractItemModel, index: QModelIndex
    ) -> None:
        previous_data = index.data(role=Qt.ItemDataRole.EditRole)
        assert isinstance(previous_data, StepData)
        match previous_data:
            case VariableDeclarationData():
                assert isinstance(editor, VariableDeclarationEditor)
                try:
                    new_data = editor.get_step_data()
                except InvalidVariableNameError:
                    return
                else:
                    model.setData(index, new_data, Qt.ItemDataRole.EditRole)
            case LinspaceLoopData():
                assert isinstance(editor, LinspaceLoopEditor)
                try:
                    new_data = editor.get_step_data()
                except InvalidVariableNameError:
                    return
                else:
                    model.setData(index, new_data, Qt.ItemDataRole.EditRole)
            case ArrangeLoopData():
                assert isinstance(editor, ArrangeLoopEditor)
                try:
                    new_data = editor.get_step_data()
                except InvalidVariableNameError:
                    return
                else:
                    model.setData(index, new_data, Qt.ItemDataRole.EditRole)
            case _:
                assert_never(previous_data)

    def set_available_names(self, names: Set[DottedVariableName]) -> None:
        available_names = {str(name) for name in names}
        self._completer.setModel(QStringListModel(available_names))


class CompoundWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)
        self._widgets = []

    def add_widget(self, widget: QWidget):
        self.layout().addWidget(widget)
        widget.installEventFilter(self)
        self._widgets.append(widget)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        watched_is_child = watched in self._widgets
        if watched_is_child and event.type() == QEvent.Type.FocusOut:
            if QApplication.focusWidget() not in self.findChildren(QWidget):
                QApplication.postEvent(self, QFocusEvent(event.type()))
            return False

        return super().eventFilter(watched, event)


class LinspaceLoopEditor(CompoundWidget):
    def __init__(self, parent, font):
        super().__init__(parent)
        self.setFont(font)
        for_label = QLabel("for ", self)
        for_label.setAttribute(Qt.WA_TranslucentBackground, True)
        for_label.setStyleSheet(f"color: {HIGHLIGHT_COLOR}")
        self.add_widget(for_label)
        self.name_editor = AutoResizeLineEdit(self)
        self.name_editor.setStyleSheet(f"color: {NAME_COLOR}")
        self.add_widget(self.name_editor)
        equal_label = QLabel(" = ", self)
        equal_label.setAttribute(Qt.WA_TranslucentBackground, True)
        self.add_widget(equal_label)
        self.start_editor = AutoResizeLineEdit(self)
        self.start_editor.setStyleSheet(f"color: {VALUE_COLOR}")
        self.add_widget(self.start_editor)
        to_label = QLabel(" to ", self)
        to_label.setAttribute(Qt.WA_TranslucentBackground, True)
        to_label.setStyleSheet(f"color: {HIGHLIGHT_COLOR}")
        self.add_widget(to_label)
        self.stop_editor = AutoResizeLineEdit(self)
        self.stop_editor.setStyleSheet(f"color: {VALUE_COLOR}")
        self.add_widget(self.stop_editor)
        with_label = QLabel(" with ", self)
        with_label.setAttribute(Qt.WA_TranslucentBackground, True)
        with_label.setStyleSheet(f"color: {HIGHLIGHT_COLOR}")
        self.add_widget(with_label)
        self.num_editor = QSpinBox(self)
        self.num_editor.setStyleSheet(f"color: {VALUE_COLOR}")
        self.num_editor.setRange(0, 9999)
        self.add_widget(self.num_editor)
        steps_label = QLabel(" steps:", self)
        steps_label.setAttribute(Qt.WA_TranslucentBackground, True)
        steps_label.setStyleSheet(f"color: {HIGHLIGHT_COLOR}")
        self.add_widget(steps_label)
        self.layout().addStretch(1)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, Qt.GlobalColor.black)
        self.setAutoFillBackground(True)
        self.setPalette(palette)

    def set_step_data(self, loop: LinspaceLoopData) -> None:
        self.name_editor.setText(str(loop.variable))
        self.start_editor.setText(str(loop.start))
        self.stop_editor.setText(str(loop.stop))
        self.num_editor.setValue(loop.num)

    def get_step_data(self) -> LinspaceLoopData:
        return LinspaceLoopData(
            variable=DottedVariableName(self.name_editor.text()),
            start=Expression(self.start_editor.text()),
            stop=Expression(self.stop_editor.text()),
            num=self.num_editor.value(),
        )

    def set_name_completer(self, completer: QCompleter) -> None:
        self.name_editor.setCompleter(completer)


class VariableDeclarationEditor(CompoundWidget):
    def __init__(self, parent, font):
        super().__init__(parent)
        self.setFont(font)
        self.name_editor = AutoResizeLineEdit(self)
        self.add_widget(self.name_editor)
        label = QLabel(" = ", self)
        label.setAttribute(Qt.WA_TranslucentBackground, True)
        self.add_widget(label)
        self.value_editor = AutoResizeLineEdit(self)
        self.add_widget(self.value_editor)
        self.layout().addStretch(1)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, Qt.GlobalColor.black)
        self.setAutoFillBackground(True)
        self.setPalette(palette)
        self.name_editor.setStyleSheet(f"color: {NAME_COLOR}")
        self.value_editor.setStyleSheet(f"color: {VALUE_COLOR}")
        self.name_editor.setPlaceholderText("Parameter name")
        self.value_editor.setPlaceholderText("Parameter value")

    def set_step_data(self, declaration: VariableDeclarationData) -> None:
        self.name_editor.setText(str(declaration.variable))
        self.value_editor.setText(str(declaration.value))

    def get_step_data(self) -> VariableDeclarationData:
        return VariableDeclarationData(
            variable=DottedVariableName(self.name_editor.text()),
            value=Expression(self.value_editor.text()),
        )

    def set_name_completer(self, completer: QCompleter) -> None:
        self.name_editor.setCompleter(completer)


class ArrangeLoopEditor(CompoundWidget):
    def __init__(self, parent, font):
        super().__init__(parent)
        self.setFont(font)
        for_label = QLabel("for ", self)
        for_label.setAttribute(Qt.WA_TranslucentBackground, True)
        for_label.setStyleSheet(f"color: {HIGHLIGHT_COLOR}")
        self.add_widget(for_label)
        self.name_editor = AutoResizeLineEdit(self)
        self.name_editor.setStyleSheet(f"color: {NAME_COLOR}")
        self.add_widget(self.name_editor)
        equal_label = QLabel(" = ", self)
        equal_label.setAttribute(Qt.WA_TranslucentBackground, True)
        self.add_widget(equal_label)
        self.start_editor = AutoResizeLineEdit(self)
        self.start_editor.setStyleSheet(f"color: {VALUE_COLOR}")
        self.add_widget(self.start_editor)
        to_label = QLabel(" to ", self)
        to_label.setAttribute(Qt.WA_TranslucentBackground, True)
        to_label.setStyleSheet(f"color: {HIGHLIGHT_COLOR}")
        self.add_widget(to_label)
        self.stop_editor = AutoResizeLineEdit(self)
        self.stop_editor.setStyleSheet(f"color: {VALUE_COLOR}")
        self.add_widget(self.stop_editor)
        with_label = QLabel(" with ", self)
        with_label.setAttribute(Qt.WA_TranslucentBackground, True)
        with_label.setStyleSheet(f"color: {HIGHLIGHT_COLOR}")
        self.add_widget(with_label)
        self.step_editor = AutoResizeLineEdit(self)
        self.step_editor.setStyleSheet(f"color: {VALUE_COLOR}")
        self.add_widget(self.step_editor)
        spacing_label = QLabel(" spacing:", self)
        spacing_label.setAttribute(Qt.WA_TranslucentBackground, True)
        spacing_label.setStyleSheet(f"color: {HIGHLIGHT_COLOR}")
        self.add_widget(spacing_label)
        self.layout().addStretch(1)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, Qt.GlobalColor.black)
        self.setAutoFillBackground(True)
        self.setPalette(palette)

    def set_step_data(self, loop: ArrangeLoopData) -> None:
        self.name_editor.setText(str(loop.variable))
        self.start_editor.setText(str(loop.start))
        self.stop_editor.setText(str(loop.stop))
        self.step_editor.setText(str(loop.step))

    def get_step_data(self) -> ArrangeLoopData:
        return ArrangeLoopData(
            variable=DottedVariableName(self.name_editor.text()),
            start=Expression(self.start_editor.text()),
            stop=Expression(self.stop_editor.text()),
            step=Expression(self.step_editor.text()),
        )

    def set_name_completer(self, completer: QCompleter) -> None:
        self.name_editor.setCompleter(completer)
