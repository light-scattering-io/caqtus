from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtWidgets import (
    QStyledItemDelegate,
    QWidget,
    QStyleOptionViewItem,
    QPushButton,
    QLineEdit,
)

from caqtus.types.expression import Expression


class DigitalTimeLaneDelegate(QStyledItemDelegate):
    def createEditor(
        self, parent: QWidget, option: QStyleOptionViewItem, index: QModelIndex
    ) -> QWidget:
        cell_value = index.data(Qt.ItemDataRole.EditRole)
        if isinstance(cell_value, bool):
            return CheckedButton(parent)
        elif isinstance(cell_value, Expression):
            return QLineEdit(parent)
        else:
            raise TypeError(f"Invalid type for value: {type(cell_value)}")

    def setEditorData(self, editor: QWidget, index: QModelIndex):
        cell_value = index.data(Qt.ItemDataRole.EditRole)
        if isinstance(cell_value, bool):
            editor: CheckedButton
            editor.setChecked(cell_value)
        elif isinstance(cell_value, Expression):
            editor: QLineEdit
            editor.setText(str(cell_value))
        else:
            raise TypeError(f"Invalid type for value: {type(cell_value)}")

    def setModelData(self, editor, model, index):
        cell_value = index.data(Qt.ItemDataRole.EditRole)
        if isinstance(cell_value, bool):
            editor: CheckedButton
            model.setData(index, editor.isChecked(), Qt.ItemDataRole.EditRole)
        elif isinstance(cell_value, Expression):
            editor: QLineEdit
            model.setData(index, Expression(editor.text()), Qt.ItemDataRole.EditRole)
        else:
            raise TypeError(f"Invalid type for value: {type(cell_value)}")


class CheckedButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setCheckable(True)
        self.toggled.connect(self.on_toggled)

    def setChecked(self, a0: bool) -> None:
        super().setChecked(a0)
        self.on_toggled(a0)

    def on_toggled(self, checked: bool):
        if checked:
            self.setText("Enabled")
        else:
            self.setText("Disabled")
