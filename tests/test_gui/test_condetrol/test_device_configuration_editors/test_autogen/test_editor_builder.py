import enum
import typing

import attrs
import pytest
from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from caqtus.gui.autogen import (
    EditorBuilder,
    StringEditor,
    TypeNotRegisteredError,
    IntegerEditor,
    generate_enum_editor,
)


def test_dispatch_simple_type():
    builder = EditorBuilder()
    builder.register_editor(str, StringEditor)
    assert builder.build_editor(str) == StringEditor


def test_not_registered_type():
    builder = EditorBuilder()
    with pytest.raises(TypeNotRegisteredError):
        builder.build_editor(str)


def test_attrs_class(qtbot: QtBot):
    builder = EditorBuilder()
    builder.register_editor(str, StringEditor)
    builder.register_editor(int, IntegerEditor)

    @attrs.define
    class MyClass:
        number: int
        channel_0: str
        channel_1: str

    MyClassEditor = builder.build_editor(MyClass)  # noqa: N806

    initial_value = MyClass(42, "abc", "test")

    editor = MyClassEditor()
    editor.set_value(initial_value)
    widget = editor.widget
    qtbot.add_widget(widget)
    editor.set_editable(False)
    editor_channel_0 = getattr(editor, "editor_channel_0")  # noqa: B009
    assert isinstance(editor_channel_0, StringEditor)
    editor_channel_0.widget.setText("check")
    assert editor.read_value() == MyClass(42, "check", "test")


def test_nested_class(qtbot: QtBot):
    builder = EditorBuilder()
    builder.register_editor(str, StringEditor)
    builder.register_editor(int, IntegerEditor)

    @attrs.define
    class Child:
        age: int

    @attrs.define
    class Parent:
        name: str
        child: Child

    initial_value = Parent(name="Julia", child=Child(age=8))

    ParentEditor = builder.build_editor(Parent)  # noqa: N806
    editor = ParentEditor()
    editor.set_value(initial_value)
    widget = editor.widget
    qtbot.add_widget(widget)
    editor.set_editable(True)
    assert editor.read_value() == initial_value


def test_literal_editor(qtbot: QtBot):
    builder = EditorBuilder()

    editor_factory = builder.build_editor(typing.Literal["abc", 123])
    editor = editor_factory()

    widget = editor.widget
    qtbot.add_widget(widget)
    assert isinstance(widget, QtWidgets.QComboBox)
    widget.setCurrentIndex(1)
    assert editor.read_value() == 123


def test_enum_editor(qtbot: QtBot):
    class MyEnum(enum.Enum):
        A = 1
        B = 2
        C = 3

    editor_factory = generate_enum_editor(MyEnum)

    editor = editor_factory()
    widget = editor.widget

    qtbot.add_widget(widget)
    widget.setCurrentIndex(1)
    assert widget.currentText() == "MyEnum.B"
    value = editor.read_value()
    typing.assert_type(value, MyEnum)
    assert value == MyEnum.B


def test_enum_dispatch(qtbot: QtBot):
    class MyEnum(enum.StrEnum):
        A = "A"
        B = "B"
        C = "C"

    builder = EditorBuilder()
    factory = builder.build_editor(MyEnum)
    editor = factory()
    qtbot.add_widget(editor)
    editor.set_value(MyEnum.B)
    assert editor.read_value() == MyEnum.B
