from __future__ import annotations

import abc
import asyncio
from collections.abc import Set
from typing import Optional, assert_never, Literal

import attrs
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QWidget, QToolBar, QStackedWidget, QLabel, QHBoxLayout

from caqtus.session import (
    ExperimentSessionMaker,
    PureSequencePath,
    ExperimentSession,
    AsyncExperimentSession,
)
from caqtus.session._return_or_raise import unwrap
from caqtus.session.path_hierarchy import PathNotFoundError
from caqtus.session.sequence import State
from caqtus.session.sequence_collection import (
    SequenceNotEditableError,
    PathIsNotSequenceError,
)
from caqtus.session.shot import TimeLanes
from caqtus.types.iteration import (
    IterationConfiguration,
    StepsConfiguration,
)
from caqtus.types.parameter import ParameterNamespace
from caqtus.types.variable_name import DottedVariableName
from .sequence_widget_ui import Ui_SequenceWidget
from ..icons import get_icon
from ..parameter_tables_editor import ParameterNamespaceEditor
from ..sequence_iteration_editors import StepsIterationEditor
from ..timelanes_editor import TimeLanesEditor
from ..timelanes_editor.extension import CondetrolLaneExtensionProtocol


class SequenceWidget(QWidget, Ui_SequenceWidget):
    """Widget for editing sequence parameters, iterations and time lanes.

    This widget is a tab widget with three tabs: one for defining initial parameters, one for editing how the
    parameters should be iterated over for the sequence, and one for editing the time lanes that specify how
    a given shot should be executed.

    This widget is (optionally) associated with a sequence and displays the iteration
    configuration and time lanes for that sequence.
    If the widget is not associated with a sequence, it will hide itself.

    When associated with a sequence, the widget is constantly watching the state of the
    sequence.
    If the sequence is not in the draft state, the iteration editor and time lanes editor
    will become read-only.
    If the sequence is in the draft state, the iteration editor and time lanes editor
    will become editable and any change will be saved.
    """

    sequence_changed = Signal(object)  # Optional[tuple[PureSequencePath, State]]
    sequence_start_requested = Signal(PureSequencePath)

    def __init__(
        self,
        session_maker: ExperimentSessionMaker,
        extension: CondetrolLaneExtensionProtocol,
        parent: Optional[QWidget] = None,
    ):
        """Initializes the sequence widget.

        The sequence widget will initially be associated with no sequence.

        Args:
            session_maker: It is used to connect to the storage system in which to look
            for the sequence.
            parent: The parent widget.
        """

        super().__init__(parent)
        self.setupUi(self)
        self.session_maker = session_maker

        with self.session_maker() as session:
            device_configurations = dict(session.default_device_configurations)

        self._state: _State = _SequenceNotSetState()
        self._state_lock = asyncio.Lock()

        self.parameters_editor = ParameterNamespaceEditor(self)
        self.parameters_editor.set_read_only(True)
        self.time_lanes_editor = TimeLanesEditor(
            extension,
            device_configurations,
            self,
        )
        self.iteration_editor = StepsIterationEditor(self)

        self.tabWidget.clear()
        self.tabWidget.addTab(self.parameters_editor, "&Globals")
        self.tabWidget.addTab(self.iteration_editor, "&Parameters")
        self.tabWidget.addTab(self.time_lanes_editor, "Time &lanes")

        self.setup_connections()

        self.tool_bar = QToolBar(self)
        self.status_widget = IconLabel(icon_position="left")
        self.tool_bar.addWidget(self.status_widget)
        self.start_sequence_action = self.tool_bar.addAction(
            get_icon("start", color=Qt.GlobalColor.darkGreen), "start"
        )
        self.start_sequence_action.triggered.connect(self._on_start_sequence_requested)
        self.interrupt_sequence_action = self.tool_bar.addAction(
            get_icon("stop"), "interrupt"
        )
        self.interrupt_sequence_action.setVisible(False)
        self.verticalLayout.insertWidget(0, self.tool_bar)
        self.stacked = QStackedWidget(self)
        self.stacked.addWidget(QWidget(self))
        self.stacked.addWidget(QWidget(self))
        self.stacked.addWidget(self.time_lanes_editor.toolbar)
        self.stacked.setCurrentIndex(0)
        self.tool_bar.addSeparator()
        self.tool_bar.addWidget(self.stacked)

        self.tabWidget.currentChanged.connect(self.stacked.setCurrentIndex)

        self._transition(_SequenceNotSetState())

    def set_available_parameter_names(
        self, parameter_names: Set[DottedVariableName]
    ) -> None:
        """Set the names of the parameters that are defined externally."""

        self.iteration_editor.set_available_parameter_names(parameter_names)

    async def exec_async(self) -> None:
        while True:
            await self.watch_sequence()
            await asyncio.sleep(10e-3)

    async def watch_sequence(self) -> None:
        if isinstance(self._state, _SequenceNotSetState):
            return
        elif isinstance(self._state, _SequenceSetState):
            old_state = self._state
            new_state = await _query_state_async(
                self._state.sequence_path, self.session_maker
            )
            # It could be that the widget state changed while we were fetching infos,
            # for example if set_sequence is called or the editors emitted editions.
            if old_state != self._state:
                return

            if new_state != self._state:
                self._transition(new_state)
        else:
            assert_never(self._state)

    def _transition(self, new_state: _State) -> None:
        match new_state:
            case _SequenceNotSetState():
                self.setVisible(False)
                self.start_sequence_action.setEnabled(False)
                self.interrupt_sequence_action.setEnabled(False)
            case _SequenceSetState(
                iterations=iterations,
                time_lanes=time_lanes,
                sequence_state=state,
                sequence_path=path,
            ):
                if not isinstance(iterations, StepsConfiguration):
                    raise NotImplementedError(f"Only supports {StepsConfiguration}")
                self.iteration_editor.set_iteration(iterations)
                self.time_lanes_editor.set_time_lanes(time_lanes)
                self._set_status_widget(path, state)
                if isinstance(new_state, _SequenceEditableState):
                    self.start_sequence_action.setEnabled(True)
                    self.interrupt_sequence_action.setEnabled(False)
                    self.time_lanes_editor.set_read_only(False)
                    self.iteration_editor.set_read_only(False)
                    self.tabWidget.setTabEnabled(0, False)
                elif isinstance(new_state, _SequenceNotEditableState):
                    self.start_sequence_action.setEnabled(False)
                    self.interrupt_sequence_action.setEnabled(False)
                    self.time_lanes_editor.set_read_only(True)
                    self.iteration_editor.set_read_only(True)
                    self.parameters_editor.set_parameters(new_state.parameters)
                    self.tabWidget.setTabEnabled(0, True)
                self.setVisible(True)
        self._state = new_state

    def _set_status_widget(self, path: PureSequencePath, state: State) -> None:
        text = " > ".join(path.parts)
        color = self.palette().text().color()
        if state.is_editable():
            icon = get_icon("editable-sequence", color=color)
        else:
            icon = get_icon("read-only-sequence", color=color)
        self.status_widget.set_text(text)
        self.status_widget.set_icon(icon)

    def _on_start_sequence_requested(self):
        assert isinstance(self._state, _SequenceEditableState)
        self.sequence_start_requested.emit(self._state.sequence_path)

    def set_sequence(self, sequence_path: Optional[PureSequencePath]) -> None:
        if sequence_path is None:
            new_state = _SequenceNotSetState()
        else:
            new_state = _query_state_sync(sequence_path, self.session_maker)
        self._transition(new_state)

    def setup_connections(self):
        self.time_lanes_editor.time_lanes_edited.connect(self.on_time_lanes_edited)
        self.iteration_editor.iteration_edited.connect(
            self.on_sequence_iteration_edited
        )

    def on_sequence_iteration_edited(self, iterations: IterationConfiguration):
        assert isinstance(self._state, _SequenceEditableState)
        with self.session_maker() as session:
            try:
                session.sequences.set_iteration_configuration(
                    self._state.sequence_path, iterations
                )
            except (PathNotFoundError, PathIsNotSequenceError):
                self._transition(_SequenceNotSetState())
            except SequenceNotEditableError:
                self._transition(
                    _query_state_sync(self._state.sequence_path, self.session_maker)
                )
            else:
                self._state = attrs.evolve(self._state, iterations=iterations)

    def on_time_lanes_edited(self, time_lanes: TimeLanes):
        assert isinstance(self._state, _SequenceEditableState)
        with self.session_maker() as session:
            try:
                session.sequences.set_time_lanes(self._state.sequence_path, time_lanes)
            except (PathNotFoundError, PathIsNotSequenceError):
                self._transition(_SequenceNotSetState())
            except SequenceNotEditableError:
                self._transition(
                    _query_state_sync(self._state.sequence_path, self.session_maker)
                )
            else:
                self._state = attrs.evolve(self._state, time_lanes=time_lanes)


class _State(abc.ABC):
    pass


@attrs.frozen
class _SequenceNotSetState(_State):
    pass


@attrs.frozen
class _SequenceSetState(_State):
    sequence_path: PureSequencePath
    sequence_state: State

    iterations: IterationConfiguration
    time_lanes: TimeLanes


@attrs.frozen
class _SequenceNotEditableState(_SequenceSetState):
    parameters: ParameterNamespace


@attrs.frozen
class _SequenceEditableState(_SequenceSetState):
    pass


async def _query_state_async(
    path: PureSequencePath, session_maker: ExperimentSessionMaker
) -> _State:
    async with session_maker.async_session() as session:
        is_sequence_result = await session.sequences.is_sequence(path)
        try:
            is_sequence = unwrap(is_sequence_result)
        except PathNotFoundError:
            return _SequenceNotSetState()
        else:
            if is_sequence:
                return await _query_sequence_state_async(path, session)
            else:
                return _SequenceNotSetState()


async def _query_sequence_state_async(
    path: PureSequencePath, session: AsyncExperimentSession
) -> _SequenceSetState:
    # These results can be unwrapped safely because we checked that the sequence
    # exists in the session.
    state = unwrap(await session.sequences.get_state(path))
    iterations = await session.sequences.get_iteration_configuration(path)
    time_lanes = await session.sequences.get_time_lanes(path)

    if state.is_editable():
        return _SequenceEditableState(
            path, iterations=iterations, time_lanes=time_lanes, sequence_state=state
        )
    else:
        parameters = await session.sequences.get_global_parameters(path)
        return _SequenceNotEditableState(
            path,
            iterations=iterations,
            time_lanes=time_lanes,
            parameters=parameters,
            sequence_state=state,
        )


def _query_state_sync(
    path: PureSequencePath, session_maker: ExperimentSessionMaker
) -> _State:
    with session_maker() as session:
        is_sequence_result = session.sequences.is_sequence(path)
        try:
            is_sequence = unwrap(is_sequence_result)
        except PathNotFoundError:
            return _SequenceNotSetState()
        else:
            if is_sequence:
                return _query_sequence_state_sync(path, session)
            else:
                return _SequenceNotSetState()


def _query_sequence_state_sync(
    path: PureSequencePath, session: ExperimentSession
) -> _SequenceSetState:
    # These results can be unwrapped safely because we checked that the sequence
    # exists in the session.
    state = unwrap(session.sequences.get_state(path))
    iterations = session.sequences.get_iteration_configuration(path)
    time_lanes = session.sequences.get_time_lanes(path)

    if state.is_editable():
        return _SequenceEditableState(
            path, iterations=iterations, time_lanes=time_lanes, sequence_state=state
        )
    else:
        parameters = session.sequences.get_global_parameters(path)
        return _SequenceNotEditableState(
            path,
            iterations=iterations,
            time_lanes=time_lanes,
            parameters=parameters,
            sequence_state=state,
        )


class IconLabel(QWidget):
    def __init__(
        self,
        parent: Optional[QWidget] = None,
        icon_position: Literal["left", "right"] = "left",
    ):
        super().__init__(parent)
        self._label = QLabel()
        self._icon = QLabel()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        if icon_position == "left":
            layout.addWidget(self._icon)
            layout.addWidget(self._label)
        else:
            layout.addWidget(self._label)
            layout.addWidget(self._icon)
        self.setLayout(layout)

    def set_text(self, text: str):
        self._label.setText(text)

    def set_icon(self, icon: Optional[QIcon]):
        if icon is None:
            self._icon.clear()
        else:
            self._icon.setPixmap(icon.pixmap(20, 20))
