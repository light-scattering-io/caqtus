from PySide6.QtCore import QSize
from hypothesis.stateful import (
    RuleBasedStateMachine,
    run_state_machine_as_test,
    rule,
    precondition,
)
from hypothesis.strategies import integers, data, text, SearchStrategy, lists, booleans

from caqtus.gui.condetrol.timelanes_editor._time_lanes_model import TimeLanesModel
from caqtus.types.timelane import DigitalTimeLane


def digital_lanes(length: int) -> SearchStrategy[DigitalTimeLane]:
    return lists(booleans(), min_size=length, max_size=length).map(DigitalTimeLane)


class TimeLaneModelMachine(RuleBasedStateMachine):
    def __init__(self, lane_extension):
        super().__init__()
        self.model = TimeLanesModel(lane_extension)

    @rule(data=data())
    def insert_column(self, data):
        previous_number_steps = self.model.columnCount()
        column = data.draw(integers(0, previous_number_steps))
        self.model.insertColumn(column)
        assert self.model.columnCount() == previous_number_steps + 1

    @precondition(lambda self: self.model.columnCount() > 0)
    @rule(data=data())
    def remove_column(self, data):
        previous_number_steps = self.model.columnCount()
        column = data.draw(integers(0, previous_number_steps - 1))
        self.model.removeColumn(column)
        assert self.model.columnCount() == previous_number_steps - 1

    @precondition(lambda self: self.model.columnCount() > 0)
    @rule(data=data())
    def change_step_name(self, data):
        number_steps = self.model.columnCount()
        column = data.draw(integers(0, number_steps - 1))
        name = data.draw(text())
        assert self.model.setData(self.model.index(0, column), name)
        assert self.model.data(self.model.index(0, column)) == name

    @precondition(lambda self: self.model.columnCount() > 0)
    @rule(data=data())
    def change_step_duration(self, data):
        number_steps = self.model.columnCount()
        column = data.draw(integers(0, number_steps - 1))
        duration = data.draw(integers(0, 100))
        duration = f"{duration} ms"
        assert self.model.setData(self.model.index(1, column), duration)
        assert self.model.data(self.model.index(1, column)) == duration

    @rule(data=data())
    def add_lane(self, data):
        already_used_names = self.model.lane_names()
        name = data.draw(text().filter(lambda name: name not in already_used_names))
        time_lane = data.draw(digital_lanes(length=self.model.number_steps()))
        self.model.insert_time_lane(name, time_lane)

    @precondition(lambda self: self.model.lane_number() > 0)
    @rule(data=data())
    def remove_lane(self, data):
        lane_number = self.model.lane_number()
        to_remove = data.draw(integers(0, lane_number - 1))
        name = self.model.get_lane_name(to_remove)
        previous_time_lanes = self.model.get_timelanes().lanes

        self.model.remove_lane(to_remove)
        previous_time_lanes.pop(name)
        assert self.model.get_timelanes().lanes == previous_time_lanes

    @precondition(lambda self: has_value(self.model))
    @rule(data=data())
    def set_value(self, data):
        step = data.draw(integers(0, self.model.number_steps() - 1))
        lane_index = data.draw(integers(0, self.model.lane_number() - 1))
        new_value = data.draw(booleans())

        self.model.setData(self.model.cell_index(lane_index, step), new_value)

        assert self.model.get_lane(lane_index)[step] == new_value

    @precondition(lambda self: has_value(self.model))
    @rule()
    def simplify(self):
        self.model.simplify()

    @precondition(lambda self: self.model.undo_stack.count() > 0)
    @rule(data=data())
    def undo_redo(self, data):
        actions_count = self.model.undo_stack.count()
        # index is defined this way such that it shrinks toward large index.
        index = actions_count - data.draw(integers(1, actions_count))
        previous_time_lanes = self.model.get_timelanes()
        self.model.undo_stack.setIndex(index)
        self.model.undo_stack.setIndex(actions_count)
        current_time_lanes = self.model.get_timelanes()
        assert previous_time_lanes == current_time_lanes

    @precondition(
        lambda self: self.model.lane_number() > 0 and self.model.number_steps() >= 2
    )
    @rule(data=data())
    def merge_cells(self, data):
        lane_index = data.draw(integers(0, self.model.lane_number() - 1))
        start_index = data.draw(integers(0, self.model.number_steps() - 2))
        end_index = data.draw(integers(start_index + 1, self.model.number_steps() - 1))
        to_expend_index = data.draw(integers(start_index, end_index))
        self.model.expand_step(to_expend_index, lane_index, start_index, end_index)
        assert self.model.span(self.model.index(lane_index + 2, start_index)) == QSize(
            end_index - start_index + 1, 1
        )


def has_value(model: TimeLanesModel):
    return (model.number_steps() > 0) and (model.lane_number() > 0)


def test_lane_model(lane_extension):
    run_state_machine_as_test(lambda: TimeLaneModelMachine(lane_extension))
