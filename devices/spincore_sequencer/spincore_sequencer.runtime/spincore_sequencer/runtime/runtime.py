import logging
from collections.abc import Sequence
from enum import IntFlag
from functools import singledispatchmethod
from typing import ClassVar

from pydantic import Field, validator

from log_exception import log_exception
from sequencer.instructions import (
    SequencerInstruction,
    SequencerPattern,
    ChannelLabel,
    Repeat,
    Concatenate,
)
from sequencer.runtime import Sequencer, Trigger, SoftwareTrigger
from . import spinapi
from .spinapi import ns

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


class SpincoreStatus(IntFlag):
    Stopped = 2**0
    Reset = 2**1
    Running = 2**2
    Waiting = 2**3


class SpincorePulseBlaster(Sequencer):
    """

    Fields:
        clock_cycle: Duration of a clock cycle in ns
        time_step: Digitization time in ns
    """

    channel_number: ClassVar[int] = 24
    clock_cycle: ClassVar[int] = 10

    board_number: int = 0
    spincore_lib_debug: bool = False
    time_step: int = Field(ge=5 * clock_cycle)
    trigger: Trigger = Field(default_factory=SoftwareTrigger)

    @validator("trigger")
    def _validate_trigger(cls, trigger: Trigger) -> Trigger:
        if not isinstance(trigger, SoftwareTrigger):
            raise NotImplementedError(
                f"Trigger type {type(trigger)} is not implemented for the Spincore PulseBlaster"
            )
        return trigger

    @classmethod
    def exposed_remote_methods(cls) -> tuple[str, ...]:
        return super().exposed_remote_methods() + ("run",)

    @log_exception(logger)
    def initialize(self) -> None:
        super().initialize()

        spinapi.pb_set_debug(self.spincore_lib_debug)

        board_count = spinapi.pb_count_boards()
        if self.board_number >= board_count:
            raise ConnectionError(
                f"Can't access board {self.board_number}\nThere are only"
                f" {board_count} boards"
            )
        if spinapi.pb_select_board(self.board_number) != 0:
            raise ConnectionError(
                f"Can't access board {self.board_number}\n{spinapi.pb_get_error()}"
            )

        if spinapi.pb_init() != 0:
            raise ConnectionError(
                f"Can't initialize board {self.board_number}\n{spinapi.pb_get_error()}"
            )
        self._add_closing_callback(spinapi.pb_close)

        spinapi.pb_core_clock(1e3 / self.clock_cycle)

    @log_exception(logger)
    def update_parameters(self, /, sequence: SequencerInstruction, **kwargs) -> None:
        sequence_duration = len(sequence) * self.time_step * 1e-9
        logger.debug(f"{sequence_duration=}")
        if spinapi.pb_start_programming(spinapi.PULSE_PROGRAM) != 0:
            raise RuntimeError(
                f"Can't start programming sequence.{spinapi.pb_get_error()}"
            )

        self._program_instruction(sequence)
        self.program_stop(sequence.split(len(sequence) - 1)[1].flatten())

        if spinapi.pb_stop_programming() != 0:
            raise RuntimeError(
                "An error occurred when finishing programming the sequence."
                f"{spinapi.pb_get_error()}"
            )
        self._set_sequence_programmed()

    @singledispatchmethod
    def _program_instruction(self, instruction: SequencerInstruction) -> int:
        raise NotImplementedError(f"Not implemented for {type(instruction)}")

    @_program_instruction.register
    @log_exception(logger)
    def _(self, pattern: SequencerPattern):
        for i in range(len(pattern)):
            values = pattern.get_values_at(i)
            outputs = [
                values[ChannelLabel(channel)] for channel in range(self.channel_number)
            ]
            self._program_continue(outputs, 1)

    @property
    def tick_duration(self) -> float:
        return self.time_step * ns

    @property
    def max_number_ticks(self) -> int:
        return int((2**32 - 1) * self.clock_cycle // self.time_step)

    @property
    def max_duration(self) -> float:
        return self.max_number_ticks * self.time_step * ns

    def _program_continue(self, output_: Sequence[bool], number_ticks: int):
        # break the duration into multiple long waits and one short wait
        number_of_repetitions, remainder = divmod(number_ticks, self.max_number_ticks)

        flags = self._output_to_flags(output_)
        if number_of_repetitions >= 1:
            # Delay multiplier must be greater than 2, so we divide by 2 the length of the wait and multiply the number
            # of repetitions by 2
            if (
                spinapi.pb_inst_pbonly(
                    flags,
                    spinapi.Inst.LONG_DELAY,
                    2 * number_of_repetitions,
                    self.max_duration / 2,
                )
                < 0
            ):
                raise RuntimeError(
                    "An error occurred when programming a long delay of duration"
                    f" {self.max_duration / 2} s with"
                    f" {2*number_of_repetitions} repetitions. "
                )
        duration = remainder * self.time_step * ns
        if spinapi.pb_inst_pbonly(flags, spinapi.Inst.CONTINUE, 0, duration) < 0:
            raise RuntimeError(
                "An error occurred when programming a continue instruction with"
                f" duration {duration} s."
            )

    @_program_instruction.register
    @log_exception(logger)
    def _(self, repeat: Repeat):
        if len(repeat.instruction) == 1:
            channel_values = repeat.instruction.flatten().get_values_at(0)
            outputs = [
                channel_values[ChannelLabel(channel)]
                for channel in range(self.channel_number)
            ]
            self._program_continue(outputs, repeat.number_repetitions)
            return
        elif len(repeat.instruction) == 2:
            for_part, end_for_part = repeat.instruction.split(1)
            middle = None
            rep = repeat.number_repetitions
        else:
            for_part, right = repeat.instruction.split(1)
            middle, end_for_part = right.split(len(right) - 1)
            rep = repeat.number_repetitions
        for_values = for_part.flatten().get_first_values()
        for_flag = self._output_to_flags(
            [
                for_values[ChannelLabel(channel)]
                for channel in range(self.channel_number)
            ]
        )
        end_for_values = end_for_part.flatten().get_last_values()
        end_for_flag = self._output_to_flags(
            [
                end_for_values[ChannelLabel(channel)]
                for channel in range(self.channel_number)
            ]
        )
        logger.debug(f"for {rep=}")
        if rep > 2**20 - 1:
            raise ValueError(
                f"Can't program a for loop with more than {2**20 - 1} repetitions."
            )
        if (
            loop_beginning := spinapi.pb_inst_pbonly(
                for_flag, spinapi.Inst.LOOP, rep, self.time_step * ns
            )
        ) < 0:
            raise RuntimeError(
                "An error occurred when programming the sequence."
                f"{spinapi.pb_get_error()}"
            )
        if middle is not None and len(middle) > 0:
            self._program_instruction(middle)

        if (
            spinapi.pb_inst_pbonly(
                end_for_flag,
                spinapi.Inst.END_LOOP,
                loop_beginning,
                self.time_step * ns,
            )
            < 0
        ):
            raise RuntimeError(
                "An error occurred when programming the sequence."
                f"{spinapi.pb_get_error()}"
            )

    @_program_instruction.register
    @log_exception(logger)
    def _(self, concatenate: Concatenate):
        for instruction in concatenate.instructions:
            self._program_instruction(instruction)

    def program_stop(self, pattern: SequencerPattern):
        values = pattern.values
        channel_values = [
            values[ChannelLabel(channel)].values
            for channel in range(self.channel_number)
        ]

        outputs = [channel_values[channel][0] for channel in range(self.channel_number)]
        flags = self._output_to_flags(outputs)
        if spinapi.pb_inst_pbonly(flags, spinapi.Inst.STOP, 0, self.time_step * ns) < 0:
            raise RuntimeError(
                "An error occurred when programming the sequence. "
                f"{spinapi.pb_get_error()}"
            )

    def _output_to_flags(self, states: Sequence[bool]) -> int:
        flags = 0
        for channel, state in zip(range(self.channel_number), states):
            flags |= int(state) << channel
        return flags

    @log_exception(logger)
    def start_sequence(self) -> None:
        super().start_sequence()
        if spinapi.pb_reset() != 0:
            raise RuntimeError(f"Can't reset the board. {spinapi.pb_get_error()}")

        if spinapi.pb_start() != 0:
            raise RuntimeError(f"Can't start the sequence. {spinapi.pb_get_error()}")

    @log_exception(logger)
    def has_sequence_finished(self) -> bool:
        super().has_sequence_finished()
        is_running = spinapi.pb_read_status() & SpincoreStatus.Running
        return not is_running
