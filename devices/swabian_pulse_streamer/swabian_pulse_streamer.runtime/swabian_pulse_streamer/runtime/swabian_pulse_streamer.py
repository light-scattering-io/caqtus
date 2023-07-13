from functools import singledispatchmethod
from typing import Optional

from pulsestreamer import (
    PulseStreamer,
    TriggerStart,
    TriggerRearm,
    Sequence as PulseStreamerSequence,
)
from pydantic import validator, Field

from sequencer.instructions import SequencerInstruction, SequencerPattern, ChannelLabel, Concatenate
from sequencer.runtime import (
    Sequencer,
    SequenceNotConfiguredError,
    Trigger,
)


class SwabianPulseStreamer(Sequencer):
    # only support digital channels at the moment
    channel_number = 8
    ip_address: str

    # only 1 ns time step supported
    time_step: int = Field(ge=1, le=1, allow_mutation=False)

    _pulse_streamer: PulseStreamer
    _sequence: Optional[PulseStreamerSequence] = None

    @validator("trigger")
    def _validate_trigger(cls, trigger: Trigger) -> Trigger:
        if not trigger.is_software_trigger():
            raise ValueError("Only supports software trigger.")
        return trigger

    def initialize(self) -> None:
        super().initialize()

        # There is no close method for the PulseStreamer class
        self._pulse_streamer = PulseStreamer(self.ip_address)
        if self.trigger.is_software_trigger():
            self._pulse_streamer.setTrigger(TriggerStart.SOFTWARE, TriggerRearm.MANUAL)
        else:
            raise ValueError("Only supports software trigger.")

    def update_parameters(self, *_, sequence: SequencerInstruction, **kwargs) -> None:
        super().update_parameters(sequence=sequence, **kwargs)
        self._sequence = self._construct_pulse_streamer_sequence(sequence)
        self._set_sequence_programmed()

    def start_sequence(self) -> None:
        super().start_sequence()
        if not self._sequence:
            raise SequenceNotConfiguredError("The sequence has not been set yet.")
        self._pulse_streamer.stream(seq=self._sequence, n_runs=1)
        if self.trigger.is_software_trigger():
            self._pulse_streamer.startNow()

    def has_sequence_finished(self) -> bool:
        super().has_sequence_finished()
        return self._pulse_streamer.hasFinished()

    @singledispatchmethod
    def _construct_pulse_streamer_sequence(
        self, instruction: SequencerInstruction
    ) -> PulseStreamerSequence:
        raise NotImplementedError(
            f"Not implemented for type of instruction {type(instruction)}."
        )

    @_construct_pulse_streamer_sequence.register
    def _(
        self, pattern: SequencerPattern
    ) -> PulseStreamerSequence:
        sequence = self._pulse_streamer.createSequence()
        values = pattern.values
        for channel in range(self.channel_number):
            channel_values = values[ChannelLabel(channel)].values
            sequence.setDigital(channel, [(1, v) for v in channel_values])
        return sequence

    @_construct_pulse_streamer_sequence.register
    def _(self, concatenate: Concatenate) -> PulseStreamerSequence:
        instructions = concatenate.instructions
        seq = self._construct_pulse_streamer_sequence(instructions[0])
        for instruction in instructions[1:]:
            seq += self._construct_pulse_streamer_sequence(instruction)
        return seq
