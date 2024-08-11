import decimal
from abc import ABC, abstractmethod
from typing import (
    Type,
    TypeVar,
    Generic,
)

import attrs

from caqtus.device.configuration import DeviceConfiguration
from ..channel_commands import ChannelOutput
from ..runtime import Sequencer
from ..trigger import Trigger, is_trigger


@attrs.define
class ChannelConfiguration(ABC):
    """Contains information to computer the output of a channel."""

    description: str = attrs.field(
        converter=str,
        on_setattr=attrs.setters.convert,
    )
    output: ChannelOutput = attrs.field(
        validator=attrs.validators.instance_of(ChannelOutput),
        on_setattr=attrs.setters.validate,
    )


@attrs.define
class DigitalChannelConfiguration(ChannelConfiguration):
    def __str__(self):
        return f"digital channel '{self.description}'"


@attrs.define
class AnalogChannelConfiguration(ChannelConfiguration):
    output_unit: str = attrs.field(
        converter=str,
        on_setattr=attrs.setters.convert,
    )

    def __str__(self):
        return f"analog channel '{self.description}' with unit {self.output_unit}"


def validate_trigger(instance, attribute, value):
    if not is_trigger(value):
        raise TypeError(f"Trigger {value} is not of type Trigger")


SequencerType = TypeVar("SequencerType", bound=Sequencer)


@attrs.define
class SequencerConfiguration(
    DeviceConfiguration[SequencerType], ABC, Generic[SequencerType]
):
    """Holds the static configuration of a sequencer device.

    Fields:
        time_step: The quantization time step used, in nanoseconds.
            The device can only update its output at times that are integer multiples
            of this time step.
            This is a decimal number to allow sub-nanosecond precision without floating
            point errors.
        channels: The configuration of the channels of the device. The length of this
            list must match the number of channels of the device.
        trigger: The trigger that starts the execution of the sequencer.
    """

    time_step: decimal.Decimal = attrs.field(
        converter=decimal.Decimal,
        validator=attrs.validators.gt(0),
        on_setattr=attrs.setters.pipe(attrs.setters.convert, attrs.setters.validate),
    )
    channels: tuple[ChannelConfiguration, ...] = attrs.field(
        converter=tuple,
        validator=attrs.validators.deep_iterable(
            member_validator=attrs.validators.instance_of(ChannelConfiguration)
        ),
        on_setattr=attrs.setters.pipe(attrs.setters.convert, attrs.setters.validate),
    )
    trigger: Trigger = attrs.field(validator=validate_trigger)

    @abstractmethod
    def channel_types(self) -> tuple[Type[ChannelConfiguration], ...]:
        """Returns the types of the channels of the device."""

        raise NotImplementedError

    @channels.validator  # type: ignore
    def validate_channels(self, _, channels):
        channel_types = self.channel_types()
        number_channels = len(channel_types)
        if len(channels) != number_channels:
            raise ValueError(
                f"The length of channels ({len(channels)}) doesn't match the number of"
                f" channels {number_channels}"
            )
        for channel, channel_type in zip(channels, channel_types, strict=True):
            if not isinstance(channel, channel_type):
                raise TypeError(
                    f"Channel {channel} is not of the expected type {channel_type}"
                )

    def __getitem__(self, item):
        return self.channels[item]
