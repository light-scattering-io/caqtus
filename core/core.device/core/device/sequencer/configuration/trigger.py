from enum import Enum
from typing import TypeAlias, TypeGuard

import attrs

from util import serialization


class TriggerEdge(Enum):
    RISING = "rising"
    FALLING = "falling"
    BOTH = "both"


serialization.register_unstructure_hook(TriggerEdge, lambda edge: edge.value)


@attrs.define
class SoftwareTrigger:
    pass


@attrs.define
class ExternalTriggerStart:
    edge: TriggerEdge = TriggerEdge.RISING


@attrs.define
class ExternalClock:
    edge: TriggerEdge = TriggerEdge.RISING


@attrs.define
class ExternalClockOnChange:
    edge: TriggerEdge = TriggerEdge.RISING


Trigger: TypeAlias = (
    SoftwareTrigger | ExternalTriggerStart | ExternalClock | ExternalClockOnChange
)


def is_trigger(value) -> TypeGuard[Trigger]:
    return isinstance(
        value,
        (SoftwareTrigger, ExternalTriggerStart, ExternalClock, ExternalClockOnChange),
    )


serialization.configure_tagged_union(Trigger, tag_name="trigger type")
