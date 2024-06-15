from hypothesis.strategies import SearchStrategy, integers, floats, builds

from caqtus.device.sequencer.instructions import Ramp


def ramp() -> SearchStrategy[Ramp]:
    return builds(
        Ramp,
        start=floats(allow_nan=False, allow_infinity=False),
        stop=floats(allow_nan=False, allow_infinity=False),
        length=integers(min_value=1, max_value=1000),
    )
