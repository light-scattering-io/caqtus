from caqtus.session.shot import DigitalTimeLane
from caqtus.session.shot.timelane import AnalogTimeLane, Ramp
from caqtus.types.expression import Expression
from caqtus.utils import serialization


def test():
    lane = AnalogTimeLane([Expression("0 mW")] + [Ramp()] * 2 + [Expression("0.7 mW")])
    s = serialization.converters["json"].unstructure(lane, AnalogTimeLane)
    print(s)
    o = serialization.converters["json"].structure(s, AnalogTimeLane)
    assert o == lane, (o, lane)


def test_1():
    lane = DigitalTimeLane(
        [True] * 2 + [False] * 3,
    )
    s = serialization.converters["json"].unstructure(lane, DigitalTimeLane)
    print(s)
    o = serialization.converters["json"].structure(s, DigitalTimeLane)
    assert o == lane, (o, lane)