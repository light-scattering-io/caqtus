from core.session.sequence.iteration_configuration import (
    StepsConfiguration,
    LinspaceLoop,
    ExecuteShot,
)
from core.types.expression import Expression
from core.types.variable_name import DottedVariableName


def test_serialization(steps_configuration: StepsConfiguration):
    assert steps_configuration == StepsConfiguration.load(
        StepsConfiguration.dump(steps_configuration)
    )


def test_number():
    steps = StepsConfiguration(
        steps=[
            LinspaceLoop(
                variable=DottedVariableName("a"),
                start=Expression("0"),
                stop=Expression("1"),
                num=10,
                sub_steps=[
                    ExecuteShot(),
                ],
            ),
        ]
    )
    assert steps.expected_number_shots() == 10
