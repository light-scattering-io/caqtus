import pytest
from PySide6.QtCore import QTimer

from caqtus.experiment_control.manager import BoundExperimentManager
from caqtus.gui.condetrol import Condetrol
from caqtus.gui.condetrol.device_configuration_editors import DeviceConfigurationsPlugin
from caqtus.session import PureSequencePath
from caqtus.session.sql import (
    SQLExperimentSessionMaker,
    Serializer,
)
from spincore_pulse_blaster.configuration import SpincoreSequencerConfiguration
from spincore_pulse_blaster.configuration_editor import (
    SpincorePulseBlasterDeviceConfigEditor,
)
from tests.fixtures import steps_configuration, time_lanes


@pytest.fixture
def session_maker(tmp_path):
    url = f"sqlite:///{tmp_path / 'database.db'}"

    serializer = Serializer.default()

    serializer.register_device_configuration(
        SpincoreSequencerConfiguration,
        SpincoreSequencerConfiguration.dump,
        SpincoreSequencerConfiguration.load,
    )

    session_maker = SQLExperimentSessionMaker.from_url(url, serializer=serializer)
    session_maker.create_tables()
    return session_maker


def test_condetrol(
    session_maker,
    steps_configuration,
    time_lanes,
):
    device_plugin = DeviceConfigurationsPlugin.default()
    device_plugin.register_editor(
        SpincoreSequencerConfiguration,
        SpincorePulseBlasterDeviceConfigEditor,
    )
    experiment_manager = BoundExperimentManager(session_maker, {})

    device_plugin.register_default_configuration(
        "Spincore sequencer", SpincoreSequencerConfiguration.default
    )
    condetrol = Condetrol(
        session_maker=session_maker,
        device_configurations_plugin=device_plugin,
        connect_to_experiment_manager=lambda: experiment_manager,
    )
    with session_maker() as session:
        sequence = session.sequences.create(
            path=PureSequencePath(r"\test"),
            iteration_configuration=steps_configuration,
            time_lanes=time_lanes,
        )

    timer = QTimer(condetrol.window)
    # timer.singleShot(0, condetrol.window.close)
    condetrol.run()
