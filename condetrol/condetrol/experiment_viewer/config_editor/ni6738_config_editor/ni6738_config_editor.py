import logging
from typing import Optional

from PyQt6.QtWidgets import QWidget

from experiment.configuration import ExperimentConfig
from ni6738_analog_card.configuration import NI6738SequencerConfiguration
from .ni6738_editor_ui import Ui_NI6738Editor
from ..channel_model import ChannelsModel
from ..color_delegate import ColorCellDelegate
from ..config_settings_editor import ConfigSettingsEditor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

us = 1e-6


class NI6738ConfigEditor(ConfigSettingsEditor, Ui_NI6738Editor):
    """NI 6738 analog card configuration widget

    This widget has fields for the device ID and the time step, as well as a table
    to configure the channels.
    """

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        tree_label: str,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(experiment_config, tree_label, parent)

        self.device_name = self.strip_device_prefix(tree_label)
        self.experiment_config = experiment_config
        self.config: NI6738SequencerConfiguration = experiment_config.get_device_config(
            self.device_name
        )

        self.setupUi(self)
        self.setup_ui_from_config(self.config)
        self.channels_table_view.setItemDelegateForColumn(1, ColorCellDelegate())

    def get_experiment_config(self) -> ExperimentConfig:
        self.write_ui_to_config(self.config)
        self.experiment_config.set_device_config(self.device_name, self.config)
        return self.experiment_config

    def setup_ui_from_config(self, config: NI6738SequencerConfiguration):
        self.device_id_line_edit.setText(config.device_id)
        self.time_step_spinbox.setValue(config.time_step / us)
        self.channels_table_view.setModel(NI6738ChannelsModel(config))
        self.channels_table_view.resizeColumnToContents(0)

    def write_ui_to_config(self, config: NI6738SequencerConfiguration):
        config.device_id = self.device_id_line_edit.text()

        # There is an issue here because 2.5 * 1e-6 == 2.4999999999999998e-06 which is
        # lower than the minimum limit of 2.5e-06, so we deal with this value explicitly
        if self.time_step_spinbox.value() == 2.5:
            new_value = 2.5e-6
        else:
            new_value = self.time_step_spinbox.value() * us
        config.time_step = new_value
        return config


class NI6738ChannelsModel(ChannelsModel):
    pass
