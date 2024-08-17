import decimal
import os

import numpy as np
from PySide6.QtWidgets import QApplication
from matplotlib import pyplot as plt

from caqtus.device.sequencer.channel_commands import Constant
from caqtus.gui.condetrol.device_configuration_editors.sequencer_configuration_editor.channel_output_editor._constant_node import (
    ConstantNode,
)
from caqtus.shot_compilation import ShotContext, SequenceContext
from caqtus.types.expression import Expression
from caqtus.types.timelane import TimeLanes
from .screenshot_output_graph import screenshot_output, screenshot_node


def generate_for_constant():
    output = Constant(Expression("10 V"))
    screenshot_output(output, "images/sequencer_outputs/constant_graph.png")
    node = ConstantNode()
    node.set_value(Expression("expression"))
    screenshot_node(node, "images/sequencer_outputs/constant_node.png")

    sequence_context = SequenceContext({}, TimeLanes(["step1"], [Expression("2 s")]))
    shot_context = ShotContext(sequence_context, {}, {})

    time_step = 3e3
    series = output.evaluate(decimal.Decimal(time_step), 0, 0, shot_context)

    fig, ax = plt.subplots()

    t = np.arange(0, len(series.values)) * time_step * 1e-9
    ax.step(t, series.values.to_pattern().array, where="post")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Output [V]")
    ax.set_ylim(0, 15)
    ax.grid(True)
    ax.set_xlim(t[0], t[-1])

    plt.savefig(
        "images/sequencer_outputs/constant_plot.png",
        bbox_inches="tight",
    )


def generate_figures():
    app = QApplication([])
    os.makedirs("images/sequencer_outputs", exist_ok=True)
    screenshot_output(None, "images/sequencer_outputs/output_node.png")
    generate_for_constant()


if __name__ == "__main__":
    generate_figures()