from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ._instructions import SequencerInstruction
from ._to_time_array import convert_to_change_arrays


def plot_instruction(
    instruction: SequencerInstruction, ax: Optional[Axes] = None
) -> Axes:
    """Plot the instruction on the given axis."""

    if ax is None:
        fig, ax = plt.subplots()

    change_times, change_values = convert_to_change_arrays(instruction)

    ax.step(change_times, change_values, where="post")
    ax.set_xlabel("Time [ticks]")
    ax.set_ylabel("Value")
    return ax
