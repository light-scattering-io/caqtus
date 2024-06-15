import functools

import graphviz

from ._instructions import SequencerInstruction, Concatenated, Repeated, Pattern
from ._ramp import Ramp


def to_graph(instruction: SequencerInstruction) -> graphviz.Digraph:
    """Convert a sequencer instruction to a graphiz graph for visualization.

    This function requires `Graphviz <https://www.graphviz.org/>`_ to be installed and
    available in the system path.

    Args:
        instruction: The instruction to convert to a graph.

    Returns:
        The graph representation of the instruction.
    """

    graph = graphviz.Digraph(graph_attr={"ordering": "in"})
    add_to_graph(instruction, graph, [0])
    return graph


def levels_to_str(levels: list[int]) -> str:
    return "_".join(map(str, levels))


@functools.singledispatch
def add_to_graph(instr, graph, levels: list[int]):
    raise NotImplementedError(f"Cannot add {type(instr)} to graph")


@add_to_graph.register
def _(instr: Concatenated, graph, levels: list[int]):
    graph.node(levels_to_str(levels), "+")
    graph.attr(rank="same")
    for i, sub_instr in enumerate(instr.instructions):
        add_to_graph(sub_instr, graph, levels + [i])
        graph.edge(levels_to_str(levels), levels_to_str(levels + [i]))
    return graph


@add_to_graph.register
def _(instr: Repeated, graph, levels: list[int]):
    graph.node(levels_to_str(levels), f"*{instr.repetitions}")
    add_to_graph(instr.instruction, graph, levels + [0])
    graph.edge(levels_to_str(levels), levels_to_str(levels + [0]))


@add_to_graph.register
def _(instr: Pattern, graph, levels: list[int]):
    graph.attr(shape="box")
    graph.node(levels_to_str(levels), str(instr))
    return graph


@add_to_graph.register
def _(instr: Ramp, graph, levels: list[int]):
    graph.node(levels_to_str(levels), str(instr))
    return graph
