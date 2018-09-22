# This module credits heavy inspiration to a similar Keras module
# https://github.com/keras-team/keras/blob/master/keras/utils/vis_utils.py

import os
from . import utils
from IPython.display import SVG

# check for pydot
try:
    import pydot
except ImportError:
    pydot = None


def _check_pydot():
    """Raise errors if `pydot` or GraphViz are not properly installed."""
    if pydot is None:
        raise ImportError('Failed to import `pydot`. Please install `pydot` in your '
                          'current environment.')
    try:
        pydot.Dot.create(pydot.Dot())
    except OSError:
        raise OSError('GraphViz must be installed with its executables included in the $PATH.')


def pipeline_to_dot(graph, output_nodes, rankdir='TB'):
    """Convert a megatron Graph to dot format for visualization.

    Parameters
    ----------
    graph : megatron.Graph
        Feature pipeline defined as a graph.
    output_nodes : megatron.Node or list of megatron.Node
        The output nodes of the pipeline determine your feature-space. Include a list
        of all nodes which you would like to be included as features in the output.
    rankdir : str ['TB' or 'LR']
        Direction of graph to plot (top to bottom or left to right).


    Returns
    -------
    pydot.Dot
        Dot representation of the Graph.
    """

    _check_pydot()
    dot = pydot.Dot()
    dot.set('rankdir', rankdir)
    dot.set('concentrate', True)
    dot.set_node_defaults(shape='record')

    # build graph
    paths = []
    output_nodes = utils.listify(output_nodes)
    for output_node in output_nodes:
        paths.append(graph._postorder_traversal(output_node))

    # add nodes
    nodes = {node for path in paths for node in path}
    for node in nodes:

        node_id = str(id(node))
        label = node.name
        pydot_node = pydot.Node(node_id, label=label)
        dot.add_node(pydot_node)

        # create edges
        for input_node in reversed(node.input_nodes):
            input_node_id = str(id(input_node))
            dot.add_edge(pydot.Edge(input_node_id, node_id))

    return dot


def graph_imshow(graph, output_nodes, rankdir='TB'):
    """Create visualization of graph within Jupyter Notebook.

    Parameters
    ----------
    graph : megatron.Graph
        Feature pipeline defined as a graph.
    output_nodes : megatron.Node or list of megatron.Node
        The output nodes of the pipeline determine your feature-space. Include a list
        of all nodes which you would like to be included as features in the output.
    rankdir : str ['TB' or 'LR']
        Direction of graph to plot (top to bottom or left to right).

    Returns
    -------
    IPython.display.SVG
        Display of graph.
    """
    dot = pipeline_to_dot(graph, output_nodes, rankdir)
    return SVG(dot.create(prog='dot', format='svg'))


def save_image(graph, output_nodes, save_path='pipeline.png', rankdir='TB'):
    """Save visualization of graph to an image file.

    Parameters
    ----------
    graph : megatron.Graph
        Feature pipeline defined as a graph.
    output_nodes : megatron.Node or list of megatron.Node
        The output nodes of the pipeline determine your feature-space. Include a list
        of all nodes which you would like to be included as features in the output.
    save_path : str
        Specify where to save the graph visualization.
    rankdir : str ['TB' or 'LR']
        Direction of graph to plot (top to bottom or left to right).
    """
    dot = pipeline_to_dot(graph, output_nodes, rankdir)
    _, extension = os.path.splitext(save_path)
    if not extension:
        extension = 'png'
    else:
        extension = extension[1:]
    dot.write(save_path, format=extension)
