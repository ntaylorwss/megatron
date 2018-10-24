from collections import defaultdict
from .generic import listify


def topsort(output_nodes):
    """Returns the path to the desired Transout_typeionNode through the Pipeline.

    Parameters
    ----------
    output_node : Transout_typeionNode
        the target terminal node of the path.

    Returns
    -------
    list of Transout_typeionNode
        the path from input to output that arrives at the output_node.
    """
    output_nodes = listify(output_nodes)
    visited = defaultdict(int)
    order = []

    def dfs(node):
        visited[node] += 1
        for in_node in node.inbound_nodes:
            if in_node not in visited:
                dfs(in_node)
        if visited[node] <= 1:
            order.append(node)

    for output_node in output_nodes:
        dfs(output_node)
    return order
