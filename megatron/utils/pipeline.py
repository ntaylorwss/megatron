from collections import defaultdict
import numpy as np
import pandas as pd


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
    visited = defaultdict(int)
    order = []
    def dfs(node):
        visited[node] += 1
        for in_node in node.inbound_nodes:
            if not in_node in visited:
                dfs(in_node)
        if visited[node] <= 1:
            order.append(node)
    for output_node in output_nodes:
        dfs(output_node)
    return order


def format_output(output_data, out_type):
    """Return arrays for single or multiple Transout_typeionNode(s) in requested out_type.

    Parameters
    ----------
    arrays : list of np.ndarray
        arrays resulting from Pipeline.transform(). Will always be a list, potentially of one.
    out_type : {'array', 'arraysframe'}
        arrays type to return as. If dataframe, colnames are node names.
    """
    if out_type== 'array':
        return output_data
    elif out_type== 'dataframe':
        if len(output_data) == 1:
            name, array = list(output_data.items()[0])
            if len(array.shape) == 2:
                names = ['{}.{}'.format(name, i) for i in range(array.shape[1])]
            arrays = pd.DataFrame(array, columns=names)
        else:
            out = []
            new_names = []
            for name, array in output_data.items():
                if len(array.shape) == 1:
                    out.append(array)
                    new_names.append(name)
                elif len(array.shape) == 2:
                    out += list(array.T)
                    new_names += ['{}{}'.format(name, i) for i in range(len(out))]
                else:
                    raise ValueError("An array has more than 2 dimensions")
            arrays = pd.DataFrame(np.stack(out).T, columns=new_names)
        return arrays
    else:
        raise ValueError("Invalid out_type; should be either 'array' or 'dataframe'")
