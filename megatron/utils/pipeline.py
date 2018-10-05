from collections import defaultdict
import numpy as np
import pandas as pd


def topsort(output_nodes):
    """Returns the path to the desired TransformationNode through the Pipeline.

    Parameters
    ----------
    output_node : TransformationNode
        the target terminal node of the path.

    Returns
    -------
    list of TransformationNode
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


def format_output(arrays, form, names):
    """Return data for single or multiple TransformationNode(s) in requested format.

    Parameters
    ----------
    arrays : list of np.ndarray
        data resulting from Pipeline.transform(). Will always be a list, potentially of one.
    form : {'array', 'dataframe'}
        data type to return as. If dataframe, colnames are node names.
    names : list of str
        names of output nodes; used when form is 'dataframe'.
    """
    name_counts = {}
    new_names = []
    for name in names:
        if name in name_counts:
            new_names.append('{}.{}'.format(name, name_counts[name]))
            name_counts[name] += 1
        else:
            new_names.append(name)
            name_counts[name] = 1
    if form== 'array':
        return {name: array for name, array in zip(new_names, arrays)}
    elif form== 'dataframe':
        if len(data) == 1:
            array, name = data[0], names[0]
            if len(array.shape) == 2:
                names = ['{}.{}'.format(name, i) for i in range(array.shape[1])]
            data = pd.DataFrame(array, columns=names)
        else:
            out = []
            new_names = []
            for array, name in zip(data, names):
                if len(array.shape) == 1:
                    out.append(array)
                    new_names.append(name)
                elif len(array.shape) == 2:
                    out += list(array.T)
                    new_names += ['{}{}'.format(name, i) for i in range(len(out))]
                else:
                    raise ValueError("An array has more than 2 dimensions")
            data = pd.DataFrame(np.stack(out).T, columns=new_names)
        return data
    else:
        raise ValueError("Invalid format; should be either 'array' or 'dataframe'")
