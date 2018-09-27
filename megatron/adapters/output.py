import numpy as np
import pandas as pd


def array(data):
    return data[0] if len(data) == 1 else data


def dataframe(arrays, names):
    """Return data from pipeline as Pandas dataframe.

    Each column corresponds to a node. If a node has 2D data,
    it creates multiple columns, named [node.name].i, for each i.

    If a node holds data that has more than 2 dimensions, this
    function will fail.

    Parameters
    ----------
    arrays : list of Numpy array(s)
        arrays corresponding to each output node.
    names : list of str(s)
        names corresponding to each output node.
    """
    if len(arrays) == 1:
        array, name = arrays[0], names[0]
        if len(array.shape) == 2:
            names = ['{}.{}'.format(name, i) for i in range(array.shape[1])]
        data = pd.DataFrame(array, columns=names)
    else:
        out = []
        new_names = []
        for array, name in zip(arrays, names):
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
