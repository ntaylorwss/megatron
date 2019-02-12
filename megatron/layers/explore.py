import math
import numpy as np
import matplotlib.pyplot as plt
from ..nodes.auxiliary import ExploreNode
from .. import utils


class Explorer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _call(self, nodes, name):
        """Produce an ExploreNode acting on the given inbound nodes.

        Parameters
        ----------
        nodes : list of megatron.Node
            nodes to be passed to the explorer function.
        name : str
            name to be associated with this ExploreNode.
        """
        nodes = utils.generic.listify(nodes)
        out_node = ExploreNode(self, nodes, name)
        for node in nodes:
            node.outbound_nodes.append(out_node)
        if all(node.output is not None for node in nodes):
            out_node.explore()
        return out_node

    def __call__(self, nodes, name):
        return self._call(nodes, name)

    def evaluate(self, *inputs):
        """Run exploration function on given input data."""
        raise NotImplementedError

    def __setitem__(self, key, value):
        self.kwargs[key] = value


class Describe(Explorer):
    """A statistical summary of the given array, similar to Pandas dataframes."""
    def __init__(self, axis=0):
        super().__init__(axis=axis)

    def explore(self, X):
        return {
            'mean': np.mean(X, axis=self.kwargs['axis']),
            'sd': np.std(X, axis=self.kwargs['axis']),
            'min': np.min(X, axis=self.kwargs['axis']),
            '25%': np.percentile(X, 25, axis=self.kwargs['axis']),
            '50%': np.percentile(X, 50, axis=self.kwargs['axis']),
            '75%': np.percentile(X, 75, axis=self.kwargs['axis']),
            'max': np.max(X, axis=self.kwargs['axis'])
        }


class Histogram(Explorer):
    def __init__(self, **plot_kwargs):
        super().__init__(**plot_kwargs)

    def explore(self, X):
        if len(X.shape) > 2:
            raise ValueError("Data has too many dimensions")
        elif len(X.shape) == 1:
            fig, ax = plt.subplots()
            ax.hist(X, **self.kwargs)
            return fig
        else:
            fig = plt.figure()
            nrows = int(math.sqrt(X.shape[1]))
            ncols = math.ceil(X.shape[1] / nrows)
            for i in range(X.shape[1]):
                ax = fig.add_subplot(nrows, ncols, i+1)
                ax.hist(X[:, i], **self.kwargs)
            fig.tight_layout()
            return fig


class Scatter(Explorer):
    def __init__(self, **plot_kwargs):
        super().__init__(**plot_kwargs)

    def explore(self, X, Y):
        if len(X.shape) > 1 or len(Y.shape) > 1:
            raise ValueError("Arrays must be one-dimensional")
        fig, ax = plt.subplots()
        ax.scatter(X, Y, **self.kwargs)
        return fig


class Correlate(Explorer):
    def explore(self, *inputs):
        if len(inputs) == 1:
            return np.corrcoef(inputs[0], rowvar=False)
        else:
            X = np.stack(inputs, axis=1)
            return np.corrcoef(X, rowvar=False)
