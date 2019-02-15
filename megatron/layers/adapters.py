from .. import utils
from .core import Layer
from ..nodes.auxiliary import KerasNode


class Sklearn(Layer):
    def __init__(self, sklearn_transformation):
        super().__init__()
        self.transformation = sklearn_transformation
        self.name = self.transformation.__class__.__name__

    @property
    def metadata(self):
        return self.transformation.__dict__

    def partial_fit(self, *inputs):
        if hasattr(self.transformation, 'partial_fit'):
            self.transformation.partial_fit(*inputs)
        else:
            msg = "Layer {} does not support partial_fit"
            raise NotImplementedError(msg.format(self.transformation.__class__.__name__))

    def fit(self, *inputs):
        self.transformation.fit(*inputs)

    def transform(self, *inputs):
        if hasattr(self.transformation, 'transform'):
            return self.transformation.transform(*inputs)
        else:
            # don't use the labels for this
            return self.transformation.predict(inputs[0])


class Keras(Layer):
    def __init__(self, keras_model):
        super().__init__(n_outputs=len(keras_model.outputs))
        self.model = keras_model
        self.n_inputs = len(self.model.inputs)
        self.name = 'KerasModel'

    def _call(self, inbound_nodes, name=None):
        if name is None and self.n_outputs > 1:
            name = [None for i in range(self.n_outputs)]
        elif len(utils.generic.listify(name)) != self.n_outputs:
            raise ValueError("Number of names does not match number of outputs")

        if self.n_outputs > 1:
            if any(node.output is not None for node in inbound_nodes):
                raise Exception("Keras nodes cannot be run in eager mode")
            out_nodes = [KerasNode(self, inbound_nodes, i) for i in range(self.n_outputs)]
            for node in inbound_nodes:
                node.outbound_nodes += out_nodes
            out = {node.name: node for node in out_nodes}
        else:
            if any(node.output is not None for node in inbound_nodes):
                raise Exception("Keras nodes cannot be run in eager mode")
            out_node = KerasNode(self, inbound_nodes)
            for node in inbound_nodes:
                node.outbound_nodes.append(out_node)
            out = out_node
        return out

    def _split_inputs(self, inputs):
        if self.n_inputs > 1 and self.n_outputs > 1:
            X = inputs[:self.n_inputs]
            Y = inputs[self.n_inputs:]
        elif self.n_inputs > 1:
            X = inputs[:-1]
            Y = inputs[-1]
        elif self.n_outputs > 1:
            X = inputs[0]
            Y = inputs[1:]
        else:
            X, Y = inputs
        return X, Y

    def partial_fit(self, *inputs):
        X, Y = self._split_inputs(list(inputs))
        self.model.train_on_batch(X, Y)

    def fit(self, *inputs, epochs):
        X, Y = self._split_inputs(list(inputs))
        self.model.fit(X, Y, epochs=epochs)

    def fit_generator(self, generator, steps_per_epoch, epochs):
        def _split_generator(generator):
            for inputs in generator:
                X, Y = self._split_inputs(inputs)
                yield X, Y
        self.model.fit_generator(_split_generator(generator),
                                 steps_per_epoch=steps_per_epoch, epochs=epochs)

    def transform(self, *inputs):
        # don't use the labels for this
        if self.n_inputs == 1:
            preds = self.model.predict(inputs[0])
        else:
            preds = self.model.predict(list(inputs[:self.n_inputs]))
        return preds
