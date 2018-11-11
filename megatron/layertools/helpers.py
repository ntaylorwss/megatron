def apply(layer, nodes, **layer_kwargs):
    """Create an instance of the Layer for each given Node with the same hyperparameters.

    The difference between map and apply is that map shares the Layer between Nodes,
    whereas apply creates separate Layers for each Node.

    Parameters
    ----------
    layer : megatron.Layer
        the Layer to apply to the Nodes.
    nodes : list of megatron.Node
        the Nodes to which the Layer should be applied.
    **layer_kwargs : kwargs
        the kwargs to be passed to the initialization of the Layer.
    """
    if isinstance(nodes, dict):
        return {key: layer(**layer_kwargs)(node) for key, node in nodes.items()}
    elif isinstance(nodes, list):
        return [layer(**layer_kwargs)(node) for node in nodes]
    else:
        raise ValueError("Can only apply a layer to multiple nodes")
