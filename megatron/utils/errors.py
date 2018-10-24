import warnings


class ShapeError(Exception):
    def __init__(self, name, input_dims, expected_dims):
        msg = "Data fed into '{}' has {} dims; should be {}-D array".format(
            name, input_dims, expected_dims)
        super().__init__(msg)


class DisconnectedError(Exception):
    def __init__(self, missing_inputs):
        base_msg = "The following inputs are not connected to your provided outputs: {}"
        msg = base_msg.format(', '.join([node.name for node in missing_inputs]))
        super().__init__(msg)


def ExtraInputsWarning(extra_inputs):
    msg = "Some input nodes provided that aren't used in the graph: {}"
    msg.format(', '.join([node.node_name for node in extra_inputs]))
    warnings.warn(msg)
