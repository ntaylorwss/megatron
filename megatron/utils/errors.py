class EagerRunException(Exception):
    def __init__(self):
        message = "Pipeline.run() should not be called when running in Eager Execution mode."
        super().__init__(message)


class ShapeError(Exception):
    def __init__(self, name, input_dims):
        msg = "Data fed into '{}' has {} dims; should be 1D array".format(name, input_dims)
        super().__init__(msg)
