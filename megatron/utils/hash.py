import hashlib
import inspect
import numpy as np
from .generic import isinstance_str


def hash_path(nodes):
    h = hashlib.md5()
    for node in nodes:
        to_str = []
        if isinstance_str(node, 'InputNode'):
            to_str.append(node.output)
        else:
            if isinstance_str(node.layer, 'Lambda'):
                to_str.append(inspect.getsource(node.layer.transform_fn))
            elif isinstance_str(node.layer, 'StatefulLayer'):
                to_str.append(inspect.getsource(node.layer.partial_fit))
                to_str.append(inspect.getsource(node.layer.fit))
                to_str.append(inspect.getsource(node.layer.transform))
                to_str += list(node.kwargs.values())
                for metadata in node.layer.metadata.values():
                    to_str.append(metadata)
            elif isinstance_str(node.layer, 'StatelessLayer'):
                to_str.append(inspect.getsource(node.layer.transform))
                to_str += list(node.layer.kwargs.values())

        for s in to_str:
            if isinstance_str(s, 'ndarray'):
                h.update(str(bytes(s)).encode())
            else:
                h.update(str(s).encode())
    return h.hexdigest()
