import hashlib

def listify(x):
    return x if isinstance(x, list) else [x]


def md5_hash(x):
    if x.__class__.__name__ == 'ndarray':
        x = bytes(x)
    return str(int(hashlib.md5(str(x).encode()).hexdigest(), 16))
