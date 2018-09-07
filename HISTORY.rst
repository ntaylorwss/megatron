# Release History

## 0.1.2

### Changes
- Added visualization module for the computation graph. Contributed by @jeremyjordan.
- Calling method on layers takes a list of nodes rather than individual positional arguments.

## 0.1.1

### Changes
- Added wrapper for sklearn functions as Transformations.
- Give all nodes names, add ability to run a node by name rather than by a variable reference.
- Relax shape validation if input is not a Numpy array.
- Removed utility dictionaries for custom errors.

### Bug Fixes
- Fixed syntax errors in Graph.


## 0.1.0

Initial release.
