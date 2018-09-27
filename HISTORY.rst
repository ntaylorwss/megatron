# Release History

## 0.2.1

### Changes
- Topological sort now operates on all output nodes at once, producing a single path.
- Caching was re-implemented to align with the new path structure.
- Adapters broken out into input and output modules.
- Rename Graph to Pipeline.

### Bug Fixes
- Lots of syntax errors because I didn't test the last release.

## 0.2.0

### Changes
- Added visualization module for the computation graph. Contributed by @jeremyjordan.
- Calling method on layers takes a list of nodes rather than individual positional arguments.
- Loading and saving Graph functionality uses Dill to correctly pickle functions.
- Added adapters for Pandas data as both Input node creator based on colnames, and as feed dict based on colnames.
- Save memory by removing data from output member of nodes during run when they're not needed anymore.
- Remove custom naming option from nodes and transformations (not Input).
- Remove ability to run nodes by string name; must be actual Node variable.
- Add FeatureSet, a grouping of Nodes for mapping transformations onto.
- Structural change of the package, breaking up core into its components.
- Rename Transformation to Layer.

### Bug Fixes
- Previously was not applying kwargs passed to Lambda init on to transform method.
- Fix postorder traversal so it doesn't duplicate nodes.

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
