# Release History

## 0.5.1
### Bug Fixes
- Node names are fixed, by delegating the responsiblity of naming solely to the node itself.

## 0.5.0
### Changes
- Add Flatten layer.
- Add Filter layer.
- Give layers a name attribute rather than manipulate the class name.
- Make more descriptive error messages; when a node fails, identify which node it is.
- Make use of progress bar for full dataset fits as well as generator fits, for Keras models.

### Bug Fixes
- Fix dataset loaders, which previously only threw an error.

## 0.4.5
### Changes
- Add Slice layer.
- Re-arrange nodes module.
- Add exploration mode to the pipeline for EDA.
- Separate out paths for transformation, evaluation, and exploration within the pipeline to avoid unnnecessary computation.
- Update visualization module to include exploration nodes in their own colour.

## 0.4.4
### Changes
- Add Normalization layer.
- Convert TimeSeries layer to Stateful so it can work across batches without loss of data.
- Add traversal method to Nodes to be able to navigate between them.
- Add 'anon' to label of Lambda layers that are anonymous functions.

## 0.4.3
### Changes
- Add ability to overwrite storage for a pipeline name/version.
- When executing eagerly, creating a pipeline destroys stored data and metadata.

### Bug Fixes
- Fix pipeline saving by re-structuring usage of dynamic classes in wrappers.

## 0.4.2
### Changes
- Merge metrics module into layers.

### Bug Fixes
- Remove extra argument in CSVData.
- Fix storage read method.
- Re-work unused data pruning logic.

## 0.4.1
### Changes
- Generalized SklearnMetric wrapper to simple Metric wrapper, which can take any function.
- Remove any attempt to create built-in metrics; they exist elsewhere and should be imported from elsewhere.
- Pipeline transform always returns a list of results, even if there's only one output node.
- Renamed parameters of storage.read.

## 0.4.0
### Changes
- Nodes no longer have names, except InputNodes.
- Since nodes don't have names, retrieving columns from cache comes by integer index.
- Input moved from nodes to layers.
- Layertools module added.
- FeatureSet has been removed. Mapping and applying a layer to a set of nodes has been added in layertools module.
- Data readers give dictionaries rather than FeatureSets.
- keep_data flag added to transform to indicate whether to wipe non-output nodes' data or not.
- transform_generator will always keep data.
- Metrics are now their own kind of node, not part of the transformation pipeline; they're run when evaluate() is called.

## 0.3.6
### Changes
- Multiple Keras models no longer possible, but fit_generator is used when fitting to a generator which makes the pipeline load in parallel with GPU training.
- Data index is a string indicating the key name from the passed in data dictionary when using transform.

## 0.3.5
### Changes
- Add wrapper for metrics.
- Add support for multiple epochs for models.
- Fix fitting to a generator so it properly supports models.
- Add pop method to FeatureSet.
- Remove Pandas formatting for output data.
- Import data loaders and generators straight from io module.

## 0.3.4a
### Bug Fixes
- Name of single unnamed node was coming up as list; fixed.

## 0.3.4
### Changes
- Allow for mix of Nodes and FeatureSets when calling Layer.
- Re-arrange the layers module slightly (remove common, move to shaping).
- Add internal utility for flattening irregular list of lists.

## 0.3.3
### Changes
- Add nrows argument to Pandas and CSV full dataset loaders.
- Break OneHot layer into one for range of numbers and one for categorical.
- Add helper decorator to vectorize a function.
- Add support for layers that output multiple nodes.

### Bug Fixes
- Fix syntax error preventing Input nodes from running.

## 0.3.2
### Bug Fixes
- Version 0.3.1 was completely uninstallable because of trying to load the version file from disk. Fixed.

## 0.3.1
### Changes
- SQLite no longer the default caching database. By default, there is no caching.
- No more cache_result parameter to transform methods. If a database is passed at init, it caches, otherwise not.
- Add support for Sklearn supervised learning models, and Keras models, as layers.

## 0.3.0
### Changes
- Now handling data generators as input.
- Pulled out fit into its own public method, as well as partial_fit.
- Create pipeline by defining inputs and outputs, rather than passing to inputs.
- Re-organize utils module.
- Add sqlite database for storing processed features in such a way that they can be queried.
- Create io module, which holds data generators / datasets for input, feature cache database for output.
- Remove SklearnTransformation wrapper; classes can now be used as long as they adhere to the fit/transform API.
- Adjust naming so that Layer and Node have their own names.
- Name argument for node is now received in the call of the Layer, not its init.
- Observations can be given an index, and by default the index is just integers. This is how cache lookup is done.
- Cache now supports multi-dimensional outputs, such as images.

## 0.2.1
### Changes
- Topological sort now operates on all output nodes at once, producing a single path.
- Caching was re-implemented to align with the new path structure.
- Adapters broken out into input and output modules.
- Rename Graph to Pipeline.
- Add color to graph visualization.

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
