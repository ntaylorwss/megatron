# Megatron: Computation Graphs for Feature Engineering in Machine Learning

Megatron is a library that facilitates fully customizable pipelines for feature engineering by implementing the process as a directed acyclic graph. Based in Numpy, it provides a comprehensive suite of transformations, as well as the ability to apply user-defined transformations, provided they accept and return particular data types, which are listed below.

## Installation and Requirements
Megatron comes with a Docker image that provides a minimal environment with all its required dependencies and a Jupyter environment. If you wish to use the package on its own, it can be installed via pip:

`pip install megatron`

And the dependencies are as follows:

- Numpy (required)
- Sklearn (optional: only if using transformation functions from sklearn with the `SklearnTransformer` wrapper)
- NLTK (optional: only if using `megatron.transforms.text` module)
- Scikit-Image (optional: only if using `megatron.transforms.image` module)

## Transforms
Megatron supports arbitrary transformations, so long as they return valid Numpy arrays. The transformation can use other data structures as intermediate tools, but each transformation must receive and return a Numpy array.

Megatron provides a growing suite of transformation functions, and these are grouped by data type under `megatron.transforms`. This means that at present, the modules are:

- `megatron.transforms.numeric`
- `megatron.transforms.image`
- `megatron.transforms.text`
- `megatron.transforms.common`

`megatron.transforms.common` holds those transformations that apply generally to any data type, such as the `time_series` function that converts an array of observations into an array of time series observations by sliding a window over the data.

All functions are used by being passed to a `megatron.Transformation` object, which takes in as parameters a transformation function and any Configuration Parameters necessary.

### Configuration Parameters
Transformation functions, which are any built-ins, or custom functions that are stateful, take keyword arguments in their initialization. These are like "hyperparameters" for the function, those that stay the same for each execution and which can be customized by the user. These functions are then called on data (arrays), which are passed as individual arguments. Transformations should take in arguments using list expansion syntax, e.g. `*inputs`, so that this can be facilitated.

As an example, here's the usage for the `TimeSeries` function from `megatron.transforms.common`, which takes in a `window_size` parameter and operates on one data argument:

```
out = megatron.transforms.TimeSeries(window_size=5, time_axis=-1)(X)
```

## Eager Execution
Megatron supports eager execution with a very simple adjustment to the code. When you define an Input node, you can call it as a function and pass it a Numpy array as data. Doing this will result in the graph being executed eagerly; the results for each Node can be found in its `output` attribute. An example to state the comparison:

```
# lazy
G = megatron.Graph(cache_path='megatron/cache')
X = megatron.Input(G, 'X', (10,))
Y = megatron.Input(G, 'Y', (10,))

Z = megatron.transforms.TimeSeries(window_size=5)(X)

data = {'X': np.ones((50, 10)),
        'Y': np.ones((50, 10))}

result = G.run(Z, data)
print(result.shape) # >>> (45, 5, 10)

# eager
data = {'X': np.random.random((50, 10)),
        'Y': np.random.random((50, 10))}

G = megatron.Graph()
X = megatron.Input(G, 'X', (10,))(data['X'])
Y = megatron.Input(G, 'Y', (10,))(data['Y'])
Z = megatron.transforms.TimeSeries(window_size=5)(X)
print(Z.output.shape) # >>> (45, 5, 10)
```

As you can see, the Node `Z` had its value computed on creation, and it was stored in `Z.output`. The same is true of all Nodes in the Graph. `Graph.run` is not used when eager execution is being applied.

## Caching
Megatron will, by default, cache the results of an execution of a Node in a space-efficient compressed file. When a Node in a computation graph is run, it checks to see if at any point along the path, a particular Node's subpath has already been previously computed and stored, and if it has, it reloads that Node and begins from that point. This will happen at the latest possible point in the executed path, including potentially the executed Node itself. This is meant to be time-efficient at the potential cost of space, though space was conserved as best as possible through compression.

Caching can be avoided by executing `Graph.run` with the keyword `cache=False`.

Caching does not apply in the case of eager execution.
