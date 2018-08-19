# Megatron: Computation Graphs for Feature Engineering

Megatron is a library that facilitates fully customizable pipelines for feature engineering by implementing the process as a directed acyclic graph. Based in Numpy, it provides a comprehensive suite of transformations, as well as the ability to apply user-defined transformations, provided they accept and return particular data types, which are listed below.

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
Each transformation function takes in the data it will operate on as its first positional arguments. It can act on as many data arguments as is necessary. Following that, it takes in any "hyperparameters", or static configuration parameters. These are passed to the transformation function as keyword arguments to the Transformation object.

As an example, here's the setup for the `time_series` function from `megatron.transforms.common`, which takes in a `window_size` parameter and operates on one data argument:

```
out = megatron.Transformation(megatron.transforms.time_series, window_size=5)(X)
```

## Eager Execution
Megatron supports eager execution with a very simple adjustment to the code. When you define a Feature variable (or any child class), you can call it as a function and pass it a Numpy array as data. Doing this will result in the graph being executed eagerly; the results for each Node can be found in its `output` attribute. An example to state the comparison:

```
# lazy
G = megatron.Graph(cache_path='megatron/cache')
X = megatron.FeatureSet(G, 'X', 10)
Y = megatron.FeatureSet(G, 'Y', 10)

Z = megatron.Transformation(megatron.transforms.time_series, window_size=5)(X)

data = {'X': np.ones((50, 10)),
        'Y': np.ones((50, 10))}

result = G.run(Z, data)
print(result.shape) # >>> (45, 5, 10)

# eager
data = {'X': np.random.random((50, 10)),
        'Y': np.random.random((50, 10))}

G = megatron.Graph()
X = megatron.FeatureSet(G, 'X', 10)(data['X'])
Y = megatron.FeatureSet(G, 'Y', 10)(data['Y'])
Z = megatron.Transformation(megatron.transforms.time_series, window_size=5)(X)
print(Z.output.shape) # >>> (45, 5, 10)
```

As you can see, the Node `Z` had its value computed on creation, and it was stored in `Z.output`. The same is true of all Nodes in the Graph. `Graph.run` is not used when eager execution is being applied.

## Caching
Megatron will, by default, cache the results of an execution of a Node in a space-efficient compressed file. When a Node in a computation graph is run, it checks to see if at any point along the path, a particular Node's subpath has already been previously computed and stored, and if it has, it reloads that Node and begins from that point. This will happen at the latest possible point in the executed path, including potentially the executed Node itself. This is meant to be time-efficient at the potential cost of space, though space was conserved as best as possible through compression.

Caching can be avoided by executing `Graph.run` with the keyword `cache=False`.

Caching does not apply in the case of eager execution.

## Data Types
Feature transformers act on and return data only in the following types, which aim to cover all forms of features one might use in a machine learning model:

- NumericFeature: A single column of numeric data; a 1-dimensional array.
- FeatureSet: Multiple related but independent columns of numeric data; a 2-dimensional array.
- Image: A representation of an image in either greyscale, RGB, or RGBA format; a 3-dimensional array.
- Text: A single column of text data; a 1-dimensional array.
