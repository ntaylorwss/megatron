# Megatron: Feature Engineering Pipelines

Megatron is a library that facilitates fully customizable pipelines for feature engineering by implementing the process as a graph of functions on Numpy/Pandas data. It can be used to clearly define complex feature engineering concepts as a series of simple functions; it also allows for clear visualization as a graph, fast re-use through caching, and the ability to fit to training data and apply fitted transformations to testing data. It supports eager execution for transparency at every step. It provides a comprehensive suite of transformations, as well as the ability to apply user-defined transformations, and transformations from other libraries like Sklearn. With a design heavily inspired by the Keras Functional API, Megatron aims to be the simplest interface to both straightforward and arbitrarily complex feature engineering.

## Sabermetrics Demo
Coming Soon.

## Installation and Requirements
Megatron comes with a Docker image that provides a minimal environment with all its required dependencies and Jupyter. If you wish to use the package on its own, it can be installed via pip:

`pip install megatron`

And the dependencies are as follows:

- Numpy (required)
- Sklearn (optional: only if using transformation functions from sklearn with the `SklearnTransformer` wrapper)
- NLTK (optional: only if using `megatron.transforms.text` module)
- Scikit-Image (optional: only if using `megatron.transforms.image` module)
- PyArrow (optional: only if loading Parquet-formatted raw data)

## Using Transformations
Transformations take keyword arguments in their initialization. These are like "hyperparameters" for the function, those that stay the same for each execution and which can be customized by the user. These functions are then called on other Nodes if in lazy execution, or on data if in eager execution (see below for a discussion on Eager Execution). Multiple arguments are passed as a list.

As an example, here's the usage for the `TimeSeries` function from `megatron.transforms.common`, which takes in a `window_size` parameter and operates on one node:

`out = megatron.transforms.TimeSeries(window_size=5)(X)`

And a `Divide` function from `megatron.transforms.numeric`, which takes in no parameters, and operates on two nodes:

`out = megatron.transforms.Divide()([X, Y])`

## Defining Custom Transformations
A Transformation is one of two things: stateless, or stateful.

- Stateless transformations, when provided the same input at any given time, will always provide the same output. They are dependent on nothing, in terms of context.
- Stateful transformations will provide different output for the same input depending on their parameters; these transformations are "fitted" to data, much like models. An example is a whitening transformation that subtracts the mean and divides by the standard deviation. This Transformation is "fitted" to the data by calculating a mean and standard deviation, and it then uses these statistics for subsequent transformations of future data.

To define a stateless transformation, simply write the function that accepts Numpy data and returns Numpy data, doing whatever arbitrary transformation you wish. Then pass this function to a `megatron.Lambda` wrapper:

```
def add_5(x):
    return x + 5

out = Lambda(add_5)(X)
```

To define a stateful transformation, you must inherit the `megatron.Transformation` class. You are responsible for writing the following methods for your new child class:

- `fit(self, *nodes)`: This should calculate any necessary parameters when passed in Numpy data as `*nodes`. A `self.metadata` dictionary is provided for the storage of these parameters, and should be used.
- `transform(self, *nodes)`: This should transform your Numpy data, given as `*nodes`. You make use of the learned parameters by accessing them in `self.metadata`.

`*nodes` is of course not necessary as a parameter name; it is intended to show that you can have an arbitrary number of parameters, each corresponding to a node's data.

## Using Sklearn Transformations
A wrapper is provided for Sklearn transformations that makes it simple to insert them into the pipeline:

`out = megatron.wrappers.SklearnTransformer(sklearn.StandardScaler())(X)`

That's it.

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
Megatron will, by default, cache the results of an execution of a Node in a space-efficient compressed file. Any Nodes extending from this cached Node will begin from the cached data, rather than re-computing it. If the same pipeline is run twice with the same data, the second execution will simply load from the cache.
Caching can be avoided by executing `Graph.run` with the keyword `cache=False`. Caching does not apply in the case of eager execution.
