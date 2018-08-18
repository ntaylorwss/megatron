# Megatron: Computation Graphs for Feature Engineering in Machine Learning

Megatron is a library that facilitates fully customizable pipelines for feature engineering by implementing the process as a directed acyclic graph. Based in Numpy, it provides a comprehensive suite of transformations, as well as the ability to apply user-defined transformations, provided they accept and return particular data types, which are listed below.

## Data Types
Feature transformers act on and return data only in the following types, which aim to cover all forms of features one might use in a machine learning model:

- NumericFeature: A single column of numeric data; a 1-dimensional array.
- FeatureSet: Multiple related but independent columns of numeric data; a 2-dimensional array.
- Image: A representation of an image in either greyscale, RGB, or RGBA format; a 3-dimensional array.
- Text: A single column of text data; a 1-dimensional array.
- TimeSeries (in progress): Multiple columns of numeric data forming a time series; a 2-dimensional array.
