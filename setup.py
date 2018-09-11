from setuptools import setup, find_packages

long_description = '''
Megatron is a framework for building computation graphs for
feature engineering in machine learning, with Numpy arrays as the data type.
Use Megatron if you want to:
    - Do feature engineering in a modular and functional way, building up features one step at a time
    - Use disk space to save time by caching feature sets for easy reloading
    - Train feature engineering modules on training data and apply them to testing data
    - Write custom functions for complex transformations, but access built-in functions for quick and common transformations
    - Build feature engineering like you build Keras models (the API is heavily inspired by Keras)
Or any combination of these.
Megatron is distributed under the MIT license.
'''

with open('VERSION') as f:
    version = f.read().strip()

setup(name='Megatron',
      version=version,
      description='A computation graph library for feature engineering with Numpy data',
      long_description=long_description,
      author='Nash Taylor',
      author_email='nashtaylor22@gmail.com',
      url='https://github.com/ntaylorwss/megatron',
      download_url='https://github.com/ntaylorwss/megatron/archive/master.zip',
      license='MIT',
      packages=find_packages(),
      install_requires=[
        'dill',
      ],
      classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
      ])
