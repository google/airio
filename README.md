# AirIO: Task-based datasets, preprocessing, and evaluation for sequence models.

**AirIO** is the next generation of
[SeqIO](http://github.com/google/seqio): a library for processing
sequential data to be fed into downstream sequence models. It uses a variety of
data loaders to create scalable data pipelines. With one line of code, the
returned dataset can be transformed to a numpy iterator and hence it is fully
compatible with other frameworks such as [JAX](https://github.com/google/jax) or
[PyTorch](https://pytorch.org/).

The main improvements over SeqIO are:

*   Clear abstractions
    *   Agnostic encapsulation over data loading and processing steps
    *   Compatible with Grain, tf.data, etc.
*   Better interfacing with other components
    *   Simple bridges to smooth decoupling
    *   Clear boundary with evaluation libraries
    *   Ability to combine a variety of data formats
*   Data pipelines are verifiable
    *   Plug in inspection and visualization tools
    *   Ability to set up tests
*   Better design patterns
    *   No global state
    *   Composition over inheritance
    *   Loose coupling with data, eval, and other API layers

Task definitions are compatible with SeqIO.

