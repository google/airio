# AirIO: Task-based datasets, preprocessing, and evaluation for sequence models.


**AirIO** is a library for loading, processing and feeding multimodal data into
sequence models. It provides simple APIs to write reusable specifications
encapsulating data loading and transformation steps in training, inference and
evaluation. AirIO supports a variety of storage formats, e.g. SSTable, and
services, e.g. TFDS, and a variety of data loaders, e.g. Grain and tf.data. It
is fully compatible with frameworks such as Jax and TensorFlow.

The following are guiding principles for AirIO development:

*   Clear abstractions
    *   Agnostic encapsulation over data loading and processing steps
    *   Compatible with Grain, tf.data, etc.
*   Clear interfaces with other components
    *   Clear boundary with evaluation libraries
    *   Ability to combine a variety of data formats
    *   Simple bridges to smooth decoupling
*   Verifiable data pipelines
    *   Plug in inspection and visualization tools
    *   Easy path to setting up tests
*   Good software design patterns
    *   No global state
    *   Composition over inheritance
    *   Loose coupling with data, eval, and other API layers

## Installation

### From source

```
git clone https://github.com/google/airio.git
cd airio
pip install -e .
```

