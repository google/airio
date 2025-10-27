# AirIO: Task-based datasets, preprocessing, and evaluation for sequence models.

🌬️ *An **A**daptive **I**nterface for **R**esearch **I/O***

**AirIO** is a library for loading, processing and feeding multimodal data into
sequence models. It provides simple APIs to write reusable specifications
encapsulating data loading and transformation steps in training, inference and
evaluation. AirIO supports a variety of storage formats, e.g. SSTable, and
services, e.g. TFDS. It is fully compatible with frameworks such as Jax and
TensorFlow. It is built using Grain while providing a decoupled and extensible
core API.

Here are a few important points about AirIO:

+   It is an evolution of the SeqIO library (https://github.com/google/seqio),
    with the main motivation of supporting the next generation of data loading
    pipelines.
+   It offers functional and performance parity with greater flexibility and
    iteration speed.
+   The API cleanly decouples data processing from model architectures and
    addresses many usability issues.
+   It improves code reusability, reduces fragmentation between frameworks, and
    speeds up model development.

## Installation
### From source

```
git clone https://github.com/google/airio.git
cd airio
pip install -e .
```

## Philosophy

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

## How does AirIO relate to SeqIO? {#seqio}

AirIO builds upon the foundations established by SeqIO to support the next
generation of data loading pipelines, offering functional and performance parity
with greater flexibility and iteration speed.

***Same core API***. AirIO provides the same core API to users, i.e. the `Task`
encapsulating a data source and a list of preprocessors, a `Mixture`
encapsulating a list of `Task`s and mixing rates, and the `get_dataset` call to
load data from `Task`s and `Mixture`s, built on top of a dataloading stack
(tf.data for SeqIO, Grain for AirIO). Ideally, AirIO should feel like SeqIO but
significantly simpler and easier to use.

***Different data loading stack***. SeqIO is built on top of tf.data. The core
AirIO APIs are fully decoupled from the underlying data loading stack. AirIO
currently provides a Grain implementation for a flexible, research-friendly
setup.

***SeqIO++***. AirIO makes simplifications to SeqIO design and usage patterns
that do not scale well, including eliminating the global registry, removing
feature converters, and excluding evaluation orchestration from the core
library.

## Should I use the core API or AirIO-Grain?

The core API contains protocols to ensure consistency across dataloaders. Most
users will interact with `airio.pygrain`. In rare cases, or when using AirIO
with a custom dataloader, you may need to import `airio.core`.

## When should I use Grain directly? {#grain}

AirIO aims to combine the usability of the SeqIO library with the functionality
of Grain, e.g. out-of-the-box determinism, etc.). AirIO could be useful to your
usecase because it (1) helps you make your SeqIO setups Grain-compatible (2)
provides useful abstractions and components to build, reuse and iterate on your
dataset specifications, and (3) provides flexibility of choice between
dataloading libraries.

If the above features are not essential for you and/or you already have a
Grain-based setup, then using Grain directly could be a good choice. Most AirIO
components, including mixing, packing, feature converters and dataset iterators,
can be used directly with Grain.

## What kind of preprocessors can I use with AirIO? {#preprocessors}

AirIO supports all `grain.Transformation`s as preprocessors. In addition, AirIO
provides helpers to transform simple python map and filter functions into
`grain.Transformation`s, and adds several important orchestration features on
top, e.g. passing unique and reproducible seeds to stochastic preprocessors,
injecting args (like sequence lengths) to preprocessors at runtime, etc.
