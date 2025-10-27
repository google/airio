# AirIO & SeqIO Comparison

This document captures the major user-facing similarities and differences
between AirIO and its predecessor, [SeqIO](https://github.com/google/seqio).

## Evolution

AirIO builds upon the foundations established by SeqIO, the data processing API
that originated in T5X (https://github.com/google-research/t5x). AirIO is
designed as a drop-in replacement for SeqIO.

The motivation behind creating a new API is to support the next generation of
data loading pipelines. AirIO offers functional and performance parity with
greater flexibility and iteration speed. The new API cleanly decouples data
processing from model architectures and addresses many usability issues. It
improves code reusability, reduces fragmentation between frameworks, and speeds
up model development.

New pipelines should use AirIO over SeqIO where possible. Existing SeqIO
pipelines should be migrated using the provided migration guide.

## Major similarities

### Core API

AirIO provides the same core API to users, i.e. the `Task` encapsulating a data
source and a list of preprocessors, a `Mixture` encapsulating a list of `Task`s
and mixing rates, and the `get_dataset` call to load data from `Task`s and
`Mixture`s, built on top of a data loading stack (tf.data for SeqIO, Grain for
AirIO). Ideally, AirIO should feel like SeqIO but significantly simpler and
easier to use (a breath of fresh “air”).

### Components

AirIO provides a familiar DataSource API to configure source data,
implementations of common preprocessors and packing. For non-trivial
preprocessors, unit tests are provided to guarantee functional equivalence
between the SeqIO and AirIO implementations.

## Major differences

### Data loading stack

SeqIO is built on top of tf.data. The core AirIO APIs, i.e. the Task, Mixture,
and preprocessor interface are fully decoupled from the underlying data loading
stack. AirIO currently provides a Grain implementation for a flexible,
research-friendly setup.

### Better determinism

SeqIO datasets are not deterministic - i.e. they may not be reproducible and
recoverable unless carefully designed as such by users. AirIO is deterministic
out-of-the-box via Grain with checkpointing.

### No global registry

The common usage pattern in SeqIO is to add every Task and Mixture to a global
registry (dict) at import time, and retrieve Tasks and Mixtures using their
string name. While this worked really well at a smaller scale, it is a major
pain point for larger projects because of breakages due to transitive imports,
difficulty in tracking task registration code, and registering of O(100K-1M)
tasks to define variants over a combination of params.

AirIO eliminates the global registry. Users can simply define functions that
return Task/Mixture objects, and invoke and configure objects directly in
training libraries. Users are free to define local, restricted registries for
discoverability and reusability.

### No feature converters

With SeqIO, it is common for users to configure a Task to perform model-agnostic
preprocessing (e.g. tokenization) and a FeatureConverter to perform
model-specific preprocessing. Over time, SeqIO FeatureConverters have become
monolithic objects obscuring the underlying preprocessing and have also grown to
encompass other preprocessing steps like packing, etc. that don’t belong in a
“feature converter”.

AirIO replaces FeatureConverters with runtime preprocessors. It allows users to
pass a list of preprocessors to the `get_dataset` call, which are applied to the
dataset in addition to the Task preprocessors. This allows configuring
preprocessors that can vary over runs, e.g. train-specific preprocessors like
packing, trimming, padding, feature conversion, etc. and eval-specific
preprocessors like few-shot prompting, etc.

AirIO also provides helpers that return a list of preprocessors corresponding to
popular SeqIO FeatureConverters in
https://github.com/google/airio/blob/main/airio/_src/pygrain/common/feature_converters.py.

### No magic keywords in preprocessor functions

In SeqIO, preprocessors with arguments named `sequence_length` and
`output_features` would be passed these values at runtime. This approach has
been modified in AirIO to minimize magic keywords and collect these args under
a descriptive name to minimize errors. To use runtime arguments, users can add
an argument to preprocessors of type
`airio.preprocessors.AirIOInjectedRuntimeArgs`, which is provided by AirIO
during runtime. This dataclass has common runtime args like `sequence_lengths`
and `split`. The name of the argument does not matter as long as it is correctly
type-annotated as `airio.preprocessors.AirIOInjectedRuntimeArgs`. For instance:

```python
def pad(
    ex: Dict[str, Any],
    runtime_args: preprocessors_lib.AirIOInjectedRuntimeArgs,
    pad_id: int,
):
  seq_lens = runtime_args.sequence_lengths
  return  {k: _pad(v, seq_lens[k], pad_id) for k, v in ex.items()}
```

In the above example, `pad_id` must be set when defining the Task (e.g.
`functools.partial(pad, pad_id=0)`, and `runtime_args` is passed by
AirIO during execution.

### No metrics

SeqIO supports configuring metrics in Tasks that can be used to evaluate models
on SeqIO Tasks. This is out of scope for AirIO.

While SeqIO Task metrics have the benefits of creating reusable and comparable
eval specs, standardizing metric implementations, etc. the evaluation
orchestration has to be coupled with the higher-level library used to run
inference and is hard to maintain and extend.
