# Packing

This doc describes packing, its technical design in AirIO, and a few limitations
to be aware of when packing and mixing datasets. AirIO's packing implementations
can be found in
https://github.com/google/airio/blob/main/airio/_src/pygrain/common/packing.py.

## Packing

Packing is a preprocessing step which concatenates multiple examples with
shorter sequences into a single (“packed”) example to minimize padding. There
are two popular variants of packing:

+   Concat-then-split packing : consecutive examples are concatenated until they
    exceed the desired sequence length, and then trimmed to the sequence length.
    This results in zero padding, but results in incomplete sequences. This is
    ok for masked language modeling.

+   Bin packing: examples are picked and grouped together without trimming to
    minimize padding (but it may not be zero). Packing quality can be improved
    by increasing the pool of examples the preprocessor can pick from, and the
    number of partially packed examples it can maintain, but it requires more
    memory. SeqIO provides a TF op and a custom C++ kernel for true-packing,
    with the C++ kernel providing better packing quality.

These preprocessors are implemented in 3 steps:

+   Batching: Theoretically the packing algorithm could pick examples to pack
    from the entire dataset, but this would require fitting the dataset into
    memory, which is often not feasible. Hence the dataset is first batched to
    create pools of examples to pick from and pack. Larger batch size generally
    leads to better packing but requires more memory. This is a many-to-one
    transformation.

+   Packing: This step applies the packing algorithm to each batch. Based on the
    sequence length of examples in a batch, the number of packed examples
    produced would be different. Thus, the resulting dataset has uneven batches
    of packed examples. This is a map transformation, because it takes a batch
    of examples and outputs a batch (of fewer, packed examples)

+   Un-batching: The final step is to “unbatch” each batch of packed examples.
    Note that each batch could produce a different number of packed examples (1
    <= examples produced <= batch size in the first step). This is a one-to-many
    transformation (flatmap).

Mixing multiple datasets for training is common practice. Users can decide to
pack before or after mixing, depending on whether they want packed examples to
contain sequences from a single dataset (mix after packing) or multiple ones
(mix before packing). Both approaches are fairly common. Supporting mixing after
packing has technical challenges described below.

## Mixing and Packing in AirIO / Grain

AirIO Tasks typically represent a raw data source and a list of preprocessors,
while AirIO Mixtures take a list of AirIO Tasks and mix them at given
proportions. AirIO provides a pure-python user interface and uses Grain to
implement the APIs. Specifically, it uses the Grain to support preprocessing,
global shuffling, repeating, mixing, etc. While Grain provides a simple and
debuggable data loading stack, its default support for flatmap and mixing have
limitations that would result in incorrect behavior when packed datasets are
mixed.

AirIO primarily uses Grain MapDataset, which defines a dataset with efficient
random access support. For transformations that may produce a variable number of
examples (filters and flatmaps), random access is non-trivial. LazyDataset
supports these with the concept of sparse datasets, i.e. some examples of the
dataset may not exist. Implementing filters with this approach is trivial, and
flat maps are supported by requiring users to set the maximum number of examples
that can be produced from a single example. For instance, let’s say you have a
dataset (non-sparse) of 4 examples, and apply a flatmap transformation with
max_fan_out set to 5, and applying the transformation produces 2, 5, 3 and 4
examples from the original dataset. The output of the transformation is a sparse
dataset of size (4 * 5 =) 20, and the actual size of the dataset is (2 + 5 + 3 +
4 =) 14. The final size is unknown until the dataset is either accessed randomly
or iterated upon. Using Grain MapDatsets, packing can be implemented as a series
of a Batch, Map and FlatMap transforms, producing a sparse dataset of packed
examples.

The problem arises when mixing sparse datasets. Mixing packed datasets requires
sampling from sparse datasets at given rates. If the dataset being mixed is
sparse, then the mixing transformation will pick examples that don’t exist in
the final dataset. This means that the actual proportion of examples in the
mixed dataset doesn’t match the desired rate. This deviation would negatively
affect training runs and is an unacceptable tradeoff.

The solution is to convert packed datasets to Grain IterDataset before mixing,
which is an iterable that does not support random access. Packed datasets can be
converted to LazyIterDatsets and then mixed to ensure correct mixing rate. This
is supported out-of-the box by AirIO. The downside is that unlike MapDataset,
IterDataset cannot be globally shuffled.
