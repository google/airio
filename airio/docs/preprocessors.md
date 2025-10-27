# Preprocessors

## Preprocessors

With AirIO, data points are dicts (possibly nested) of numpy arrays.
Most preprocessors are map or filter functions that are applied over the
dataset. AirIO configures simple numpy-based functions as Task preprocessors
using these wrappers:

+   `airio.MapFnTransform`
+   `airio.RandomMapFnTransform`
+   `airio.FilterFnTransform`

These are similar to SeqIO’s `map_over_dataset` utility.

Note: A preprocessor can often be decomposed into a list of simpler
preprocessors, e.g. the popular `seqio.EncDecFeatureConverter` is composed of
packing (supported out-of-the-box), trimming (map), padding (map) and a feature
conversion step (map), implemented by the
`get_t5x_enc_dec_feature_converter_preprocessors` method in
https://github.com/google/airio/blob/main/airio/_src/pygrain/common/feature_converters.py.
Your preprocessors list can be extended with lists of preprocessors representing
a composite preprocessor.

## Runtime Arguments

Sometimes a preprocessor needs arguments passed during runtime, e.g. a padding
preprocessor needs to know the sequence length to pad to, which could vary for
every execution and is configured at runtime.

To use runtime arguments, add an argument to your preprocessor of type
`airio.AirIOInjectedRuntimeArgs`, which is provided by AirIO during runtime.
This dataclass has common runtime args like `sequence_lengths` and `split`. The
name of the argument does not matter; it must be correctly type-annotated as
`airio.AirIOInjectedRuntimeArgs`. For instance:

```python
def pad(
    ex: Dict[str, Any],
    runtime_args: airio.AirIOInjectedRuntimeArgs,
    pad_id: int,
):
  seq_lens = runtime_args.sequence_lengths
  return  {k: _pad(v, seq_lens[k], pad_id) for k, v in ex.items()}
```

In the above example, the `pad_id` arg has to be set when defining the Task
(e.g. `functools.partial(pad, pad_id=0)`, the `runtime_args` arg will be passed
by AirIO during execution.

Note: In SeqIO, preprocessors with arguments named `sequence_length` and
`output_features` would be passed these values at runtime. This approach has
been modified in AirIO to minimize magic keywords and collect these args under
a descriptive name to minimize errors.

## Runtime preprocessors

AirIO allows passing a list of preprocessors called `runtime_preprocessors` to
the `get_dataset` call, which are applied to the dataset in addition to the
`Task` preprocessors. This allows configuring train-specific preprocessors like
packing, trimming, padding, feature conversion, etc. eval-specific preprocessors
like few-shot prompting, etc. that can vary over runs.

Note: With SeqIO, it was common for users to configure a Task to perform
model-agnostic preprocessing (e.g. tokenization) and a FeatureConverter to
perform model-specific preprocessing. AirIO replaces FeatureConverters with
runtime preprocessors for better usability.

## Common preprocessor implementations

**General preprocessors**: Common preprocessors like trimming, padding and
rekeying are provided in
https://github.com/google/airio/blob/main/airio/_src/pygrain/common/preprocessors.py.
Configure relevant args using `functools.partial` and use the appropriate
wrapper, e.g. `airio.MapFnTransform` to pass these as preprocessors. e.g.

```python
pad_fn = functools.partial(preprocessors.pad, pad_id=pad_id)
prep = airio.MapFnTransform(pad_fn)
```

**Packing**: AirIO provides common implementations of packing out-of-the-box in
https://github.com/google/airio/blob/main/airio/_src/pygrain/common/packing.py.
For first-fit packing, pass `SingleBinTruePackIterPreprocessor` or
`MultiBinTruePackIterPreprocessor` to `preprocessors` in your `Task` or
runtime_preprocessors in `get_dataset`. For concat-then-split packing, use
`NoamPackMapPreprocessor`. Note that it's important to shuffle after
concat-then-split packing, because it potentially splits examples into adjacent
packed examples (AirIO always shuffles after applying all preprocessors).

Here is how these map to packing implementations in SeqIO:

+   `SingleBinTruePackIterPreprocessor` is the exact functional equivalent of
    setting `pack=True` and `use_custom_ops=False` in a
    `seqio.FeatureConverter`, which is first-fit packing with a single
    partially-packed example.
+   `MultiBinTruePackIterPreprocessor` is the exact functional equivalent of
    setting `pack=True` and `use_custom_ops=True` in a `seqio.FeatureConverter`,
    which is first-fit packing with upto 1000 partially-packed examples.
+   `NoamPackIterPreprocessor` is the exact functional equivalent of running
    `t5.data.preprocessors.reduce_concat_tokens` and
    `t5.data.preprocessors.split_tokens`, which is concat-and-split packing,
    also commonly referred to as Noam-packing.

**Feature Converters**: AirIO provides numpy implementations of commonly used
SeqIO feature converters in
https://github.com/google/airio/blob/main/airio/_src/pygrain/common/feature_converters.py.
Instead of monolithic objects that obscure the underlying preprocessing steps,
these are implemented as lists of simpler map fns like trimming, padding, and
feature conversion steps, which can be passed to `runtime_preprocessors` in the
`get_dataset` call (or less commonly, as `preprocessors` of a `Task`). Use the
following helper corresponding to the desired FeatureConverter:

+   `EncDecFeatureConverter`: `get_t5x_enc_dec_feature_converter_preprocessors`
+   `LMFeatureConverter`: `get_t5x_lm_feature_converter_preprocessors`
+   `PrefixLMFeatureConverter`:
    `get_t5x_prefix_lm_feature_converter_preprocessors`
