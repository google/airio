# SeqIO Migration

This document provides instructions to migrate SeqIO Tasks and Mixtures to
AirIO.

## General note on Grain migration

If you’re migrating from SeqIO to AirIO, you will transition the underlying data
engine used (and hence, dataset iterator type) from tf.data to Grain. Beyond
that, your setup may still be using TensorFlow in the following ways:

+   Source data may be stored as tf.Examples protos
+   Source data may be stored TFRecord files
+   Source data may be parsed as tf.Tensors
+   TF-ops (not tf.data ops) may be used in preprocessors to manipulate the
    data.

Grain only requires you to convert source files to ArrayRecord; all other
TensorFlow components in your setup can run with Grain in eager mode and can
stay unchanged, allowing for a quicker migration. If the TensorFlow dependency
is undesirable, you can parse your tf.Example protos as numpy arrays without
importing TensorFlow, and replace the TF ops in your preprocessors with numpy /
JAX operations. You can also move away from tf.Example protos if you own the
pipeline that writes your source data to files.

It is up to you to make a decision on which TensorFlow components to make
TF-less. Wherever applicable, we provide instructions for both approaches in
this migration guide. Additionally, we provide an end-to-end migration example
of preserving TensorFlow components and one of fully moving away from
TensorFlow.

## Converting a `seqio.Task` to an `airio.GrainTask`

### Change the import

Change `import seqio` to `import airio.pygrain as airio`.

### Convert the data source

Follow the instructions below based on the SeqIO DataSource you are using:

+   `seqio.TfdsDataSource`: Change to `airio.TfdsDataSource`. The arguments are
    unchanged. The TF-less version of TFDS is used, which returns the data as
    numpy arrays (and also eliminates the TF dependency).

    Note: `airio.TfdsDataSource` only supports TFDS datasets stored in the
    ArrayRecord format.

+   `seqio.TFExampleDataSource`, `seqio.ProtoDataSource`,
    `seqio.FileDataSource`: Change to
    `airio.ArrayRecordDataSource(split_to_filepattern)`. Remove all other args.

    A major change from SeqIO for file-based data sources is that AirIO
    returns raw bytes for each record. Users must configure an appropriate parse
    fn as the first preprocessor in their Task. If your data is not stored in
    tf.Example, then pass your `decode_proto_fn` to the Task preprocessors as
    shown in the example below:

    ```python
    seqio.Task(
        "name",
        source=seqio.ProtoDataSource(split_to_filepattern, decode_proto_fn),
        output_features=[...]
        preprocessors=[...],
    )
    ```

    becomes:

    <pre><code class="lang-python">
    <ins>airio.GrainTask(</ins>
        "name",
        source=seqio.ProtoDataSource(split_to_filepattern<del>, decode_proto_fn</del>),
        preprocessors=[<ins>airio.MapFnTransform(decode_proto_fn), </ins>...],
    )
    </code></pre>

    If your data is stored in `tf.Example`s you can parse them into
    `tf.Tensor`s, which allows you to continue using TF ops in your
    preprocessors. For example:

    ```python
    feature_description = {"text": tf.io.FixedLenFeature([], tf.string)}
    seqio.Task(
        "name",
        source=seqio.TFExampleDataSource(split_to_filepattern, feature_description),
        preprocessors=[...],
    )
    ```

    becomes:

    <pre><code class="lang-python">
    <ins>def parse_fn(pb):</ins>
      feature_description = {"text": tf.io.FixedLenFeature([], tf.string)}
      <ins>return tf.io.parse_single_example(pb, feature_description)</ins>

    <ins>airio.GrainTask(</ins>
        "name",
        source=airio.ArrayRecordDataSource(split_to_filepattern,
        preprocessors=[
            <ins>airio.MapFnTransform(parse_fn),</ins>
            ...
        ],
    )
    </code></pre>

    If you'd like to avoid the TF dependency and convert the TF ops in your
    preprocessors to numpy and/or jax operations, then you can use Grain's
    `fast_proto_parser` to parse the tf.Examples as numpy arrays without
    introducing a TF dependency. For example:

    ```python
    feature_description = {"text": tf.io.FixedLenFeature([], tf.string)}
    seqio.Task(
        "name",
        source=seqio.TFExampleDataSource(split_to_filepattern, feature_description),
        preprocessors=[...],
    )
    ```

    becomes:

    <pre><code class="lang-python">
    <del>feature_description = {"text": tf.io.FixedLenFeature([], tf.string)}</del>
    <ins>import grain.python as grain</ins>

    <ins>airio.GrainTask(</ins>
        "name",
        source=airio.ArrayRecordDataSource(split_to_filepattern,
        preprocessors=[
            <ins>airio.MapFnTransform(</ins>
                <ins>grain.fast_proto.parse_tf_example</ins>
            <ins>),</ins>
            ...
        ],
    )
    </code></pre>

    The `reader_cls` is redundant because only ArrayRecord files are supported.
    The `file_shuffle_buffer_size`, `cycle_length` and `block_length` args are
    not supported.

+   `seqio.FunctionDataSource`: Change to `airio.FunctionDataSource`. The
    arguments are unchanged. Update your `dataset_fn` to return a sequence of
    dicts of numpy arrays instead of a `tf.data.Dataset`. In most cases you can
    easily convert your dataset by applying the following map:

    ```python
    def to_numpy(d):
      return tf.nest.map_structure(lambda x: x.numpy(), d)
    ds = ds.map(to_numpy)
    ```

    Remove the `shuffle_files` and `seed` args from the `dataset_fn` you pass to
    the `FunctionDataSource`, and remove any shuffling done in the dataset fn.
    Buffer-based shuffling done by tf.data is not required because Grain samples
    from the data source using a globally shuffled index.

    Unlike SeqIO, the AirIO `FunctionDataSource` is backed by an in-memory
    Grain data source which materializes the entire dataset in-memory for
    random access. Therefore, you may run into memory issues after migration; in
    such cases, write your data out to file and use a file-based DataSource.

### Convert the preprocessors

Note: In general, if your preprocessors use TF ops, you can continue to use them
in eager mode in Grain. If you want to remove the TF dependency, then you can
replace the TF ops in your preprocessors with numpy or jax operations.

#### Functions annotated with `seqio.map_over_dataset`

+   For **non-stochastic functions**, i.e. `num_seeds=None`, remove the
    `seqio.map_over_dataset` annotation from the function definition and wrap
    the function with `airio.MapFnTransform` in the Task definition. e.g.

    ```python
    @seqio.map_over_dataset
    def my_fn(ex):
      ...

    Task(..., preprocessors=[my_fn], ...)
    ```

    becomes:

    <pre><code class="lang-python">
    <del>@seqio.map_over_dataset</del>
    def my_fn(ex):
      ...

    <del>Task(..., preprocessors=[my_fn], ...)</del>
    <ins>GrainTask(..., preprocessors=[airio.MapFnTransform(my_fn)], ...)</ins>
    </code></pre>

+   For **stochastic functions with a single seed**, i.e. `num_seeds=1`, remove
    the `seqio.map_over_dataset` annotation from the function definition and
    wrap the function with `airio.RandomMapFnTransform` in the Task definition.
    e.g.

    ```python
    @seqio.map_over_dataset(num_seeds=1)
    def my_fn(ex, seed):
      ...

    Task(..., preprocessors=[my_fn], ...)
    ```

    becomes:

    <pre><code class="lang-python">
    <del>@seqio.map_over_dataset(num_seeds=1)</del>
    def my_fn(ex, seed):
      ...

    <del>Task(..., preprocessors=[my_fn], ...)</del>
    <ins>GrainTask(..., preprocessors=[airio.RandomMapFnTransform(my_fn)], ...)</ins>
    </code></pre>

+   For **stochastic functions with multiple seeds**, i.e. `num_seeds=N`, remove
    the `seqio.map_over_dataset` annotation from the function definition, update
    your function to take a single `seed` instead of multiple `seeds`, use
    `tf.random.split` (or `jax.random.split` if you'd like to move away from TF)
    to split it into multiple seeds, and wrap the function with
    `airio.RandomMapFnTransform` in the Task definition. e.g.

    ```python
    @seqio.map_over_dataset(num_seeds=N)
    def my_fn(ex, seeds):
      ...

    Task(..., preprocessors=[my_fn], ...)
    ```

    becomes:

    <pre><code class="lang-python">
    <del>@seqio.map_over_dataset(num_seeds=N)</del>
    def my_fn(ex, seed):
      <ins>seeds = tf.random.split(seed, N)</ins>
      <ins># OR</ins>
      <ins>seeds = jax.random.split(seed, N)</ins>
      ...

    <del>Task(..., preprocessors=[my_fn], ...)</del>
    <ins>GrainTask(..., preprocessors=[airio.RandomMapFnTransform(my_fn)], ...)</ins>
    </code></pre>

    Preprocessors always taking a single seed, and optionally splitting into
    multiple seeds if needed, allows for a simpler and more stable API.

Note: Applying the wrapper during Task definition instead of the function
definition allows testing the function directly instead of using the transformed
function.

#### Filter preprocessors

If you have a preprocessor that filters your dataset, define the filter function
for a single example and wrap the function with `airio.FilterFnTransform` in the
Task definition. e.g.

```python

def my_filter(dataset):

  def my_filter_fn(ex):
    ...

  return dataset.filter(my_filter_fn)

Task(..., preprocessors=[my_filter], ...)
```

becomes:

<pre><code class="lang-python">
<del>def my_filter(dataset):</del>

def my_filter_fn(ex):
  ...

<del>return dataset.filter(my_filter_fn)</del>

<del>Task(..., preprocessors=[my_filter], ...)</del>
<ins>GrainTask(..., preprocessors=[airio.FilterFnTransform(my_filter_fn)], ...)</ins>
</code></pre>

#### Grain transformations

Any `grain.Transformation` instance can be directly passed to your Task's
preprocessors list.

#### tf.data transformations

If your preprocessor is a `tf.data` transformation, i.e. takes a
`tf.data.Dataset` and produces a `tf.data.Dataset`, simplify it into one or more
map, random map or filter functions and configure them using the airio function
wrappers described above. If this is not possible, try converting your
transformation into a `grain.Transformation`.

### Remove registration

AirIO doesn't support the Task and Mixture registries. Instead of adding your
Task to the TaskRegistry, define a function that returns the Task instance.
Then, instead of configuring a Task name to look up the Task from the registry,
invoke the function and configure the Task instance directly. e.g.

```python
seqio.TaskRegistry.add(
    "my_task_name",
    source=seqio.TfdsDataSource(...),
    preprocessors=[...],
    output_features=...,
    metric_fns=[...])
```

becomes:

<pre><code class="lang-python">
<del>seqio.TaskRegistry.add(</del>
<ins>def get_my_task():</ins>
  <ins>return airio.GrainTask(</ins>
      "my_task_name",
      source=airio.TfdsDataSource(...),
      preprocessors=[...]
      <del>output_features=...,</del>
      <del>metric_fns=[...]</del>)

</code></pre>

It’s common to loop over a set of params and register multiple variants of a
Task. Here’s how to migrate these to the new pattern:

```python
for a, b, c in itertools.product(A, B, C):
  ... setup params ...
  task_name = get_task_name(a, b, c)
  seqio.TaskRegistry.add(
      task_name,
      source=source,
      preprocessors=[...],
      output_features=output_features,
      metric_fns=[...])
```

becomes:

<pre><code class="lang-python">

<ins>def get_task(a, b, c):</ins>
  ... setup params ...
  task_name = get_task_name(a, b, c)
  <del>seqio.TaskRegistry.add(</del>
  <ins>return airio.GrainTask(</ins>
      task_name,
      source=source,
      preprocessors=[...],
      <del>output_features=...,</del>
      <del>metric_fns=[...]</del>))

for a, b, c in itertools.product(A, B, C):
  task = get_task(a, b, c)
</code></pre>

### Remove metrics

If you configure `metric_fns` or `metric_objs` in your Task, remove these. AirIO
doesn't support metrics.

### Convert FeatureConverters

If you were configuring a `feature_converter` to pass to the `seqio.get_dataset`
call, replace this with a list of `runtime_preprocessors`. The
`feature_converters` module in `airio.pygrain_common` provides helpers to return
a list of preprocessors corresponding to canonical SeqIO FeatureConverters.

For T5X:

+   `EncDecFeatureConverter` ->
    `get_t5x_enc_dec_feature_converter_preprocessors`
+   `LMFeatureConverter` -> `get_t5x_lm_feature_converter_preprocessors`
+   `PrefixLMFeatureConverter` ->
    `get_t5x_prefix_lm_feature_converter_preprocessors`

For example:

```python
feature_converter = seqio.PrefixLMFeatureConverter(
    pack=True,
    use_custom_packing_ops=False,
    apply_length_check=True,
    bos_id=0,
    loss_on_target_only=True,
    passthrough_features=[...],
)
```

becomes

<pre><code class="lang-python">
import airio.pygrain_common

<del>feature_converter = seqio.PrefixLMFeatureConverter(</del>
<ins>runtime_preprocessors = (
    airio.pygrain_common.get_t5x_prefix_lm_feature_converter_preprocessors(</ins>
        pack=True,
        <ins>use_multi_bin_packing</ins>=False,
        <del>apply_length_check=True,</del>
        bos_id=0,
        <ins>pad_id=0,</ins>
        loss_on_targets_only=True,
        passthrough_feature_keys=[...],
    )
)
</code></pre>

The `apply_length_check` arg is also removed; longer sequences are allowed and
will be trimmed. AirIO also requires users to pass the `pad_id` (and `bos_id`)
explicitly to avoid mistakes (these are commonly set to 0).

Note: AirIO's packing implementation is functionally exactly equivalent to
SeqIO's packing implementation. Change `use_custom_packing_ops` to
`use_multi_bin_packing`. `use_custom_packing_ops=False` does first-fit packing
with a single partially-packed example. `use_custom_packing_ops=True` does
first-fit packing with upto 1000 partially-packed examples, which improves
packing quality but requires possibly checkpointing many partially-packed
examples.

Note: The SeqIO DecoderFeatureConverter is a wrapper over the LM and PrefixLM
FeatureConverters that applies PrefixLM if the "inputs" feature is present in
examples, and applies the LM FeatureConverter otherwise. AirIO doesn't provide
this wrapper and instead encourages users to configure the appropriate feature
converter explicitly to avoid mistakes.

### [Optional] Convert runtime args (`sequence_length`)

If your preprocessor uses the `sequence_length` arg, which is passed during
runtime, update your preprocessor to take an arg of type
`airio.AirIOInjectedRuntimeArgs` instead. The `airio.AirIOInjectedRuntimeArgs`
dataclass is provided by AirIO during runtime and has common runtime args like
`sequence_lengths` and `split`. The name of the argument does not matter; it
must be correctly type-annotated as `airio.AirIOInjectedRuntimeArgs`. e.g.

```python
@seqio.map_over_dataset
def my_fn(ex, sequence_length):
  ...

Task(..., preprocessors=[my_fn], ...)
```

becomes:

<pre><code class="lang-python">
<del>@seqio.map_over_dataset</del>
def my_fn(ex, <del>sequence_length</del> <ins>runtime_args: airio.AirIOInjectedRuntimeArgs</ins>):
  <ins>sequence_length = runtime_args.sequence_lengths</ins>
  ...

<del>Task(..., preprocessors=[my_fn], ...)</del>
<ins>GrainTask(..., preprocessors=[airio.MapFnTransform(my_fn)], ...)</ins>
</code></pre>

Note: AirIO uses the `airio.AirIOInjectedRuntimeArgs` dataclass instead of
arguments named `sequence_length` to minimize magic keywords and collect these
args under a descriptive name to minimize errors.

### [Optional] Convert `output_features` and tokenization

If you set `output_features` in your Task for tokenization and other
preprocessors, update them as follows:

1.  **Convert your vocabulary**: If you're using
    `seqio.SentencepieceVocabulary`, convert it to
    `airio.SentencepieceVocabulary`. The args remain unchanged.

2.  **Convert your output_features**: Convert your `Mapping[str, seqio.Feature]`
    to `Mapping[str, airio.TokenizerConfig]`. `seqio.Feature` and
    `airio.TokenizerConfig` have the same fields except for `dtype` and `rank`,
    which are no longer needed.

3.  **Create a Tokenizer**: Create an `airio.Tokenizer` instance with the
    tokenizer configs. Wrap the tokenizer instance with `airio.MapFnTransform`
    in your Task preprocessors. You can also pass `copy_pretokenized` (True by
    default). Unlike SeqIO, the `with_eos` option is no longer required. EOS
    token is added if `add_eos` is set to `True` in the TokenizerConfig for the
    feature; hence, both `seqio.tokenize` and `seqio.tokenize_and_append_eos`
    can be replaced by the same airio.Tokenizer instance. Note that the default
    value for `add_eos` in `TokenizerConfig` is True; please be sure to set it
    to False if needed.

4.  **Remove `output_features`** from your Task. AirIO decouples tokenization
    from the Task instance.

Combining all steps, the following:

```python
vocab = seqio.SentencepieceVocabulary(...)
output_features = {
    "feature": seqio.Feature(
        vocab, add_eos=..., dtype=..., rank=..., required=...
    )
}
Task(..., preprocessors=[seqio.tokenize], output_features=output_features)
```

becomes:

<pre><code class="lang-python">
vocab = <del>seqio</del> <ins>airio</ins>.SentencepieceVocabulary(...)
<del>output_features = { ... } </del>
<ins>tokenizer_configs = {"feature": airio.TokenizerConfig(vocab, add_eos=...)}
tokenizer = airio.Tokenizer(tokenizer_configs)</ins>
<del>Task(..., preprocessors=[seqio.tokenize], output_features=output_features)</del>
<ins>GrainTask(..., preprocessors=[airio.MapFnTransform(tokenizer)])</ins>
</code></pre>

If you have other preprocessors that take the `output_features` arg, which is
passed by SeqIO during runtime, update the preprocessor to use a `Mapping[str,
airio.TokenizerConfig]`. This is a no-op unless your preprocessor uses the
`rank` and/or `dtype` fields, which can instead be inferred directly from the
feature, e.g. `dtype=ex[feature_key].dtype`). Then pass the tokenizer configs to
your processor when defining the Task.

```python
@seqio.map_over_dataset
def my_fn(ex, output_features):
  ...

vocab = seqio.SentencepieceVocabulary(...)
output_features = {
    "feature": seqio.Feature(
        vocab, add_eos=..., dtype=..., rank=..., required=...
    )
}
Task(..., preprocessors=[my_fn], output_features=output_features)
```

becomes:

<pre><code class="lang-python">
<del>@seqio.map_over_dataset</del>
def my_fn(ex, <del>output_features</del> <ins>tokenizer_configs</ins>):
  ...

vocab = <del>seqio</del> <ins>airio</ins>.SentencepieceVocabulary(...)
<del>output_features = { ... } </del>
<ins>tokenizer_configs = {"feature": airio.TokenizerConfig(vocab, add_eos=...)}
transform = functools.partial(my_fn, tokenizer_configs=tokenizer_configs)</ins>
<del>Task(..., preprocessors=[my_fn], output_features=output_features)</del>
<ins>GrainTask(..., preprocessors=[airio.MapFnTransform(transform)])</ins>
</code></pre>

### [Optional] Remove CacheDatasetPlaceholder

Remove `seqio.CacheDatasetPlaceholder` from your Task preprocessors. AirIO
supports determinism on-the-fly via Grain and hence, doesn't support
materializing Tasks.

## Converting a `seqio.Mixture` to an `airio.GrainMixture`

### Convert each Task

Convert `seqio.Task` to `airio.GrainTask` using the instructions above.

### Update your Mixture

Convert your `seqio.Mixture` to `airio.GrainMixture`. A few things to note:

+   Pass a list of `airio.GrainTask` instances instead of Task names.
+   AirIO doesn't support `default_rate`; instead pass an explicit rate for each
    Task to avoid mistakes. e.g. `Mixture(tasks, default_rate=0.5)` becomes
    `GrainMixture(tasks, [0.5]*len(tasks))`

### Remove registration

See the instructions in the earlier section to migrate away from the global
registry.

## Advanced use cases

### Packing

AirIO provides common implementations of packing out-of-the-box in
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

### Task Migration

In this example, we use PyGrain's `fast_proto_parser` to load tf.Examples from
file into numpy arrays, and update the preprocessors to process numpy arrays.
The following SeqIO Task definition:

```python
import seqio
import tensorflow as tf

def register_nqo_v001_task() -> None:
  # source
  file_path = ".../natural_questions_open-train.array_record*"
  feature_description = {
      "question": tf.io.FixedLenFeature([], tf.string),
      "answer": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
  }
  source = seqio.TFExampleDataSource(
      {"train": file_path},
      feature_description=feature_description,
      reader_cls=ArrayRecordDataset,
  )

  # tokenizer
  vocab_path = ".../sentencepiece.model"
  extra_ids = 100
  vocab = seqio.SentencePieceVocabulary(vocab_path, extra_ids)
  tokenizer_configs = {
      "inputs": seqio.Feature(vocabulary=vocab),
      "targets": seqio.Feature(vocabulary=vocab),
  }

  # preprocessor
  @seqio.map_over_dataset
  def question(ex: dict[str, str]) -> dict[str, str]:
    return {
        "inputs": "nq question: " + ex["question"],
        "targets": tf.strings.reduce_join(ex["answer"], separator=", "),
    }

  seqio.TaskRegistry.add(
      name="dummy_task",
      source=source,
      preprocessors=[
          question,
          seqio.preprocessors.tokenize,
      ],
      output_features=tokenizer_configs,
  )
```

becomes:

<pre><code class="lang-python">
<del>import seqio</del>
<del>import tensorflow as tf</del>
<ins>import airio.pygrain as airio</ins>
<ins>import grain.python as grain</ins>

<del>def register_nqo_v001_task() -> None:</del>
<ins>def get_nqo_v001_task() -> airio.GrainTask:</ins>
  # source
  file_path = ".../natural_questions_open-train.array_record@1"
  <del>feature_description = {</del>
      <del>"question": tf.io.FixedLenFeature([], tf.string),</del>
      <del>"answer": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),</del>
  <del>}</del>
  <del>source = seqio.TFExampleDataSource(</del>
  <ins>source = airio.ArrayRecordDataSource(</ins>
      {"train": file_path},
      <del>feature_description=feature_description,</del>
      <del>reader_cls=ArrayRecordDataset,</del>
  )

  # tokenizer
  vocab_path = ".../sentencepiece.model"
  extra_ids = 100
  vocab = <ins>airio</ins>.SentencePieceVocabulary(vocab_path, extra_ids)
  tokenizer_configs = {
      "inputs": <ins>airio.TokenizerConfig(vocab=vocab)</ins>,
      "targets": <ins>airio.TokenizerConfig(vocab=vocab)</ins>,
  }

  # preprocessor
  <del>@seqio.map_over_dataset</del>
  def question(ex: dict[str, str]) -> dict[str, str]:
    return {
        "inputs": "nq question: ".encode() + ex["question"]<ins>[0],  # everything is parsed as an array</ins>
        <del>"targets": tf.strings.reduce_join(ex["answer"], separator=", "),</del>
        <ins>"targets": ", ".encode().join(ex["answer"]),  # tf-less</ins>
    }


  <del>seqio.TaskRegistry.add(</del>
  <ins>return airio.GrainTask(</ins>
      name="dummy_airio_task",
      source=source,
      preprocessors=[
          <ins>airio.MapFnTransform(</ins>
              <ins>grain.fast_proto.parse_tf_example,  # parse as numpy arrays</ins>
          <ins>),</ins>
          <ins>airio.MapFnTransform(</ins>question<ins>)</ins>,
          <del>seqio.preprocessors.tokenize,</del>
          <ins>airio.MapFnTransform(</ins>
              <ins>airio.Tokenizer(</ins>
                  <ins>tokenizer_configs=tokenizer_configs,</ins>
                  <ins>copy_pretokenized=True,</ins>
              <ins>)</ins>
          <ins>),</ins>
      ],
      <del>output_features=tokenizer_configs,</del>
  )
</code></pre>

Here is how the code to load data from the SeqIO Task would change:

```python
register_nqo_v001_task()
task = seqio.get_mixture_or_task("dummy_task")
feature_converter = seqio.PrefixLMFeatureConverter(
    pack=True,
    use_custom_packing_ops=True,
    apply_length_check=False,
    bos_id=0,
    loss_on_targets_only=True,
    passthrough_features=None,
)
ds = seqio.get_dataset(
    task,
    task_feature_lengths={"inputs": 32, "targets": 32},
    dataset_split="train",
    shuffle=True,
    seed=42,  # arbitrary choice
    feature_converter=feature_converter,
    batch_size=8,
)
print(next(iter(ds)))
```

becomes:

<pre><code class="lang-python">
import airio.pygrain_common as airio_common

<del>task = seqio.get_mixture_or_task("dummy_task")</del>
<ins>task = get_nqo_v001_task()</ins>
<del>feature_converter = seqio.EncDecFeatureConverter(</del>
<ins>runtime_preps = airio_common.feature_converters.get_t5x_prefix_lm_feature_converter_preprocessors(</ins>
    pack=True,
    <ins>use_multi_bin_packing</ins>=True,
    <del>apply_length_check=False,</del>
    bos_id=0,
    <ins>pad_id=0,</ins>
    loss_on_targets_only=True,
    <ins>target_has_suffix=False,</ins>
    <ins>passthrough_feature_keys</ins>=[],
)
ds = <ins>task</ins>.get_dataset(
    sequence_lengths={"inputs": 32, "targets": 32},
    split="train",
    shuffle=True,
    seed=42,  # arbitrary choice
    <del>feature_converter=feature_converter,</del>
    <ins>runtime_preprocessors=runtime_preps,</ins>
    batch_size=8,
)
<ins>print(next(ds))</ins>

</code></pre>

### Mixture Migration

In this example, we will preserve the TF ops in the preprocessor and parse the
examples as `tf.Tensor`s. Note that we'll use `SentencePieceVocabulary` from
`seqio` instead of `airio.pygrain` to tokenize `tf.Tensor`s.

The following SeqIO Mixture definition:

```python
import seqio
import tensorflow as tf

TYDIQA_LANGS = ['ar', 'bn', 'fi', 'id', 'ko', 'ru', 'sw', 'te']

def register_multilingual_tydiqa_mixture() -> None:
  # source
  file_paths = {
      "ar": ".../tydi_qa-translate-train-ar.array_record*",
      "bn": ".../tydi_qa-translate-train-bn.array_record*",
      "fi": ".../tydi_qa-translate-train-fi.array_record*",
      "id": ".../tydi_qa-translate-train-id.array_record*",
      "ko": ".../tydi_qa-translate-train-ko.array_record*",
      "ru": ".../tydi_qa-translate-train-ru.array_record*",
      "sw": ".../tydi_qa-translate-train-sw.array_record*",
      "te": ".../tydi_qa-translate-train-te.array_record*",
  }
  feature_description = {
      'answers/answer_start': tf.io.FixedLenFeature([1], tf.int64),
      'answers/text': tf.io.FixedLenFeature([1], tf.string),
      'context': tf.io.FixedLenFeature([], tf.string),
      'question': tf.io.FixedLenFeature([], tf.string),
      'id': tf.io.FixedLenFeature([], tf.string),
  }

  # tokenizer
  vocab_path = 'gs://t5-data/vocabs/cc_all.32000/sentencepiece.model'
  extra_ids = 100
  vocab = seqio.SentencePieceVocabulary(vocab_path, extra_ids)
  tokenizer_configs = {
      'inputs': seqio.Feature(vocabulary=vocab),
      'targets': seqio.Feature(vocabulary=vocab),
  }

  # preprocessor
  @seqio.map_over_dataset
  def xquad(ex):
    def _pad_punctuation(text):
      text = tf.strings.regex_replace(text, r'([^_\s\p{N}\p{L}\p{M}])', r' \1 ')
      text = tf.strings.regex_replace(text, r'\s+', ' ')
      return text

    def _string_join(lst):
      out = tf.strings.join(lst, separator=' ')
      return tf.strings.regex_replace(out, r'\s+', ' ')

    a = _pad_punctuation(ex['answers/text'])
    q = _pad_punctuation(ex['question'])
    c = _pad_punctuation(ex['context'])
    inputs = _string_join(['question:', q, 'context:', c])
    return {
        'inputs': inputs,
        'targets': a[0],
        'id': ex['id'],
        'context': c,
        'question': q,
        'answers': a,
    }

  # sub-tasks
  tasks = []
  for lang in TYDIQA_LANGS:
    task_name = f'mt5_tydiqa_translate_train.{lang}'
    source = seqio.TFExampleDataSource(
        {"train": file_paths[lang]},
        feature_description=feature_description,
        reader_cls=ArrayRecordDataset,
    )
    seqio.TaskRegistry.add(
        task_name,
        source=source,
        preprocessors=[
            xquad,
            seqio.preprocessors.tokenize,
        ],
        output_features=tokenizer_configs,
    )
    tasks.append(task_name)
  seqio.MixtureRegistry.add('mt5_tydiqa', tasks, default_rate=1.0)
```

becomes:

<pre><code class="lang-python">
<del>import seqio</del>
import airio.pygrain as airio
import tensorflow as tf

TYDIQA_LANGS = ['ar', 'bn', 'fi', 'id', 'ko', 'ru', 'sw', 'te']


<del>def register_multilingual_tydiqa_mixture() -> None:</del>
<ins>def get_multilingual_tydiqa_mixture() -> airio.GrainMixture:</ins>
  # source
  file_paths = {
      "ar": ".../tydi_qa-translate-train-ar.array_record@1",
      "bn": ".../tydi_qa-translate-train-bn.array_record@1",
      "fi": ".../tydi_qa-translate-train-fi.array_record@1",
      "id": ".../tydi_qa-translate-train-id.array_record@1",
      "ko": ".../tydi_qa-translate-train-ko.array_record@1",
      "ru": ".../tydi_qa-translate-train-ru.array_record@1",
      "sw": ".../tydi_qa-translate-train-sw.array_record@1",
      "te": ".../tydi_qa-translate-train-te.array_record@1",
  }
  feature_description = {
      'answers/answer_start': tf.io.FixedLenFeature([1], tf.int64),
      'answers/text': tf.io.FixedLenFeature([1], tf.string),
      'context': tf.io.FixedLenFeature([], tf.string),
      'question': tf.io.FixedLenFeature([], tf.string),
      'id': tf.io.FixedLenFeature([], tf.string),
  }
  <ins>def parse_fn(pb):</ins>
    <ins>return tf.io.parse_single_example(pb, feature_description)</ins>

  # tokenizer
  vocab_path = 'gs://t5-data/vocabs/cc_all.32000/sentencepiece.model'
  extra_ids = 100
  vocab = seqio.SentencePieceVocabulary(vocab_path, extra_ids)
  tokenizer_configs = {
      'inputs': <ins>airio.TokenizerConfig(vocab=vocab),</ins>
      'targets': <ins>airio.TokenizerConfig(vocab=vocab),</ins>
  }

  # preprocessor
  <del>seqio.map_over_dataset</del>
  def xquad(ex):
    def _pad_punctuation(text):
      text = tf.strings.regex_replace(text, r'([^_\s\p{N}\p{L}\p{M}])', r' \1 ')
      text = tf.strings.regex_replace(text, r'\s+', ' ')
      return text

    def _string_join(lst):
      out = tf.strings.join(lst, separator=' ')
      return tf.strings.regex_replace(out, r'\s+', ' ')

    a = _pad_punctuation(ex['answers/text'])
    q = _pad_punctuation(ex['question'])
    c = _pad_punctuation(ex['context'])
    inputs = _string_join(['question:', q, 'context:', c])
    return {
        'inputs': inputs,
        'targets': a[0],
        'id': ex['id'],
        'context': c,
        'question': q,
        'answers': a,
    }

  # sub-tasks
  tasks = []
  for lang in TYDIQA_LANGS:
    task_name = f'mt5_tydiqa_translate_train.{lang}'
    <del>source = seqio.TFExampleDataSource(</del>
    source = airio.ArrayRecordDataSource(
        {"train": file_paths[lang]},
    )
    <del>seqio.TaskRegistry.add(</del>
    <ins>task = airio.GrainTask(</ins>
        task_name,
        source=source,
        preprocessors=[
            <ins>airio.MapFnTransform(parse_fn),  # parse as tensors</ins>
            <ins>airio.MapFnTransform(</ins>xquad<ins>)</ins>,
            <del>seqio.preprocessors.tokenize,</del>
            <ins>airio.MapFnTransform(</ins>
                <ins>airio.Tokenizer(</ins>
                    <ins>tokenizer_configs=tokenizer_configs,</ins>
                    <ins>copy_pretokenized=True,</ins>
                <ins>)</ins>
            <ins>),</ins>
        ],
        <del>output_features=tokenizer_configs,</del>
    )
    tasks.append(<ins>task</ins>)
  <ins>proportions = [1.0] * len(tasks)<ins>
  <del>seqio.MixtureRegistry.add('mt5_tydiqa', tasks, default_rate=1.0)</del>
  <ins>return airio.GrainMixture('mt5_tydiqa', tasks, proportions)</ins>
</code></pre>

Here is how the code to load data from the SeqIO Mixture would change:

```python
register_multilingual_tydiqa_mixture()
mix = seqio.get_mixture_or_task("mt5_tydiqa")
feature_converter = seqio.EncDecFeatureConverter(
    pack=True,
    use_custom_packing_ops=True,
    apply_length_check=False,
    bos_id=0,
    loss_on_targets_only=True,
    passthrough_features=None,
)
ds = seqio.get_dataset(
    mix,
    task_feature_lengths={"inputs": 32, "targets": 32},
    dataset_split="train",
    shuffle=True,
    seed=42,  # arbitrary choice
    feature_converter=feature_converter,
    batch_size=8,
)
print(next(iter(ds)))
```

becomes:

<pre><code class="lang-python">
import airio.pygrain_common as airio_common

<del>mix = seqio.get_mixture_or_task("dummy_task")</del>
<ins>mix = get_multilingual_tydiqa_mixture()</ins>
<del>feature_converter = seqio.EncDecFeatureConverter(</del>
<ins>runtime_preps = airio_common.feature_converters.get_t5x_prefix_lm_feature_converter_preprocessors(</ins>
    pack=True,
    <ins>use_multi_bin_packing</ins>=True,
    <del>apply_length_check=False,</del>
    bos_id=0,
    <ins>pad_id=0,</ins>
    loss_on_targets_only=True,
    <ins>target_has_suffix=False,</ins>
    <ins>passthrough_feature_keys</ins>=[],
)
ds = <ins>mix</ins>.get_dataset(
    sequence_lengths={"inputs": 32, "targets": 32},
    split="train",
    shuffle=True,
    seed=42,  # arbitrary choice
    <del>feature_converter=feature_converter,</del>
    <ins>runtime_preprocessors=runtime_preps,</ins>
    batch_size=8,
)
<ins>print(next(ds))</ins>

</code></pre>
