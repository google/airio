# AirIO Quickstart

## Define a Task

Following is an example of an AirIO Task. This Task uses an arrayrecord file
from the TFDS natural_questions_open dataset as source, appends a prompt and
tokenizes the data using the provided sentencepiece vocabulary.

```python
import airio.pygrain as airio
import grain.python as grain

def get_nqo_v001_task() -> airio.GrainTask:
  # source
  file_path = ".../datasets-arrayrecord/natural_questions_open/1.0.0/natural_questions_open-train.array_record@1"

  # tokenizer
  vocab_path = ".../vocabs/sentencepiece.model"
  extra_ids = 100
  vocab = airio.SentencePieceVocabulary(vocab_path, extra_ids)
  tokenizer_configs = {
      "inputs": airio.TokenizerConfig(vocab=vocab),
      "targets": airio.TokenizerConfig(vocab=vocab),
  }

  # preprocessor
  def question(ex: dict[str, str]) -> dict[str, str]:
    prefix = "nq question: ".encode()
    separator = ", ".encode()
    return {
        "inputs": prefix + ex["question"][0],  # everything is parsed as an array
        "targets": separator.join(ex["answer"]),
    }

  return airio.GrainTask(
      name="dummy_airio_task",
      source=airio.ArrayRecordDataSource({"train": file_path}),
      preprocessors=[
          airio.MapFnTransform(
              grain.fast_proto.parse_tf_example,  # parse as numpy arrays
          ),
          airio.MapFnTransform(question),
          airio.MapFnTransform(
              airio.Tokenizer(
                  tokenizer_configs=tokenizer_configs,
                  copy_pretokenized=True,
              )
          ),
      ],
  )
```

## Load data from the Task

Here’s how to load data from the Task defined above:

```python
task = get_nqo_v001_task()
ds = task.get_dataset(
    sequence_lengths={"inputs": 32, "targets": 32},
    split="train",
    shuffle=True,
    seed=42
)
print(next(ds))

> {'inputs': array([    3,    29,  1824,   822,    10,    84,    13,   175,    47,
>            8,   943,   189,   538,    12,  1715,     8, 18279,  2315]),
> 'inputs_pretokenized': b'nq question: which of these was the 50th state to join the united states',
> 'targets': array([13394]),
> 'targets_pretokenized': b'Hawaii'}
```

Note: Unlike SeqIO, AirIO's GrainTask.get_dataset() does not trim examples by
default. Users must explicitly add a preprocessor to trim the examples.

## Load data from the Task with packing and feature converter

```python
import airio.pygrain_common as airio_common

task = get_nqo_v001_task()
runtime_preps = airio_common.feature_converters.get_t5x_prefix_lm_feature_converter_preprocessors(
    pack=True,
    use_multi_bin_packing=True,
    bos_id=0,
    pad_id=0,
    loss_on_targets_only=True,
    target_has_suffix=False,
    passthrough_feature_keys=[],
)
ds = task.get_dataset(
    sequence_lengths={"inputs": 32, "targets": 32},
    split="train",
    shuffle=True,
    seed=42,  # arbitrary choice
    runtime_preprocessors=runtime_preps,
)
print(next(ds))

> {'decoder_target_tokens': array([...]),
>  'decoder_input_tokens': array([...]),
>  'decoder_loss_weights': array([...]),
>  'decoder_positions': array([...]),
>  'decoder_segment_ids': array([...]),
>  'decoder_causal_attention': array([...])}
```

## Define a Mixture

Following is an example of an AirIO Mixture. This example first defines Tasks
for several multilingual C4 datasets, and then defines a Mixture over them with
equal proportions. The preprocessors applied to the examples in each Task are
formatting, tokenization with the provided sentencepiece vocabulary, and the
T5-style span corruption (a canonical implementation of which is provided by
AirIO).

```python
import airio.pygrain as airio
import airio.pygrain_common as airio_common

MC4_LANGUAGES = ["de", "en", "fr"]


def get_mc4_mixture() -> airio.GrainMixture:
  # source
  file_paths = {
      "de": ".../c4-de.array_record-00000-of-xxxxx",
      "en": ".../c4-en.array_record-00000-of-xxxxx",
      "fr": ".../c4-fr.array_record-00000-of-xxxxx",
  }

  # formatting preprocessor
  def format_fn(ex):
    return {"inputs": "", "targets": ex["text"][0]}

  # tokenizer
  vocab_path = ".../t5-data/vocabs/sentencepiece.model"
  extra_ids = 100
  vocab = airio.SentencePieceVocabulary(vocab_path, extra_ids)
  tokenizer_configs = {
      "inputs": airio.TokenizerConfig(vocab=vocab),
      "targets": airio.TokenizerConfig(vocab=vocab),
  }

  # sub-tasks
  tasks = []
  for lang in MC4_LANGUAGES:
    task = airio.GrainTask(
        f"mc4.{lang}",
        source=airio.ArrayRecordDataSource({"train": file_paths[lang]}),
        preprocessors=[
            airio.MapFnTransform(
                grain.fast_proto.parse_tf_example,  # parse as numpy arrays
            ),
            airio.MapFnTransform(format_fn),
            airio.MapFnTransform(
                airio.Tokenizer(
                    tokenizer_configs=tokenizer_configs,
                    copy_pretokenized=False,
                )
            ),
            airio_common.span_corruption.create_span_corruption_transform(
                tokenizer_configs
            ),
        ],
    )
    tasks.append(task)
  proportions = [1.0] * len(tasks)
  return airio.GrainMixture("mc4", tasks, proportions)
```

## Load data from the Mixture

Here’s how to load data from the Mixture defined above:

```python
mix = get_mc4_mixture()
ds = mix.get_dataset(
    sequence_lengths={"inputs": 32, "targets": 32},
    split="train",
    shuffle=True,
    seed=42,  # arbitrary choice
)
next(ds)

> {'inputs': <tf.Tensor: shape=(31,), dtype=int64, numpy=
> array([   18,    18,    89, 15342,     5,   221,     5,  2442, 32099,
>         2825,    18,    18,    89, 15342,     5,   221,     3, 18023,
>           17,    67, 14167,   193,   510,  4837, 17302,    29, 18411,
>         2256,  6972,   218, 32098])>, 'targets': <tf.Tensor: shape=(7,), dtype=int64, numpy=array([32099,     5,  3727, 32098,    67,  5613,  1880])>}
```

## Load data from the Mixture with feature conversion

Examples are already packed as part of the span corruption preprocessor.

```python
mix = get_mc4_mixture()

runtime_preps = airio_common.feature_converters.get_t5x_prefix_lm_feature_converter_preprocessors(
    pack=False,  # packing is already done as part of the span corruption preprocessor.
    use_multi_bin_packing=False,
    bos_id=0,
    pad_id=0,
    loss_on_targets_only=True,
    target_has_suffix=False,
    passthrough_feature_keys=[],
)
ds = mix.get_dataset(
    sequence_lengths={"inputs": 32, "targets": 32},
    split="train",
    shuffle=True,
    seed=42,
    batch_size=8,
    runtime_preprocessors=runtime_preps,
)
print(next(ds))

> {'decoder_target_tokens': array([...]),
>  'decoder_input_tokens': array([...]),
>  'decoder_loss_weights': array([...]),
>  'decoder_positions': array([...]),
>  'decoder_segment_ids': array([...]),
>  'decoder_causal_attention': array([...])}
```

## `get_dataset` params
Here are a few other options you can pass to `get_dataset`:

+   To shard the dataset into `N` shards and load the `i`-th shard, pass
    `shard_info = airio.ShardInfo(index=i, num_shards=N)`.
+   To repeat the dataset, pass `num_epochs=M`.
+   To batch the dataset, pass `batch_size=B`.

`GrainTask.get_dataset` also supports the following advanced features (more
details in the next section):

+   `runtime_preprocessors`: You can pass a list of preprocessors to apply to
    the dataset in addition to the Task preprocessors. This allows configuring
    train-specific preprocessors like packing, trimming, padding, feature
    conversion, etc. eval-specific preprocessors like few-shot prompting, and so
    on.
+   `num_workers`: Use this to control the number of child processes created by
    Grain to parallelize the processing of the input pipeline.

