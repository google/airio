# Copyright 2024 The AirIO Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Microbenchmarks for AirIO dataset_providers functions."""


import functools
import os
from typing import Dict, Sequence

import airio.core as airio_core
import airio.pygrain as airio
import airio.pygrain_common as airio_common
import google_benchmark
import grain.python as grain
import jax
import numpy as np
import tensorflow_datasets as tfds


_SOURCE_NAME = "imdb_reviews"
_SOURCE_NUM_EXAMPLES = 3
_SOURCE_SPLITS = ("train", "test", "unsupervised")
_TEST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_data"
)
_SENTENCEPIECE_VOCAB = airio.SentencePieceVocabulary(
    os.path.join(_TEST_DIR, "sentencepiece", "sentencepiece.model")
)
_TOKENIZER_CONFIG = airio.TokenizerConfig(vocab=_SENTENCEPIECE_VOCAB)


def _imdb_preprocessor(raw_example: Dict[str, str]) -> Dict[str, str]:
  final_example = {"inputs": "imdb " + raw_example["text"]}
  raw_label = str(raw_example["label"])
  if raw_label == "0":
    final_example["targets"] = "negative"
  elif raw_label == "1":
    final_example["targets"] = "positive"
  else:
    final_example["targets"] = "invalid"
  return final_example


def _create_preprocessors() -> Sequence[grain.Transformation]:
  return [
      airio.MapFnTransform(_imdb_preprocessor),
      airio.MapFnTransform(
          airio.Tokenizer(
              tokenizer_configs={
                  "inputs": _TOKENIZER_CONFIG,
                  "targets": _TOKENIZER_CONFIG,
              },
          )
      ),
  ]


def _create_runtime_preprocessors() -> Sequence[grain.Transformation]:
  return airio_common.feature_converters.get_t5x_enc_dec_feature_converter_preprocessors(
      pack=False,
      use_multi_bin_packing=False,
      passthrough_feature_keys=["inputs", "targets"],
      pad_id=0,
      bos_id=0,
  )


def _create_source(
    source_name: str = _SOURCE_NAME,
    splits: Sequence[str] | None = None,
    num_examples: int = _SOURCE_NUM_EXAMPLES,
) -> airio.TfdsDataSource:
  """Creates a basic TfdsDataSource."""
  if splits is None:
    splits = _SOURCE_SPLITS
  with tfds.testing.mock_data(num_examples):
    return airio.TfdsDataSource(tfds_name=source_name, splits=splits)


def _create_fn_src(num_elements=5):
  def _dataset_fn(split: str):
    del split
    return np.arange(num_elements)

  return airio.FunctionDataSource(dataset_fn=_dataset_fn, splits=["train"])


def _create_task(
    source: airio_core.DataSource | None = None,
    preprocessors: Sequence[grain.Transformation] | None = None,
    task_name: str = "dummy_airio_task",
    num_elements: int = _SOURCE_NUM_EXAMPLES,
    idx: int = 1,
) -> airio.GrainTask:
  """Create a simple task."""
  if source is None:

    def dataset_fn(split: str):
      del split
      return np.arange(num_elements)

    source = airio.FunctionDataSource(dataset_fn=dataset_fn, splits=["train"])
  if preprocessors is None:

    def map_fn(ex, idx):
      return {"idx": idx, "val": ex}

    preprocessors = [airio.MapFnTransform(functools.partial(map_fn, idx=idx))]
  return airio.GrainTask(
      name=task_name,
      source=source,
      preprocessors=preprocessors,
  )


def _create_mixture(
    tasks: Sequence[airio.GrainTask] | None = None,
    proportions: Sequence[float] | None = None,
):
  if tasks is None:
    tasks = [_create_task(idx=1), _create_task(idx=2)]
  if proportions is None:
    proportions = [2.0, 1.0]
  return airio.GrainMixture(
      name="test_mix",
      tasks=tasks,
      proportions=proportions,
  )


class _TestFilterLazyDatasetIterator(
    airio.preprocessors.lazy_dataset.LazyDatasetIterator
):
  """Iterator that filters elements based on an int threshold."""

  def __init__(
      self,
      parent: airio.preprocessors.lazy_dataset.LazyDatasetIterator,
      threshold: int,
  ):
    super().__init__()
    self._parent = parent
    self._threshold = threshold
    self._index = 0

  def __next__(self):
    while True:
      elem = next(self._parent)
      if elem > self._threshold:
        return elem

  def get_state(self):
    return {"parent": self._parent.get_state(), "threshold": self._threshold}

  def set_state(self, state):
    self._parent.set_state(state["parent"])
    self._threshold = state["threshold"]


class TestFilterLazyIterDataset(
    airio.preprocessors.lazy_dataset.LazyIterDataset
):
  """LazyIterDataset thatfilters elements based on an int threshold."""

  def __init__(
      self,
      parent: airio.preprocessors.lazy_dataset.LazyIterDataset,
      threshold: int,
  ):
    super().__init__(parent)
    self._threshold = threshold

  def __iter__(self) -> _TestFilterLazyDatasetIterator:
    return _TestFilterLazyDatasetIterator(
        self._parent.__iter__(),
        threshold=self._threshold,
    )


@google_benchmark.register
def task_create(state):
  source = _create_source()
  while state:
    airio.GrainTask(
        name=f"{_SOURCE_NAME}_task",
        source=source,
        preprocessors=[],
    )


@google_benchmark.register
def task_create_with_preprocessors(state):
  source = _create_source()
  preprocessors = _create_preprocessors()
  while state:
    airio.GrainTask(
        name=f"{_SOURCE_NAME}_task",
        source=source,
        preprocessors=preprocessors,
    )


@google_benchmark.register
def task_num_input_examples(state):
  task = _create_task(
      source=_create_source(), preprocessors=_create_preprocessors()
  )
  while state:
    _ = task.num_input_examples(split="train")


@google_benchmark.register
def task_get_dataset(state):
  task = _create_task(
      source=_create_source(), preprocessors=_create_preprocessors()
  )
  while state:
    _ = task.get_dataset(split="train", shuffle=False)


@google_benchmark.register
def task_get_dataset_with_runtime_preps_without_batching(state):
  task = _create_task(
      source=_create_source(), preprocessors=_create_preprocessors()
  )
  runtime_preprocessors = _create_runtime_preprocessors()
  while state:
    _ = task.get_dataset(
        split="train",
        runtime_preprocessors=runtime_preprocessors,
        shuffle=False,
    )


@google_benchmark.register
def task_get_dataset_batched_with_sequence_lengths(state):
  task = _create_task(
      source=_create_source(), preprocessors=_create_preprocessors()
  )
  sequence_lengths = {"inputs": 20, "targets": 10}
  runtime_preprocessors = _create_runtime_preprocessors()
  while state:
    _ = task.get_dataset(
        sequence_lengths=sequence_lengths,
        split="train",
        runtime_preprocessors=runtime_preprocessors,
        batch_size=2,
        shuffle=False,
    )


@google_benchmark.register
def task_get_dataset_with_shard_info(state):
  task = _create_task(
      source=_create_source(), preprocessors=_create_preprocessors()
  )
  while state:
    _ = task.get_dataset(shard_info=airio.ShardInfo(index=0, num_shards=1))


@google_benchmark.register
def task_get_dataset_with_lazy_iter_prep(state):
  """Analogous to the DatasetProvidersTest with the same name."""

  def test_map_fn(ex, idx):
    return {"idx": idx, "val": ex}

  map_transform_idx_1 = airio.MapFnTransform(
      functools.partial(test_map_fn, idx=1)
  )
  task_with_iter = _create_task(
      source=_create_fn_src(num_elements=10),
      preprocessors=[
          airio.preprocessors.LazyIterTransform(
              lambda ds, *_: TestFilterLazyIterDataset(ds, threshold=4),
              update_runtime_args=lambda x: x,
          ),
          map_transform_idx_1,
      ],
      task_name="test_task_with_iter",
  )
  while state:
    _ = task_with_iter.get_dataset(shuffle=False)


@google_benchmark.register
def task_get_dataset_with_lazy_iter_prep_with_runtime_preps_and_batching(state):
  """Analogous to the DatasetProvidersTest with the same name."""

  def simple_to_imdb_map_fn(ex, rargs: airio.AirIOInjectedRuntimeArgs):
    return {
        "inputs_pretokenized": f"{ex}",
        "inputs": np.array([ex] * rargs.sequence_lengths["inputs"]),
        "targets_pretokenized": f"{ex + 1}",
        "targets": np.array([ex + 1] * rargs.sequence_lengths["targets"]),
    }

  simple_to_imdb_prep = airio.MapFnTransform(simple_to_imdb_map_fn)

  task_with_iter = _create_task(
      source=_create_fn_src(num_elements=10),
      preprocessors=[
          airio.preprocessors.LazyIterTransform(
              lambda ds, *_: TestFilterLazyIterDataset(ds, threshold=4),
              update_runtime_args=lambda x: x,
          ),
          simple_to_imdb_prep,
      ],
      task_name="test_task_with_iter",
  )
  sequence_lengths = {"inputs": 2, "targets": 1}
  while state:
    _ = task_with_iter.get_dataset(
        sequence_lengths=sequence_lengths,
        shuffle=False,
        runtime_preprocessors=_create_runtime_preprocessors(),
        batch_size=4,
    )


@google_benchmark.register
def task_get_dataset_with_none_elements(state):
  """Analogous to the DatasetProvidersTest with the same name."""

  def test_map_fn(ex, idx):
    return {"idx": idx, "val": ex}

  map_transform_idx_1 = airio.MapFnTransform(
      functools.partial(test_map_fn, idx=1)
  )
  task_with_none = _create_task(
      source=_create_fn_src(num_elements=10),
      preprocessors=[
          airio.FilterFnTransform(lambda x: x > 4),
          map_transform_idx_1,
      ],
      task_name="test_task_with_none",
  )
  while state:
    _ = task_with_none.get_dataset(shuffle=False)


@google_benchmark.register
def task_get_dataset_with_none_elements_with_runtime_preps_and_batching(state):
  """Analogous to the DatasetProvidersTest with the same name."""

  def simple_to_imdb_map_fn(ex, rargs: airio.AirIOInjectedRuntimeArgs):
    return {
        "inputs_pretokenized": f"{ex}",
        "inputs": np.array([ex] * rargs.sequence_lengths["inputs"]),
        "targets_pretokenized": f"{ex + 1}",
        "targets": np.array([ex + 1] * rargs.sequence_lengths["targets"]),
    }

  simple_to_imdb_prep = airio.MapFnTransform(simple_to_imdb_map_fn)
  task_with_iter = _create_task(
      source=_create_fn_src(num_elements=10),
      preprocessors=[
          airio.FilterFnTransform(lambda x: x > 4),
          simple_to_imdb_prep,
      ],
      task_name="test_task_with_none",
  )
  sequence_lengths = {"inputs": 2, "targets": 1}
  runtime_preprocessors = _create_runtime_preprocessors()
  while state:
    _ = task_with_iter.get_dataset(
        sequence_lengths=sequence_lengths,
        shuffle=False,
        runtime_preprocessors=runtime_preprocessors,
        batch_size=4,
    )


@google_benchmark.register
def task_get_dataset_by_step_without_runtime_preps(state):
  task = _create_task(
      source=_create_source(), preprocessors=_create_preprocessors()
  )
  while state:
    _ = task.get_dataset_by_step(num_records=1)


@google_benchmark.register
def task_get_dataset_by_step_with_runtime_preps(state):
  task = _create_task(
      source=_create_source(), preprocessors=_create_preprocessors()
  )
  sequence_lengths = {"inputs": 20, "targets": 10}
  runtime_preprocessors = _create_runtime_preprocessors()
  while state:
    _ = task.get_dataset_by_step(
        num_records=1,
        sequence_lengths=sequence_lengths,
        batch_size=2,
        runtime_preprocessors=runtime_preprocessors,
        shuffle=False,
    )


@google_benchmark.register
def task_get_dataset_by_step_without_transformations(state):
  task = _create_task(source=_create_source(), preprocessors=[])
  while state:
    _ = task.get_dataset_by_step(num_records=1)


@google_benchmark.register
def function_get_dataset(state):
  task = _create_task(
      source=_create_source(), preprocessors=_create_preprocessors()
  )
  while state:
    _ = airio_core.dataset_providers.get_dataset(task)


@google_benchmark.register
def task_get_updated_runtime_args(state):
  """Analogous to the DatasetProvidersTest with the same name."""

  def update_runtime_args_1(args):
    args.sequence_lengths.update({"new_val": 5})
    return args

  def update_runtime_args_2(args):
    args.sequence_lengths.update({"another_val": 7})
    return args

  prep_1 = airio.MapFnTransform(
      lambda x: x,
      update_runtime_args=update_runtime_args_1,
  )
  prep_2 = airio.MapFnTransform(
      lambda x: x,
      update_runtime_args=update_runtime_args_2,
  )
  task = airio.GrainTask(
      "test", source=_create_source(), preprocessors=[prep_1, prep_2]
  )
  runtime_args = airio.AirIOInjectedRuntimeArgs(
      sequence_lengths={"val": 3}, split="train"
  )
  while state:
    _ = task.get_updated_runtime_args(runtime_args, runtime_preprocessors=None)


@google_benchmark.register
def function_get_vocabularies(state):
  task = _create_task(source=_create_source(), preprocessors=[])
  while state:
    _ = airio_core.dataset_providers.get_vocabularies(task)


@google_benchmark.register
def task_builder_from_task(state):
  task = _create_task(source=_create_source(), preprocessors=[])
  while state:
    _ = airio.GrainTaskBuilder.from_task(task)


@google_benchmark.register
def task_builder_build(state):
  task_builder = airio.GrainTaskBuilder(
      task_name="dummy_airio_task",
      source=_create_source(),
      preprocessors=_create_preprocessors(),
  )
  while state:
    _ = task_builder.build()


@google_benchmark.register
def task_get_dataset_with_runtime_args(state):
  """Analogous to the DatasetProvidersTest with the same name."""

  def simple_to_imdb_map_fn(ex, rargs: airio.AirIOInjectedRuntimeArgs):
    return {
        "inputs_pretokenized": f"{ex}",
        "inputs": np.array([ex] * rargs.sequence_lengths["inputs"]),
        "targets_pretokenized": f"{ex + 1}",
        "targets": np.array([ex + 1] * rargs.sequence_lengths["targets"]),
    }

  simple_task = _create_task(
      source=_create_fn_src(),
      preprocessors=[airio.MapFnTransform(simple_to_imdb_map_fn)],
  )
  while state:
    _ = simple_task.get_dataset(
        sequence_lengths={"inputs": 20, "targets": 10}, shuffle=False
    )


@google_benchmark.register
def task_get_dataset_by_step_with_runtime_args(state):
  """Analogous to the DatasetProvidersTest with the same name."""

  def simple_to_imdb_map_fn(ex, rargs: airio.AirIOInjectedRuntimeArgs):
    return {
        "inputs_pretokenized": f"{ex}",
        "inputs": np.array([ex] * rargs.sequence_lengths["inputs"]),
        "targets_pretokenized": f"{ex + 1}",
        "targets": np.array([ex + 1] * rargs.sequence_lengths["targets"]),
    }

  simple_task = _create_task(
      source=_create_fn_src(),
      preprocessors=[airio.MapFnTransform(simple_to_imdb_map_fn)],
  )
  while state:
    _ = simple_task.get_dataset_by_step(
        sequence_lengths={"inputs": 20, "targets": 10}, shuffle=False
    )


# TODO(b/314832206): Figure out why this throws a type error.
# @google_benchmark.register
# def task_switch_to_lazy_dataset(state):
#   """Analogous to the DatasetProvidersTest with the same name."""
#   def lazy_id_fn(
#       ds: lazy_dataset.LazyMapDataset,
#       rargs: preprocessors_lib.AirIOInjectedRuntimeArgs,
#   ):
#     del rargs
#     return ds
#   preprocessors = _create_preprocessors() + [
#       preprocessors_lib.LazyMapTransform(
#           lazy_id_fn,
#           update_runtime_args=lambda rargs: rargs,
#           produces_none_elements=False,
#       ),
#   ]
#   task = _create_task(source=_create_source(), preprocessors=preprocessors)
#   while state:
#     _ = task.get_dataset(split="train", shuffle=False)


@google_benchmark.register
def mixture_runtime_args_updated_by_task(state):
  """Analogous to the MixtureTest with the same name."""

  def update_runtime_args_fn(rargs):
    return airio.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 20, "targets": 10}, split=rargs.split
    )

  task = _create_task(
      source=_create_source(), preprocessors=_create_preprocessors()
  )
  task_with_runtime_args_update = (
      airio.GrainTaskBuilder.from_task(task)
      .set_preprocessors(
          task.get_preprocessors()
          + [
              airio.MapFnTransform(
                  lambda x: x, update_runtime_args=update_runtime_args_fn
              ),
          ]
      )
      .build()
  )
  mix = airio.GrainMixture(
      name="test_mix",
      tasks=[task_with_runtime_args_update],
      proportions=[1.0],
  )
  while state:
    _ = mix.get_dataset(
        sequence_lengths={"xyz": 5, "abc": 7},  # will be updated
        shuffle=False,
        shard_info=airio.ShardInfo(index=0, num_shards=2),
        num_epochs=1,
        runtime_preprocessors=_create_runtime_preprocessors(),
    )


@google_benchmark.register
def simple_mixture(state):
  mix = _create_mixture()
  while state:
    _ = mix.get_dataset(shuffle=False)


@google_benchmark.register
def mixture_sharding(state):
  mix = _create_mixture()
  while state:
    _ = mix.get_dataset(
        shuffle=False,
        shard_info=airio.ShardInfo(index=0, num_shards=2),
    )


@google_benchmark.register
def mixture_shuffling(state):
  mix = _create_mixture()
  while state:
    _ = mix.get_dataset(
        shuffle=True,
        seed=42,
        shard_info=airio.ShardInfo(index=0, num_shards=2),
    )


@google_benchmark.register
def multi_epoch(state):
  mix = _create_mixture()
  while state:
    _ = mix.get_dataset(
        shuffle=True,
        seed=42,
        shard_info=airio.ShardInfo(index=0, num_shards=2),
        num_epochs=2,
    )


@google_benchmark.register
def multi_epoch_with_stochastic_preprocessor(state):
  """Analogous to the MixtureTest with the same name."""

  def test_random_map_fn(ex, rng):
    ex["var"] = int(jax.random.randint(rng, [], 0, 20))
    return ex

  simple_task_1 = _create_task(idx=1)
  simple_task_2 = _create_task(idx=2)

  def test_map_fn(ex, idx):
    return {"idx": idx, "val": ex}

  map_transform_idx_1 = airio.MapFnTransform(
      functools.partial(test_map_fn, idx=1)
  )
  map_transform_idx_2 = airio.MapFnTransform(
      functools.partial(test_map_fn, idx=2)
  )
  task1 = (
      airio.GrainTaskBuilder.from_task(simple_task_1)
      .set_preprocessors([
          map_transform_idx_1,
          airio.RandomMapFnTransform(test_random_map_fn),
      ])
      .build()
  )
  task2 = (
      airio.GrainTaskBuilder.from_task(simple_task_2)
      .set_preprocessors([
          map_transform_idx_2,
          airio.RandomMapFnTransform(test_random_map_fn),
      ])
      .build()
  )
  mix = _create_mixture(tasks=[task1, task2])
  while state:
    _ = mix.get_lazy_dataset(
        None,
        "train",
        shuffle=True,
        seed=42,
        shard_info=airio.ShardInfo(index=0, num_shards=2),
        num_epochs=2,
    )


@google_benchmark.register
def indefinite_repeat(state):
  mix = _create_mixture()
  while state:
    _ = mix.get_dataset(
        shuffle=False,
        shard_info=airio.ShardInfo(index=0, num_shards=2),
        num_epochs=None,
    )


@google_benchmark.register
def mixture_with_different_sources_and_preprocessors(state):
  """Analogous to the MixtureTest with the same name."""
  imdb_task = _create_task(
      source=_create_source(), preprocessors=_create_preprocessors()
  )

  def simple_to_imdb_map_fn(ex, rargs: airio.AirIOInjectedRuntimeArgs):
    return {
        "inputs_pretokenized": f"{ex}",
        "inputs": np.array([ex] * rargs.sequence_lengths["inputs"]),
        "targets_pretokenized": f"{ex + 1}",
        "targets": np.array([ex + 1] * rargs.sequence_lengths["targets"]),
    }

  simple_to_imdb_task = (
      airio.GrainTaskBuilder.from_task(_create_task())
      .set_preprocessors([
          airio.MapFnTransform(simple_to_imdb_map_fn),
      ])
      .build()
  )
  mix = _create_mixture(
      tasks=[imdb_task, simple_to_imdb_task], proportions=[1.0, 1.0]
  )
  while state:
    _ = mix.get_dataset(
        sequence_lengths={"inputs": 20, "targets": 10},
        shuffle=False,
        shard_info=airio.ShardInfo(index=0, num_shards=2),
        num_epochs=1,
    )


@google_benchmark.register
def mixture_with_runtime_preps(state):
  """Analogous to the MixtureTest with the same name."""
  imdb_task = _create_task(
      source=_create_source(), preprocessors=_create_preprocessors()
  )

  def simple_to_imdb_map_fn(ex, rargs: airio.AirIOInjectedRuntimeArgs):
    return {
        "inputs_pretokenized": f"{ex}",
        "inputs": np.array([ex] * rargs.sequence_lengths["inputs"]),
        "targets_pretokenized": f"{ex + 1}",
        "targets": np.array([ex + 1] * rargs.sequence_lengths["targets"]),
    }

  simple_to_imdb_task = (
      airio.GrainTaskBuilder.from_task(_create_task())
      .set_preprocessors([
          airio.MapFnTransform(simple_to_imdb_map_fn),
      ])
      .build()
  )
  mix = _create_mixture(
      tasks=[imdb_task, simple_to_imdb_task], proportions=[1.0, 1.0]
  )
  sequence_lengths = {"inputs": 20, "targets": 10}
  while state:
    _ = mix.get_dataset(
        sequence_lengths=sequence_lengths,
        shuffle=False,
        shard_info=airio.ShardInfo(index=0, num_shards=2),
        num_epochs=1,
        runtime_preprocessors=_create_runtime_preprocessors(),
    )


@google_benchmark.register
def mixture_with_runtime_preps_and_batching(state):
  """Analogous to the MixtureTest with the same name."""
  imdb_task = _create_task(
      source=_create_source(), preprocessors=_create_preprocessors()
  )

  def simple_to_imdb_map_fn(ex, rargs: airio.AirIOInjectedRuntimeArgs):
    return {
        "inputs_pretokenized": f"{ex}",
        "inputs": np.array([ex] * rargs.sequence_lengths["inputs"]),
        "targets_pretokenized": f"{ex + 1}",
        "targets": np.array([ex + 1] * rargs.sequence_lengths["targets"]),
    }

  simple_to_imdb_task = (
      airio.GrainTaskBuilder.from_task(_create_task())
      .set_preprocessors([
          airio.MapFnTransform(simple_to_imdb_map_fn),
      ])
      .build()
  )
  mix = _create_mixture(
      tasks=[imdb_task, simple_to_imdb_task], proportions=[1.0, 1.0]
  )
  sequence_lengths = {"inputs": 20, "targets": 10}
  while state:
    _ = mix.get_dataset(
        sequence_lengths=sequence_lengths,
        shuffle=False,
        shard_info=airio.ShardInfo(index=0, num_shards=2),
        num_epochs=1,
        runtime_preprocessors=_create_runtime_preprocessors(),
        batch_size=2,
    )


@google_benchmark.register
def mixture_with_batching_only(state):
  mix = _create_mixture(proportions=[1.0, 1.0])
  while state:
    _ = mix.get_dataset(
        sequence_lengths={"inputs": 20, "targets": 10},
        shuffle=False,
        shard_info=airio.ShardInfo(index=0, num_shards=2),
        num_epochs=1,
        runtime_preprocessors=None,
        batch_size=2,
    )


@google_benchmark.register
def mixing_with_iter_test(state):
  """Analogous to the MixtureTest with the same name."""

  def test_map_fn(ex, idx):
    return {"idx": idx, "val": ex}

  map_transform_idx_1 = airio.MapFnTransform(
      functools.partial(test_map_fn, idx=1)
  )
  map_transform_idx_2 = airio.MapFnTransform(
      functools.partial(test_map_fn, idx=2)
  )
  # Mix datasets that produce None elements and verify that mixture length and
  # mixing rate are correct
  task_with_none = _create_task(
      source=_create_fn_src(num_elements=10),
      preprocessors=[
          airio.FilterFnTransform(lambda x: x > 4),
          map_transform_idx_1,
      ],
      task_name="test_task_with_none",
  )
  ordinary_task = _create_task(
      source=_create_fn_src(num_elements=10),
      preprocessors=[map_transform_idx_2],
      task_name="ordinary_task",
  )
  mix = _create_mixture(
      tasks=[task_with_none, ordinary_task], proportions=[1.0, 1.0]
  )
  while state:
    _ = mix.get_dataset(shuffle=False)


@google_benchmark.register
def mixing_with_iter_test_with_runtime_preps_and_batching(state):
  """Analogous to the MixtureTest with the same name."""

  def simple_to_imdb_map_fn(ex, rargs: airio.AirIOInjectedRuntimeArgs):
    return {
        "inputs_pretokenized": f"{ex}",
        "inputs": np.array([ex] * rargs.sequence_lengths["inputs"]),
        "targets_pretokenized": f"{ex + 1}",
        "targets": np.array([ex + 1] * rargs.sequence_lengths["targets"]),
    }

  simple_to_imdb_prep = airio.MapFnTransform(simple_to_imdb_map_fn)

  # Mix datasets that produce None elements and verify that mixture length and
  # mixing rate are correct.
  task_with_none = _create_task(
      source=_create_fn_src(num_elements=10),
      preprocessors=[
          airio.FilterFnTransform(lambda x: x > 4),
          simple_to_imdb_prep,
      ],
      task_name="test_task_with_none",
  )
  ordinary_task = _create_task(
      source=_create_fn_src(num_elements=10),
      preprocessors=[
          simple_to_imdb_prep,
      ],
      task_name="ordinary_task",
  )
  mix = _create_mixture(
      tasks=[task_with_none, ordinary_task], proportions=[1.0, 1.0]
  )
  sequence_lengths = {"inputs": 2, "targets": 1}
  while state:
    _ = mix.get_dataset(
        sequence_lengths=sequence_lengths,
        shuffle=False,
        runtime_preprocessors=_create_runtime_preprocessors(),
        batch_size=4,
    )


@google_benchmark.register
def mixing_with_lazy_iter_preprocessor(state):
  """Analogous to the MixtureTest with the same name."""

  def test_map_fn(ex, idx):
    return {"idx": idx, "val": ex}

  map_transform_idx_1 = airio.MapFnTransform(
      functools.partial(test_map_fn, idx=1)
  )
  map_transform_idx_2 = airio.MapFnTransform(
      functools.partial(test_map_fn, idx=2)
  )
  # Mix tasks with LazyIter preprocessors and verify that mixture length and
  # mixing rate are correct.
  task_with_iter = _create_task(
      source=_create_fn_src(num_elements=10),
      preprocessors=[
          airio.preprocessors.LazyIterTransform(
              lambda ds, *_: TestFilterLazyIterDataset(ds, threshold=4),
              update_runtime_args=lambda x: x,
          ),
          map_transform_idx_1,
      ],
      task_name="test_task_with_iter",
  )
  ordinary_task = _create_task(
      source=_create_fn_src(num_elements=10),
      preprocessors=[map_transform_idx_2],
      task_name="ordinary_task",
  )
  mix = _create_mixture(
      tasks=[task_with_iter, ordinary_task], proportions=[1.0, 1.0]
  )
  while state:
    _ = mix.get_dataset(shuffle=False)


@google_benchmark.register
def mixing_with_lazy_iter_preprocessor_with_runtime_preps_and_batching(state):
  """Analogous to the MixtureTest with the same name."""

  def simple_to_imdb_map_fn(ex, rargs: airio.AirIOInjectedRuntimeArgs):
    return {
        "inputs_pretokenized": f"{ex}",
        "inputs": np.array([ex] * rargs.sequence_lengths["inputs"]),
        "targets_pretokenized": f"{ex + 1}",
        "targets": np.array([ex + 1] * rargs.sequence_lengths["targets"]),
    }

  simple_to_imdb_prep = airio.MapFnTransform(simple_to_imdb_map_fn)

  # Mix tasks with LazyIter preprocessors and verify that mixture length and
  # mixing rate are correct.
  task_with_none = _create_task(
      source=_create_fn_src(num_elements=10),
      preprocessors=[
          airio.preprocessors.LazyIterTransform(
              lambda ds, *_: TestFilterLazyIterDataset(ds, threshold=4),
              update_runtime_args=lambda x: x,
          ),
          simple_to_imdb_prep,
      ],
      task_name="test_task_with_none",
  )
  ordinary_task = _create_task(
      source=_create_fn_src(num_elements=10),
      preprocessors=[
          simple_to_imdb_prep,
      ],
      task_name="ordinary_task",
  )
  mix = _create_mixture(
      tasks=[task_with_none, ordinary_task], proportions=[1.0, 1.0]
  )
  sequence_lengths = {"inputs": 2, "targets": 1}
  while state:
    _ = mix.get_dataset(
        sequence_lengths=sequence_lengths,
        shuffle=False,
        runtime_preprocessors=_create_runtime_preprocessors(),
        batch_size=4,
    )


if __name__ == "__main__":
  google_benchmark.main()
