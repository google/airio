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

"""Tests for airio.pygrain.dataset_providers with multiprocessing."""

import functools
import multiprocessing as mp  # pylint:disable=unused-import
import os
from typing import Dict, Sequence
from unittest import mock

from absl.testing import absltest
from airio._src.core import data_sources
from airio._src.core import dataset_providers as core_dataset_providers
# Import "preprocessors" as "preprocessors_lib" to prevent naming conflicts with
# "preprocessors" attrs in this file.
from airio._src.core import preprocessors as core_preprocessors_lib
from airio._src.core import test_utils
from airio._src.core import tokenizer
from airio._src.pygrain import dataset_providers
from airio._src.pygrain import preprocessors as preprocessors_lib
from airio._src.pygrain.common import feature_converters
import grain.python as grain
import numpy as np
from seqio import vocabularies
import tensorflow_datasets as tfds


lazy_dataset = grain.experimental.lazy_dataset
_SOURCE_NAME = "imdb_reviews"
_SOURCE_NUM_EXAMPLES = 3
_SOURCE_SPLITS = {"train", "test", "unsupervised"}


class _TestFilterLazyDatasetIterator(lazy_dataset.LazyDatasetIterator):
  """Iterator that filters elements based on an int threshold."""

  def __init__(
      self,
      parent: lazy_dataset.LazyDatasetIterator,
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


class TestFilterLazyIterDataset(lazy_dataset.LazyIterDataset):
  """LazyIterDataset thatfilters elements based on an int threshold."""

  def __init__(
      self,
      parent: lazy_dataset.LazyIterDataset,
      threshold: int,
  ):
    super().__init__(parent)
    self._threshold = threshold

  def __iter__(self) -> _TestFilterLazyDatasetIterator:
    return _TestFilterLazyDatasetIterator(
        self._parent.__iter__(),
        threshold=self._threshold,
    )


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


def _create_sentencepiece_vocab() -> vocabularies.SentencePieceVocabulary:
  test_data_dir = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "../../test_data",
  )
  sentencepiece_vocab = vocabularies.SentencePieceVocabulary(
      os.path.join(test_data_dir, "sentencepiece", "sentencepiece.model")
  )
  return sentencepiece_vocab


def _create_tokenizer_config() -> tokenizer.TokenizerConfig:
  return tokenizer.TokenizerConfig(vocab=_create_sentencepiece_vocab())


def _create_preprocessors() -> (
    Sequence[preprocessors_lib.PyGrainAirIOPreprocessor]
):
  tokenizer_config = _create_tokenizer_config()
  return [
      core_preprocessors_lib.MapFnTransform(_imdb_preprocessor),
      core_preprocessors_lib.MapFnTransform(
          tokenizer.Tokenizer(
              tokenizer_configs={
                  "inputs": tokenizer_config,
                  "targets": tokenizer_config,
              },
          )
      ),
  ]


def _create_runtime_preprocessors() -> (
    Sequence[preprocessors_lib.PyGrainAirIOPreprocessor]
):
  return feature_converters.get_t5x_enc_dec_feature_converter_preprocessors(
      pack=False,
      use_multi_bin_packing=False,
      passthrough_feature_keys=[],
      pad_id=0,
      bos_id=0,
  )


def _create_source(
    source_name: str = _SOURCE_NAME,
    splits: Sequence[str] | None = None,
    num_examples: int = _SOURCE_NUM_EXAMPLES,
) -> data_sources.TfdsDataSource:
  """Creates a basic TfdsDataSource."""
  if splits is None:
    splits = _SOURCE_SPLITS
  with tfds.testing.mock_data(num_examples):
    return data_sources.TfdsDataSource(tfds_name=source_name, splits=splits)


def _create_fn_src(num_elements=5):
  def _dataset_fn(split: str):
    del split
    return np.arange(num_elements)

  return data_sources.FunctionDataSource(
      dataset_fn=_dataset_fn, splits=["train"]
  )


def _create_task(
    source: data_sources.DataSource | None,
    preprocessors: (
        Sequence[preprocessors_lib.PyGrainAirIOPreprocessor] | None
    ) = None,
    task_name: str = "dummy_airio_task",
) -> dataset_providers.GrainTask:
  """Create example AirIO task."""
  return dataset_providers.GrainTask(
      name=task_name,
      source=source,
      preprocessors=preprocessors,
  )


class DatasetProvidersMultiprocessingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    def test_map_fn(ex, idx):
      return {"idx": idx, "val": ex}

    def simple_to_imdb_map_fn(
        ex, rargs: core_preprocessors_lib.AirIOInjectedRuntimeArgs
    ):
      return {
          "inputs_pretokenized": f"{ex}",
          "inputs": np.array([ex] * rargs.sequence_lengths["inputs"]),
          "targets_pretokenized": f"{ex + 1}",
          "targets": np.array([ex + 1] * rargs.sequence_lengths["targets"]),
      }

    imdb_source = _create_source(
        source_name=_SOURCE_NAME,
        splits=_SOURCE_SPLITS,
        num_examples=_SOURCE_NUM_EXAMPLES,
    )
    self._map_transform_idx_1 = core_preprocessors_lib.MapFnTransform(
        functools.partial(test_map_fn, idx=1)
    )
    self._simple_to_imdb_prep = core_preprocessors_lib.MapFnTransform(
        simple_to_imdb_map_fn
    )
    self._simple_task_1 = _create_task(
        task_name="test_task1",
        source=_create_fn_src(),
        preprocessors=[self._map_transform_idx_1],
    )
    self._imdb_task = _create_task(
        source=imdb_source, preprocessors=_create_preprocessors()
    )
    self._simple_to_imdb_task = (
        dataset_providers.GrainTaskBuilder.from_task(self._simple_task_1)
        .set_preprocessors([
            core_preprocessors_lib.MapFnTransform(simple_to_imdb_map_fn),
        ])
        .build()
    )

  def test_mixture_with_runtime_preps_and_batching(self):
    mix = dataset_providers.GrainMixture(
        name="test_mix",
        tasks=[self._imdb_task, self._simple_to_imdb_task],
        proportions=[1.0, 1.0],
    )
    sequence_lengths = {"inputs": 20, "targets": 10}
    ds = mix.get_dataset(
        sequence_lengths=sequence_lengths,
        shuffle=False,
        shard_info=core_dataset_providers.ShardInfo(index=0, num_shards=2),
        num_epochs=1,
        runtime_preprocessors=_create_runtime_preprocessors(),
        batch_size=2,
        num_workers=2,
    )
    expected_first_batch = {
        "decoder_input_tokens": [
            [0, 3, 15, 7, 6, 8, 24, 8, 25, 4],  # imdb task
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # simple task
        ],
        "decoder_loss_weights": [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # imdb task
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # simple task
        ],
        "decoder_target_tokens": [
            [3, 15, 7, 6, 8, 24, 8, 25, 4, 0],  # imdb task
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # imdb task
        ],
        "encoder_input_tokens": [
            [
                3,
                8,
                14,
                21,
                2,
                3,
                4,
                2,
                13,
                3,
                5,
                20,
                2,
                4,
                2,
                20,
                2,
                4,
                0,
                0,
            ],  # imdb task
            # simple task
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
    }
    self.assertDictEqual(
        {k: v.tolist() for k, v in next(ds).items()}, expected_first_batch
    )
    # Worker state should reflect 2 workers.
    self.assertDictEqual(
        ds.get_state()["workers_state"],
        {"0": {"next_index": 0}, "1": {"next_index": 0}},
    )

  def test_mixing_with_iter_test_with_runtime_preps_and_batching(self):
    # Mix datasets that produce None elements and verify that mixture length and
    # mixing rate are correct.
    task_with_none = _create_task(
        source=_create_fn_src(num_elements=10),
        preprocessors=[
            core_preprocessors_lib.FilterFnTransform(lambda x: x > 4),
            self._simple_to_imdb_prep,
        ],
        task_name="test_task_with_none",
    )
    ordinary_task = _create_task(
        source=_create_fn_src(num_elements=10),
        preprocessors=[
            self._simple_to_imdb_prep,
        ],
        task_name="ordinary_task",
    )
    mix = dataset_providers.GrainMixture(
        name="test_mix",
        tasks=[task_with_none, ordinary_task],
        proportions=[1.0, 1.0],
    )
    sequence_lengths = {"inputs": 2, "targets": 1}
    ds = mix.get_dataset(
        sequence_lengths=sequence_lengths,
        shuffle=False,
        runtime_preprocessors=_create_runtime_preprocessors(),
        batch_size=4,
        num_workers=2,
    )
    expected_dataset = [
        {  # task 2 ex 1, task 1 ex 7, task 2 ex 3, task 1 ex 9
            "decoder_input_tokens": [[0], [0], [0], [0]],
            "decoder_loss_weights": [[1], [1], [1], [1]],
            "decoder_target_tokens": [[1], [7], [3], [9]],
            "encoder_input_tokens": [[0, 0], [6, 6], [2, 2], [8, 8]],
        },
        {  # task 2 ex 2, task 1 ex 6, task 2 ex 4, task 1 ex 8
            "decoder_input_tokens": [[0], [0], [0], [0]],
            "decoder_loss_weights": [[1], [1], [1], [1]],
            "decoder_target_tokens": [[2], [6], [4], [8]],
            "encoder_input_tokens": [[1, 1], [5, 5], [3, 3], [7, 7]],
        },
        {  # task 2 ex 5, no more examples in task 1 on this worker.
            "decoder_input_tokens": [[0]],
            "decoder_loss_weights": [[1]],
            "decoder_target_tokens": [[5]],
            "encoder_input_tokens": [[4, 4]],
        },
        {  # task 2 ex 6, task 1 ex 10, task 2 ex 8
            "decoder_input_tokens": [[0], [0], [0]],
            "decoder_loss_weights": [[1], [1], [1]],
            "decoder_target_tokens": [[6], [10], [8]],
            "encoder_input_tokens": [[5, 5], [9, 9], [7, 7]],
        },
    ]
    expected_keys = [
        "decoder_input_tokens",
        "decoder_loss_weights",
        "decoder_target_tokens",
        "encoder_input_tokens",
    ]
    for actual, expected in zip(ds, expected_dataset):
      for k in expected_keys:
        np.testing.assert_array_equal(actual[k], expected[k])
    # Worker state should reflect 2 workers.
    self.assertDictEqual(
        ds.get_state()["workers_state"],
        {
            "0": {
                "parent": {
                    "parent": {
                        "parent": {
                            "parents": [{"next_index": 0}, {"next_index": 0}],
                            "index": 0,
                            "stop": False,
                        },
                        "index_for_rng": 0,
                    },
                    "index_for_rng": 0,
                },
                "index_for_rng": 0,
            },
            "1": {
                "parent": {
                    "parent": {
                        "parent": {
                            "parents": [{"next_index": 0}, {"next_index": 0}],
                            "index": 0,
                            "stop": False,
                        },
                        "index_for_rng": 0,
                    },
                    "index_for_rng": 0,
                },
                "index_for_rng": 0,
            },
        },
    )

  def test_task_get_dataset_with_lazy_iter_prep_with_runtime_preps_and_batching(
      self,
  ):
    task_with_iter = _create_task(
        source=_create_fn_src(num_elements=10),
        preprocessors=[
            preprocessors_lib.LazyIterTransform(
                lambda ds, unused_args, unused_rng: TestFilterLazyIterDataset(
                    ds, threshold=4
                ),
                update_runtime_args=lambda x: x,
            ),
            self._simple_to_imdb_prep,
        ],
        task_name="test_task_with_iter",
    )
    sequence_lengths = {"inputs": 2, "targets": 1}
    ds = task_with_iter.get_dataset(
        sequence_lengths=sequence_lengths,
        shuffle=False,
        runtime_preprocessors=_create_runtime_preprocessors(),
        batch_size=4,
        num_workers=2,
    )
    expected_dataset = [
        {  # worker 1 - example 2 and 3
            "decoder_input_tokens": [[0], [0]],
            "decoder_loss_weights": [[1], [1]],
            "decoder_target_tokens": [[7], [9]],
            "encoder_input_tokens": [[6, 6], [8, 8]],
        },
        {  # worker 2 - example 1, 4 and 5
            "decoder_input_tokens": [[0], [0], [0]],
            "decoder_loss_weights": [[1], [1], [1]],
            "decoder_target_tokens": [[6], [8], [10]],
            "encoder_input_tokens": [[5, 5], [7, 7], [9, 9]],
        },
    ]
    expected_keys = [
        "decoder_input_tokens",
        "decoder_loss_weights",
        "decoder_target_tokens",
        "encoder_input_tokens",
    ]
    for actual, expected in zip(ds, expected_dataset):
      self.assertSequenceEqual(sorted(actual.keys()), sorted(expected_keys))
      for k in expected_keys:
        np.testing.assert_array_equal(actual[k], expected[k])
    # Worker state should reflect 2 workers.
    self.assertDictEqual(
        ds.get_state()["workers_state"],
        {
            "0": {
                "parent": {
                    "parent": {
                        "parent": {
                            "parent": {
                                "parent": {"next_index": 0},
                                "threshold": 4,
                            },
                            "index_for_rng": 0,
                        },
                        "index_for_rng": 0,
                    },
                    "index_for_rng": 0,
                },
                "index_for_rng": 0,
            },
            "1": {
                "parent": {
                    "parent": {
                        "parent": {
                            "parent": {
                                "parent": {"next_index": 0},
                                "threshold": 4,
                            },
                            "index_for_rng": 0,
                        },
                        "index_for_rng": 0,
                    },
                    "index_for_rng": 0,
                },
                "index_for_rng": 0,
            },
        },
    )

  def test_task_get_dataset_with_none_elements_with_runtime_preps_and_batching(
      self,
  ):
    task_with_iter = _create_task(
        source=_create_fn_src(num_elements=10),
        preprocessors=[
            core_preprocessors_lib.FilterFnTransform(lambda x: x > 4),
            self._simple_to_imdb_prep,
        ],
        task_name="test_task_with_none",
    )
    sequence_lengths = {"inputs": 2, "targets": 1}
    ds = task_with_iter.get_dataset(
        sequence_lengths=sequence_lengths,
        shuffle=False,
        runtime_preprocessors=_create_runtime_preprocessors(),
        batch_size=4,
        num_workers=2,
    )
    expected_dataset = [
        {
            "decoder_input_tokens": [[0], [0]],
            "decoder_loss_weights": [[1], [1]],
            "decoder_target_tokens": [[7], [9]],
            "encoder_input_tokens": [[6, 6], [8, 8]],
        },
        {
            "decoder_input_tokens": [[0], [0], [0]],
            "decoder_loss_weights": [[1], [1], [1]],
            "decoder_target_tokens": [[6], [8], [10]],
            "encoder_input_tokens": [[5, 5], [7, 7], [9, 9]],
        },
    ]
    expected_keys = [
        "decoder_input_tokens",
        "decoder_loss_weights",
        "decoder_target_tokens",
        "encoder_input_tokens",
    ]
    for actual, expected in zip(ds, expected_dataset):
      self.assertSequenceEqual(sorted(actual.keys()), sorted(expected_keys))
      for k in expected_keys:
        np.testing.assert_array_equal(actual[k], expected[k])
    # This uses the DataLoader, and doesn't have workers_state.
    self.assertEqual(ds.get_state()["worker_count"], 2)

  def test_task_lazy_dataset_batch_across_epochs(self):
    # Create a Task with 3 elements.
    test_task = _create_task(
        source=_create_fn_src(num_elements=3),
        preprocessors=[],
    )
    # Repeat for two epochs and batch with size 2
    test_task._switch_to_lazy_dataset = mock.Mock(return_value=True)
    ds = test_task.get_dataset(shuffle=False, num_epochs=2, batch_size=2)
    ds = list(ds)
    # In the 2nd batched example, the first element is from the first epoch, and
    # the second element is from the second epoch
    expected_ds = [[0, 1], [2, 0], [1, 2]]
    for actual, expected in zip(ds, expected_ds, strict=True):
      np.testing.assert_array_equal(actual, expected)

  def test_task_get_dataset_batched_with_sequence_lengths(self):
    source = _create_source(
        source_name=_SOURCE_NAME,
        splits=_SOURCE_SPLITS,
        num_examples=_SOURCE_NUM_EXAMPLES,
    )
    task = _create_task(source=source, preprocessors=_create_preprocessors())
    sequence_lengths = {"inputs": 20, "targets": 10}
    ds = task.get_dataset(
        sequence_lengths=sequence_lengths,
        split="train",
        runtime_preprocessors=_create_runtime_preprocessors(),
        batch_size=2,
        shuffle=False,
        num_workers=2,
    )
    expected_first_batch = [
        {
            "decoder_input_tokens": [
                [0, 3, 15, 7, 6, 8, 24, 8, 25, 4],
                [0, 3, 15, 7, 6, 8, 24, 8, 25, 4],
            ],
            "decoder_loss_weights": [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, False],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, False],
            ],
            "decoder_target_tokens": [
                [3, 15, 7, 6, 8, 24, 8, 25, 4, 0],
                [3, 15, 7, 6, 8, 24, 8, 25, 4, 0],
            ],
            "encoder_input_tokens": [
                [
                    3,
                    8,
                    14,
                    21,
                    2,
                    3,
                    4,
                    2,
                    13,
                    3,
                    5,
                    20,
                    2,
                    4,
                    2,
                    20,
                    2,
                    4,
                    0,
                    0,
                ],
                [
                    3,
                    8,
                    14,
                    21,
                    2,
                    3,
                    5,
                    13,
                    21,
                    20,
                    21,
                    5,
                    13,
                    2,
                    20,
                    20,
                    2,
                    0,
                    0,
                    0,
                ],
            ],
        },
        {
            "decoder_input_tokens": [[0, 3, 22, 4, 2, 18, 8, 25, 4, 0]],
            "decoder_loss_weights": [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]],
            "decoder_target_tokens": [[3, 22, 4, 2, 18, 8, 25, 4, 0, 0]],
            "encoder_input_tokens": [[
                3,
                8,
                14,
                21,
                2,
                3,
                20,
                2,
                3,
                5,
                8,
                2,
                13,
                8,
                21,
                13,
                8,
                2,
                21,
                2,
            ]],
        },
    ]
    test_utils.assert_datasets_equal(ds, expected_first_batch)
    # This uses the DataLoader, and doesn't have workers_state.
    self.assertEqual(ds.get_state()["worker_count"], 2)


class IterAndPrefetchTest(absltest.TestCase):

  def test_read_configs_none(self):
    self.assertEqual(
        dataset_providers._get_read_options(None), grain.ReadOptions()
    )

  def test_read_configs_0(self):
    self.assertEqual(
        dataset_providers._get_read_options(0), grain.ReadOptions()
    )

  def test_read_configs_10(self):
    self.assertEqual(
        dataset_providers._get_read_options(10),
        grain.ReadOptions(num_threads=10),
    )

  def test_iter_and_prefetch_none_multiprocessing(self):
    ds = lazy_dataset.RangeLazyMapDataset(10)
    ds = dataset_providers._iter_and_prefetch(
        ds, num_workers=None, num_prefetch_threads=2
    )
    self.assertDictEqual(iter(ds).get_state(), {"next_index": 0})

  def test_iter_and_prefetch_zero_multiprocessing(self):
    ds = lazy_dataset.RangeLazyMapDataset(10)
    ds = dataset_providers._iter_and_prefetch(
        ds, num_workers=0, num_prefetch_threads=2
    )
    self.assertDictEqual(iter(ds).get_state(), {"next_index": 0})

  def test_iter_and_prefetch_with_multiprocessing(self):
    ds = lazy_dataset.RangeLazyMapDataset(10)
    ds = dataset_providers._iter_and_prefetch(
        ds, num_workers=2, num_prefetch_threads=2
    )
    self.assertDictEqual(
        iter(ds).get_state(),
        {
            "workers_state": {"0": {"next_index": 0}, "1": {"next_index": 0}},
            "iterations_to_skip": {"0": 0, "1": 0},
            "last_worker_index": -1,
        },
    )


if __name__ == "__main__":
  mp.handle_test_main(absltest.main)
