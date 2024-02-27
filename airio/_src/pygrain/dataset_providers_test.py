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

"""Tests for airio.pygrain.dataset_providers."""

import functools
import os
from typing import Dict, Sequence
from unittest import mock

from absl.testing import absltest
from airio._src.core import data_sources as core_data_sources
from airio._src.core import dataset_providers as core_dataset_providers
# Import "preprocessors" as "preprocessors_lib" to prevent naming conflicts with
# "preprocessors" attrs in this file.
from airio._src.core import preprocessors as core_preprocessors_lib
from airio._src.core import test_utils
from airio._src.core import tokenizer
from airio._src.pygrain import data_sources
from airio._src.pygrain import dataset_providers
from airio._src.pygrain import preprocessors as preprocessors_lib
from airio._src.pygrain.common import feature_converters
import grain.python as grain
import jax
import numpy as np
from seqio import vocabularies
import tensorflow_datasets as tfds



lazy_dataset = grain.experimental.lazy_dataset
_SOURCE_NAME = "imdb_reviews"
_SOURCE_NUM_EXAMPLES = 3
_SOURCE_SPLITS = frozenset(["train", "test", "unsupervised"])


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
      preprocessors_lib.MapFnTransform(_imdb_preprocessor),
      preprocessors_lib.MapFnTransform(
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
    source: core_data_sources.DataSource | None,
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


def _create_task_builder(
    source: core_data_sources.DataSource | None,
    preprocessors: (
        Sequence[preprocessors_lib.PyGrainAirIOPreprocessor] | None
    ) = None,
    task_name: str = "dummy_airio_task",
) -> dataset_providers.GrainTaskBuilder:
  return dataset_providers.GrainTaskBuilder(
      task_name=task_name,
      source=source,
      preprocessors=preprocessors,
  )


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


class DatasetProviderBaseTest(absltest.TestCase):

  @mock.patch.multiple(
      core_dataset_providers.DatasetProviderBase, __abstractmethods__=set()
  )
  def test_protocol(self):
    base = core_dataset_providers.DatasetProviderBase
    self.assertIsNone(base.get_dataset(self, split=""))
    self.assertIsNone(base.num_input_examples(self, split=""))


class TaskTest(absltest.TestCase):

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

    self._map_transform_idx_1 = preprocessors_lib.MapFnTransform(
        functools.partial(test_map_fn, idx=1)
    )
    self._simple_to_imdb_prep = preprocessors_lib.MapFnTransform(
        simple_to_imdb_map_fn
    )

  def test_create_task_with_source_only_succeeds(self):
    task = _create_task(source=_create_source(), preprocessors=None)
    self.assertIsInstance(task.source, core_data_sources.DataSource)
    self.assertIsInstance(task.source, data_sources.TfdsDataSource)
    self.assertEmpty(task.get_preprocessors())

  def test_create_task_with_source_and_empty_preprocessors_succeeds(self):
    task = _create_task(source=_create_source(), preprocessors=[])
    self.assertIsInstance(task.source, core_data_sources.DataSource)
    self.assertIsInstance(task.source, data_sources.TfdsDataSource)
    self.assertEmpty(task.get_preprocessors())

  def test_create_task(self):
    source = _create_source(splits=_SOURCE_SPLITS)
    task = _create_task(
        source=source,
        preprocessors=_create_preprocessors(),
        task_name="dummy_airio_task",
    )
    self.assertIsInstance(task.source, core_data_sources.DataSource)
    self.assertIsInstance(task.source, data_sources.TfdsDataSource)
    self.assertEqual(task.name, "dummy_airio_task")
    self.assertEqual(task.splits, _SOURCE_SPLITS)

  def test_empty_splits(self):
    source = _create_source(splits=[])
    task = _create_task(source=source, preprocessors=_create_preprocessors())
    self.assertEmpty(task.splits)

  def test_none_splits(self):
    with tfds.testing.mock_data(_SOURCE_NUM_EXAMPLES):
      source = data_sources.TfdsDataSource(tfds_name=_SOURCE_NAME, splits=None)
    task = _create_task(source=source, preprocessors=_create_preprocessors())
    self.assertEmpty(task.splits)

  def test_num_input_examples(self):
    source = _create_source(
        splits=_SOURCE_SPLITS,
        num_examples=_SOURCE_NUM_EXAMPLES,
    )
    task = _create_task(source=source, preprocessors=_create_preprocessors())
    num_examples = task.num_input_examples(split="train")
    self.assertEqual(num_examples, _SOURCE_NUM_EXAMPLES)

  def test_task_get_dataset(self):
    source = _create_source(
        source_name=_SOURCE_NAME,
        splits=_SOURCE_SPLITS,
        num_examples=_SOURCE_NUM_EXAMPLES,
    )
    task = _create_task(source=source, preprocessors=_create_preprocessors())
    ds = task.get_dataset(split="train", shuffle=False)
    expected = [
        {
            "inputs_pretokenized": "imdb ebc   ahgjefjhfe",
            "inputs": [
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
            ],
            "targets_pretokenized": "positive",
            "targets": [3, 15, 7, 6, 8, 24, 8, 25, 4],
        },
        {
            "inputs_pretokenized": "imdb hj aijbcidcibdg",
            "inputs": [
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
            ],
            "targets_pretokenized": "negative",
            "targets": [3, 22, 4, 2, 18, 8, 25, 4],
        },
        {
            "inputs_pretokenized": "imdb acdhdacfhhjb",
            "inputs": [
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
            ],
            "targets_pretokenized": "positive",
            "targets": [3, 15, 7, 6, 8, 24, 8, 25, 4],
        },
    ]
    test_utils.assert_datasets_equal(ds, expected)

  def test_task_get_dataset_with_runtime_preps_without_batching(self):
    source = _create_source(
        source_name=_SOURCE_NAME,
        splits=_SOURCE_SPLITS,
        num_examples=_SOURCE_NUM_EXAMPLES,
    )
    task = _create_task(source=source, preprocessors=_create_preprocessors())
    ds = task.get_dataset(
        split="train",
        runtime_preprocessors=_create_runtime_preprocessors(),
        shuffle=False,
    )
    expected = [
        {
            "encoder_input_tokens": [
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
            ],
            "decoder_target_tokens": [3, 15, 7, 6, 8, 24, 8, 25, 4],
            "decoder_input_tokens": [0, 3, 15, 7, 6, 8, 24, 8, 25],
            "decoder_loss_weights": [1, 1, 1, 1, 1, 1, 1, 1, 1],
        },
        {
            "encoder_input_tokens": [
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
            ],
            "decoder_target_tokens": [3, 22, 4, 2, 18, 8, 25, 4],
            "decoder_input_tokens": [0, 3, 22, 4, 2, 18, 8, 25],
            "decoder_loss_weights": [1, 1, 1, 1, 1, 1, 1, 1],
        },
        {
            "encoder_input_tokens": [
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
            ],
            "decoder_target_tokens": [3, 15, 7, 6, 8, 24, 8, 25, 4],
            "decoder_input_tokens": [0, 3, 15, 7, 6, 8, 24, 8, 25],
            "decoder_loss_weights": [1, 1, 1, 1, 1, 1, 1, 1, 1],
        },
    ]
    test_utils.assert_datasets_equal(ds, expected)

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
    )
    expected_first_batch = [
        {
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
                ],
            ],
            "decoder_target_tokens": [
                [3, 15, 7, 6, 8, 24, 8, 25, 4, 0],
                [3, 22, 4, 2, 18, 8, 25, 4, 0, 0],
            ],
            "decoder_input_tokens": [
                [0, 3, 15, 7, 6, 8, 24, 8, 25, 4],
                [0, 3, 22, 4, 2, 18, 8, 25, 4, 0],
            ],
            "decoder_loss_weights": [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            ],
        },
        {
            "encoder_input_tokens": [[
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
            ]],
            "decoder_target_tokens": [[3, 15, 7, 6, 8, 24, 8, 25, 4, 0]],
            "decoder_input_tokens": [[0, 3, 15, 7, 6, 8, 24, 8, 25, 4]],
            "decoder_loss_weights": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 0]],
        },
    ]
    test_utils.assert_datasets_equal(ds, expected_first_batch)

  def test_task_get_dataset_with_shard_info(self):
    source = _create_source(num_examples=_SOURCE_NUM_EXAMPLES)
    task = _create_task(source=source, preprocessors=_create_preprocessors())
    ds = task.get_dataset(
        shard_info=core_dataset_providers.ShardInfo(index=0, num_shards=1)
    )
    num_examples = 0
    for _ in ds:
      num_examples += 1
    self.assertEqual(num_examples, _SOURCE_NUM_EXAMPLES)

  def test_task_get_dataset_nonexistent_split(self):
    source = _create_source(splits=_SOURCE_SPLITS)
    task = _create_task(source=source, preprocessors=_create_preprocessors())
    with self.assertRaisesRegex(ValueError, "Split nonexistent not found in"):
      task.get_dataset(split="nonexistent")

  def test_task_get_dataset_with_lazy_iter_prep(self):
    task_with_iter = _create_task(
        source=_create_fn_src(num_elements=10),
        preprocessors=[
            preprocessors_lib.LazyIterTransform(
                lambda ds, unused_args, unused_rng: TestFilterLazyIterDataset(
                    ds, threshold=4
                ),
                update_runtime_args=lambda x: x,
            ),
            self._map_transform_idx_1,
        ],
        task_name="test_task_with_iter",
    )
    ds = task_with_iter.get_dataset(shuffle=False)
    expected_dataset = [
        {"idx": 1, "val": 5},
        {"idx": 1, "val": 6},
        {"idx": 1, "val": 7},
        {"idx": 1, "val": 8},
        {"idx": 1, "val": 9},
    ]
    self.assertListEqual(list(ds), expected_dataset)

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
    )
    expected_dataset = [
        {
            "decoder_input_tokens": [[0], [0], [0], [0]],
            "decoder_loss_weights": [[1], [1], [1], [1]],
            "decoder_target_tokens": [[6], [7], [8], [9]],
            "encoder_input_tokens": [[5, 5], [6, 6], [7, 7], [8, 8]],
        },
        {
            "decoder_input_tokens": [[0]],
            "decoder_loss_weights": [[1]],
            "decoder_target_tokens": [[10]],
            "encoder_input_tokens": [[9, 9]],
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

  def test_task_get_dataset_with_none_elements(self):
    task_with_none = _create_task(
        source=_create_fn_src(num_elements=10),
        preprocessors=[
            preprocessors_lib.FilterFnTransform(lambda x: x > 4),
            self._map_transform_idx_1,
        ],
        task_name="test_task_with_none",
    )
    ds = task_with_none.get_dataset(shuffle=False)
    expected_dataset = [
        {"idx": 1, "val": 5},
        {"idx": 1, "val": 6},
        {"idx": 1, "val": 7},
        {"idx": 1, "val": 8},
        {"idx": 1, "val": 9},
    ]
    self.assertListEqual(list(ds), expected_dataset)

  def test_task_get_dataset_with_none_elements_with_runtime_preps_and_batching(
      self,
  ):
    task_with_iter = _create_task(
        source=_create_fn_src(num_elements=10),
        preprocessors=[
            preprocessors_lib.FilterFnTransform(lambda x: x > 4),
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
    )
    expected_dataset = [
        {
            "decoder_input_tokens": [[0], [0], [0], [0]],
            "decoder_loss_weights": [[1], [1], [1], [1]],
            "decoder_target_tokens": [[6], [7], [8], [9]],
            "encoder_input_tokens": [[5, 5], [6, 6], [7, 7], [8, 8]],
        },
        {
            "decoder_input_tokens": [[0]],
            "decoder_loss_weights": [[1]],
            "decoder_target_tokens": [[10]],
            "encoder_input_tokens": [[9, 9]],
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

  def test_task_get_dataset_by_step_without_runtime_preps(self):
    source = _create_source(source_name=_SOURCE_NAME)
    task = _create_task(source=source, preprocessors=_create_preprocessors())
    ds_by_step = task.get_dataset_by_step(num_records=1)
    expected = [
        [{
            "text": "ebc   ahgjefjhfe",
            "label": 1,
        }],
        [{
            "inputs": "imdb ebc   ahgjefjhfe",
            "targets": "positive",
        }],
        [{
            "inputs_pretokenized": "imdb ebc   ahgjefjhfe",
            "inputs": [
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
            ],
            "targets_pretokenized": "positive",
            "targets": [3, 15, 7, 6, 8, 24, 8, 25, 4],
        }],
    ]
    for i, step in enumerate(expected):
      test_utils.assert_datasets_equal(ds_by_step[i], step)

  def test_task_get_dataset_by_step_with_runtime_preps(self):
    source = _create_source()
    task = _create_task(source=source, preprocessors=_create_preprocessors())
    sequence_lengths = {"inputs": 20, "targets": 10}
    ds_by_step = task.get_dataset_by_step(
        num_records=1,
        sequence_lengths=sequence_lengths,
        batch_size=2,
        runtime_preprocessors=_create_runtime_preprocessors(),
        shuffle=False,
    )
    expected = [
        # Original data.
        [{"label": 1, "text": "ebc   ahgjefjhfe"}],
        # IMDB preprocessor.
        [{"inputs": "imdb ebc   ahgjefjhfe", "targets": "positive"}],
        # Tokenizer.
        [{
            "inputs_pretokenized": "imdb ebc   ahgjefjhfe",
            "inputs": [
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
            ],
            "targets_pretokenized": "positive",
            "targets": [3, 15, 7, 6, 8, 24, 8, 25, 4],
        }],
        # Trim.
        [{
            "inputs_pretokenized": "imdb ebc   ahgjefjhfe",
            "inputs": [
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
            ],
            "targets_pretokenized": "positive",
            "targets": [3, 15, 7, 6, 8, 24, 8, 25, 4],
        }],  # Pad.
        [{
            "inputs_pretokenized": "imdb ebc   ahgjefjhfe",
            "inputs": [
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
            "targets_pretokenized": "positive",
            "targets": [3, 15, 7, 6, 8, 24, 8, 25, 4, 0],
        }],
        # Convert to model features.
        [{
            "encoder_input_tokens": [
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
            "decoder_target_tokens": [3, 15, 7, 6, 8, 24, 8, 25, 4, 0],
            "decoder_input_tokens": [0, 3, 15, 7, 6, 8, 24, 8, 25, 4],
            "decoder_loss_weights": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        }],
        # Batching.
        [{
            "decoder_input_tokens": [[0, 3, 15, 7, 6, 8, 24, 8, 25, 4]],
            "decoder_loss_weights": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 0]],
            "decoder_target_tokens": [[3, 15, 7, 6, 8, 24, 8, 25, 4, 0]],
            "encoder_input_tokens": [[
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
            ]],
        }],
    ]
    for i, step in enumerate(expected):
      test_utils.assert_datasets_equal(ds_by_step[i], step)

  def test_task_get_dataset_by_step_without_transformations(self):
    source = _create_source(source_name=_SOURCE_NAME)
    task = _create_task(source=source, preprocessors=[])
    ds_by_step = task.get_dataset_by_step(num_records=1)
    expected = [
        [{
            "text": "ebc   ahgjefjhfe",
            "label": 1,
        }],
    ]
    test_utils.assert_datasets_equal(ds_by_step[0], expected[0])

  def test_task_get_dataset_by_step_invalid_num_records(self):
    source = _create_source()
    task = _create_task(source=source, preprocessors=[])
    ds_by_step = task.get_dataset_by_step(num_records=-1)
    self.assertLen(
        list(ds_by_step[0]),
        dataset_providers.DEFAULT_NUM_RECORDS_TO_INSPECT,
    )
    ds_by_step = task.get_dataset_by_step(num_records=0)
    self.assertLen(
        list(ds_by_step[0]),
        dataset_providers.DEFAULT_NUM_RECORDS_TO_INSPECT,
    )
    ds_by_step = task.get_dataset_by_step(
        num_records=dataset_providers.MAX_NUM_RECORDS_TO_INSPECT + 1,
    )
    self.assertLen(
        list(ds_by_step[0]), dataset_providers.MAX_NUM_RECORDS_TO_INSPECT
    )

  def test_get_updated_runtime_args(self):
    def update_runtime_args_1(args):
      args.sequence_lengths.update({"new_val": 5})
      return args

    def update_runtime_args_2(args):
      args.sequence_lengths.update({"another_val": 7})
      return args

    prep_1 = preprocessors_lib.MapFnTransform(
        lambda x: x,
        update_runtime_args=update_runtime_args_1,
    )
    prep_2 = preprocessors_lib.MapFnTransform(
        lambda x: x,
        update_runtime_args=update_runtime_args_2,
    )
    task = dataset_providers.GrainTask(
        "test", source=_create_source(), preprocessors=[prep_1, prep_2]
    )
    runtime_args = core_preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 3}, split="train"
    )
    updated_runtime_args = task.get_updated_runtime_args(
        runtime_args, runtime_preprocessors=None
    )
    expected_runtime_args = core_preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 3, "new_val": 5, "another_val": 7},
        split="train",
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

  def test_task_get_dataset_with_runtime_args(self):
    def simple_to_imdb_map_fn(
        ex, rargs: core_preprocessors_lib.AirIOInjectedRuntimeArgs
    ):
      return {
          "inputs_pretokenized": f"{ex}",
          "inputs": np.array([ex] * rargs.sequence_lengths["inputs"]),
          "targets_pretokenized": f"{ex + 1}",
          "targets": np.array([ex + 1] * rargs.sequence_lengths["targets"]),
      }

    simple_task = _create_task(
        task_name="test_task1",
        source=_create_fn_src(),
        preprocessors=[preprocessors_lib.MapFnTransform(simple_to_imdb_map_fn)],
    )
    ds = simple_task.get_dataset(
        sequence_lengths={"inputs": 20, "targets": 10}, shuffle=False
    )
    expected = [
        {
            "inputs_pretokenized": "0",
            "inputs": [0] * 20,
            "targets_pretokenized": "1",
            "targets": [1] * 10,
        },
        {
            "inputs_pretokenized": "1",
            "inputs": [1] * 20,
            "targets_pretokenized": "2",
            "targets": [2] * 10,
        },
        {
            "inputs_pretokenized": "2",
            "inputs": [2] * 20,
            "targets_pretokenized": "3",
            "targets": [3] * 10,
        },
        {
            "inputs_pretokenized": "3",
            "inputs": [3] * 20,
            "targets_pretokenized": "4",
            "targets": [4] * 10,
        },
        {
            "inputs_pretokenized": "4",
            "inputs": [4] * 20,
            "targets_pretokenized": "5",
            "targets": [5] * 10,
        },
    ]
    test_utils.assert_datasets_equal(list(ds), expected)

  def test_task_get_dataset_by_step_with_runtime_args(self):
    def simple_to_imdb_map_fn(
        ex, rargs: core_preprocessors_lib.AirIOInjectedRuntimeArgs
    ):
      return {
          "inputs_pretokenized": f"{ex}",
          "inputs": np.array([ex] * rargs.sequence_lengths["inputs"]),
          "targets_pretokenized": f"{ex + 1}",
          "targets": np.array([ex + 1] * rargs.sequence_lengths["targets"]),
      }

    simple_task = _create_task(
        task_name="test_task1",
        source=_create_fn_src(),
        preprocessors=[preprocessors_lib.MapFnTransform(simple_to_imdb_map_fn)],
    )
    ds = simple_task.get_dataset_by_step(
        sequence_lengths={"inputs": 20, "targets": 10}, shuffle=False
    )
    expected = [
        [0, 1],
        [
            {
                "inputs_pretokenized": "0",
                "inputs": [0] * 20,
                "targets_pretokenized": "1",
                "targets": [1] * 10,
            },
            {
                "inputs_pretokenized": "1",
                "inputs": [1] * 20,
                "targets_pretokenized": "2",
                "targets": [2] * 10,
            },
        ],
    ]
    # src
    self.assertListEqual(list(ds[0]), expected[0])
    # preprocessed
    test_utils.assert_datasets_equal(ds[1], expected[1])

  def test_task_switch_to_lazy_dataset(self):
    # Add a preprocessor that consumes a `LazyMapDataset`, and verify that
    # Task.get_dataset() works correctly by running preprocessing using
    # lazy_dataset instead of DataLoader operations.
    def lazy_id_fn(
        ds: lazy_dataset.LazyMapDataset,
        rargs: core_preprocessors_lib.AirIOInjectedRuntimeArgs,
        rng: jax.Array,
    ):
      del rargs, rng
      return ds

    preprocessors = _create_preprocessors() + [
        preprocessors_lib.LazyMapTransform(
            lazy_id_fn,
            update_runtime_args=lambda rargs: rargs,
            produces_none_elements=False,
            requires_non_none_elements=False,
        )
    ]
    source = _create_source(
        source_name=_SOURCE_NAME,
        splits=_SOURCE_SPLITS,
        num_examples=_SOURCE_NUM_EXAMPLES,
    )

    task = _create_task(source=source, preprocessors=preprocessors)
    ds = task.get_dataset(split="train", shuffle=False)
    expected = [
        {
            "inputs_pretokenized": "imdb ebc   ahgjefjhfe",
            "inputs": [
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
            ],
            "targets_pretokenized": "positive",
            "targets": [3, 15, 7, 6, 8, 24, 8, 25, 4],
        },
        {
            "inputs_pretokenized": "imdb hj aijbcidcibdg",
            "inputs": [
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
            ],
            "targets_pretokenized": "negative",
            "targets": [3, 22, 4, 2, 18, 8, 25, 4],
        },
        {
            "inputs_pretokenized": "imdb acdhdacfhhjb",
            "inputs": [
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
            ],
            "targets_pretokenized": "positive",
            "targets": [3, 15, 7, 6, 8, 24, 8, 25, 4],
        },
    ]
    test_utils.assert_datasets_equal(ds, expected)

  def test_task_switch_to_lazy_dataset_runtime_preprocessors(self):
    # Add a preprocessor that consumes a `LazyMapDataset`, and verify that
    # Task.get_dataset() works correctly by running preprocessing using
    # lazy_dataset instead of DataLoader operations.
    def lazy_id_fn(
        ds: lazy_dataset.LazyMapDataset,
        rargs: core_preprocessors_lib.AirIOInjectedRuntimeArgs,
        rng: jax.Array,
    ):
      del rargs, rng
      return ds

    preprocessors = _create_preprocessors()
    runtime_preprocessors = [
        preprocessors_lib.LazyMapTransform(
            lazy_id_fn,
            update_runtime_args=lambda rargs: rargs,
            produces_none_elements=False,
            requires_non_none_elements=False,
        )
    ]
    source = _create_source(
        source_name=_SOURCE_NAME,
        splits=_SOURCE_SPLITS,
        num_examples=_SOURCE_NUM_EXAMPLES,
    )

    task = _create_task(source=source, preprocessors=preprocessors)
    ds = task.get_dataset(
        split="train",
        shuffle=False,
        runtime_preprocessors=runtime_preprocessors,
    )
    expected = [
        {
            "inputs_pretokenized": "imdb ebc   ahgjefjhfe",
            "inputs": [
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
            ],
            "targets_pretokenized": "positive",
            "targets": [3, 15, 7, 6, 8, 24, 8, 25, 4],
        },
        {
            "inputs_pretokenized": "imdb hj aijbcidcibdg",
            "inputs": [
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
            ],
            "targets_pretokenized": "negative",
            "targets": [3, 22, 4, 2, 18, 8, 25, 4],
        },
        {
            "inputs_pretokenized": "imdb acdhdacfhhjb",
            "inputs": [
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
            ],
            "targets_pretokenized": "positive",
            "targets": [3, 15, 7, 6, 8, 24, 8, 25, 4],
        },
    ]
    test_utils.assert_datasets_equal(ds, expected)

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

  def test_task_lazy_dataset_batch_after_shuffle(self):
    test_task = _create_task(
        source=_create_fn_src(num_elements=10),
        preprocessors=[],
    )
    test_task._switch_to_lazy_dataset = mock.Mock(return_value=True)
    ds = test_task.get_dataset(shuffle=True, seed=94043, batch_size=5)
    ds = list(ds)
    expected_ds = [[5, 3, 7, 4, 2], [8, 6, 0, 1, 9]]
    for actual, expected in zip(ds, expected_ds, strict=True):
      np.testing.assert_array_equal(actual, expected)

  def test_task_lazy_dataset_runtime_preprocessors_after_shuffle(self):
    class ElemAndIdMapDataset(lazy_dataset.LazyMapDataset):
      """Returns a pair of the element and its index."""

      def __len__(self):
        return len(self._parent)

      def __getitem__(self, index):
        if isinstance(index, slice):
          return self.slice(index)
        return (self._parent[index], index)

    def prep_fn(ds, runtime_args, rng):
      del runtime_args, rng
      return ElemAndIdMapDataset(ds)

    runtime_prep = preprocessors_lib.LazyMapTransform(
        prep_fn,
        update_runtime_args=lambda rargs: rargs,
        produces_none_elements=False,
        requires_non_none_elements=False,
    )
    test_task = _create_task(
        source=_create_fn_src(num_elements=10), preprocessors=[]
    )
    test_task._switch_to_lazy_dataset = mock.Mock(return_value=True)
    ds = test_task.get_dataset(
        shuffle=True, seed=94043, runtime_preprocessors=[runtime_prep]
    )
    ds = list(ds)

    expected_ds = [
        (5, 0),  # 6th elem is at 1st position after shuffle.
        (3, 1),  # 3rd elem is at 2nd position after shuffle.
        (7, 2),  # and so on.
        (4, 3),
        (2, 4),  # Note: the ids are not shuffled, only the examples, meaning
        (8, 5),  # that the shuffling ran before the runtime preprocessor.
        (6, 6),
        (0, 7),
        (1, 8),
        (9, 9),
    ]
    for actual, expected in zip(ds, expected_ds, strict=True):
      np.testing.assert_array_equal(actual, expected)

  def test_task_requires_non_none_prep_raises_error(self):
    produces_none_prep = preprocessors_lib.LazyMapTransform(
        lambda ds, *_: ds,
        update_runtime_args=lambda args: args,
        produces_none_elements=True,
        requires_non_none_elements=False,
    )
    requires_non_none_prep = preprocessors_lib.LazyMapTransform(
        lambda ds, *_: ds,
        update_runtime_args=lambda args: args,
        produces_none_elements=True,
        requires_non_none_elements=True,
    )
    test_task = _create_task(
        source=_create_fn_src(num_elements=10),
        preprocessors=[produces_none_prep, requires_non_none_prep],
    )
    test_task._switch_to_lazy_dataset = mock.Mock(return_value=True)
    with self.assertRaisesRegex(
        ValueError,
        "There are preprocessors in this Task that produce none elements.*",
    ):
      _ = test_task.get_dataset()

  def test_task_requires_non_none_prep_converts_ds_to_iter(self):
    class FilterMapDataset(lazy_dataset.LazyMapDataset):
      """Filters out the 3rd element."""

      def __len__(self):
        return len(self._parent)

      def __getitem__(self, index):
        if isinstance(index, slice):
          return self.slice(index)
        if index > 2 and index < 4:
          return None
        return self._parent[index]

    def prep_fn(ds, runtime_args, rng):
      del runtime_args, rng
      return FilterMapDataset(ds)

    make_dict_prep = preprocessors_lib.MapFnTransform(lambda x: {"val": x})
    filter_prep = preprocessors_lib.LazyMapTransform(
        prep_fn,
        update_runtime_args=lambda rargs: rargs,
        produces_none_elements=True,
        requires_non_none_elements=False,
    )
    test_task = _create_task(
        source=_create_fn_src(num_elements=10),
        preprocessors=[make_dict_prep, filter_prep],
    )
    test_task._switch_to_lazy_dataset = mock.Mock(return_value=True)
    ds = test_task.get_dataset(shuffle=False, batch_size=2)
    ds = list(ds)
    # The none element between '2' and '4' is removed when the dataset is
    # converted to an iterator because requires_non_none_elements is True
    # for batching.
    expected_ds = [
        {"val": [0, 1]},
        {"val": [2, 4]},
        {"val": [5, 6]},
        {"val": [7, 8]},
        {"val": [9]},
    ]
    for actual, expected in zip(ds, expected_ds, strict=True):
      np.testing.assert_array_equal(actual["val"], expected["val"])

  def test_task_initial_ds_has_non_none_elems(self):
    requires_non_none_prep = preprocessors_lib.LazyMapTransform(
        lambda ds, *_: ds,
        update_runtime_args=lambda args: args,
        produces_none_elements=True,
        requires_non_none_elements=True,
    )
    test_task = _create_task(
        source=_create_fn_src(num_elements=10),
        preprocessors=[requires_non_none_prep],
    )
    test_task._switch_to_lazy_dataset = mock.Mock(return_value=True)
    ds = test_task.get_dataset(shuffle=False, batch_size=2)
    ds = list(ds)
    expected_ds = [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9],
    ]
    for actual, expected in zip(ds, expected_ds, strict=True):
      np.testing.assert_array_equal(actual, expected)

  def test_task_no_preps_has_non_none_elems(self):
    requires_non_none_prep = preprocessors_lib.LazyMapTransform(
        lambda ds, *_: ds,
        update_runtime_args=lambda args: args,
        produces_none_elements=True,
        requires_non_none_elements=True,
    )
    test_task = _create_task(
        source=_create_fn_src(num_elements=10),
        preprocessors=[],
    )
    test_task._switch_to_lazy_dataset = mock.Mock(return_value=True)
    ds = test_task.get_dataset(
        shuffle=False,
        runtime_preprocessors=[requires_non_none_prep],
        batch_size=2,
    )
    ds = list(ds)
    expected_ds = [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9],
    ]
    for actual, expected in zip(ds, expected_ds, strict=True):
      np.testing.assert_array_equal(actual, expected)



class TaskBuilderTest(absltest.TestCase):

  def test_task_builder_from_task_copies_params_correctly(self):
    """Verify that the TaskBuilder object is created with correct params."""
    task = _create_task(
        source=_create_source(),
        task_name="dummy_airio_task",
        preprocessors=_create_preprocessors(),
    )
    task_builder = dataset_providers.GrainTaskBuilder.from_task(task)
    self.assertEqual(task_builder._task_name, "dummy_airio_task")
    self.assertEqual(task_builder._source, task.source)
    self.assertEqual(task_builder._preprocessors, task.get_preprocessors())

  def test_task_builder_build_copies_task_correctly(self):
    task_name = "dummy_airio_task"
    source = _create_source()
    preprocessors = _create_preprocessors()
    task_builder = _create_task_builder(
        source=source,
        preprocessors=preprocessors,
        task_name=task_name,
    )
    new_task = task_builder.build()
    self.assertEqual(new_task.name, task_name)
    self.assertEqual(new_task.source, source)
    self.assertEqual(new_task.get_preprocessors(), preprocessors)

  def test_task_builder_set_name_updates_name_correctly(self):
    source = _create_source()
    preprocessors = _create_preprocessors()
    task_builder = _create_task_builder(
        source=source,
        preprocessors=preprocessors,
        task_name="dummy_airio_task",
    )
    task_builder.set_task_name("new_dummy_task")
    new_task = task_builder.build()
    self.assertEqual(new_task.name, "new_dummy_task")
    # Verify rest of the properties are unchanged.
    self.assertEqual(new_task.source, source)
    self.assertEqual(new_task.get_preprocessors(), preprocessors)

  def test_task_builder_set_preprocessors_updates_preprocessors_correctly(self):
    task_name = "dummy_airio_task"
    source = _create_source()
    task_builder = _create_task_builder(
        source=source,
        preprocessors=_create_preprocessors(),
        task_name=task_name,
    )
    new_preprocessors = [preprocessors_lib.MapFnTransform(_imdb_preprocessor)]
    task_builder.set_preprocessors(new_preprocessors)
    new_task = task_builder.build()
    self.assertEqual(new_task.get_preprocessors(), new_preprocessors)
    # Verify rest of the properties are unchanged.
    self.assertEqual(new_task.name, task_name)
    self.assertEqual(new_task.source, source)

  def test_task_builder_set_data_source_updates_source_correctly(self):
    task_name = "dummy_airio_task"
    preprocessors = _create_preprocessors()
    task_builder = _create_task_builder(
        source=_create_source(),
        preprocessors=preprocessors,
        task_name=task_name,
    )
    new_splits = ["train"]
    new_source = _create_source(splits=new_splits)
    task_builder.set_data_source(new_source)
    new_task = task_builder.build()
    self.assertEqual(new_task.source, new_source)
    # Verify rest of the properties are unchanged.
    self.assertEqual(new_task.name, task_name)
    self.assertEqual(new_task.get_preprocessors(), preprocessors)

  def test_task_builder_raises_error_when_source_is_none(self):
    task_builder = _create_task_builder(
        source=None, preprocessors=_create_preprocessors()
    )
    with self.assertRaisesRegex(
        ValueError, "Source has not been set on this task builder."
    ):
      task_builder.build()

  def test_task_builder_raises_error_when_preprocessors_is_none(self):
    task_builder = _create_task_builder(
        source=_create_source(), preprocessors=None
    )
    with self.assertRaisesRegex(
        ValueError, "Preprocessors have not been set on this task builder."
    ):
      task_builder.build()

  def test_task_builder_repr(self):
    task_builder = _create_task_builder(
        source=_create_source(), task_name="dummy_airio_task"
    )
    self.assertStartsWith(
        repr(task_builder),
        "GrainTaskBuilder(task_name=dummy_airio_task,"
        " source=<airio._src.pygrain.data_sources.TfdsDataSource",
    )



class MixtureTest(absltest.TestCase):

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
    self._map_transform_idx_1 = preprocessors_lib.MapFnTransform(
        functools.partial(test_map_fn, idx=1)
    )
    self._map_transform_idx_2 = preprocessors_lib.MapFnTransform(
        functools.partial(test_map_fn, idx=2)
    )
    self._simple_to_imdb_prep = preprocessors_lib.MapFnTransform(
        simple_to_imdb_map_fn
    )
    self._simple_task_1 = _create_task(
        task_name="test_task1",
        source=_create_fn_src(),
        preprocessors=[self._map_transform_idx_1],
    )
    self._simple_task_2 = _create_task(
        task_name="test_task2",
        source=_create_fn_src(),
        preprocessors=[self._map_transform_idx_2],
    )
    self._imdb_task = _create_task(
        source=imdb_source, preprocessors=_create_preprocessors()
    )
    self._simple_to_imdb_task = (
        dataset_providers.GrainTaskBuilder.from_task(self._simple_task_1)
        .set_preprocessors([
            preprocessors_lib.MapFnTransform(simple_to_imdb_map_fn),
        ])
        .build()
    )

  def test_mixture_runtime_args_updated_by_task(self):
    def update_runtime_args_fn(rargs):
      return core_preprocessors_lib.AirIOInjectedRuntimeArgs(
          sequence_lengths={"inputs": 20, "targets": 10}, split=rargs.split
      )

    task_with_runtime_args_update = (
        dataset_providers.GrainTaskBuilder.from_task(self._imdb_task)
        .set_preprocessors(
            self._imdb_task._preprocessors
            + [
                preprocessors_lib.MapFnTransform(
                    lambda x: x, update_runtime_args=update_runtime_args_fn
                ),
            ]
        )
        .build()
    )
    mix = dataset_providers.GrainMixture(
        name="test_mix",
        tasks=[task_with_runtime_args_update],
        proportions=[1.0],
    )
    ds = mix.get_dataset(
        sequence_lengths={"xyz": 5, "abc": 7},  # will be updated
        shuffle=False,
        shard_info=core_dataset_providers.ShardInfo(index=0, num_shards=2),
        num_epochs=1,
        runtime_preprocessors=_create_runtime_preprocessors(),
    )
    expected = [
        {
            "encoder_input_tokens": [
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
            "decoder_target_tokens": [3, 15, 7, 6, 8, 24, 8, 25, 4, 0],
            "decoder_input_tokens": [0, 3, 15, 7, 6, 8, 24, 8, 25, 4],
            "decoder_loss_weights": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        },
        {
            "encoder_input_tokens": [
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
            ],
            "decoder_target_tokens": [3, 22, 4, 2, 18, 8, 25, 4, 0, 0],
            "decoder_input_tokens": [0, 3, 22, 4, 2, 18, 8, 25, 4, 0],
            "decoder_loss_weights": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        },
    ]
    test_utils.assert_datasets_equal(ds, expected)

  def test_mixture_with_non_grain_tasks_fails_on_get_dataset(self):
    non_grain_task = core_dataset_providers.TaskBuilder.from_task(
        self._simple_task_1
    ).build()
    with self.assertRaisesRegex(
        ValueError,
        f"Task '{non_grain_task.name}' is not a GrainTask or GrainMixture.",
    ):
      dataset_providers.GrainMixture(
          name="test_mix",
          tasks=[non_grain_task],
          proportions=[1.0],
      )

  def test_simple_mixture(self):
    mix = dataset_providers.GrainMixture(
        name="test_mix",
        tasks=[self._simple_task_1, self._simple_task_2],
        proportions=[1.0, 1.0],
    )
    ds = mix.get_dataset(shuffle=False)
    self.assertListEqual(
        list(ds),
        [
            {"idx": 1, "val": 0},  # task 1, ex 0
            {"idx": 2, "val": 0},  # task 2, ex 0
            {"idx": 1, "val": 1},  # task 1, ex 1
            {"idx": 2, "val": 1},  # task 2, ex 1
            {"idx": 1, "val": 2},  # task 1, ex 2
            {"idx": 2, "val": 2},  # task 2, ex 2
            {"idx": 1, "val": 3},  # task 1, ex 3
            {"idx": 2, "val": 3},  # task 2, ex 3
            {"idx": 1, "val": 4},  # task 1, ex 4
            {"idx": 2, "val": 4},  # task 2, ex 4
        ],
    )

  def test_simple_mixture_stop_on_empty(self):
    mix = dataset_providers.GrainMixture(
        name="test_mix",
        tasks=[self._simple_task_1, self._simple_task_2],
        proportions=[2.0, 1.0],
    )
    ds = mix.get_dataset(shuffle=False)
    self.assertListEqual(
        list(ds),
        [
            {"idx": 1, "val": 0},  # task 1, ex 0
            {"idx": 1, "val": 1},  # task 1, ex 1
            {"idx": 2, "val": 0},  # task 2, ex 0
            {"idx": 1, "val": 2},  # task 1, ex 2
            {"idx": 1, "val": 3},  # task 1, ex 3
            {"idx": 2, "val": 1},  # task 2, ex 1
            {"idx": 1, "val": 4},  # task 1, ex 4
            # task 1 dataset now empty
        ],
    )

  def test_mixture_sharding(self):
    mix = dataset_providers.GrainMixture(
        name="test_mix",
        tasks=[self._simple_task_1, self._simple_task_2],
        proportions=[2.0, 1.0],
    )
    ds = mix.get_dataset(
        shuffle=False,
        shard_info=core_dataset_providers.ShardInfo(index=0, num_shards=2),
    )
    self.assertListEqual(
        list(ds),
        [
            {"idx": 1, "val": 0},  # task 1, ex 0
            {"idx": 1, "val": 1},  # task 1, ex 1
            {"idx": 2, "val": 0},  # task 2, ex 0
            {"idx": 1, "val": 2},  # task 1, ex 2
            # task 1 dataset now empty
        ],
    )

  def test_mixture_shuffling(self):
    mix = dataset_providers.GrainMixture(
        name="test_mix",
        tasks=[self._simple_task_1, self._simple_task_2],
        proportions=[2.0, 1.0],
    )
    ds = mix.get_dataset(
        shuffle=True,
        seed=42,
        shard_info=core_dataset_providers.ShardInfo(index=0, num_shards=2),
    )
    self.assertListEqual(
        list(ds),
        [
            {"idx": 1, "val": 2},
            {"idx": 1, "val": 1},
            {"idx": 2, "val": 2},
            {"idx": 1, "val": 0},
            # task 1 dataset now empty
        ],
    )

  @mock.patch(
      "airio._src.pygrain.lazy_dataset_transforms.ConcatLazyMapDataset",
      new_callable=mock.NonCallableMock,
  )
  def test_single_epoch_concat_not_called(self, unused_mock_concat_fn):
    mix = dataset_providers.GrainMixture(
        name="test_mix",
        tasks=[self._simple_task_1, self._simple_task_2],
        proportions=[1.0, 1.0],
    )
    ds = mix.get_dataset(shuffle=False)
    self.assertListEqual(
        list(ds),
        [
            {"idx": 1, "val": 0},  # task 1, ex 0
            {"idx": 2, "val": 0},  # task 2, ex 0
            {"idx": 1, "val": 1},  # task 1, ex 1
            {"idx": 2, "val": 1},  # task 2, ex 1
            {"idx": 1, "val": 2},  # task 1, ex 2
            {"idx": 2, "val": 2},  # task 2, ex 2
            {"idx": 1, "val": 3},  # task 1, ex 3
            {"idx": 2, "val": 3},  # task 2, ex 3
            {"idx": 1, "val": 4},  # task 1, ex 4
            {"idx": 2, "val": 4},  # task 2, ex 4
        ],
    )
    with self.assertRaisesRegex(
        TypeError, "'NonCallableMock' object is not callable"
    ):
      _ = mix.get_dataset(shuffle=False, num_epochs=2)

  def test_multi_epoch(self):
    mix = dataset_providers.GrainMixture(
        name="test_mix",
        tasks=[self._simple_task_1, self._simple_task_2],
        proportions=[2.0, 1.0],
    )
    ds = mix.get_dataset(
        shuffle=True,
        seed=42,
        shard_info=core_dataset_providers.ShardInfo(index=0, num_shards=2),
        num_epochs=2,
    )
    self.assertListEqual(
        list(ds),
        [
            {"idx": 1, "val": 2},
            {"idx": 1, "val": 1},
            {"idx": 2, "val": 2},
            {"idx": 1, "val": 0},
            # epoch 1 end, no overlapping examples
            {"idx": 1, "val": 1},
            {"idx": 2, "val": 1},
            {"idx": 1, "val": 0},
            {"idx": 1, "val": 2},
            {"idx": 2, "val": 0},
            # task 1 dataset now empty
        ],
        # Note: We get an odd number of examples with num_epochs = 2. This is
        # because we mix after repeating - mixing datasets of length 10 (2
        # epochs). If we repeated after mixing, we'd mix datasets of length 5
        # and get 4 examples, and then repeat to get 8 examples. Repeating
        # earlier enables passing different seeds to epochs for preprocessing.
    )

  def test_multi_epoch_with_stochastic_preprocessor(self):
    def test_random_map_fn(ex, rng):
      ex["var"] = int(jax.random.randint(rng, [], 0, 20))
      return ex

    task1 = (
        dataset_providers.GrainTaskBuilder.from_task(self._simple_task_1)
        .set_preprocessors([
            self._map_transform_idx_1,
            preprocessors_lib.RandomMapFnTransform(test_random_map_fn),
        ])
        .build()
    )
    task2 = (
        dataset_providers.GrainTaskBuilder.from_task(self._simple_task_2)
        .set_preprocessors([
            self._map_transform_idx_2,
            preprocessors_lib.RandomMapFnTransform(test_random_map_fn),
        ])
        .build()
    )
    mix = dataset_providers.GrainMixture(
        name="test_mix", tasks=[task1, task2], proportions=[2.0, 1.0]
    )
    ds = mix.get_lazy_dataset(
        None,
        "train",
        shuffle=True,
        seed=42,
        shard_info=core_dataset_providers.ShardInfo(index=0, num_shards=2),
        num_epochs=2,
    )
    self.assertListEqual(
        list(ds),
        [
            {"idx": 1, "val": 2, "var": 5},  # task 1 ex 2
            {"idx": 1, "val": 1, "var": 9},  # task 1 ex 1
            {"idx": 2, "val": 2, "var": 5},  # task 2 ex 2
            {"idx": 1, "val": 0, "var": 13},  # task 1 ex 0
            # epoch 1 end, no overlapping examples
            {"idx": 1, "val": 1, "var": 19},  # task 1 ex 1
            {"idx": 2, "val": 1, "var": 9},  # task 2 ex 1
            {"idx": 1, "val": 0, "var": 5},  # task 1 ex 0
            {"idx": 1, "val": 2, "var": 17},  # task 1 ex 2
            {"idx": 2, "val": 0, "var": 13},  # task 2 ex 0
            # task 1 dataset now empty
        ],
    )

  def test_indefinite_repeat(self):
    mix = dataset_providers.GrainMixture(
        name="test_mix",
        tasks=[self._simple_task_1, self._simple_task_2],
        proportions=[2.0, 1.0],
    )
    ds = mix.get_dataset(
        shuffle=False,
        shard_info=core_dataset_providers.ShardInfo(index=0, num_shards=2),
        num_epochs=None,
    )
    self.assertListEqual(
        [next(ds) for _ in range(13)],
        [
            {"idx": 1, "val": 0},
            {"idx": 1, "val": 1},
            {"idx": 2, "val": 0},
            {"idx": 1, "val": 2},
            {"idx": 1, "val": 0},  # task 1 starts repeating
            {"idx": 2, "val": 1},
            {"idx": 1, "val": 1},
            {"idx": 1, "val": 2},
            {"idx": 2, "val": 2},
            {"idx": 1, "val": 0},
            {"idx": 1, "val": 1},
            {"idx": 2, "val": 0},  # task 2 starts repeating
            {"idx": 1, "val": 2},
        ],
    )

  def test_mixture_with_different_sources_and_preprocessors(self):
    mix = dataset_providers.GrainMixture(
        name="test_mix",
        tasks=[self._imdb_task, self._simple_to_imdb_task],
        proportions=[1.0, 1.0],
    )
    ds = mix.get_dataset(
        sequence_lengths={"inputs": 20, "targets": 10},
        shuffle=False,
        shard_info=core_dataset_providers.ShardInfo(index=0, num_shards=2),
        num_epochs=1,
    )
    expected = [
        {  # imdb task
            "inputs_pretokenized": "imdb ebc   ahgjefjhfe",
            "inputs": [
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
            ],
            "targets_pretokenized": "positive",
            "targets": [3, 15, 7, 6, 8, 24, 8, 25, 4],
        },
        {  # simple task
            "inputs_pretokenized": "0",
            "inputs": [0] * 20,
            "targets_pretokenized": "1",
            "targets": [1] * 10,
        },
        {  # imdb task
            "inputs_pretokenized": "imdb hj aijbcidcibdg",
            "inputs": [
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
            ],
            "targets_pretokenized": "negative",
            "targets": [3, 22, 4, 2, 18, 8, 25, 4],
        },
        {  # simple task
            "inputs_pretokenized": "1",
            "inputs": [1] * 20,
            "targets_pretokenized": "2",
            "targets": [2] * 10,
        },
        # imdb task now empty.
    ]
    test_utils.assert_datasets_equal(ds, expected)

  def test_mixture_with_different_output_features_fail_batching(self):
    mix = dataset_providers.GrainMixture(
        name="test_mix",
        tasks=[self._imdb_task, self._simple_task_1],
        proportions=[1.0, 1.0],
    )
    ds = mix.get_dataset(
        shuffle=False,
        shard_info=core_dataset_providers.ShardInfo(index=0, num_shards=2),
        num_epochs=1,
        batch_size=2,
    )
    with self.assertRaisesRegex(
        ValueError,
        "(The two structures don't have the same nested structure|Dict key"
        " mismatch;).*",
    ):
      _ = next(ds)

  def test_mixture_with_runtime_preps(self):
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
    )
    expected = [
        {  # imdb task
            "encoder_input_tokens": [
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
            "decoder_target_tokens": [3, 15, 7, 6, 8, 24, 8, 25, 4, 0],
            "decoder_input_tokens": [0, 3, 15, 7, 6, 8, 24, 8, 25, 4],
            "decoder_loss_weights": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        },
        {  # simple task
            "encoder_input_tokens": [0] * 20,
            "decoder_target_tokens": [1] * 10,
            "decoder_input_tokens": [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "decoder_loss_weights": [1] * 10,
        },
        {  # imdb task
            "encoder_input_tokens": [
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
            ],
            "decoder_target_tokens": [3, 22, 4, 2, 18, 8, 25, 4, 0, 0],
            "decoder_input_tokens": [0, 3, 22, 4, 2, 18, 8, 25, 4, 0],
            "decoder_loss_weights": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        },
        {  # simple task
            "encoder_input_tokens": [1] * 20,
            "decoder_target_tokens": [2] * 10,
            "decoder_input_tokens": [0, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            "decoder_loss_weights": [1] * 10,
        },
        # imdb task now empty
    ]
    test_utils.assert_datasets_equal(ds, expected)

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

  def test_mixture_with_batching_only(self):
    mix = dataset_providers.GrainMixture(
        name="test_mix",
        tasks=[self._simple_task_1, self._simple_task_2],
        proportions=[1.0, 1.0],
    )
    ds = mix.get_dataset(
        sequence_lengths={"inputs": 20, "targets": 10},
        shuffle=False,
        shard_info=core_dataset_providers.ShardInfo(index=0, num_shards=2),
        num_epochs=1,
        runtime_preprocessors=None,
        batch_size=2,
    )
    self.assertDictEqual(
        {k: v.tolist() for k, v in next(ds).items()},
        {"idx": [1, 2], "val": [0, 0]},
    )

  def test_mixing_with_iter_test(self):
    # Mix datasets that produce None elements and verify that mixture length and
    # mixing rate are correct
    task_with_none = _create_task(
        source=_create_fn_src(num_elements=10),
        preprocessors=[
            preprocessors_lib.FilterFnTransform(lambda x: x > 4),
            self._map_transform_idx_1,
        ],
        task_name="test_task_with_none",
    )
    ordinary_task = _create_task(
        source=_create_fn_src(num_elements=10),
        preprocessors=[self._map_transform_idx_2],
        task_name="ordinary_task",
    )
    mix = dataset_providers.GrainMixture(
        name="test_mix",
        tasks=[task_with_none, ordinary_task],
        proportions=[1.0, 1.0],
    )
    ds = mix.get_dataset(shuffle=False)
    expected_dataset = [
        {"idx": 2, "val": 0},  # task 2, example 1
        {"idx": 1, "val": 5},  # task 1, example 6 (prev examples filtered)
        {"idx": 2, "val": 1},  # task 2, example 2
        {"idx": 1, "val": 6},  # task 1, example 7
        {"idx": 2, "val": 2},  # task 2, example 3
        {"idx": 1, "val": 7},  # task 1, example 8
        {"idx": 2, "val": 3},  # task 2, example 4
        {"idx": 1, "val": 8},  # task 1, example 9
        {"idx": 2, "val": 4},  # task 2, example 5
        {"idx": 1, "val": 9},  # task 1, example 10 (last example)
        {"idx": 2, "val": 5},  # task 2, example 6
        # There are 4 more elements available in `ordinary_task`, but mixing
        # stops here because there are no more elements in `task_with_none`. The
        # desired mixing rate is achieved.
    ]
    self.assertListEqual(list(ds), expected_dataset)

  def test_mixing_with_iter_test_with_runtime_preps_and_batching(self):
    # Mix datasets that produce None elements and verify that mixture length and
    # mixing rate are correct.
    task_with_none = _create_task(
        source=_create_fn_src(num_elements=10),
        preprocessors=[
            preprocessors_lib.FilterFnTransform(lambda x: x > 4),
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
    )
    expected_dataset = [
        {  # task 2 ex 1, task 1 ex 6, task 2 ex 2, task 1 ex 7
            "decoder_input_tokens": [[0], [0], [0], [0]],
            "decoder_loss_weights": [[1], [1], [1], [1]],
            "decoder_target_tokens": [[1], [6], [2], [7]],
            "encoder_input_tokens": [[0, 0], [5, 5], [1, 1], [6, 6]],
        },
        {  # task 2 ex 3, task 1 ex 8, task 2 ex 4, task 1 ex 9
            "decoder_input_tokens": [[0], [0], [0], [0]],
            "decoder_loss_weights": [[1], [1], [1], [1]],
            "decoder_target_tokens": [[3], [8], [4], [9]],
            "encoder_input_tokens": [[2, 2], [7, 7], [3, 3], [8, 8]],
        },
        {  # task 2 ex 5, task 1 ex 10 (last example), task 2 ex 6
            "decoder_input_tokens": [[0], [0], [0]],
            "decoder_loss_weights": [[1], [1], [1]],
            "decoder_target_tokens": [[5], [10], [6]],
            "encoder_input_tokens": [[4, 4], [9, 9], [5, 5]],
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

  def test_mixing_with_lazy_iter_preprocessor(self):
    # Mix tasks with LazyIter preprocessors and verify that mixture length and
    # mixing rate are correct.
    task_with_iter = _create_task(
        source=_create_fn_src(num_elements=10),
        preprocessors=[
            preprocessors_lib.LazyIterTransform(
                lambda ds, unused_args, unused_rng: TestFilterLazyIterDataset(
                    ds, threshold=4
                ),
                update_runtime_args=lambda x: x,
            ),
            self._map_transform_idx_1,
        ],
        task_name="test_task_with_iter",
    )
    ordinary_task = _create_task(
        source=_create_fn_src(num_elements=10),
        preprocessors=[self._map_transform_idx_2],
        task_name="ordinary_task",
    )
    mix = dataset_providers.GrainMixture(
        name="test_mix",
        tasks=[task_with_iter, ordinary_task],
        proportions=[1.0, 1.0],
    )
    ds = mix.get_dataset(shuffle=False)
    expected_dataset = [
        {"idx": 2, "val": 0},  # task 2, example 1
        {"idx": 1, "val": 5},  # task 1, example 6 (prev examples filtered)
        {"idx": 2, "val": 1},  # task 2, example 2
        {"idx": 1, "val": 6},  # task 1, example 7
        {"idx": 2, "val": 2},  # task 2, example 3
        {"idx": 1, "val": 7},  # task 1, example 8
        {"idx": 2, "val": 3},  # task 2, example 4
        {"idx": 1, "val": 8},  # task 1, example 9
        {"idx": 2, "val": 4},  # task 2, example 5
        {"idx": 1, "val": 9},  # task 1, example 10 (last example)
        {"idx": 2, "val": 5},  # task 2, example 6
        # There are 4 more elements available in `ordinary_task`, but mixing
        # stops here because there are no more elements in `task_with_none`. The
        # desired mixing rate is achieved.
    ]
    self.assertListEqual(list(ds), expected_dataset)

  def test_mixing_with_lazy_iter_preprocessor_with_runtime_preps_and_batching(
      self,
  ):
    # Mix tasks with LazyIter preprocessors and verify that mixture length and
    # mixing rate are correct.
    task_with_none = _create_task(
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
    )
    expected_dataset = [
        {  # task 2 ex 1, task 1 ex 6, task 2 ex 2, task 1 ex 7
            "decoder_input_tokens": [[0], [0], [0], [0]],
            "decoder_loss_weights": [[1], [1], [1], [1]],
            "decoder_target_tokens": [[1], [6], [2], [7]],
            "encoder_input_tokens": [[0, 0], [5, 5], [1, 1], [6, 6]],
        },
        {  # task 2 ex 3, task 1 ex 8, task 2 ex 4, task 1 ex 9
            "decoder_input_tokens": [[0], [0], [0], [0]],
            "decoder_loss_weights": [[1], [1], [1], [1]],
            "decoder_target_tokens": [[3], [8], [4], [9]],
            "encoder_input_tokens": [[2, 2], [7, 7], [3, 3], [8, 8]],
        },
        {  # task 2 ex 5, task 1 ex 10 (last example), task 2 ex 6
            "decoder_input_tokens": [[0], [0], [0]],
            "decoder_loss_weights": [[1], [1], [1]],
            "decoder_target_tokens": [[5], [10], [6]],
            "encoder_input_tokens": [[4, 4], [9, 9], [5, 5]],
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



class MixturePropertiesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tasks = []
    for i in range(5):
      self.tasks.append(
          _create_task(
              source=_create_fn_src(),
              preprocessors=[],
              task_name=f"test_task_{i}",
          )
      )
    self.simple_mix = dataset_providers.GrainMixture(
        name="test_mix_1",
        tasks=self.tasks[:3],
        proportions=[1.0, 0.5, 2.0],
    )
    self.mix_of_mix = dataset_providers.GrainMixture(
        name="test_mix_2",
        tasks=[self.simple_mix, self.tasks[3]],
        proportions=[0.5, 0.7],
    )
    self.mix_of_mix_of_mix = dataset_providers.GrainMixture(
        name="test_mix_3",
        tasks=[self.simple_mix, self.mix_of_mix, self.tasks[4]],
        proportions=[0.5, 0.7, 0.8],
    )

  def test_tasks_or_mixtures(self):
    self.assertListEqual(self.simple_mix.tasks_or_mixtures, self.tasks[:3])
    self.assertListEqual(
        self.mix_of_mix.tasks_or_mixtures, [self.simple_mix, self.tasks[3]]
    )
    self.assertListEqual(
        self.mix_of_mix_of_mix.tasks_or_mixtures,
        [self.simple_mix, self.mix_of_mix, self.tasks[4]],
    )

  def test_total_proportions(self):
    self.assertAlmostEqual(self.simple_mix.total_proportion, 3.5)
    self.assertAlmostEqual(self.mix_of_mix.total_proportion, 1.2)
    self.assertAlmostEqual(self.mix_of_mix_of_mix.total_proportion, 2.0)

  def test_get_proportion(self):
    self.assertAlmostEqual(self.simple_mix.get_proportion(self.tasks[0]), 1.0)
    self.assertAlmostEqual(self.simple_mix.get_proportion(self.tasks[1]), 0.5)
    self.assertAlmostEqual(self.simple_mix.get_proportion(self.tasks[2]), 2.0)
    self.assertAlmostEqual(self.simple_mix.get_proportion(self.tasks[3]), 0.0)
    self.assertAlmostEqual(self.simple_mix.get_proportion(self.tasks[4]), 0.0)

    self.assertAlmostEqual(
        self.mix_of_mix.get_proportion(self.tasks[0]), 0.5 * (1.0 / 3.5)
    )
    self.assertAlmostEqual(
        self.mix_of_mix.get_proportion(self.tasks[1]), 0.5 * (0.5 / 3.5)
    )
    self.assertAlmostEqual(
        self.mix_of_mix.get_proportion(self.tasks[2]), 0.5 * (2.0 / 3.5)
    )
    self.assertAlmostEqual(self.mix_of_mix.get_proportion(self.tasks[3]), 0.7)
    self.assertAlmostEqual(self.mix_of_mix.get_proportion(self.tasks[4]), 0.0)

    self.assertAlmostEqual(
        self.mix_of_mix_of_mix.get_proportion(self.tasks[0]),
        0.5 * (1.0 / 3.5) + 0.7 * (0.5 / 1.2) * (1.0 / 3.5),
    )
    self.assertAlmostEqual(
        self.mix_of_mix_of_mix.get_proportion(self.tasks[1]),
        0.5 * (0.5 / 3.5) + 0.7 * (0.5 / 1.2) * (0.5 / 3.5),
    )
    self.assertAlmostEqual(
        self.mix_of_mix_of_mix.get_proportion(self.tasks[2]),
        0.5 * (2.0 / 3.5) + 0.7 * (0.5 / 1.2) * (2.0 / 3.5),
    )
    self.assertAlmostEqual(
        self.mix_of_mix_of_mix.get_proportion(self.tasks[3]), 0.7 * (0.7 / 1.2)
    )
    self.assertAlmostEqual(
        self.mix_of_mix_of_mix.get_proportion(self.tasks[4]), 0.8
    )

  def test_leaf_tasks(self):
    self.assertListEqual(self.simple_mix.leaf_tasks, self.tasks[:3])
    self.assertListEqual(self.mix_of_mix.leaf_tasks, self.tasks[:4])
    self.assertListEqual(self.mix_of_mix_of_mix.leaf_tasks, self.tasks)

  def test_splits(self):
    self.assertSequenceEqual(self.simple_mix.splits, ["train"])
    self.assertSequenceEqual(self.mix_of_mix.splits, ["train"])
    self.assertSequenceEqual(self.mix_of_mix_of_mix.splits, ["train"])

  def test_num_input_examples(self):
    self.assertEqual(self.simple_mix.num_input_examples("train"), 3 * 5)
    self.assertEqual(self.mix_of_mix.num_input_examples("train"), 3 * 5 + 5)
    self.assertEqual(
        self.mix_of_mix_of_mix.num_input_examples("train"),
        3 * 5 + (3 * 5 + 5) + 5,
    )

  def test_tasks_and_proportions_mismatch(self):
    with self.assertRaisesRegex(
        ValueError,
        "Mixture invalid_mix must have same number of tasks and proportions.*",
    ):
      _ = dataset_providers.GrainMixture(
          "invalid_mix", [self.tasks[0], self.tasks[1]], [1.0]
      )

  def test_duplicate_tasks(self):
    with self.assertRaisesRegex(
        ValueError,
        "Mixture invalid_mix has duplicate tasks.*",
    ):
      _ = dataset_providers.GrainMixture(
          "invalid_mix", [self.tasks[0], self.tasks[0]], [1.0, 1.0]
      )


class EvenSplitTest(absltest.TestCase):

  def test_even_split_one_shard(self):
    # Splitting into one shard returns the entire interval.
    interval = dataset_providers._even_split(
        num_examples=20, shard_index=0, shard_count=1
    )
    self.assertTupleEqual(interval, (0, 20))  # All 20 elements.

  def test_even_split_two_shards(self):
    # Splitting into two shards returns half the interval. Since the number is
    # not perfectly divisible, the first shard has the extra element.
    intervals = []
    for i in range(2):
      intervals.append(
          dataset_providers._even_split(
              num_examples=21, shard_index=i, shard_count=2
          )
      )
    self.assertTupleEqual(intervals[0], (0, 11))  # First 11 elements.
    self.assertTupleEqual(intervals[1], (11, 21))  # Last 10 elements.

  def test_even_split_three_shards(self):
    # Splitting into three shards returns a third of the interval. Since the
    # number is not perfectly divisible, an extra element is added to the first
    # and second shards. The third shard has one less element.
    intervals = []
    for i in range(3):
      intervals.append(
          dataset_providers._even_split(
              num_examples=8, shard_index=i, shard_count=3
          )
      )
    self.assertTupleEqual(intervals[0], (0, 3))  # First 3 elements.
    self.assertTupleEqual(intervals[1], (3, 6))  # Next 3 elements.
    self.assertTupleEqual(intervals[2], (6, 8))  # Last 2 elements.


class DatasetProvidersTest(absltest.TestCase):

  def test_get_dataset(self):
    source = _create_source(
        source_name=_SOURCE_NAME,
        num_examples=_SOURCE_NUM_EXAMPLES,
    )
    task = _create_task(source=source, preprocessors=_create_preprocessors())
    ds = core_dataset_providers.get_dataset(task)
    expected = [
        {
            "inputs_pretokenized": "imdb ebc   ahgjefjhfe",
            "inputs": [
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
            ],
            "targets_pretokenized": "positive",
            "targets": [3, 15, 7, 6, 8, 24, 8, 25, 4],
        },
        {
            "inputs_pretokenized": "imdb hj aijbcidcibdg",
            "inputs": [
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
            ],
            "targets_pretokenized": "negative",
            "targets": [3, 22, 4, 2, 18, 8, 25, 4],
        },
        {
            "inputs_pretokenized": "imdb acdhdacfhhjb",
            "inputs": [
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
            ],
            "targets_pretokenized": "positive",
            "targets": [3, 15, 7, 6, 8, 24, 8, 25, 4],
        },
    ]
    test_utils.assert_datasets_equal(ds, expected)



if __name__ == "__main__":
  absltest.main()
