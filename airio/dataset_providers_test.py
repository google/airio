# Copyright 2023 The AirIO Authors.
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

"""Tests for airio.dataset_providers."""

import functools
import os
from typing import Dict, Sequence
from unittest import mock

from absl.testing import absltest
# from absl.testing import parameterized
from airio import data_sources
from airio import dataset_providers
from airio import feature_converters
from airio import preprocessors as airio_preps
from airio import test_utils
from airio import tokenizer
import jax
import numpy as np
from seqio import vocabularies
import tensorflow_datasets as tfds


_SOURCE_NAME = "imdb_reviews"
_SOURCE_NUM_EXAMPLES = 3
_SOURCE_SPLITS = {"train", "test", "unsupervised"}


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


def _create_tokenizer_config() -> Dict[str, str]:
  test_data_dir = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "test_data"
  )
  sentencepiece_vocab = vocabularies.SentencePieceVocabulary(
      os.path.join(test_data_dir, "sentencepiece", "sentencepiece.model")
  )
  return tokenizer.TokenizerConfig(vocab=sentencepiece_vocab)


def _create_preprocessors() -> Sequence[dataset_providers.GrainPreprocessor]:
  tokenizer_config = _create_tokenizer_config()
  return [
      airio_preps.MapFnTransform(_imdb_preprocessor),
      airio_preps.MapFnTransform(
          functools.partial(
              tokenizer.tokenize,
              tokenizer_configs={
                  "inputs": tokenizer_config,
                  "targets": tokenizer_config,
              },
          )
      ),
  ]


def _create_feature_converter() -> feature_converters.PyGrainFeatureConverter:
  return feature_converters.PyGrainEncDecFeatureConverter()


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
    preprocessors: Sequence[dataset_providers.GrainPreprocessor] | None = None,
    task_name: str = "dummy_airio_task",
) -> dataset_providers.Task:
  """Create example AirIO task."""
  return dataset_providers.Task(
      name=task_name,
      source=source,
      preprocessors=preprocessors,
  )


def _create_task_builder(
    source: data_sources.DataSource | None,
    preprocessors: Sequence[dataset_providers.GrainPreprocessor] | None = None,
    task_name: str = "dummy_airio_task",
) -> dataset_providers.TaskBuilder:
  return dataset_providers.TaskBuilder(
      task_name=task_name,
      source=source,
      preprocessors=preprocessors,
  )


class DatasetProviderBaseTest(absltest.TestCase):

  @mock.patch.multiple(
      dataset_providers.DatasetProviderBase, __abstractmethods__=set()
  )
  def test_protocol(self):
    base = dataset_providers.DatasetProviderBase
    self.assertIsNone(base.get_dataset(self, split=""))
    self.assertIsNone(base.num_input_examples(self, split=""))


class DatasetProvidersTest(absltest.TestCase):

  def test_create_task_with_source_only_succeeds(self):
    task = _create_task(source=_create_source(), preprocessors=None)
    self.assertIsInstance(task.source, data_sources.DataSource)
    self.assertIsInstance(task.source, data_sources.TfdsDataSource)
    self.assertEmpty(task.get_preprocessors())

  def test_create_task_with_source_and_empty_preprocessors_succeeds(self):
    task = _create_task(source=_create_source(), preprocessors=[])
    self.assertIsInstance(task.source, data_sources.DataSource)
    self.assertIsInstance(task.source, data_sources.TfdsDataSource)
    self.assertEmpty(task.get_preprocessors())

  def test_create_task(self):
    source = _create_source(splits=_SOURCE_SPLITS)
    task = _create_task(
        source=source,
        preprocessors=_create_preprocessors(),
        task_name="dummy_airio_task",
    )
    self.assertIsInstance(task.source, data_sources.DataSource)
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

  def test_task_get_dataset_with_feature_converter_without_batching(self):
    source = _create_source(
        source_name=_SOURCE_NAME,
        splits=_SOURCE_SPLITS,
        num_examples=_SOURCE_NUM_EXAMPLES,
    )
    task = _create_task(source=source, preprocessors=_create_preprocessors())
    ds = task.get_dataset(
        split="train",
        feature_converter=_create_feature_converter(),
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
            "decoder_input_tokens": [3, 15, 7, 6, 8, 24, 8, 25, 4],
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
            "decoder_input_tokens": [3, 22, 4, 2, 18, 8, 25, 4],
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
            "decoder_input_tokens": [3, 15, 7, 6, 8, 24, 8, 25, 4],
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
    ds = task.get_dataset(
        sequence_lengths={"inputs": 20, "targets": 10},
        split="train",
        feature_converter=_create_feature_converter(),
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
                [3, 15, 7, 6, 8, 24, 8, 25, 4, 0],
                [3, 22, 4, 2, 18, 8, 25, 4, 0, 0],
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
            "decoder_input_tokens": [[3, 15, 7, 6, 8, 24, 8, 25, 4, 0]],
            "decoder_loss_weights": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 0]],
        },
    ]
    test_utils.assert_datasets_equal(ds, expected_first_batch)

  def test_task_get_dataset_with_shard_info(self):
    source = _create_source(num_examples=_SOURCE_NUM_EXAMPLES)
    task = _create_task(source=source, preprocessors=_create_preprocessors())
    ds = task.get_dataset(
        shard_info=dataset_providers.ShardInfo(index=0, num_shards=1)
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

  def test_task_get_dataset_by_step_without_feature_converter(self):
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

  def test_task_get_dataset_by_step_with_feature_converter(self):
    source = _create_source()
    task = _create_task(source=source, preprocessors=_create_preprocessors())
    feature_converter = _create_feature_converter()
    ds_by_step = task.get_dataset_by_step(
        num_records=1,
        sequence_lengths={"inputs": 20, "targets": 10},
        batch_size=2,
        feature_converter=feature_converter,
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
        # Keep features specified in sequence_lengths only.
        [{
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
            "targets": [3, 15, 7, 6, 8, 24, 8, 25, 4],
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
            ],
            "decoder_target_tokens": [3, 15, 7, 6, 8, 24, 8, 25, 4],
            "decoder_input_tokens": [3, 15, 7, 6, 8, 24, 8, 25, 4],
            "decoder_loss_weights": [1, 1, 1, 1, 1, 1, 1, 1, 1],
        }],
        # Trim/Pad Operation.
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
            "decoder_input_tokens": [3, 15, 7, 6, 8, 24, 8, 25, 4, 0],
            "decoder_loss_weights": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        }],
        # Batching.
        [{
            "decoder_input_tokens": [[3, 15, 7, 6, 8, 24, 8, 25, 4, 0]],
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
        list(ds_by_step[0]), dataset_providers.DEFAULT_NUM_RECORDS_TO_INSPECT
    )
    ds_by_step = task.get_dataset_by_step(num_records=0)
    self.assertLen(
        list(ds_by_step[0]), dataset_providers.DEFAULT_NUM_RECORDS_TO_INSPECT
    )
    ds_by_step = task.get_dataset_by_step(
        num_records=dataset_providers.MAX_NUM_RECORDS_TO_INSPECT + 1,
    )
    self.assertLen(
        list(ds_by_step[0]), dataset_providers.MAX_NUM_RECORDS_TO_INSPECT
    )

  def test_get_dataset(self):
    source = _create_source(
        source_name=_SOURCE_NAME,
        num_examples=_SOURCE_NUM_EXAMPLES,
    )
    task = _create_task(source=source, preprocessors=_create_preprocessors())
    ds = dataset_providers.get_dataset(task)
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

  def test_task_builder_from_task_copies_params_correctly(self):
    """Verify that the TaskBuilder object is created with correct params."""
    task = _create_task(
        source=_create_source(),
        task_name="dummy_airio_task",
        preprocessors=_create_preprocessors(),
    )
    task_builder = dataset_providers.TaskBuilder.from_task(task)
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
    new_preprocessors = [airio_preps.MapFnTransform(_imdb_preprocessor)]
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
        "TaskBuilder(task_name=dummy_airio_task,"
        " source=<airio.data_sources.TfdsDataSource",
    )


class MixtureTest(absltest.TestCase):
  def setUp(self):
    super().setUp()

    def test_map_fn(ex, idx):
      return {"idx": idx, "val": ex}

    # TODO(b/294122943): Pass runtime args to this preprocessor.
    def simple_to_imdb_map_fn(ex):
      return {
          "inputs_pretokenized": f"{ex}",
          "inputs": np.array([ex] * 20),
          "targets_pretokenized": f"{ex + 1}",
          "targets": np.array([ex + 1] * 10),
      }

    imdb_source = _create_source(
        source_name=_SOURCE_NAME,
        splits=_SOURCE_SPLITS,
        num_examples=_SOURCE_NUM_EXAMPLES,
    )
    self._map_transform_idx_1 = airio_preps.MapFnTransform(
        functools.partial(test_map_fn, idx=1)
    )
    self._map_transform_idx_2 = airio_preps.MapFnTransform(
        functools.partial(test_map_fn, idx=2)
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
        dataset_providers.TaskBuilder.from_task(self._simple_task_1)
        .set_preprocessors([
            airio_preps.MapFnTransform(simple_to_imdb_map_fn),
        ])
        .build()
    )

  def test_simple_mixture(self):
    mix = dataset_providers.Mixture(
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
    mix = dataset_providers.Mixture(
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
    mix = dataset_providers.Mixture(
        name="test_mix",
        tasks=[self._simple_task_1, self._simple_task_2],
        proportions=[2.0, 1.0],
    )
    ds = mix.get_dataset(
        shuffle=False,
        shard_info=dataset_providers.ShardInfo(index=0, num_shards=2),
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
    mix = dataset_providers.Mixture(
        name="test_mix",
        tasks=[self._simple_task_1, self._simple_task_2],
        proportions=[2.0, 1.0],
    )
    ds = mix.get_dataset(
        shuffle=True,
        seed=42,
        shard_info=dataset_providers.ShardInfo(index=0, num_shards=2),
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
      "airio.lazy_dataset_transforms.ConcatLazyMapDataset",
      new_callable=mock.NonCallableMock,
  )
  def test_single_epoch_concat_not_called(self, unused_mock_concat_fn):
    mix = dataset_providers.Mixture(
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
    mix = dataset_providers.Mixture(
        name="test_mix",
        tasks=[self._simple_task_1, self._simple_task_2],
        proportions=[2.0, 1.0],
    )
    ds = mix.get_dataset(
        shuffle=True,
        seed=42,
        shard_info=dataset_providers.ShardInfo(index=0, num_shards=2),
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
        dataset_providers.TaskBuilder.from_task(self._simple_task_1)
        .set_preprocessors([
            self._map_transform_idx_1,
            airio_preps.RandomMapFnTransform(test_random_map_fn),
        ])
        .build()
    )
    task2 = (
        dataset_providers.TaskBuilder.from_task(self._simple_task_2)
        .set_preprocessors([
            self._map_transform_idx_2,
            airio_preps.RandomMapFnTransform(test_random_map_fn),
        ])
        .build()
    )
    mix = dataset_providers.Mixture(
        name="test_mix", tasks=[task1, task2], proportions=[2.0, 1.0]
    )
    ds = mix.get_lazy_dataset(
        None,
        "train",
        shuffle=True,
        seed=42,
        shard_info=dataset_providers.ShardInfo(index=0, num_shards=2),
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
    mix = dataset_providers.Mixture(
        name="test_mix",
        tasks=[self._simple_task_1, self._simple_task_2],
        proportions=[2.0, 1.0],
    )
    ds = mix.get_dataset(
        shuffle=False,
        shard_info=dataset_providers.ShardInfo(index=0, num_shards=2),
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
    mix = dataset_providers.Mixture(
        name="test_mix",
        tasks=[self._imdb_task, self._simple_to_imdb_task],
        proportions=[1.0, 1.0],
    )
    ds = mix.get_dataset(
        shuffle=False,
        shard_info=dataset_providers.ShardInfo(index=0, num_shards=2),
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
    mix = dataset_providers.Mixture(
        name="test_mix",
        tasks=[self._imdb_task, self._simple_task_1],
        proportions=[1.0, 1.0],
    )
    ds = mix.get_dataset(
        shuffle=False,
        shard_info=dataset_providers.ShardInfo(index=0, num_shards=2),
        num_epochs=1,
        batch_size=2
    )
    with self.assertRaisesRegex(
        ValueError,
        "The two structures don't have the same nested structure.*",
    ):
      _ = next(ds)

  def test_mixture_with_feature_converter(self):
    mix = dataset_providers.Mixture(
        name="test_mix",
        tasks=[self._imdb_task, self._simple_to_imdb_task],
        proportions=[1.0, 1.0],
    )
    ds = mix.get_dataset(
        shuffle=False,
        shard_info=dataset_providers.ShardInfo(index=0, num_shards=2),
        num_epochs=1,
        feature_converter=_create_feature_converter(),
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
            ],
            "decoder_target_tokens": [3, 15, 7, 6, 8, 24, 8, 25, 4],
            "decoder_input_tokens": [3, 15, 7, 6, 8, 24, 8, 25, 4],
            "decoder_loss_weights": [1, 1, 1, 1, 1, 1, 1, 1, 1],
        },
        {  # simple task
            "encoder_input_tokens": [0] * 20,
            "decoder_target_tokens": [1] * 10,
            "decoder_input_tokens": [1] * 10,
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
            "decoder_target_tokens": [3, 22, 4, 2, 18, 8, 25, 4],
            "decoder_input_tokens": [3, 22, 4, 2, 18, 8, 25, 4],
            "decoder_loss_weights": [1, 1, 1, 1, 1, 1, 1, 1],
        },
        {  # simple task
            "encoder_input_tokens": [1] * 20,
            "decoder_target_tokens": [2] * 10,
            "decoder_input_tokens": [2] * 10,
            "decoder_loss_weights": [1] * 10,
        },
        # imdb task now empty
    ]
    test_utils.assert_datasets_equal(ds, expected)

  def test_mixture_with_feature_converter_and_batching(self):
    mix = dataset_providers.Mixture(
        name="test_mix",
        tasks=[self._imdb_task, self._simple_to_imdb_task],
        proportions=[1.0, 1.0],
    )
    ds = mix.get_dataset(
        sequence_lengths={"inputs": 20, "targets": 10},
        shuffle=False,
        shard_info=dataset_providers.ShardInfo(index=0, num_shards=2),
        num_epochs=1,
        feature_converter=_create_feature_converter(),
        batch_size=2,
    )
    expected_first_batch = {
        "decoder_input_tokens": [
            [3, 15, 7, 6, 8, 24, 8, 25, 4, 0],  # imdb task
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # simple task
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
    mix = dataset_providers.Mixture(
        name="test_mix",
        tasks=[self._simple_task_1, self._simple_task_2],
        proportions=[1.0, 1.0],
    )
    ds = mix.get_dataset(
        sequence_lengths={"inputs": 20, "targets": 10},
        shuffle=False,
        shard_info=dataset_providers.ShardInfo(index=0, num_shards=2),
        num_epochs=1,
        feature_converter=None,
        batch_size=2,
    )
    self.assertDictEqual(
        {k: v.tolist() for k, v in next(ds).items()},
        {"idx": [1, 2], "val": [0, 0]},
    )


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
    self.simple_mix = dataset_providers.Mixture(
        name="test_mix_1",
        tasks=self.tasks[:3],
        proportions=[1.0, 0.5, 2.0],
    )
    self.mix_of_mix = dataset_providers.Mixture(
        name="test_mix_2",
        tasks=[self.simple_mix, self.tasks[3]],
        proportions=[0.5, 0.7],
    )
    self.mix_of_mix_of_mix = dataset_providers.Mixture(
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
      _ = dataset_providers.Mixture(
          "invalid_mix", [self.tasks[0], self.tasks[1]], [1.0]
      )

  def test_duplicate_tasks(self):
    with self.assertRaisesRegex(
        ValueError,
        "Mixture invalid_mix has duplicate tasks.*",
    ):
      _ = dataset_providers.Mixture(
          "invalid_mix", [self.tasks[0], self.tasks[0]], [1.0, 1.0]
      )


if __name__ == "__main__":
  absltest.main()
