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
from airio import data_sources
from airio import dataset_providers
from airio import feature_converters
from airio import preprocessors as airio_preps
from airio import test_utils
from airio import tokenizer
import grain.python as grain
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

  def test_replace_batch_transform_workaround(self):
    map_fn_1 = airio_preps.MapFnTransform(lambda x: x + 1)
    map_fn_2 = airio_preps.MapFnTransform(lambda x: x + 2)
    old_transforms = [
        map_fn_1,
        grain.Batch(batch_size=5),
        map_fn_2,
        grain.Batch(batch_size=10),
    ]
    new_transforms = dataset_providers._replace_batch_transform(old_transforms)
    expected_transforms = [
        map_fn_1,
        grain.BatchOperation(batch_size=5),
        map_fn_2,
        grain.BatchOperation(batch_size=10),
    ]
    self.assertListEqual(new_transforms, expected_transforms)


if __name__ == "__main__":
  absltest.main()
