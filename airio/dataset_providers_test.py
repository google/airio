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
from airio import test_utils
from airio import tokenizer
import grain.python as grain
import seqio
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
      grain.MapOperation(map_function=_imdb_preprocessor),
      grain.MapOperation(
          functools.partial(
              tokenizer.tokenize,
              tokenizer_configs={
                  "inputs": tokenizer_config,
                  "targets": tokenizer_config,
              },
          )
      ),
  ]


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
    partial_task = dataset_providers.Task(
        name="dummy_airio_task", source=_create_source(), preprocessors=None
    )
    self.assertIsInstance(partial_task.source, data_sources.DataSource)
    self.assertIsInstance(partial_task.source, data_sources.TfdsDataSource)

  def test_create_task_without_preprocessors_get_preprocessors_fails(self):
    partial_task = dataset_providers.Task(
        name="dummy_airio_task", source=_create_source(), preprocessors=None
    )
    with self.assertRaisesRegex(
        ValueError, "Preprocessors have not been set on this task."
    ):
      partial_task.get_preprocessors()

  def test_create_task_with_preprocessors_only_succeeds(self):
    preprocessors = _create_preprocessors()
    partial_task = dataset_providers.Task(
        name="dummy_airio_task", source=None, preprocessors=preprocessors
    )
    self.assertEqual(partial_task.get_preprocessors(), preprocessors)
    self.assertIsNone(partial_task.source)

  def test_get_dataset_fails_on_partial_task_without_source(self):
    partial_task = dataset_providers.Task(
        name="dummy_airio_task",
        source=None,
        preprocessors=[],
    )
    with self.assertRaisesRegex(
        ValueError,
        "Both source and preprocessors must be set before calling"
        " get_dataset().",
    ):
      partial_task.get_dataset(split="train")

  def test_get_dataset_fails_on_partial_task_without_preprocessors(self):
    partial_task = dataset_providers.Task(
        name="dummy_airio_task",
        source=_create_source(splits=_SOURCE_SPLITS),
        preprocessors=None,
    )
    with self.assertRaisesRegex(
        ValueError,
        "Both source and preprocessors must be set before calling"
        " get_dataset().",
    ):
      partial_task.get_dataset(split="train")

  def test_create_empty_task(self):
    """Verify behavior when neither source nor preprocessors are set."""
    with self.assertRaisesRegex(
        ValueError, "Either source or preprocessors must be set."
    ):
      dataset_providers.Task(
          name="dummy_empty_task_no_source_no_preprocessors",
          source=None,
          preprocessors=None,
      )

  def test_create_task_without_source_can_set_source(self):
    task = dataset_providers.Task(
        name="dummy_airio_task",
        source=None,
        preprocessors=_create_preprocessors(),
    )
    task.set_data_source(source=_create_source())
    self.assertIsInstance(task.source, data_sources.DataSource)
    self.assertIsInstance(task.source, data_sources.TfdsDataSource)

  def test_create_task_without_preprocessors_can_set_preprocessors(self):
    task = dataset_providers.Task(
        name="dummy_airio_task", source=_create_source(), preprocessors=None
    )
    preprocessors = _create_preprocessors()
    task.set_preprocessors(preprocessors)
    self.assertEqual(task.get_preprocessors(), preprocessors)

  def test_create_task_with_source_and_preprocessors_cannot_set_either(self):
    task = dataset_providers.Task(
        name="dummy_airio_task",
        source=_create_source(),
        preprocessors=_create_preprocessors(),
    )
    with self.assertRaisesRegex(
        ValueError, "Source has already been set on this task."
    ):
      task.set_data_source(source=_create_source())
    with self.assertRaisesRegex(
        ValueError, "Preprocessors have already been set on this task."
    ):
      task.set_preprocessors(_create_preprocessors())

  def test_create_task(self):
    task = dataset_providers.Task(
        name="dummy_airio_task",
        source=_create_source(splits=_SOURCE_SPLITS),
        preprocessors=_create_preprocessors(),
    )
    self.assertIsInstance(task.source, data_sources.DataSource)
    self.assertIsInstance(task.source, data_sources.TfdsDataSource)
    self.assertEqual(task.name, "dummy_airio_task")
    self.assertEqual(task.splits, _SOURCE_SPLITS)

  def test_empty_splits(self):
    with tfds.testing.mock_data(_SOURCE_NUM_EXAMPLES):
      source = data_sources.TfdsDataSource(tfds_name=_SOURCE_NAME, splits=[])
    task = dataset_providers.Task(
        name="dummy_airio_task",
        source=source,
        preprocessors=_create_preprocessors(),
    )
    self.assertEmpty(task.splits)

  def test_none_splits(self):
    with tfds.testing.mock_data(_SOURCE_NUM_EXAMPLES):
      source = data_sources.TfdsDataSource(tfds_name=_SOURCE_NAME, splits=None)
    task = dataset_providers.Task(
        name="dummy_airio_task",
        source=source,
        preprocessors=_create_preprocessors(),
    )
    self.assertEmpty(task.splits)

  def test_num_input_examples(self):
    source = _create_source(
        splits=_SOURCE_SPLITS,
        num_examples=_SOURCE_NUM_EXAMPLES,
    )
    task = dataset_providers.Task(
        name="dummy_airio_task",
        source=source,
        preprocessors=_create_preprocessors(),
    )
    num_examples = task.num_input_examples(split="train")
    self.assertEqual(num_examples, _SOURCE_NUM_EXAMPLES)

  def test_task_get_dataset(self):
    source = _create_source(
        splits=_SOURCE_SPLITS,
        num_examples=_SOURCE_NUM_EXAMPLES,
    )
    task = dataset_providers.Task(
        name="dummy_airio_task",
        source=source,
        preprocessors=_create_preprocessors(),
    )
    ds = task.get_dataset(split="train")
    num_examples = 0
    for _ in ds:
      num_examples += 1
    self.assertEqual(num_examples, _SOURCE_NUM_EXAMPLES)

  def test_task_get_dataset_with_shard_info(self):
    source = _create_source(
        splits=_SOURCE_SPLITS,
        num_examples=_SOURCE_NUM_EXAMPLES,
    )
    task = dataset_providers.Task(
        name="dummy_airio_task",
        source=source,
        preprocessors=_create_preprocessors(),
    )
    shard_info = seqio.ShardInfo(index=0, num_shards=1)
    ds = task.get_dataset(split="train", shard_info=shard_info)
    num_examples = 0
    for _ in ds:
      num_examples += 1
    self.assertEqual(num_examples, _SOURCE_NUM_EXAMPLES)

  def test_task_get_dataset_nonexistent_split(self):
    source = _create_source(
        source_name=_SOURCE_NAME,
        splits=_SOURCE_SPLITS,
    )
    task = dataset_providers.Task(
        name="dummy_airio_task",
        source=source,
        preprocessors=_create_preprocessors(),
    )
    with self.assertRaisesRegex(ValueError, "Split nonexistent not found in"):
      task.get_dataset(split="nonexistent")

  def test_task_get_dataset_by_step(self):
    source = _create_source()
    task = dataset_providers.Task(
        name="dummy_airio_task",
        source=source,
        preprocessors=_create_preprocessors(),
    )
    ds_by_step = task.get_dataset_by_step(num_records=1, shuffle=False)
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

  def test_task_get_dataset_by_step_without_transformations(self):
    source = _create_source()
    task = dataset_providers.Task(
        name="dummy_airio_task",
        source=source,
        preprocessors=[],
    )
    ds_by_step = task.get_dataset_by_step(num_records=1, shuffle=False)
    expected = [
        [{
            "text": "ebc   ahgjefjhfe",
            "label": 1,
        }],
    ]
    test_utils.assert_datasets_equal(ds_by_step[0], expected[0])

  def test_task_get_dataset_by_step_invalid_num_records(self):
    source = _create_source()
    task = dataset_providers.Task(
        name="dummy_airio_task",
        source=source,
        preprocessors=[],
    )
    ds_by_step = task.get_dataset_by_step(num_records=-1, shuffle=False)
    self.assertLen(
        list(ds_by_step[0]), dataset_providers.DEFAULT_NUM_RECORDS_TO_INSPECT
    )
    ds_by_step = task.get_dataset_by_step(num_records=0, shuffle=False)
    self.assertLen(
        list(ds_by_step[0]), dataset_providers.DEFAULT_NUM_RECORDS_TO_INSPECT
    )
    ds_by_step = task.get_dataset_by_step(
        num_records=dataset_providers.MAX_NUM_RECORDS_TO_INSPECT + 1,
        shuffle=False,
    )
    self.assertLen(
        list(ds_by_step[0]), dataset_providers.MAX_NUM_RECORDS_TO_INSPECT
    )

  def test_get_dataset(self):
    source = _create_source(
        source_name=_SOURCE_NAME,
        splits=_SOURCE_SPLITS,
        num_examples=_SOURCE_NUM_EXAMPLES,
    )
    task = dataset_providers.Task(
        name="dummy_airio_task",
        source=source,
        preprocessors=_create_preprocessors(),
    )
    ds = dataset_providers.get_dataset(task, split="train")
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

  def test_task_num_input_examples_throws_error(self):
    """Verify that num_input_examples throws error source is not set."""
    task = dataset_providers.Task(
        name="dummy_airio_task", source=None, preprocessors=[]
    )
    with self.assertRaisesRegex(
        ValueError, "Source has not been set on this task object."
    ):
      task.num_input_examples(split="train")


if __name__ == "__main__":
  absltest.main()
