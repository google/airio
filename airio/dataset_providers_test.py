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


class DatasetProviderBaseTest(absltest.TestCase):

  @mock.patch.multiple(
      dataset_providers.DatasetProviderBase, __abstractmethods__=set()
  )
  def test_protocol(self):
    base = dataset_providers.DatasetProviderBase
    self.assertIsNone(base.get_dataset(self, split=""))
    self.assertIsNone(base.num_input_examples(self, split=""))


class DatasetProvidersTest(absltest.TestCase):

  def _create_source(
      self,
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
      self, source: data_sources.DataSource, task_name: str = "dummy_airio_task"
  ) -> dataset_providers.Task:
    """Create example AirIO task."""

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

    test_data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_data"
    )
    sentencepiece_vocab = vocabularies.SentencePieceVocabulary(
        os.path.join(test_data_dir, "sentencepiece", "sentencepiece.model")
    )
    tokenizer_config = tokenizer.TokenizerConfig(vocab=sentencepiece_vocab)
    return dataset_providers.Task(
        name=task_name,
        source=source,
        preprocessors=[
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
        ],
    )

  def test_create_task(self):
    source = self._create_source(splits=_SOURCE_SPLITS)
    task = self._create_task(source=source, task_name="dummy_airio_task")
    self.assertIsInstance(task.source, data_sources.DataSource)
    self.assertIsInstance(task.source, data_sources.TfdsDataSource)
    self.assertEqual(task.name, "dummy_airio_task")
    self.assertEqual(task.splits, _SOURCE_SPLITS)

  def test_empty_splits(self):
    with tfds.testing.mock_data(_SOURCE_NUM_EXAMPLES):
      source = data_sources.TfdsDataSource(tfds_name=_SOURCE_NAME, splits=[])
    task = self._create_task(source)
    self.assertEmpty(task.splits)

  def test_none_splits(self):
    with tfds.testing.mock_data(_SOURCE_NUM_EXAMPLES):
      source = data_sources.TfdsDataSource(tfds_name=_SOURCE_NAME, splits=None)
    task = self._create_task(source)
    self.assertEmpty(task.splits)

  def test_num_input_examples(self):
    source = self._create_source(
        splits=_SOURCE_SPLITS,
        num_examples=_SOURCE_NUM_EXAMPLES,
    )
    task = self._create_task(source)
    num_examples = task.num_input_examples(split="train")
    self.assertEqual(num_examples, _SOURCE_NUM_EXAMPLES)

  def test_task_get_dataset(self):
    source = self._create_source(
        splits=_SOURCE_SPLITS,
        num_examples=_SOURCE_NUM_EXAMPLES,
    )
    task = self._create_task(source)
    ds = task.get_dataset(split="train")
    num_examples = 0
    for _ in ds:
      num_examples += 1
    self.assertEqual(num_examples, _SOURCE_NUM_EXAMPLES)

  def test_task_get_dataset_with_shard_info(self):
    source = self._create_source(
        splits=_SOURCE_SPLITS,
        num_examples=_SOURCE_NUM_EXAMPLES,
    )
    task = self._create_task(source)
    shard_info = seqio.ShardInfo(index=0, num_shards=1)
    ds = task.get_dataset(split="train", shard_info=shard_info)
    num_examples = 0
    for _ in ds:
      num_examples += 1
    self.assertEqual(num_examples, _SOURCE_NUM_EXAMPLES)

  def test_task_get_dataset_nonexistent_split(self):
    source = self._create_source(
        source_name=_SOURCE_NAME,
        splits=_SOURCE_SPLITS,
    )
    task = self._create_task(source)
    with self.assertRaisesRegex(ValueError, "Split nonexistent not found in"):
      task.get_dataset(split="nonexistent")

  def test_get_dataset(self):
    source = self._create_source(
        source_name=_SOURCE_NAME,
        splits=_SOURCE_SPLITS,
        num_examples=_SOURCE_NUM_EXAMPLES,
    )
    task = self._create_task(source)
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


if __name__ == "__main__":
  absltest.main()
