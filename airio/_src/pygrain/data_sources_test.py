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

"""Tests for airio.pygrain.data_sources."""

import os
from typing import Sequence
from unittest import mock

from absl.testing import absltest
from airio._src.core import data_sources as core_data_sources
from airio._src.pygrain import data_sources
import tensorflow_datasets as tfds

import multiprocessing


_SOURCE_NAME = "imdb_reviews"
_SOURCE_NUM_EXAMPLES = 3
_SOURCE_SPLITS = {"train", "test", "unsupervised"}


class ArrayRecordDataSourceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../test_data",
        "classification",
    )

  def _create_data_source(
      self,
      splits: Sequence[str] | None = None,
  ):
    """Creates a basic ArrayRecordDataSource."""

    if splits is None:
      splits = _SOURCE_SPLITS

    split_to_filepattern = {}
    for split in splits:
      split_to_filepattern[split] = os.path.join(
          self.test_data_dir, "classification.array_record@2"
      )

    return data_sources.ArrayRecordDataSource(
        split_to_filepattern=split_to_filepattern,
    )

  def test_create(self):
    source = data_sources.ArrayRecordDataSource([])
    self.assertIsInstance(source, core_data_sources.DataSource)
    self.assertIsInstance(source, data_sources.ArrayRecordDataSource)

  def test_get_data_source(self):
    source = self._create_data_source(splits=_SOURCE_SPLITS)
    for split in _SOURCE_SPLITS:
      data_source = source.get_data_source(split)
      self.assertLen(data_source, 10)

  def test_get_data_source_nonexistent_split(self):
    source = self._create_data_source(splits=_SOURCE_SPLITS)
    with self.assertRaisesRegex(ValueError, "Split nonexistent not found in"):
      source.get_data_source("nonexistent")

  def test_num_input_examples(self):
    source = self._create_data_source(splits=_SOURCE_SPLITS)
    for split in _SOURCE_SPLITS:
      self.assertEqual(source.num_input_examples(split), 10)

  def test_num_input_examples_nonexistent_split(self):
    source = self._create_data_source(splits=_SOURCE_SPLITS)
    with self.assertRaisesRegex(ValueError, "Split nonexistent not found in"):
      source.num_input_examples("nonexistent")

  def test_splits(self):
    source = self._create_data_source(splits=_SOURCE_SPLITS)
    self.assertEqual(_SOURCE_SPLITS, source.splits)





class JsonDataSourceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../test_data",
        "classification",
    )

  def _create_data_source(
      self,
      splits: Sequence[str] | None = None,
  ):
    """Creates a basic JsonDataSource."""
    if splits is None:
      splits = _SOURCE_SPLITS
    split_to_filepattern = {}
    for split in splits:
      split_to_filepattern[split] = os.path.join(
          self.test_data_dir, "classification.json"
      )
    return data_sources.JsonDataSource(
        split_to_filepattern=split_to_filepattern,
    )

  def test_create(self):
    source = data_sources.JsonDataSource([])
    self.assertIsInstance(source, core_data_sources.DataSource)
    self.assertIsInstance(source, data_sources.JsonDataSource)

  def test_get_data_source(self):
    source = self._create_data_source(splits=_SOURCE_SPLITS)
    for split in _SOURCE_SPLITS:
      data_source = source.get_data_source(split)
      self.assertLen(data_source, 5)
      data_source.close()
      data_source.unlink()

  def test_get_data_source_nonexistent_split(self):
    source = self._create_data_source(splits=_SOURCE_SPLITS)
    with self.assertRaisesRegex(ValueError, "Split nonexistent not found in"):
      source.get_data_source("nonexistent")

  def test_num_input_examples(self):
    source = self._create_data_source(
        splits=_SOURCE_SPLITS,
    )
    for split in _SOURCE_SPLITS:
      self.assertEqual(source.num_input_examples(split), 5)

  def test_num_input_examples_nonexistent_split(self):
    source = self._create_data_source(splits=_SOURCE_SPLITS)
    with self.assertRaisesRegex(ValueError, "Split nonexistent not found in"):
      source.num_input_examples("nonexistent")

  def test_splits(self):
    source = self._create_data_source(splits=_SOURCE_SPLITS)
    self.assertEqual(_SOURCE_SPLITS, source.splits)





class TfdsDataSourceTest(absltest.TestCase):

  def _create_data_source(
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

  def test_create(self):
    source = self._create_data_source()
    self.assertIsInstance(source, core_data_sources.DataSource)
    self.assertIsInstance(source, data_sources.TfdsDataSource)

  def test_create_single_split(self):
    with tfds.testing.mock_data(_SOURCE_NUM_EXAMPLES):
      source = data_sources.TfdsDataSource(
          tfds_name=_SOURCE_NAME, splits="train"
      )
    self.assertIsInstance(source, core_data_sources.DataSource)
    self.assertIsInstance(source, data_sources.TfdsDataSource)

  def test_get_data_source(self):
    source = self._create_data_source(
        num_examples=_SOURCE_NUM_EXAMPLES,
        splits=_SOURCE_SPLITS,
    )
    for split in _SOURCE_SPLITS:
      data_source = source.get_data_source(split)
      self.assertLen(data_source, _SOURCE_NUM_EXAMPLES)

  def test_get_data_source_nonexistent_split(self):
    source = self._create_data_source(
        source_name=_SOURCE_NAME,
        splits=_SOURCE_SPLITS,
    )
    with self.assertRaisesRegex(ValueError, "Split nonexistent not found in"):
      source.get_data_source("nonexistent")

  def test_num_input_examples(self):
    source = self._create_data_source(
        num_examples=_SOURCE_NUM_EXAMPLES,
        splits=_SOURCE_SPLITS,
    )
    for split in _SOURCE_SPLITS:
      self.assertEqual(source.num_input_examples(split), _SOURCE_NUM_EXAMPLES)

  def test_num_input_examples_nonexistent_split(self):
    source = self._create_data_source(splits=_SOURCE_SPLITS)
    with self.assertRaisesRegex(ValueError, "Split nonexistent not found in"):
      source.num_input_examples("nonexistent")

  def test_splits(self):
    source = self._create_data_source(
        splits=_SOURCE_SPLITS,
    )
    self.assertEqual(_SOURCE_SPLITS, source.splits)

  def test_empty_splits(self):
    with tfds.testing.mock_data(_SOURCE_NUM_EXAMPLES):
      source = data_sources.TfdsDataSource(tfds_name=_SOURCE_NAME, splits=[])
    self.assertEmpty(source.splits)

  def test_none_splits(self):
    with tfds.testing.mock_data(_SOURCE_NUM_EXAMPLES):
      source = data_sources.TfdsDataSource(tfds_name=_SOURCE_NAME, splits=None)
    self.assertEmpty(source.splits)



if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
