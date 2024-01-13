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

"""Tests for airio.grain.data_sources."""

import os
from typing import Sequence
from unittest import mock

from absl.testing import absltest
import airio
from airio.grain import data_sources

import multiprocessing


_SOURCE_NAME = "imdb_reviews"
_SOURCE_NUM_EXAMPLES = 3
_SOURCE_SPLITS = {"train", "test", "unsupervised"}


class ArrayRecordDataSourceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test_data",
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
    self.assertIsInstance(source, airio.data_sources.DataSource)
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
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test_data",
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
    self.assertIsInstance(source, airio.data_sources.DataSource)
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





if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
