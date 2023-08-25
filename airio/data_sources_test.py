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

"""Tests for airio.data_sources."""

import os
from typing import Sequence
from unittest import mock

from absl.testing import absltest
from airio import data_sources
import numpy as np
import tensorflow_datasets as tfds

_SOURCE_NAME = "imdb_reviews"
_SOURCE_NUM_EXAMPLES = 3
_SOURCE_SPLITS = {"train", "test", "unsupervised"}


class DataSourceTest(absltest.TestCase):

  @mock.patch.multiple(data_sources.DataSource, __abstractmethods__=set())
  def test_protocol(self):
    source = data_sources.DataSource
    self.assertIsNone(source.get_data_source(self, split=""))
    self.assertIsNone(source.num_input_examples(self, split=""))


class DatasetFnCallableTest(absltest.TestCase):

  @mock.patch.multiple(
      data_sources.DatasetFnCallable, __abstractmethods__=set()
  )
  def test_protocol(self):
    dataset_function = data_sources.DatasetFnCallable
    self.assertIsNone(dataset_function.__call__(self, split=""))


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
    self.assertIsInstance(source, data_sources.DataSource)
    self.assertIsInstance(source, data_sources.TfdsDataSource)

  def test_create_single_split(self):
    with tfds.testing.mock_data(_SOURCE_NUM_EXAMPLES):
      source = data_sources.TfdsDataSource(
          tfds_name=_SOURCE_NAME, splits="train"
      )
    self.assertIsInstance(source, data_sources.DataSource)
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


class FunctionDataSourceTest(absltest.TestCase):

  def _create_data_source(
      self,
      splits: Sequence[str] | None = None,
      num_examples: int = _SOURCE_NUM_EXAMPLES,
  ):
    """Creates a basic FunctionDataSource."""

    if splits is None:
      splits = _SOURCE_SPLITS

    def _generate_dataset(split: str):
      if split not in splits:
        raise ValueError(f"Split {split} not found in {splits}.")
      return np.array(range(num_examples))

    return data_sources.FunctionDataSource(
        dataset_fn=_generate_dataset, splits=splits
    )

  def test_create(self):
    source = self._create_data_source()
    self.assertIsInstance(source, data_sources.DataSource)
    self.assertIsInstance(source, data_sources.FunctionDataSource)

  def test_get_data_source(self):
    source = self._create_data_source(
        num_examples=_SOURCE_NUM_EXAMPLES,
        splits=_SOURCE_SPLITS,
    )
    for split in _SOURCE_SPLITS:
      data_source = source.get_data_source(split)
      self.assertLen(data_source, _SOURCE_NUM_EXAMPLES)

  def test_get_data_source_nonexistent_split(self):
    source = self._create_data_source(splits=_SOURCE_SPLITS)
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
    source = self._create_data_source(splits=_SOURCE_SPLITS)
    self.assertEqual(_SOURCE_SPLITS, source.splits)


class SSTableDataSourceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_data", "digits"
    )

  def _create_data_source(
      self,
      splits: Sequence[str] | None = None,
  ):
    """Creates a basic FunctionDataSource."""

    if splits is None:
      splits = _SOURCE_SPLITS

    split_to_filepattern = {}
    split_to_keypath = {}
    for split in splits:
      split_to_filepattern[split] = os.path.join(
          self.test_data_dir, "digits@2.sst"
      )
      split_to_keypath[split] = os.path.join(
          self.test_data_dir, "digits_keys@1.sst"
      )

    return data_sources.SSTableDataSource(
        split_to_filepattern=split_to_filepattern,
        split_to_keypath=split_to_keypath,
    )

  def test_create(self):
    source = self._create_data_source()
    self.assertIsInstance(source, data_sources.DataSource)
    self.assertIsInstance(source, data_sources.SSTableDataSource)

  def test_create_splits_do_not_match(self):
    split_to_filepattern = {}
    split_to_keypath = {}
    split_to_filepattern["train"] = os.path.join(
        self.test_data_dir, "digits@2.sst"
    )
    split_to_keypath["test"] = os.path.join(
        self.test_data_dir, "digits_keys@1.sst"
    )
    with self.assertRaisesRegex(ValueError, "do not match"):
      data_sources.SSTableDataSource(
          split_to_filepattern=split_to_filepattern,
          split_to_keypath=split_to_keypath,
      )

  def test_get_data_source(self):
    source = self._create_data_source(splits=_SOURCE_SPLITS)
    for split in _SOURCE_SPLITS:
      data_source = source.get_data_source(split)
      self.assertLen(data_source, 5)

  def test_get_data_source_nonexistent_split(self):
    source = self._create_data_source(splits=_SOURCE_SPLITS)
    with self.assertRaisesRegex(ValueError, "Split nonexistent not found in"):
      source.get_data_source("nonexistent")

  def test_num_input_examples(self):
    source = self._create_data_source(splits=_SOURCE_SPLITS)
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
  absltest.main()
