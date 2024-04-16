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

import json
import os
from typing import Callable, Dict, Sequence
from unittest import mock

from absl.testing import absltest
from airio._src.core import data_sources as core_data_sources
from airio._src.pygrain import data_sources
from airio._src.pygrain import dataset_providers
from airio._src.pygrain import preprocessors
import grain.python as grain
import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import multiprocessing


_SOURCE_NAME = "imdb_reviews"
_SOURCE_NUM_EXAMPLES = 3
_SOURCE_SPLITS = frozenset(["train", "test", "unsupervised"])


def _get_dataset(
    src: core_data_sources.DataSource,
    split: str,
    parse_fn: Callable[bytes, Dict[str, np.ndarray]],
):
  return dataset_providers.GrainTask(
      "dummy_task",
      source=src,
      preprocessors=[preprocessors.MapFnTransform(parse_fn)],
  ).get_dataset(split=split, shuffle=False)


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
    source = data_sources.ArrayRecordDataSource({})
    self.assertIsInstance(source, core_data_sources.DataSource)
    self.assertIsInstance(source, data_sources.ArrayRecordDataSource)

  def test_get_data_source(self):
    source = self._create_data_source(splits=_SOURCE_SPLITS)
    for split in _SOURCE_SPLITS:
      data_source = source.get_data_source(split)
      self.assertLen(data_source, 10)

  def test_get_dataset_with_fast_proto_parser(self):
    source = self._create_data_source(splits=_SOURCE_SPLITS)
    ds = _get_dataset(
        source,
        "train",
        grain.experimental.fast_proto_parser.parse_tf_example,
    )
    actual_ds = list(ds)
    expected_ds = [
        {"label": [0], "text": [b"abc"]},
        {"label": [1], "text": [b"def"]},
        {"label": [0], "text": [b"ghi"]},
        {"label": [1], "text": [b"jkl"]},
        {"label": [0], "text": [b"mno"]},
        {"label": [0], "text": [b"pqr"]},
        {"label": [1], "text": [b"stu"]},
        {"label": [0], "text": [b"vwx"]},
        {"label": [1], "text": [b"yza"]},
        {"label": [0], "text": [b"bcd"]},
    ]
    for actual, expected in zip(actual_ds, expected_ds, strict=True):
      for k in ["label", "text"]:
        np.testing.assert_equal(actual[k], expected[k])

  def test_get_dataset_with_tf_feature_description(self):
    feature_description = {
        "text": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    source = self._create_data_source(splits=_SOURCE_SPLITS)

    def parse_fn(ex, feature_description):
      tensor = tf.io.parse_single_example(ex, feature_description)
      return tf.nest.map_structure(lambda x: x.numpy(), tensor)

    ds = _get_dataset(
        source, "train", lambda x: parse_fn(x, feature_description)
    )
    actual_ds = list(ds)
    expected_examples = [
        {"label": 0, "text": b"abc"},
        {"label": 1, "text": b"def"},
        {"label": 0, "text": b"ghi"},
        {"label": 1, "text": b"jkl"},
        {"label": 0, "text": b"mno"},
        {"label": 0, "text": b"pqr"},
        {"label": 1, "text": b"stu"},
        {"label": 0, "text": b"vwx"},
        {"label": 1, "text": b"yza"},
        {"label": 0, "text": b"bcd"},
    ]
    self.assertListEqual(actual_ds, expected_examples)

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





class DatasetFnCallableTest(absltest.TestCase):

  @mock.patch.multiple(
      data_sources.DatasetFnCallable, __abstractmethods__=set()
  )
  def test_protocol(self):
    dataset_function = data_sources.DatasetFnCallable
    self.assertIsNone(dataset_function.__call__(self, split=""))


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
    self.assertIsInstance(source, core_data_sources.DataSource)
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
    source = data_sources.JsonDataSource({})
    self.assertIsInstance(source, core_data_sources.DataSource)
    self.assertIsInstance(source, data_sources.JsonDataSource)

  def test_get_data_source(self):
    source = self._create_data_source(splits=_SOURCE_SPLITS)
    for split in _SOURCE_SPLITS:
      data_source = source.get_data_source(split)
      self.assertLen(data_source, 5)
      data_source.close()
      data_source.unlink()

  def test_get_dataset(self):
    def _parse_json(ex):
      ex = json.loads(ex)
      ex = jax.tree.map(np.asarray, ex)
      return ex

    source = self._create_data_source(splits=_SOURCE_SPLITS)
    ds = _get_dataset(source, "train", _parse_json)
    actual_ds = list(ds)
    expected_ds = [
        {"text": "abc", "label": 0},
        {"text": "def", "label": 1},
        {"text": "ghi", "label": 0},
        {"text": "jkl", "label": 1},
        {"text": "mno", "label": 0},
    ]
    for actual, expected in zip(actual_ds, expected_ds, strict=True):
      for k in ["text", "label"]:
        np.testing.assert_equal(actual[k], expected[k])

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

  def test_get_dataset(self):
    source = self._create_data_source(
        num_examples=_SOURCE_NUM_EXAMPLES,
        splits=_SOURCE_SPLITS,
    )
    ds = _get_dataset(source, "train", lambda x: x)
    actual_ds = list(ds)
    print(actual_ds)
    expected_ds = [
        {"label": 1, "text": "ebc   ahgjefjhfe"},
        {"label": 0, "text": "hj aijbcidcibdg"},
        {"label": 1, "text": "acdhdacfhhjb"},
    ]
    for actual, expected in zip(actual_ds, expected_ds, strict=True):
      for k in ["label", "text"]:
        np.testing.assert_equal(actual[k], expected[k])

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

  def test_splits_map(self):
    with tfds.testing.mock_data(_SOURCE_NUM_EXAMPLES):
      source = data_sources.TfdsDataSource(
          tfds_name=_SOURCE_NAME,
          splits={"my_split": "train", "my_other_split": "test"},
      )
    expected_splits = frozenset(["my_split", "my_other_split"])
    self.assertEqual(source.splits, expected_splits)
    for split in expected_splits:
      data_source = source.get_data_source(split)
      self.assertLen(data_source, 3)



if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
