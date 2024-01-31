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

"""Tests for packing."""

import functools
import json
from absl.testing import absltest
from absl.testing import parameterized
from airio.core import preprocessors as preprocessors_lib
from airio.pygrain_common import packing
from airio.pygrain_common import preprocessors
import grain.python as grain
import numpy as np

lazy_dataset = grain.experimental.lazy_dataset


class MultiBinPackingMapTest(parameterized.TestCase):

  def test_pack_single_feature(self):
    # 5 elements of variable sequence length.
    input_elements = [[1, 2, 3, 4], [5, 6], [11, 12, 13, 14], [7], [8]]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(np.asarray)
    packer = packing.MultiBinPacker(feature_lengths=4, num_partial_examples=10)
    ds = packing.PackLazyMapDataset(ds, pool_size=10, packer=packer)
    ds_iter = iter(ds)

    # Elements are tuples with (inputs, inputs_segment_ids, inputs_positions).
    expected_elements = [
        # First element was already fully packed on 'inputs'.
        ([1, 2, 3, 4], [1, 1, 1, 1], [0, 1, 2, 3]),
        # Third element was fully packed and hence produced out of order.
        ([11, 12, 13, 14], [1, 1, 1, 1], [0, 1, 2, 3]),
        # Second, fourth and fifth element packed together.
        ([5, 6, 7, 8], [1, 1, 2, 3], [0, 1, 0, 0]),
    ]
    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      # Elements are tuples with (inputs, inputs_segment_ids, inputs_positions).
      np.testing.assert_array_equal(actual, expected)

  # Same as above but elements are dictionaries.
  @parameterized.parameters(
      "inputs",
      "inputs_segment_ids",
      "inputs_positions",
  )
  def test_pack_single_feature_in_dict(self, feature: str):
    input_elements = [
        {
            "inputs": [1, 2, 3, 4],
        },
        {
            "inputs": [5, 6],
        },
        {
            "inputs": [11, 12, 13, 14],
        },
        {
            "inputs": [7],
        },
        {
            "inputs": [8],
        },
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    packer = packing.MultiBinPacker(
        feature_lengths={"inputs": 4}, num_partial_examples=10
    )
    ds = packing.PackLazyMapDataset(ds, pool_size=10, packer=packer)
    ds_iter = iter(ds)

    expected_elements = [
        # First element was already fully packed on 'inputs'.
        {
            "inputs": [1, 2, 3, 4],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
        },
        # Third element was fully packed and hence produced out of order.
        {
            "inputs": [11, 12, 13, 14],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
        },
        # Second, fourth and fifth element packed together.
        {
            "inputs": [5, 6, 7, 8],
            "inputs_segment_ids": [1, 1, 2, 3],
            "inputs_positions": [0, 1, 0, 0],
        },
    ]

    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      # Compare keys.
      self.assertSequenceEqual(sorted(actual), sorted(expected))
      np.testing.assert_array_equal(actual[feature], expected[feature])

  @parameterized.parameters(
      "inputs",
      "inputs_segment_ids",
      "inputs_positions",
      "targets",
      "targets_segment_ids",
      "targets_positions",
  )
  def test_pack_multiple_features_same_sequences_length(self, feature: str):
    input_elements = [
        {
            "inputs": [1, 2, 3, 4],
            "targets": [10, 20],
        },
        {
            "inputs": [5, 6],
            "targets": [30, 40, 50],
        },
        {
            "inputs": [11, 12, 13, 14],
            "targets": [31, 41, 51],
        },
        {
            "inputs": [7],
            "targets": [60],
        },
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    packer = packing.MultiBinPacker(
        feature_lengths={"inputs": 4, "targets": 4}, num_partial_examples=10
    )
    ds = packing.PackLazyMapDataset(ds, pool_size=10, packer=packer)
    ds_iter = iter(ds)

    expected_elements = [
        # First element was already fully packed on 'inputs'.
        {
            "inputs": [1, 2, 3, 4],
            "targets": [10, 20],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
            "targets_segment_ids": [1, 1],
            "targets_positions": [0, 1],
        },
        # Second and fourth element packed together.
        {
            "inputs": [5, 6, 7],
            "inputs_segment_ids": [1, 1, 2],
            "inputs_positions": [0, 1, 0],
            "targets": [30, 40, 50, 60],
            "targets_segment_ids": [1, 1, 1, 2],
            "targets_positions": [0, 1, 2, 0],
        },
        # The third element has fully packed "inputs", but not "targets", so is
        # added as a partially packed example instead of being yielded.
        {
            "inputs": [11, 12, 13, 14],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
            "targets": [31, 41, 51],
            "targets_segment_ids": [1, 1, 1],
            "targets_positions": [0, 1, 2],
        },
    ]
    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      # Compare keys.
      self.assertSequenceEqual(sorted(actual), sorted(expected))
      np.testing.assert_array_equal(actual[feature], expected[feature])

  @parameterized.parameters(
      "inputs",
      "inputs_segment_ids",
      "inputs_positions",
      "targets",
      "targets_segment_ids",
      "targets_positions",
  )
  def test_pack_multiple_features_different_sequences_length(
      self, feature: str
  ):
    input_elements = [
        {
            "inputs": [1, 2, 3, 4],
            "targets": [10, 20],
        },
        {
            "inputs": [5, 6],
            "targets": [30, 40, 50],
        },
        {
            "inputs": [11, 12, 13, 14],
            "targets": [31],
        },
        {
            "inputs": [7],
            "targets": [60],
        },
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    packer = packing.MultiBinPacker(
        feature_lengths={"inputs": 6, "targets": 4}, num_partial_examples=10
    )
    ds = packing.PackLazyMapDataset(ds, pool_size=10, packer=packer)
    ds_iter = iter(ds)

    expected_elements = [
        {  # 2nd and 3rd example create a perfect fit, thus yielded first.
            "inputs": [5, 6, 11, 12, 13, 14],
            "inputs_segment_ids": [1, 1, 2, 2, 2, 2],
            "inputs_positions": [0, 1, 0, 1, 2, 3],
            "targets": [30, 40, 50, 31],
            "targets_segment_ids": [1, 1, 1, 2],
            "targets_positions": [0, 1, 2, 0],
        },
        {
            "inputs": [1, 2, 3, 4, 7],
            "inputs_segment_ids": [1, 1, 1, 1, 2],
            "inputs_positions": [0, 1, 2, 3, 0],
            "targets": [10, 20, 60],
            "targets_segment_ids": [1, 1, 2],
            "targets_positions": [0, 1, 0],
        },
    ]
    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      np.testing.assert_array_equal(actual[feature], expected[feature])

  def test_do_not_pack_marked_features(self):
    input_elements = [
        {
            "inputs": [1, 2, 3, 4],
            "targets": [10, 20],
            "id": [1, 1],
        },
        {
            "inputs": [5, 6],
            "targets": [30, 40, 50],
            "id": [1, 2],
        },
        {
            "inputs": [11, 12, 13, 14],
            "targets": [31],
            "id": [2, 1],
        },
        {
            "inputs": [7],
            "targets": [60],
            "id": [2, 2],
        },
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    packer = packing.MultiBinPacker(
        feature_lengths={"inputs": 6, "targets": 4, "id": -1},
        num_partial_examples=10,
    )
    ds = packing.PackLazyMapDataset(ds, pool_size=10, packer=packer)
    ds_iter = iter(ds)
    expected_ids = [[[1, 2], [2, 1]], [[1, 1], [2, 2]]]
    for actual, expected in zip(ds_iter, expected_ids, strict=True):
      np.testing.assert_array_equal(actual["id"], expected)

  @parameterized.parameters(
      "input_tokens",
      "input_tokens_segment_ids",
      "input_tokens_positions",
      "input_vectors",
      "input_vectors_segment_ids",
      "input_vectors_positions",
  )
  def test_pack_two_dimensional_features(self, feature: str):
    input_elements = [
        {
            "input_tokens": [1, 2, 3],
            "input_vectors": [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
        },
        {
            "input_tokens": [4, 5],
            "input_vectors": [[3, 4, 5], [4, 5, 6]],
        },
        {
            "input_tokens": [6],
            "input_vectors": [[5, 6, 7]],
        },
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    packer = packing.MultiBinPacker(
        feature_lengths={"input_tokens": 3, "input_vectors": 3},
        num_partial_examples=10,
    )
    ds = packing.PackLazyMapDataset(ds, pool_size=10, packer=packer)
    ds_iter = iter(ds)

    expected_elements = [
        {
            "input_tokens": [1, 2, 3],
            "input_tokens_segment_ids": [1, 1, 1],
            "input_tokens_positions": [0, 1, 2],
            "input_vectors": [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
            "input_vectors_segment_ids": [1, 1, 1],
            "input_vectors_positions": [0, 1, 2],
        },
        {
            "input_tokens": [4, 5, 6],
            "input_tokens_segment_ids": [1, 1, 2],
            "input_tokens_positions": [0, 1, 0],
            "input_vectors": [[3, 4, 5], [4, 5, 6], [5, 6, 7]],
            "input_vectors_segment_ids": [1, 1, 2],
            "input_vectors_positions": [0, 1, 0],
        },
    ]
    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      np.testing.assert_array_equal(actual[feature], expected[feature])

  @parameterized.parameters(
      "inputs",
      "inputs_segment_ids",
      "inputs_positions",
  )
  def test_airio_pack_preprocessor(self, feature: str):
    input_elements = [
        {
            "inputs": [1, 2, 3, 4],
        },
        {
            "inputs": [5, 6],
        },
        {
            "inputs": [11, 12, 13, 14],
        },
        {
            "inputs": [7],
        },
        {
            "inputs": [8],
        },
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 4}, split="unused"
    )
    packer = packing.MultiBinPacker(
        num_partial_examples=10,
        # feature_lengths unset, will be set before packing.
    )
    transform = packing.AirIOPackDatasetMapPreprocessor(
        pool_size=10, packer=packer
    )
    unused_rng = None
    ds = transform(ds, runtime_args, unused_rng)
    updated_runtime_args = transform.update_runtime_args(runtime_args)
    ds_iter = iter(ds)

    # Verify updated runtime args
    expected_updated_runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={
            "inputs": 4,
            "inputs_segment_ids": 4,
            "inputs_positions": 4,
        },
        split="unused",
    )
    self.assertEqual(updated_runtime_args, expected_updated_runtime_args)

    # Verify packed examples.
    expected_elements = [
        # First element was already fully packed on 'inputs'.
        {
            "inputs": [1, 2, 3, 4],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
        },
        # Third element was fully packed and hence produced out of order.
        {
            "inputs": [11, 12, 13, 14],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
        },
        # Second, fourth and fifth element packed together.
        {
            "inputs": [5, 6, 7, 8],
            "inputs_segment_ids": [1, 1, 2, 3],
            "inputs_positions": [0, 1, 0, 0],
        },
    ]

    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      # Compare keys.
      self.assertSequenceEqual(sorted(actual), sorted(expected))
      np.testing.assert_array_equal(actual[feature], expected[feature])

  @parameterized.parameters(
      "inputs",
      "inputs_segment_ids",
      "inputs_positions",
  )
  def test_pack_single_feature_multiple_pools(self, feature: str):
    input_elements = [
        {
            "inputs": [1, 2, 3, 4],
        },
        {
            "inputs": [5, 6],
        },
        {
            "inputs": [11, 12, 13, 14],
        },
        {
            "inputs": [7],
        },
        {
            "inputs": [8],
        },
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    packer = packing.MultiBinPacker(
        feature_lengths={"inputs": 4}, num_partial_examples=10
    )
    ds = packing.PackLazyMapDataset(ds, pool_size=4, packer=packer)
    ds_iter = iter(ds)

    expected_elements = [
        # First element was already fully packed on 'inputs'.
        {
            "inputs": [1, 2, 3, 4],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
        },
        # Third element was fully packed and hence produced out of order.
        {
            "inputs": [11, 12, 13, 14],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
        },
        # Second and fourth element packed together.
        {
            "inputs": [5, 6, 7],
            "inputs_segment_ids": [1, 1, 2],
            "inputs_positions": [0, 1, 0],
        },
        # Fifth element is in a different pool.
        {
            "inputs": [8],
            "inputs_segment_ids": [1],
            "inputs_positions": [0],
        },
    ]

    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      # Compare keys.
      self.assertSequenceEqual(sorted(actual), sorted(expected))
      np.testing.assert_array_equal(actual[feature], expected[feature])

  @parameterized.parameters(
      "inputs",
      "inputs_segment_ids",
      "inputs_positions",
  )
  def test_pack_single_feature_with_padding(self, feature: str):
    input_elements = [
        {"inputs": [1, 2, 3, 4]},
        {"inputs": [5, 6]},
        {"inputs": [11, 12, 13, 14]},
        {"inputs": [7]},
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 4}, split="unused"
    )
    packer = packing.MultiBinPacker(
        num_partial_examples=10,
        # feature_lengths unset, will be set before packing.
    )
    packing_preprocessor = packing.AirIOPackDatasetMapPreprocessor(
        pool_size=10, packer=packer
    )
    unused_rng = None
    ds = packing_preprocessor(ds, runtime_args, unused_rng)
    updated_runtime_args = packing_preprocessor.update_runtime_args(
        runtime_args
    )
    pad_fn = functools.partial(
        preprocessors.pad, runtime_args=updated_runtime_args
    )
    ds = ds.map(pad_fn)
    ds_iter = iter(ds)

    expected_elements = [
        # First element was already fully packed on 'inputs'.
        {
            "inputs": [1, 2, 3, 4],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
        },
        # The third element is fully packed, hence yielded out of order.
        {
            "inputs": [11, 12, 13, 14],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
        },
        # Second and fourth element packed together (with padding).
        {
            "inputs": [5, 6, 7, 0],
            "inputs_segment_ids": [1, 1, 2, 0],
            "inputs_positions": [0, 1, 0, 0],
        },
    ]
    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      # Compare keys.
      self.assertSequenceEqual(sorted(actual), sorted(expected))
      np.testing.assert_array_equal(actual[feature], expected[feature])

  def test_multi_bin_packer_feature_unset_fails(self):
    ex = {"inputs": np.asarray([1, 2, 3, 4])}
    packer = packing.MultiBinPacker(num_partial_examples=10)
    with self.assertRaisesRegex(ValueError, "feature_lengths must be set.*"):
      packer.fit_example(ex)
    with self.assertRaisesRegex(ValueError, "feature_lengths must be set.*"):
      packer.get_packed_feature_lengths()

  def test_multi_bin_packer_no_packed_example(self):
    packer = packing.MultiBinPacker(
        feature_lengths={"inputs": 4}, num_partial_examples=10
    )
    with self.assertRaisesRegex(ValueError, "No packed examples.*"):
      packer.get_packed_example()


class NoamPackingMapTest(parameterized.TestCase):

  def test_pack_single_feature(self):
    # 5 elements of variable sequence length.
    input_elements = [[1, 2, 3, 4], [5, 6], [11, 12, 13, 14], [7]]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(np.asarray)
    packer = packing.NoamPacker(feature_lengths=4)
    ds = packing.PackLazyMapDataset(ds, pool_size=10, packer=packer)
    ds_iter = iter(ds)

    expected_elements = [
        # First element was already fully packed on 'inputs'.
        [1, 2, 3, 4],
        # Second and third (partial) element packed together.
        [5, 6, 11, 12],
        # Third (remainder) and fourth element packed together.
        [13, 14, 7],
    ]
    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      # Elements are tuples with (inputs, inputs_segment_ids, inputs_positions).
      np.testing.assert_array_equal(actual, expected)

  def test_pack_single_feature_in_dict(self):
    input_elements = [
        {"inputs": [1, 2, 3, 4]},
        {"inputs": [5, 6]},
        {"inputs": [11, 12, 13, 14]},
        {"inputs": [7]},
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    packer = packing.NoamPacker(feature_lengths={"inputs": 4})
    ds = packing.PackLazyMapDataset(ds, pool_size=10, packer=packer)
    ds_iter = iter(ds)

    expected_elements = [
        # First element was already fully packed on 'inputs'.
        {
            "inputs": [1, 2, 3, 4],
        },
        # Second and third (partial) element packed together.
        {
            "inputs": [5, 6, 11, 12],
        },
        # Third (remainder) and fourth element packed together.
        {
            "inputs": [13, 14, 7],
        },
    ]

    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      # Compare keys.
      self.assertSequenceEqual(sorted(actual), sorted(expected))
      np.testing.assert_array_equal(actual["inputs"], expected["inputs"])

  def test_pack_multiple_slices_from_examples(self):
    input_elements = [
        {"inputs": [0, 1, 2, 3, 4]},
        {"inputs": [5]},
        {"inputs": [6]},
        {"inputs": [7]},
        {"inputs": [8]},
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    packer = packing.NoamPacker(feature_lengths={"inputs": 2})
    ds = packing.PackLazyMapDataset(ds, pool_size=10, packer=packer)
    ds_iter = iter(ds)

    expected_elements = [
        {"inputs": [0, 1]},
        {"inputs": [2, 3]},
        {"inputs": [4, 5]},
        {"inputs": [6, 7]},
        {"inputs": [8]},
    ]

    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      # Compare keys.
      self.assertSequenceEqual(sorted(actual), sorted(expected))
      np.testing.assert_array_equal(actual["inputs"], expected["inputs"])

  @parameterized.parameters(
      "inputs",
      "targets",
  )
  def test_pack_multiple_features_same_sequences_length(self, feature: str):
    input_elements = [
        {
            "inputs": [1, 2, 3, 4],
            "targets": [10, 20, 30, 40],
        },
        {
            "inputs": [5, 6],
            "targets": [30, 40],
        },
        {
            "inputs": [11, 12, 13, 14],
            "targets": [31, 41, 51, 61],
        },
        {
            "inputs": [7],
            "targets": [60],
        },
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    packer = packing.NoamPacker(feature_lengths={"inputs": 4, "targets": 4})
    ds = packing.PackLazyMapDataset(ds, pool_size=10, packer=packer)
    ds_iter = iter(ds)

    expected_elements = [
        # First element was already fully packed.
        {
            "inputs": [1, 2, 3, 4],
            "targets": [10, 20, 30, 40],
        },
        # Second and third (partial) elements packed together.
        {
            "inputs": [5, 6, 11, 12],
            "targets": [30, 40, 31, 41],
        },
        # Third (remainder) and fourth elements packed together.
        {
            "inputs": [13, 14, 7],
            "targets": [51, 61, 60],
        },
    ]
    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      # Compare keys.
      self.assertSequenceEqual(sorted(actual), sorted(expected))
      np.testing.assert_array_equal(actual[feature], expected[feature])

  @parameterized.parameters(
      "inputs",
      "targets",
  )
  def test_pack_multiple_features_different_sequences_length(
      self, feature: str
  ):
    input_elements = [
        {
            "inputs": [1, 2, 3, 4, 5, 6, 7, 8],
            "targets": [10, 20, 30, 40],
        },
        {
            "inputs": [5, 6, 7, 8, 9, 1],
            "targets": [30, 40, 50],
        },
        {
            "inputs": [11, 12],
            "targets": [31],
        },
        {
            "inputs": [7, 9],
            "targets": [60],
        },
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    packer = packing.NoamPacker(feature_lengths={"inputs": 6, "targets": 3})
    ds = packing.PackLazyMapDataset(ds, pool_size=10, packer=packer)
    ds_iter = iter(ds)

    expected_elements = [
        {  # First element is sliced to produce first packed element.
            "inputs": [1, 2, 3, 4, 5, 6],
            "targets": [10, 20, 30],
        },
        {  # First (remainder) and second (partial) elements are packed.
            "inputs": [7, 8, 5, 6, 7, 8],
            "targets": [40, 30, 40],
        },
        {  # Second (remainder) third and fourth elements are packed.
            "inputs": [9, 1, 11, 12, 7, 9],
            "targets": [50, 31, 60],
        },
    ]
    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      np.testing.assert_array_equal(actual[feature], expected[feature])

  def test_do_not_pack_marked_features(self):
    input_elements = [
        {
            "inputs": [1, 2, 3, 4],
            "targets": [10],
        },
        {
            "inputs": [5, 6],
            "targets": [30],
        },
        {
            "inputs": [11, 12, 13, 14],
            "targets": [31],
        },
        {
            "inputs": [7],
            "targets": [60],
        },
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    packer = packing.NoamPacker(
        feature_lengths={"inputs": 6, "targets": -1},
    )
    ds = packing.PackLazyMapDataset(ds, pool_size=10, packer=packer)
    ds_iter = iter(ds)
    expected_targets = [[[10], [30]], [[31], [60]]]
    for actual, expected in zip(ds_iter, expected_targets, strict=True):
      np.testing.assert_array_equal(actual["targets"], expected)

  @parameterized.parameters(
      "input_tokens",
      "input_vectors",
  )
  def test_pack_two_dimensional_features(self, feature: str):
    input_elements = [
        {
            "input_tokens": [1, 2, 3],
            "input_vectors": [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
        },
        {
            "input_tokens": [4, 5],
            "input_vectors": [[3, 4, 5], [4, 5, 6]],
        },
        {
            "input_tokens": [6],
            "input_vectors": [[5, 6, 7]],
        },
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    packer = packing.NoamPacker(
        feature_lengths={"input_tokens": 2, "input_vectors": 2}
    )
    ds = packing.PackLazyMapDataset(ds, pool_size=10, packer=packer)
    ds_iter = iter(ds)

    expected_elements = [
        {  # First element is sliced to produce first packed element.
            "input_tokens": [1, 2],
            "input_vectors": [[0, 1, 2], [1, 2, 3]],
        },
        {  # First (remainder) and second (partial) element are packed together.
            "input_tokens": [3, 4],
            "input_vectors": [[2, 3, 4], [3, 4, 5]],
        },
        {  # Second (remainder) and third element are packed together.
            "input_tokens": [5, 6],
            "input_vectors": [[4, 5, 6], [5, 6, 7]],
        },
    ]
    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      np.testing.assert_array_equal(actual[feature], expected[feature])

  def test_airio_pack_preprocessor(self):
    input_elements = [
        {"inputs": [1, 2, 3, 4]},
        {"inputs": [5, 6]},
        {"inputs": [11, 12, 13, 14]},
        {"inputs": [7]},
        {"inputs": [8]},
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 4}, split="unused"
    )
    packer = packing.NoamPacker()  # feature_lengths will be set before packing.
    transform = packing.AirIOPackDatasetMapPreprocessor(
        pool_size=10, packer=packer
    )
    unused_rng = None
    ds = transform(ds, runtime_args, unused_rng)
    updated_runtime_args = transform.update_runtime_args(runtime_args)
    ds_iter = iter(ds)

    # Verify updated runtime args
    expected_updated_runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 4},
        split="unused",
    )
    self.assertEqual(updated_runtime_args, expected_updated_runtime_args)

    # Verify packed examples.
    expected_elements = [
        # First element was already fully packed.
        {
            "inputs": [1, 2, 3, 4],
        },
        # Second and third (partial) elements packed together.
        {
            "inputs": [5, 6, 11, 12],
        },
        # Third (remainder), fourth and fifth elements packed together.
        {
            "inputs": [13, 14, 7, 8],
        },
    ]

    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      # Compare keys.
      self.assertSequenceEqual(sorted(actual), sorted(expected))
      np.testing.assert_array_equal(actual["inputs"], expected["inputs"])

  def test_max_fan_out(self):
    input_elements = [
        {"inputs": [1, 2, 3, 4]},
        {"inputs": [5, 6]},
        {"inputs": [11, 12, 13, 14]},
        {"inputs": [7]},
        {"inputs": [8]},
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 2}, split="unused"
    )
    packer = packing.NoamPacker()  # feature_lengths will be set before packing.
    transform = packing.AirIOPackDatasetMapPreprocessor(
        pool_size=5, packer=packer
    )
    unused_rng = None
    ds = transform(ds, runtime_args, unused_rng)
    ds_iter = iter(ds)

    # Since pool_size (= max_fan_out) is 5, packing should produce no more than
    # 5 examples.
    expected_elements = [
        # First element is distributed across 2 packed examples.
        {"inputs": [1, 2]},
        {"inputs": [3, 4]},
        # Second element.
        {"inputs": [5, 6]},
        # Third element is distributed across 2 packed examples.
        {"inputs": [11, 12]},
        {"inputs": [13, 14]},
        # Fourth and fifth elements are dropped.
    ]

    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      # Compare keys.
      self.assertSequenceEqual(sorted(actual), sorted(expected))
      np.testing.assert_array_equal(actual["inputs"], expected["inputs"])

  def test_pack_single_feature_with_padding(self):
    input_elements = [
        {"inputs": [1, 2, 3, 4]},
        {"inputs": [5, 6]},
        {"inputs": [11, 12, 13, 14]},
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 4}, split="unused"
    )
    packer = packing.NoamPacker()  # feature_lengths will be set before packing.
    packing_preprocessor = packing.AirIOPackDatasetMapPreprocessor(
        pool_size=10, packer=packer
    )
    unused_rng = None
    ds = packing_preprocessor(ds, runtime_args, unused_rng)
    updated_runtime_args = packing_preprocessor.update_runtime_args(
        runtime_args
    )
    pad_fn = functools.partial(
        preprocessors.pad, runtime_args=updated_runtime_args
    )
    ds = ds.map(pad_fn)

    ds_iter = iter(ds)

    expected_elements = [
        # First element was already fully packed.
        {
            "inputs": [1, 2, 3, 4],
        },
        # Second and third (partial) elements packed together.
        {
            "inputs": [5, 6, 11, 12],
        },
        # Remainder of third element, plus padding.
        {
            "inputs": [13, 14, 0, 0],
        },
    ]
    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      # Compare keys.
      self.assertSequenceEqual(sorted(actual), sorted(expected))
      np.testing.assert_array_equal(actual["inputs"], expected["inputs"])

  def test_pack_single_feature_multiple_pools(self):
    input_elements = [
        {"inputs": [1, 2, 3, 4]},
        {"inputs": [5, 6]},
        {"inputs": [11, 12, 13, 14]},
        {"inputs": [7]},
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    packer = packing.NoamPacker(feature_lengths={"inputs": 4})
    ds = packing.PackLazyMapDataset(ds, pool_size=3, packer=packer)
    ds_iter = iter(ds)

    expected_elements = [
        # First element was already fully packed on 'inputs'.
        {
            "inputs": [1, 2, 3, 4],
        },
        # Second and third (partial) element packed together.
        {
            "inputs": [5, 6, 11, 12],
        },
        # Third element (remainder). No more elements in the pool.
        {
            "inputs": [13, 14],
        },
        # Fourth element is in a separate pool.
        {
            "inputs": [7],
        },
    ]

    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      # Compare keys.
      self.assertSequenceEqual(sorted(actual), sorted(expected))
      np.testing.assert_array_equal(actual["inputs"], expected["inputs"])

  def test_noam_packer_feature_unset_fails(self):
    ex = {"inputs": np.asarray([1, 2, 3, 4])}
    packer = packing.NoamPacker()
    with self.assertRaisesRegex(ValueError, "feature_lengths must be set.*"):
      packer.fit_example(ex)
    with self.assertRaisesRegex(ValueError, "feature_lengths must be set.*"):
      packer.get_packed_feature_lengths()

  def test_noam_packer_no_packed_example(self):
    packer = packing.NoamPacker(feature_lengths={"inputs": 4})
    with self.assertRaisesRegex(ValueError, "No packed examples.*"):
      packer.get_packed_example()


class MultiBinPackingIterTest(parameterized.TestCase):

  def test_pack_single_feature(self):
    # 5 elements of variable sequence length.
    input_elements = [[1, 2, 3, 4], [5, 6], [11, 12, 13, 14], [7], [8]]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(np.asarray)
    ds = ds.to_iter_dataset()
    packer = packing.MultiBinPacker(feature_lengths=4, num_partial_examples=10)
    ds = packing.PackLazyIterDataset(ds, packer=packer)
    # Elements are tuples with (inputs, inputs_segment_ids, inputs_positions).
    expected_elements = [
        # First element was already fully packed on 'inputs'.
        ([1, 2, 3, 4], [1, 1, 1, 1], [0, 1, 2, 3]),
        # Third element was fully packed and hence produced out of order.
        ([11, 12, 13, 14], [1, 1, 1, 1], [0, 1, 2, 3]),
        # Second, fourth and fifth element packed together.
        ([5, 6, 7, 8], [1, 1, 2, 3], [0, 1, 0, 0]),
    ]
    for actual, expected in zip(ds, expected_elements, strict=True):
      # Elements are tuples with (inputs, inputs_segment_ids, inputs_positions).
      np.testing.assert_array_equal(actual, expected)

  # Same as above but elements are dictionaries.
  @parameterized.parameters(
      "inputs",
      "inputs_segment_ids",
      "inputs_positions",
  )
  def test_pack_single_feature_in_dict(self, feature: str):
    input_elements = [
        {"inputs": [1, 2, 3, 4]},
        {"inputs": [5, 6]},
        {"inputs": [11, 12, 13, 14]},
        {"inputs": [7]},
        {"inputs": [8]},
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    packer = packing.MultiBinPacker(
        feature_lengths={"inputs": 4}, num_partial_examples=10
    )
    ds = packing.PackLazyIterDataset(ds, packer=packer)

    expected_elements = [
        # First element was already fully packed on 'inputs'.
        {
            "inputs": [1, 2, 3, 4],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
        },
        # Third element was fully packed and hence produced out of order.
        {
            "inputs": [11, 12, 13, 14],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
        },
        # Second, fourth and fifth element packed together.
        {
            "inputs": [5, 6, 7, 8],
            "inputs_segment_ids": [1, 1, 2, 3],
            "inputs_positions": [0, 1, 0, 0],
        },
    ]

    for actual, expected in zip(ds, expected_elements, strict=True):
      # Compare keys.
      self.assertSequenceEqual(sorted(actual), sorted(expected))
      np.testing.assert_array_equal(actual[feature], expected[feature])

  @parameterized.parameters(
      "inputs",
      "inputs_segment_ids",
      "inputs_positions",
      "targets",
      "targets_segment_ids",
      "targets_positions",
  )
  def test_pack_multiple_features_same_sequences_length(self, feature: str):
    input_elements = [
        {
            "inputs": [1, 2, 3, 4],
            "targets": [10, 20],
        },
        {
            "inputs": [5, 6],
            "targets": [30, 40, 50],
        },
        {
            "inputs": [11, 12, 13, 14],
            "targets": [31, 41, 51],
        },
        {
            "inputs": [7],
            "targets": [60],
        },
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    packer = packing.MultiBinPacker(
        feature_lengths={"inputs": 4, "targets": 4}, num_partial_examples=10
    )
    ds = packing.PackLazyIterDataset(ds, packer=packer)

    expected_elements = [
        # First element was already fully packed on 'inputs'.
        {
            "inputs": [1, 2, 3, 4],
            "targets": [10, 20],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
            "targets_segment_ids": [1, 1],
            "targets_positions": [0, 1],
        },
        # Second and fourth element packed together.
        {
            "inputs": [5, 6, 7],
            "inputs_segment_ids": [1, 1, 2],
            "inputs_positions": [0, 1, 0],
            "targets": [30, 40, 50, 60],
            "targets_segment_ids": [1, 1, 1, 2],
            "targets_positions": [0, 1, 2, 0],
        },
        # The third element has fully packed "inputs", but not "targets", so is
        # added as a partially packed example instead of being yielded.
        {
            "inputs": [11, 12, 13, 14],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
            "targets": [31, 41, 51],
            "targets_segment_ids": [1, 1, 1],
            "targets_positions": [0, 1, 2],
        },
    ]
    for actual, expected in zip(ds, expected_elements, strict=True):
      # Compare keys.
      self.assertSequenceEqual(sorted(actual), sorted(expected))
      np.testing.assert_array_equal(actual[feature], expected[feature])

  @parameterized.parameters(
      "inputs",
      "inputs_segment_ids",
      "inputs_positions",
      "targets",
      "targets_segment_ids",
      "targets_positions",
  )
  def test_pack_multiple_features_different_sequences_length(
      self, feature: str
  ):
    input_elements = [
        {
            "inputs": [1, 2, 3, 4],
            "targets": [10, 20],
        },
        {
            "inputs": [5, 6],
            "targets": [30, 40, 50],
        },
        {
            "inputs": [11, 12, 13, 14],
            "targets": [31],
        },
        {
            "inputs": [7],
            "targets": [60],
        },
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    packer = packing.MultiBinPacker(
        feature_lengths={"inputs": 6, "targets": 4}, num_partial_examples=10
    )
    ds = packing.PackLazyIterDataset(ds, packer=packer)

    expected_elements = [
        {  # 2nd and 3rd example create a perfect fit, thus yielded first.
            "inputs": [5, 6, 11, 12, 13, 14],
            "inputs_segment_ids": [1, 1, 2, 2, 2, 2],
            "inputs_positions": [0, 1, 0, 1, 2, 3],
            "targets": [30, 40, 50, 31],
            "targets_segment_ids": [1, 1, 1, 2],
            "targets_positions": [0, 1, 2, 0],
        },
        {
            "inputs": [1, 2, 3, 4, 7],
            "inputs_segment_ids": [1, 1, 1, 1, 2],
            "inputs_positions": [0, 1, 2, 3, 0],
            "targets": [10, 20, 60],
            "targets_segment_ids": [1, 1, 2],
            "targets_positions": [0, 1, 0],
        },
    ]
    for actual, expected in zip(ds, expected_elements, strict=True):
      np.testing.assert_array_equal(actual[feature], expected[feature])

  def test_do_not_pack_marked_features(self):
    input_elements = [
        {
            "inputs": [1, 2, 3, 4],
            "targets": [10, 20],
            "id": [1, 1],
        },
        {
            "inputs": [5, 6],
            "targets": [30, 40, 50],
            "id": [1, 2],
        },
        {
            "inputs": [11, 12, 13, 14],
            "targets": [31],
            "id": [2, 1],
        },
        {
            "inputs": [7],
            "targets": [60],
            "id": [2, 2],
        },
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    packer = packing.MultiBinPacker(
        feature_lengths={"inputs": 6, "targets": 4, "id": -1},
        num_partial_examples=10,
    )
    ds = packing.PackLazyIterDataset(ds, packer=packer)

    expected_ids = [[[1, 2], [2, 1]], [[1, 1], [2, 2]]]
    for actual, expected in zip(ds, expected_ids, strict=True):
      np.testing.assert_array_equal(actual["id"], expected)

  @parameterized.parameters(
      "input_tokens",
      "input_tokens_segment_ids",
      "input_tokens_positions",
      "input_vectors",
      "input_vectors_segment_ids",
      "input_vectors_positions",
  )
  def test_pack_two_dimensional_features(self, feature: str):
    input_elements = [
        {
            "input_tokens": [1, 2, 3],
            "input_vectors": [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
        },
        {
            "input_tokens": [4, 5],
            "input_vectors": [[3, 4, 5], [4, 5, 6]],
        },
        {
            "input_tokens": [6],
            "input_vectors": [[5, 6, 7]],
        },
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    packer = packing.MultiBinPacker(
        feature_lengths={"input_tokens": 3, "input_vectors": 3},
        num_partial_examples=10,
    )
    ds = packing.PackLazyIterDataset(ds, packer=packer)

    expected_elements = [
        {
            "input_tokens": [1, 2, 3],
            "input_tokens_segment_ids": [1, 1, 1],
            "input_tokens_positions": [0, 1, 2],
            "input_vectors": [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
            "input_vectors_segment_ids": [1, 1, 1],
            "input_vectors_positions": [0, 1, 2],
        },
        {
            "input_tokens": [4, 5, 6],
            "input_tokens_segment_ids": [1, 1, 2],
            "input_tokens_positions": [0, 1, 0],
            "input_vectors": [[3, 4, 5], [4, 5, 6], [5, 6, 7]],
            "input_vectors_segment_ids": [1, 1, 2],
            "input_vectors_positions": [0, 1, 0],
        },
    ]
    for actual, expected in zip(ds, expected_elements, strict=True):
      np.testing.assert_array_equal(actual[feature], expected[feature])

  @parameterized.parameters(
      "inputs",
      "inputs_segment_ids",
      "inputs_positions",
  )
  def test_airio_pack_preprocessor(self, feature: str):
    input_elements = [
        {
            "inputs": [1, 2, 3, 4],
        },
        {
            "inputs": [5, 6],
        },
        {
            "inputs": [11, 12, 13, 14],
        },
        {
            "inputs": [7],
        },
        {
            "inputs": [8],
        },
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 4}, split="unused"
    )
    packer = packing.MultiBinPacker(
        num_partial_examples=10,
        # feature_lengths unset, will be set before packing.
    )
    transform = packing.AirIOPackDatasetIterPreprocessor(packer=packer)
    unused_rng = None
    ds = transform(ds, runtime_args, unused_rng)
    updated_runtime_args = transform.update_runtime_args(runtime_args)

    # Verify updated runtime args
    expected_updated_runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={
            "inputs": 4,
            "inputs_segment_ids": 4,
            "inputs_positions": 4,
        },
        split="unused",
    )
    self.assertEqual(updated_runtime_args, expected_updated_runtime_args)

    # Verify packed examples.
    expected_elements = [
        # First element was already fully packed on 'inputs'.
        {
            "inputs": [1, 2, 3, 4],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
        },
        # Third element was fully packed and hence produced out of order.
        {
            "inputs": [11, 12, 13, 14],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
        },
        # Second, fourth and fifth element packed together.
        {
            "inputs": [5, 6, 7, 8],
            "inputs_segment_ids": [1, 1, 2, 3],
            "inputs_positions": [0, 1, 0, 0],
        },
    ]

    for actual, expected in zip(ds, expected_elements, strict=True):
      # Compare keys.
      self.assertSequenceEqual(sorted(actual), sorted(expected))
      np.testing.assert_array_equal(actual[feature], expected[feature])

  @parameterized.parameters(
      "inputs",
      "inputs_segment_ids",
      "inputs_positions",
  )
  def test_pack_single_feature_with_padding(self, feature: str):
    input_elements = [
        {"inputs": [1, 2, 3, 4]},
        {"inputs": [5, 6]},
        {"inputs": [11, 12, 13, 14]},
        {"inputs": [7]},
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 4}, split="unused"
    )
    packer = packing.MultiBinPacker(
        num_partial_examples=10,
        # feature_lengths unset, will be set before packing.
    )
    packing_preprocessor = packing.AirIOPackDatasetIterPreprocessor(
        packer=packer
    )
    unused_rng = None
    ds = packing_preprocessor(ds, runtime_args, unused_rng)
    updated_runtime_args = packing_preprocessor.update_runtime_args(
        runtime_args
    )
    pad_fn = functools.partial(
        preprocessors.pad, runtime_args=updated_runtime_args
    )
    ds = ds.map(pad_fn)

    expected_elements = [
        # First element was already fully packed on 'inputs'.
        {
            "inputs": [1, 2, 3, 4],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
        },
        # The third element is fully packed, hence yielded out of order.
        {
            "inputs": [11, 12, 13, 14],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
        },
        # Second and fourth element packed together (with padding).
        {
            "inputs": [5, 6, 7, 0],
            "inputs_segment_ids": [1, 1, 2, 0],
            "inputs_positions": [0, 1, 0, 0],
        },
    ]
    for actual, expected in zip(ds, expected_elements, strict=True):
      # Compare keys.
      self.assertSequenceEqual(sorted(actual), sorted(expected))
      np.testing.assert_array_equal(actual[feature], expected[feature])

  def test_checkpointing_with_json_serialization(self):
    input_elements = [[1, 2], [3], [4], [5, 6], [11], [12], [13], [14, 7, 8]]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(np.asarray)
    ds = ds.to_iter_dataset()
    packer = packing.MultiBinPacker(feature_lengths=2, num_partial_examples=10)
    ds = packing.PackLazyIterDataset(ds, packer=packer)
    ds_iter = iter(ds)

    max_steps = 6
    values_without_interruption = []
    checkpoints = []

    for _ in range(max_steps):
      st = ds_iter.get_state()
      checkpoints.append(json.dumps(st))  # pytype: disable=attribute-error
      values_without_interruption.append(next(ds_iter))
    for starting_step in [0, 1, 3, 5]:
      ds_iter.set_state(json.loads(checkpoints[starting_step]))  # pytype: disable=attribute-error
      for i in range(starting_step, max_steps):
        np.testing.assert_array_equal(
            next(ds_iter), values_without_interruption[i]
        )


class NoamPackingIterTest(parameterized.TestCase):

  def test_pack_single_feature(self):
    # 5 elements of variable sequence length.
    input_elements = [[1, 2, 3, 4], [5, 6], [11, 12, 13, 14], [7]]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(np.asarray)
    ds = ds.to_iter_dataset()
    packer = packing.NoamPacker(feature_lengths=4)
    ds = packing.PackLazyIterDataset(ds, packer=packer)

    expected_elements = [
        # First element was already fully packed on 'inputs'.
        [1, 2, 3, 4],
        # Second and third (partial) element packed together.
        [5, 6, 11, 12],
        # Third (remainder) and fourth element packed together.
        [13, 14, 7],
    ]
    for actual, expected in zip(ds, expected_elements, strict=True):
      # Elements are tuples with (inputs, inputs_segment_ids, inputs_positions).
      np.testing.assert_array_equal(actual, expected)

  def test_pack_single_feature_in_dict(self):
    input_elements = [
        {"inputs": [1, 2, 3, 4]},
        {"inputs": [5, 6]},
        {"inputs": [11, 12, 13, 14]},
        {"inputs": [7]},
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    packer = packing.NoamPacker(feature_lengths={"inputs": 4})
    ds = packing.PackLazyIterDataset(ds, packer=packer)

    expected_elements = [
        # First element was already fully packed on 'inputs'.
        {
            "inputs": [1, 2, 3, 4],
        },
        # Second and third (partial) element packed together.
        {
            "inputs": [5, 6, 11, 12],
        },
        # Third (remainder) and fourth element packed together.
        {
            "inputs": [13, 14, 7],
        },
    ]

    for actual, expected in zip(ds, expected_elements, strict=True):
      # Compare keys.
      self.assertSequenceEqual(sorted(actual), sorted(expected))
      np.testing.assert_array_equal(actual["inputs"], expected["inputs"])

  def test_pack_multiple_slices_from_examples(self):
    input_elements = [
        {"inputs": [0, 1, 2, 3, 4]},
        {"inputs": [5]},
        {"inputs": [6]},
        {"inputs": [7]},
        {"inputs": [8]},
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    packer = packing.NoamPacker(feature_lengths={"inputs": 2})
    ds = packing.PackLazyIterDataset(ds, packer=packer)

    # expected: [1, 2], [3, 4], [5, 6], ..., [18, 19]
    expected_elements = [
        {"inputs": [0, 1]},
        {"inputs": [2, 3]},
        {"inputs": [4, 5]},
        {"inputs": [6, 7]},
        {"inputs": [8]},
    ]

    for actual, expected in zip(ds, expected_elements, strict=True):
      # Compare keys.
      self.assertSequenceEqual(sorted(actual), sorted(expected))
      np.testing.assert_array_equal(actual["inputs"], expected["inputs"])

  @parameterized.parameters(
      "inputs",
      "targets",
  )
  def test_pack_multiple_features_same_sequences_length(self, feature: str):
    input_elements = [
        {
            "inputs": [1, 2, 3, 4],
            "targets": [10, 20, 30, 40],
        },
        {
            "inputs": [5, 6],
            "targets": [30, 40],
        },
        {
            "inputs": [11, 12, 13, 14],
            "targets": [31, 41, 51, 61],
        },
        {
            "inputs": [7],
            "targets": [60],
        },
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    packer = packing.NoamPacker(feature_lengths={"inputs": 4, "targets": 4})
    ds = packing.PackLazyIterDataset(ds, packer=packer)

    expected_elements = [
        # First element was already fully packed.
        {
            "inputs": [1, 2, 3, 4],
            "targets": [10, 20, 30, 40],
        },
        # Second and third (partial) elements packed together.
        {
            "inputs": [5, 6, 11, 12],
            "targets": [30, 40, 31, 41],
        },
        # Third (remainder) and fourth elements packed together.
        {
            "inputs": [13, 14, 7],
            "targets": [51, 61, 60],
        },
    ]
    for actual, expected in zip(ds, expected_elements, strict=True):
      # Compare keys.
      self.assertSequenceEqual(sorted(actual), sorted(expected))
      np.testing.assert_array_equal(actual[feature], expected[feature])

  @parameterized.parameters(
      "inputs",
      "targets",
  )
  def test_pack_multiple_features_different_sequences_length(
      self, feature: str
  ):
    input_elements = [
        {
            "inputs": [1, 2, 3, 4, 5, 6, 7, 8],
            "targets": [10, 20, 30, 40],
        },
        {
            "inputs": [5, 6, 7, 8, 9, 1],
            "targets": [30, 40, 50],
        },
        {
            "inputs": [11, 12],
            "targets": [31],
        },
        {
            "inputs": [7, 9],
            "targets": [60],
        },
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    packer = packing.NoamPacker(feature_lengths={"inputs": 6, "targets": 3})
    ds = packing.PackLazyIterDataset(ds, packer=packer)

    expected_elements = [
        {  # First element is sliced to produce first packed element.
            "inputs": [1, 2, 3, 4, 5, 6],
            "targets": [10, 20, 30],
        },
        {  # First (remainder) and second (partial) elements are packed.
            "inputs": [7, 8, 5, 6, 7, 8],
            "targets": [40, 30, 40],
        },
        {  # Second (remainder) third and fourth elements are packed.
            "inputs": [9, 1, 11, 12, 7, 9],
            "targets": [50, 31, 60],
        },
    ]
    for actual, expected in zip(ds, expected_elements, strict=True):
      np.testing.assert_array_equal(actual[feature], expected[feature])

  def test_do_not_pack_marked_features(self):
    input_elements = [
        {
            "inputs": [1, 2, 3, 4],
            "targets": [10],
        },
        {
            "inputs": [5, 6],
            "targets": [30],
        },
        {
            "inputs": [11, 12, 13, 14],
            "targets": [31],
        },
        {
            "inputs": [7],
            "targets": [60],
        },
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    packer = packing.NoamPacker(
        feature_lengths={"inputs": 6, "targets": -1},
    )
    ds = packing.PackLazyIterDataset(ds, packer=packer)

    expected_targets = [[[10], [30]], [[31], [60]]]
    for actual, expected in zip(ds, expected_targets, strict=True):
      np.testing.assert_array_equal(actual["targets"], expected)

  def test_do_not_pack_marked_features_sliced_across_multiple(self):
    input_elements = [
        {"id": 0, "inputs": [0]},
        {"id": 1, "inputs": [1, 2, 3, 4, 5]},
        {"id": 2, "inputs": [5, 6, 7, 8, 9]},
        {"id": 3, "inputs": [10]},
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    packer = packing.NoamPacker(
        feature_lengths={"inputs": 2, "id": -1},
    )
    ds = packing.PackLazyIterDataset(ds, packer=packer)
    expected_ids = [[0, 1], [1], [1], [2], [2], [2, 3]]
    for actual, expected in zip(ds, expected_ids, strict=True):
      np.testing.assert_array_equal(actual["id"], expected)

  @parameterized.parameters(
      "input_tokens",
      "input_vectors",
  )
  def test_pack_two_dimensional_features(self, feature: str):
    input_elements = [
        {
            "input_tokens": [1, 2, 3],
            "input_vectors": [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
        },
        {
            "input_tokens": [4, 5],
            "input_vectors": [[3, 4, 5], [4, 5, 6]],
        },
        {
            "input_tokens": [6],
            "input_vectors": [[5, 6, 7]],
        },
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    packer = packing.NoamPacker(
        feature_lengths={"input_tokens": 2, "input_vectors": 2}
    )
    ds = packing.PackLazyIterDataset(ds, packer=packer)

    expected_elements = [
        {  # First element is sliced to produce first packed element.
            "input_tokens": [1, 2],
            "input_vectors": [[0, 1, 2], [1, 2, 3]],
        },
        {  # First (remainder) and second (partial) element are packed together.
            "input_tokens": [3, 4],
            "input_vectors": [[2, 3, 4], [3, 4, 5]],
        },
        {  # Second (remainder) and third element are packed together.
            "input_tokens": [5, 6],
            "input_vectors": [[4, 5, 6], [5, 6, 7]],
        },
    ]
    for actual, expected in zip(ds, expected_elements, strict=True):
      np.testing.assert_array_equal(actual[feature], expected[feature])

  def test_airio_pack_preprocessor(self):
    input_elements = [
        {"inputs": [1, 2, 3, 4]},
        {"inputs": [5, 6]},
        {"inputs": [11, 12, 13, 14]},
        {"inputs": [7]},
        {"inputs": [8]},
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 4}, split="unused"
    )
    packer = packing.NoamPacker()  # feature_lengths will be set before packing.
    transform = packing.AirIOPackDatasetIterPreprocessor(packer=packer)
    unused_rng = None
    ds = transform(ds, runtime_args, unused_rng)
    updated_runtime_args = transform.update_runtime_args(runtime_args)

    # Verify updated runtime args
    expected_updated_runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 4},
        split="unused",
    )
    self.assertEqual(updated_runtime_args, expected_updated_runtime_args)

    # Verify packed examples.
    expected_elements = [
        # First element was already fully packed.
        {
            "inputs": [1, 2, 3, 4],
        },
        # Second and third (partial) elements packed together.
        {
            "inputs": [5, 6, 11, 12],
        },
        # Third (remainder), fourth and fifth elements packed together.
        {
            "inputs": [13, 14, 7, 8],
        },
    ]

    for actual, expected in zip(ds, expected_elements, strict=True):
      # Compare keys.
      self.assertSequenceEqual(sorted(actual), sorted(expected))
      np.testing.assert_array_equal(actual["inputs"], expected["inputs"])

  def test_pack_single_feature_with_padding(self):
    input_elements = [
        {"inputs": [1, 2, 3, 4]},
        {"inputs": [5, 6]},
        {"inputs": [11, 12, 13, 14]},
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 4}, split="unused"
    )
    packer = packing.NoamPacker()  # feature_lengths will be set before packing.
    packing_preprocessor = packing.AirIOPackDatasetIterPreprocessor(
        packer=packer
    )
    unused_rng = None
    ds = packing_preprocessor(ds, runtime_args, unused_rng)
    updated_runtime_args = packing_preprocessor.update_runtime_args(
        runtime_args
    )
    pad_fn = functools.partial(
        preprocessors.pad, runtime_args=updated_runtime_args
    )
    ds = ds.map(pad_fn)

    expected_elements = [
        # First element was already fully packed.
        {
            "inputs": [1, 2, 3, 4],
        },
        # Second and third (partial) elements packed together.
        {
            "inputs": [5, 6, 11, 12],
        },
        # Remainder of third element, plus padding.
        {
            "inputs": [13, 14, 0, 0],
        },
    ]
    for actual, expected in zip(ds, expected_elements, strict=True):
      # Compare keys.
      self.assertSequenceEqual(sorted(actual), sorted(expected))
      np.testing.assert_array_equal(actual["inputs"], expected["inputs"])

  def test_yield_packed_examples_at_the_end(self):
    input_elements = [[1, 2, 3, 4, 5, 6]]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(np.asarray)
    ds = ds.to_iter_dataset()
    packer = packing.NoamPacker(feature_lengths=2)
    ds = packing.PackLazyIterDataset(ds, packer=packer)

    expected_elements = [[1, 2], [3, 4], [5, 6]]
    for actual, expected in zip(ds, expected_elements, strict=True):
      np.testing.assert_array_equal(actual, expected)

  def test_yield_partially_packed_examples_at_the_end(self):
    input_elements = [[1, 2, 3]]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(np.asarray)
    ds = ds.to_iter_dataset()
    packer = packing.NoamPacker(feature_lengths=2)
    ds = packing.PackLazyIterDataset(ds, packer=packer)

    expected_elements = [[1, 2], [3]]
    for actual, expected in zip(ds, expected_elements, strict=True):
      np.testing.assert_array_equal(actual, expected)

  def test_checkpointing_with_json_serialization(self):
    # 5 elements of variable sequence length.
    input_elements = [[1, 2, 3, 4], [5, 6], [11, 12, 13, 14], [7]]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(np.asarray)
    ds = ds.to_iter_dataset()
    packer = packing.NoamPacker(feature_lengths=2)
    ds = packing.PackLazyIterDataset(ds, packer=packer)
    ds_iter = iter(ds)

    max_steps = 6
    values_without_interruption = []
    checkpoints = []

    for _ in range(max_steps):
      st = ds_iter.get_state()
      checkpoints.append(json.dumps(st))  # pytype: disable=attribute-error
      values_without_interruption.append(next(ds_iter))
    for starting_step in [0, 1, 3, 5]:
      ds_iter.set_state(json.loads(checkpoints[starting_step]))  # pytype: disable=attribute-error
      for i in range(starting_step, max_steps):
        np.testing.assert_array_equal(
            next(ds_iter), values_without_interruption[i]
        )


class CommonPackersTest(absltest.TestCase):

  def test_noam_map_pack(self):
    input_elements = [
        {"inputs": [1, 2, 3, 4]},
        {"inputs": [5, 6]},
        {"inputs": [11, 12, 13, 14]},
        {"inputs": [7]},
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 4}, split="unused"
    )
    unused_rng = None
    ds = packing.NoamPackMapPreprocessor(ds, runtime_args, unused_rng)
    updated_runtime_args = packing.NoamPackMapPreprocessor.update_runtime_args(
        runtime_args
    )
    ds_iter = iter(ds)

    expected_elements = [
        {"inputs": [1, 2, 3, 4]},
        {"inputs": [5, 6, 11, 12]},
        {"inputs": [13, 14, 7]},
    ]
    self.assertEqual(updated_runtime_args, runtime_args)
    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      # Compare keys.
      self.assertSequenceEqual(sorted(actual), sorted(expected))
      np.testing.assert_array_equal(actual["inputs"], expected["inputs"])

  def test_single_bin_map_pack(self):
    input_elements = [
        {"inputs": [1, 2, 3, 4]},
        {"inputs": [5, 6, 7]},
        {"inputs": [11, 12, 13, 14]},
        {"inputs": [8, 9]},
        {"inputs": [10]},
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 4}, split="unused"
    )
    unused_rng = None
    ds = packing.SingleBinTruePackMapPreprocessor(ds, runtime_args, unused_rng)
    updated_runtime_args = (
        packing.SingleBinTruePackMapPreprocessor.update_runtime_args(
            runtime_args
        )
    )
    ds_iter = iter(ds)

    expected_elements = [
        {  # Fully packed, yielded immediately.
            "inputs": [1, 2, 3, 4],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
        },
        {  # Fully packed, yielded immediately.
            "inputs": [11, 12, 13, 14],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
        },
        {  # Partially packed example gets yielded when fitting fourth example.
            "inputs": [5, 6, 7],
            "inputs_segment_ids": [1, 1, 1],
            "inputs_positions": [0, 1, 2],
        },
        {  # Fourth and fifth examples are packed together.
            "inputs": [8, 9, 10],
            "inputs_segment_ids": [1, 1, 2],
            "inputs_positions": [0, 1, 0],
        },
    ]
    expected_runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={
            "inputs": 4,
            "inputs_segment_ids": 4,
            "inputs_positions": 4,
        },
        split="unused",
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)
    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      # Compare keys.
      self.assertSequenceEqual(sorted(actual), sorted(expected))
      for feature in ["inputs", "inputs_segment_ids", "inputs_positions"]:
        np.testing.assert_array_equal(actual[feature], expected[feature])

  def test_multi_bin_map_pack(self):
    input_elements = [
        {"inputs": [1, 2, 3, 4]},
        {"inputs": [5, 6, 7]},
        {"inputs": [11, 12, 13, 14]},
        {"inputs": [8, 9]},
        {"inputs": [10]},
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 4}, split="unused"
    )
    unused_rng = None
    ds = packing.MultiBinTruePackMapPreprocessor(ds, runtime_args, unused_rng)
    updated_runtime_args = (
        packing.MultiBinTruePackMapPreprocessor.update_runtime_args(
            runtime_args
        )
    )
    ds_iter = iter(ds)

    expected_elements = [
        {  # Fully packed, yielded immediately.
            "inputs": [1, 2, 3, 4],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
        },
        {  # Fully packed, yielded immediately.
            "inputs": [11, 12, 13, 14],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
        },
        {  # Third and fifth examples are packed together.
            "inputs": [5, 6, 7, 10],
            "inputs_segment_ids": [1, 1, 1, 2],
            "inputs_positions": [0, 1, 2, 0],
        },
        {  # Remaining partially packed example.
            "inputs": [8, 9],
            "inputs_segment_ids": [1, 1],
            "inputs_positions": [0, 1],
        },
    ]
    expected_runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={
            "inputs": 4,
            "inputs_segment_ids": 4,
            "inputs_positions": 4,
        },
        split="unused",
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)
    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      # Compare keys.
      self.assertSequenceEqual(sorted(actual), sorted(expected))
      for feature in ["inputs", "inputs_segment_ids", "inputs_positions"]:
        np.testing.assert_array_equal(actual[feature], expected[feature])

  def test_noam_iter_pack(self):
    input_elements = [
        {"inputs": [1, 2, 3, 4]},
        {"inputs": [5, 6]},
        {"inputs": [11, 12, 13, 14]},
        {"inputs": [7]},
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 4}, split="unused"
    )
    unused_rng = None
    ds = packing.NoamPackIterPreprocessor(ds, runtime_args, unused_rng)
    updated_runtime_args = packing.NoamPackIterPreprocessor.update_runtime_args(
        runtime_args
    )

    expected_elements = [
        {"inputs": [1, 2, 3, 4]},
        {"inputs": [5, 6, 11, 12]},
        {"inputs": [13, 14, 7]},
    ]
    self.assertEqual(updated_runtime_args, runtime_args)
    for actual, expected in zip(ds, expected_elements, strict=True):
      # Compare keys.
      self.assertSequenceEqual(sorted(actual), sorted(expected))
      np.testing.assert_array_equal(actual["inputs"], expected["inputs"])

  def test_single_bin_iter_pack(self):
    input_elements = [
        {"inputs": [1, 2, 3, 4]},
        {"inputs": [5, 6, 7]},
        {"inputs": [11, 12, 13, 14]},
        {"inputs": [8, 9]},
        {"inputs": [10]},
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 4}, split="unused"
    )
    unused_rng = None
    ds = packing.SingleBinTruePackIterPreprocessor(ds, runtime_args, unused_rng)
    updated_runtime_args = (
        packing.SingleBinTruePackIterPreprocessor.update_runtime_args(
            runtime_args
        )
    )

    expected_elements = [
        {  # Fully packed, yielded immediately.
            "inputs": [1, 2, 3, 4],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
        },
        {  # Fully packed, yielded immediately.
            "inputs": [11, 12, 13, 14],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
        },
        {  # Partially packed example gets yielded when fitting fourth example.
            "inputs": [5, 6, 7],
            "inputs_segment_ids": [1, 1, 1],
            "inputs_positions": [0, 1, 2],
        },
        {  # Fourth and fifth examples are packed together.
            "inputs": [8, 9, 10],
            "inputs_segment_ids": [1, 1, 2],
            "inputs_positions": [0, 1, 0],
        },
    ]
    expected_runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={
            "inputs": 4,
            "inputs_segment_ids": 4,
            "inputs_positions": 4,
        },
        split="unused",
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)
    for actual, expected in zip(ds, expected_elements, strict=True):
      # Compare keys.
      self.assertSequenceEqual(sorted(actual), sorted(expected))
      for feature in ["inputs", "inputs_segment_ids", "inputs_positions"]:
        np.testing.assert_array_equal(actual[feature], expected[feature])

  def test_multi_bin_iter_pack(self):
    input_elements = [
        {"inputs": [1, 2, 3, 4]},
        {"inputs": [5, 6, 7]},
        {"inputs": [11, 12, 13, 14]},
        {"inputs": [8, 9]},
        {"inputs": [10]},
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 4}, split="unused"
    )
    unused_rng = None
    ds = packing.MultiBinTruePackIterPreprocessor(ds, runtime_args, unused_rng)
    updated_runtime_args = (
        packing.MultiBinTruePackIterPreprocessor.update_runtime_args(
            runtime_args
        )
    )

    expected_elements = [
        {  # Fully packed, yielded immediately.
            "inputs": [1, 2, 3, 4],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
        },
        {  # Fully packed, yielded immediately.
            "inputs": [11, 12, 13, 14],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
        },
        {  # Third and fifth examples are packed together.
            "inputs": [5, 6, 7, 10],
            "inputs_segment_ids": [1, 1, 1, 2],
            "inputs_positions": [0, 1, 2, 0],
        },
        {  # Remaining partially packed example.
            "inputs": [8, 9],
            "inputs_segment_ids": [1, 1],
            "inputs_positions": [0, 1],
        },
    ]
    expected_runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={
            "inputs": 4,
            "inputs_segment_ids": 4,
            "inputs_positions": 4,
        },
        split="unused",
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)
    for actual, expected in zip(ds, expected_elements, strict=True):
      # Compare keys.
      self.assertSequenceEqual(sorted(actual), sorted(expected))
      for feature in ["inputs", "inputs_segment_ids", "inputs_positions"]:
        np.testing.assert_array_equal(actual[feature], expected[feature])


if __name__ == "__main__":
  absltest.main()
