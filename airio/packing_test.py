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

"""Tests for packing."""

from absl.testing import absltest
from absl.testing import parameterized
from airio import packing
import grain.python as grain
import numpy as np

lazy_dataset = grain.experimental.lazy_dataset


class PackingTest(parameterized.TestCase):

  def test_pack_single_feature(self):
    # 5 elements of variable sequence length.
    input_elements = [[1, 2, 3, 4], [5, 6], [11, 12, 13, 14], [7], [8]]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(np.asarray)
    ds = packing.PackLazyMapDataset(
        ds, feature_lengths=4, pool_size=10, num_partial_examples=10
    )
    print("gaurav", list(iter(ds)))
    ds_iter = iter(ds)

    # Elements are tuples with (inputs, inputs_segment_ids, inputs_positions).
    expected_elements = [
        # First element was already fully packed on 'inputs'.
        ([1, 2, 3, 4], [1, 1, 1, 1], [0, 1, 2, 3]),
        # Third element was fully packed and hence produced out of order.
        ([11, 12, 13, 14], [1, 1, 1, 1], [0, 1, 2, 3]),
        # Second, fourth and five element packed together.
        ([5, 6, 7, 8], [1, 1, 2, 3], [0, 1, 0, 0]),
    ]
    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      # Elements are tuples with (inputs, inputs_segment_ids, inputs_positions).
      self.assertLen(actual, 3)
      np.testing.assert_array_equal(actual, expected)

  def test_pack_single_feature_remainder_is_padded(self):
    # 4 elements of variable sequence length.
    input_elements = [[1, 2, 3, 4], [5, 6], [11, 12, 13, 14], [7]]
    ds = lazy_dataset.SourceLazyMapDataset(input_elements)
    ds = ds.map(np.asarray)
    ds = packing.PackLazyMapDataset(
        ds, feature_lengths=4, pool_size=10, num_partial_examples=10
    )
    ds_iter = iter(ds)

    # Elements are tuples with (inputs, inputs_segment_ids, inputs_positions).
    expected_elements = [
        # First element was already fully packed on 'inputs'.
        ([1, 2, 3, 4], [1, 1, 1, 1], [0, 1, 2, 3]),
        # Third element was fully packed and hence produced out of order.
        ([11, 12, 13, 14], [1, 1, 1, 1], [0, 1, 2, 3]),
        # Second and fourth element packed together (plus padding).
        ([5, 6, 7, 0], [1, 1, 2, 0], [0, 1, 0, 0]),
    ]

    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      # Elements are tuples with (inputs, inputs_segment_ids, inputs_positions).
      self.assertLen(actual, 3)
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
    ds = packing.PackLazyMapDataset(
        ds, feature_lengths={"inputs": 4}, pool_size=10, num_partial_examples=10
    )
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
    ds = packing.PackLazyMapDataset(
        ds,
        feature_lengths={"inputs": 4, "targets": 4},
        pool_size=10,
        num_partial_examples=10,
    )
    ds_iter = iter(ds)

    expected_elements = [
        # First element was already fully packed on 'inputs'.
        {
            "inputs": [1, 2, 3, 4],
            "targets": [10, 20, 0, 0],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
            "targets_segment_ids": [1, 1, 0, 0],
            "targets_positions": [0, 1, 0, 0],
        },
        # Second and fourth element packed together.
        {
            "inputs": [5, 6, 7, 0],
            "inputs_segment_ids": [1, 1, 2, 0],
            "inputs_positions": [0, 1, 0, 0],
            "targets": [30, 40, 50, 60],
            "targets_segment_ids": [1, 1, 1, 2],
            "targets_positions": [0, 1, 2, 0],
        },
        {
            "inputs": [11, 12, 13, 14],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
            "targets": [31, 41, 51, 0],
            "targets_segment_ids": [1, 1, 1, 0],
            "targets_positions": [0, 1, 2, 0],
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
    ds = packing.PackLazyMapDataset(
        ds,
        feature_lengths={"inputs": 6, "targets": 4},
        pool_size=10,
        num_partial_examples=10,
    )
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
            "inputs": [1, 2, 3, 4, 7, 0],
            "inputs_segment_ids": [1, 1, 1, 1, 2, 0],
            "inputs_positions": [0, 1, 2, 3, 0, 0],
            "targets": [10, 20, 60, 0],  # 50 gets dropped.
            "targets_segment_ids": [1, 1, 2, 0],
            "targets_positions": [0, 1, 0, 0],
        },
    ]
    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      np.testing.assert_array_equal(actual[feature], expected[feature])

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
    ds = packing.PackLazyMapDataset(
        ds,
        feature_lengths={"input_tokens": 3, "input_vectors": 3},
        pool_size=10,
        num_partial_examples=10,
    )
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


if __name__ == "__main__":
  absltest.main()
