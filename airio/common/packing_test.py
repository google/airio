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

import functools
from absl.testing import absltest
from absl.testing import parameterized
from airio import preprocessors as preprocessors_lib
from airio.common import packing
from airio.common import preprocessors
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
    ds = packing.PackLazyMapDataset(
        ds,
        feature_lengths={"inputs": 6, "targets": 4, "id": -1},
        pool_size=10,
        num_partial_examples=10,
    )
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
    transform = packing.AirIOPackDatasetPreprocessor(
        pool_size=10, num_partial_examples=10
    )
    ds, updated_runtime_args = transform(ds, runtime_args)
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
    packing_preprocessor = packing.AirIOPackDatasetPreprocessor(
        pool_size=10, num_partial_examples=10
    )
    ds, updated_runtime_args = packing_preprocessor(ds, runtime_args)
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


if __name__ == "__main__":
  absltest.main()
