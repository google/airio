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

"""Tests for preprocessors."""

from absl.testing import absltest
from absl.testing import parameterized
from airio import preprocessors as preprocessors_lib
from airio.grain.common import constants
from airio.grain.common import preprocessors
import grain.python as grain
import numpy as np

lazy_dataset = grain.experimental.lazy_dataset


class TrimPreprocessorsTest(parameterized.TestCase):

  @parameterized.parameters(1, 4, 7, 9)
  def test_trim_1d(self, length: int):
    input_examples = [{"inputs": list(range(i))} for i in range(10)]
    ds = lazy_dataset.SourceLazyMapDataset(input_examples)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": length}, split="test"
    )
    ds = ds.map(lambda x: preprocessors.trim(x, runtime_args))
    for i, d in enumerate(ds):
      np.testing.assert_array_equal(d["inputs"], list(range(i))[:length])

  @parameterized.parameters(1, 4, 7, 9)
  def test_trim_2d(self, length: int):
    # [[1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3], [3, 3]] ...
    input_examples = [{"inputs": [[i, i]] * i} for i in range(10)]
    ds = lazy_dataset.SourceLazyMapDataset(input_examples)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": length}, split="test"
    )
    ds = ds.map(lambda x: preprocessors.trim(x, runtime_args))
    for i, d in enumerate(ds):
      expected = ([[i, i]] * i)[:length]
      np.testing.assert_array_equal(d["inputs"], expected)

  @parameterized.parameters(1, 4, 7, 9)
  def test_trim_multiple_features(self, length: int):
    input_examples = [
        {"inputs": list(range(i)), "targets": list(range(2 * i))}
        for i in range(10)
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_examples)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": length, "targets": 2 * length}, split="test"
    )
    ds = ds.map(lambda x: preprocessors.trim(x, runtime_args))
    for i, d in enumerate(ds):
      np.testing.assert_array_equal(d["inputs"], list(range(i))[:length])
      np.testing.assert_array_equal(
          d["targets"], list(range(2 * i))[: 2 * length]
      )

  def test_skip_marked_features(self):
    input_examples = [
        {"to_trim": list(range(i)), "not_to_trim": list(range(i))}
        for i in range(10)
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_examples)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"to_trim": 5, "not_to_trim": constants.SKIP_FEATURE},
        split="test",
    )
    ds = ds.map(lambda x: preprocessors.trim(x, runtime_args))
    for i, d in enumerate(ds):
      np.testing.assert_array_equal(d["to_trim"], list(range(i))[:5])
      np.testing.assert_array_equal(d["not_to_trim"], list(range(i)))

  def test_skip_missing_features(self):
    input_examples = [
        {"to_trim": list(range(i)), "not_to_trim": list(range(i))}
        for i in range(10)
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_examples)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"to_trim": 5}, split="test"
    )
    ds = ds.map(lambda x: preprocessors.trim(x, runtime_args))
    for i, d in enumerate(ds):
      np.testing.assert_array_equal(d["to_trim"], list(range(i))[:5])
      np.testing.assert_array_equal(d["not_to_trim"], list(range(i)))

  def test_trim_multirank(self):
    input_examples = [
        {
            "inputs": [[[7, 8, 5, 6, 1]], [[1, 2, 3, 4, 5]]],
            "targets": [[3, 0.5], [9, 3], [1, 2]],
        },
        {
            "inputs": [[[8, 4, 9, 3, 5, 7, 9, 5]]],
            "targets": [[4, 1.2], [1, 1]],
        },
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_examples)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": [1, 1, 4], "targets": [2, 1]}, split="test"
    )
    ds = ds.map(lambda x: preprocessors.trim(x, runtime_args))
    expected = [
        {
            "inputs": [[[7, 8, 5, 6]]],
            "targets": [[3], [9]],
        },
        {
            "inputs": [[[8, 4, 9, 3]]],
            "targets": [[4], [1]],
        },
    ]
    for act, exp in zip(list(ds), expected):
      for k in ["inputs", "targets"]:
        np.testing.assert_array_equal(act[k], exp[k])

  def test_trim_multirank_rank_mismatch(self):
    input_examples = [
        {
            "inputs": [[[7, 8, 5, 6, 1]], [[1, 2, 3, 4, 5]]],
            "targets": [[3, 0.5], [9, 3], [1, 2]],
        },
        {
            "inputs": [[[8, 4, 9, 3, 5, 7, 9, 5]]],
            "targets": [[4, 1.2], [1, 1]],
        },
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_examples)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": [1, 1, 4, 5], "targets": [2, 1]},
        split="test",
    )
    ds = ds.map(lambda x: preprocessors.trim(x, runtime_args))
    with self.assertRaisesRegex(ValueError, "Rank mismatch:.*"):
      _ = list(ds)


class PadPreprocessorsTest(absltest.TestCase):

  def test_pad_1d(self):
    input_examples = [{"inputs": list(range(i))} for i in range(10)]
    ds = lazy_dataset.SourceLazyMapDataset(input_examples)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 10}, split="test"
    )
    ds = ds.map(lambda x: preprocessors.pad(x, runtime_args))
    for i, d in enumerate(ds):
      np.testing.assert_array_equal(
          d["inputs"], list(range(i)) + [0] * (10 - i)
      )

  def test_pad_2d(self):
    # [[1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3], [3, 3]] ...
    input_examples = [{"inputs": [[i, i]] * i} for i in range(1, 10)]
    ds = lazy_dataset.SourceLazyMapDataset(input_examples)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 10}, split="test"
    )
    ds = ds.map(lambda x: preprocessors.pad(x, runtime_args))
    for i, d in zip(range(1, 10), ds):
      expected = [[i, i]] * i + [[0, 0]] * (10 - i)
      np.testing.assert_array_equal(d["inputs"], expected)

  def test_pad_multiple_features(self):
    input_examples = [
        {"inputs": list(range(i)), "targets": list(range(2 * i))}
        for i in range(10)
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_examples)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 10, "targets": 20}, split="test"
    )
    ds = ds.map(lambda x: preprocessors.pad(x, runtime_args))
    for i, d in enumerate(ds):
      np.testing.assert_array_equal(
          d["inputs"], list(range(i)) + [0] * (10 - i)
      )
      np.testing.assert_array_equal(
          d["targets"], list(range(2 * i)) + [0] * (20 - 2 *i)
      )

  def test_feature_too_long(self):
    input_examples = [{"inputs": list(range(i))} for i in range(10)]
    ds = lazy_dataset.SourceLazyMapDataset(input_examples)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 5}, split="test"
    )
    ds = ds.map(lambda x: preprocessors.pad(x, runtime_args))
    with self.assertRaisesRegex(ValueError, "Length of feature 'inputs'.*"):
      _ = list(ds)

  def test_skip_marked_features(self):
    input_examples = [
        {"to_pad": list(range(i)), "not_to_pad": list(range(i))}
        for i in range(10)
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_examples)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"to_pad": 10, "not_to_pad": constants.SKIP_FEATURE},
        split="test",
    )
    ds = ds.map(lambda x: preprocessors.pad(x, runtime_args))
    for i, d in enumerate(ds):
      np.testing.assert_array_equal(
          d["to_pad"], list(range(i)) + [0] * (10 - i)
      )
      np.testing.assert_array_equal(d["not_to_pad"], list(range(i)))

  def test_skip_missing_features(self):
    input_examples = [
        {"to_pad": list(range(i)), "not_to_pad": list(range(i))}
        for i in range(10)
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_examples)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"to_pad": 10}, split="test"
    )
    ds = ds.map(lambda x: preprocessors.pad(x, runtime_args))
    for i, d in enumerate(ds):
      np.testing.assert_array_equal(
          d["to_pad"], list(range(i)) + [0] * (10 - i)
      )
      np.testing.assert_array_equal(d["not_to_pad"], list(range(i)))

  def test_pad_multirank(self):
    input_examples = [
        {
            "inputs": [[[7, 8, 5]], [[1, 2, 3]]],
            "targets": [[3, 0.5], [9, 0], [1, 2]],
        },
        {
            "inputs": [[[8, 4]]],
            "targets": [[4, 1.2], [1, 1]],
        },
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_examples)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": [2, 1, 5], "targets": [3, 3]}, split="test"
    )
    ds = ds.map(lambda x: preprocessors.pad(x, runtime_args))
    expected = [
        {
            "inputs": [[[7, 8, 5, 0, 0]], [[1, 2, 3, 0, 0]]],
            "targets": [[3, 0.5, 0], [9, 0, 0], [1, 2, 0]],
        },
        {
            "inputs": [[[8, 4, 0, 0, 0]], [[0, 0, 0, 0, 0]]],
            "targets": [[4, 1.2, 0], [1, 1, 0], [0, 0, 0]],
        },
    ]
    for act, exp in zip(list(ds), expected):
      for k in ["inputs", "targets"]:
        np.testing.assert_array_equal(act[k], exp[k])

  def test_pad_multirank_feature_too_long(self):
    input_examples = [
        {
            "inputs": [[[7, 8, 5]], [[1, 2, 3]]],
            "targets": [[3, 0.5], [9, 0], [1, 2]],
        },
        {
            "inputs": [[[8, 4, 5, 3, 1, 6]]],
            "targets": [[4, 1.2], [1, 1]],
        },
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_examples)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": [2, 1, 5], "targets": [3, 3]}, split="test"
    )
    ds = ds.map(lambda x: preprocessors.pad(x, runtime_args))
    with self.assertRaisesRegex(ValueError, "Shape of feature 'inputs'.*"):
      _ = list(ds)

  def test_pad_multirank_rank_mismatch(self):
    input_examples = [
        {
            "inputs": [[[7, 8, 5, 6, 1]], [[1, 2, 3, 4, 5]]],
            "targets": [[3, 0.5], [9, 3], [1, 2]],
        },
        {
            "inputs": [[[8, 4, 9, 3, 5, 7, 9, 5]]],
            "targets": [[4, 1.2], [1, 1]],
        },
    ]
    ds = lazy_dataset.SourceLazyMapDataset(input_examples)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": [1, 1, 4, 5], "targets": [2, 1]},
        split="test",
    )
    ds = ds.map(lambda x: preprocessors.pad(x, runtime_args))
    with self.assertRaisesRegex(ValueError, "Rank mismatch:.*"):
      _ = list(ds)


if __name__ == "__main__":
  absltest.main()
