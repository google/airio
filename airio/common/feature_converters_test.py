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

"""Tests for feature_converters."""

from absl.testing import absltest
from airio.common import feature_converters
import numpy as np


class FeatureConvertersTest(absltest.TestCase):

  def test_autoregressive_inputs_unpacked(self):
    x = np.asarray([3, 8, 9, 5, 1, 0, 0])
    actual = feature_converters.make_autoregressive_inputs(
        x, sequence_ids=None, bos_id=0
    )
    expected = [0, 3, 8, 9, 5, 1, 0]
    np.testing.assert_array_equal(actual, expected)
    self.assertEqual(actual.dtype, np.int64)

  def test_autoregressive_inputs_unpacked_with_bos_id(self):
    x = np.asarray([3, 8, 9, 5, 1, 0, 0])
    actual = feature_converters.make_autoregressive_inputs(
        x, sequence_ids=None, bos_id=1
    )
    expected = [1, 3, 8, 9, 5, 1, 0]
    np.testing.assert_array_equal(actual, expected)
    self.assertEqual(actual.dtype, np.int64)

  def test_autoregressive_inputs_unpacked_2d(self):
    x = np.asarray([[3, 8, 1, 0, 0], [9, 5, 2, 0, 6]])
    actual = feature_converters.make_autoregressive_inputs(
        x, sequence_ids=None, bos_id=0
    )
    expected = [[0, 0, 0, 0, 0], [3, 8, 1, 0, 0]]
    np.testing.assert_array_equal(actual, expected)
    self.assertEqual(actual.dtype, np.int64)

  def test_autoregressive_inputs_packed(self):
    x = np.asarray([3, 8, 1, 9, 1, 5, 4, 1, 0, 0])
    sequence_ids = np.asarray([1, 1, 1, 2, 2, 3, 3, 3, 0, 0])
    actual = feature_converters.make_autoregressive_inputs(
        x,
        sequence_ids=sequence_ids,
        bos_id=0,
    )
    expected = [0, 3, 8, 0, 9, 0, 5, 4, 0, 0]
    np.testing.assert_array_equal(actual, expected)
    self.assertEqual(actual.dtype, np.int64)

  def test_autoregressive_inputs_packed_with_bos_id(self):
    x = np.asarray([3, 8, 1, 9, 1, 5, 4, 1, 0, 0])
    sequence_ids = np.asarray([1, 1, 1, 2, 2, 3, 3, 3, 0, 0])
    actual = feature_converters.make_autoregressive_inputs(
        x,
        sequence_ids=sequence_ids,
        bos_id=1,
    )
    expected = [1, 3, 8, 1, 9, 1, 5, 4, 1, 0]
    np.testing.assert_array_equal(actual, expected)
    self.assertEqual(actual.dtype, np.int64)

  def test_autoregressive_inputs_packed_2d(self):
    x = np.asarray([[3, 8, 1, 0, 0], [9, 5, 2, 0, 6]])
    sequence_ids = np.asarray([1, 2])
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        (
            "Only 1-D sequences are supported with packing. "
            "Got a packed 2-D sequence."
        ),
    ):
      feature_converters.make_autoregressive_inputs(
          x, sequence_ids=sequence_ids, bos_id=0
      )

  def test_autoregressive_inputs_packed_non_eos(self):
    # In the correct input format, x[4] should have been 1 (EOS).
    x = np.asarray([3, 8, 1, 9, 6, 5, 4, 1, 0, 0])
    # sequence_id is correctly formatted.
    sequence_ids = np.asarray([1, 1, 1, 2, 2, 3, 3, 3, 0, 0])
    actual = feature_converters.make_autoregressive_inputs(
        x, sequence_ids=sequence_ids, bos_id=0,
    )
    # The incorrect x[4] should not affect the output as long as the sequence_id
    # is correct.
    expected = [0, 3, 8, 0, 9, 0, 5, 4, 0, 0]
    np.testing.assert_array_equal(actual, expected)
    self.assertEqual(actual.dtype, np.int64)

  def test_autoregressive_inputs_different_dtypes(self):
    x = np.asarray([3, 8, 1, 9.9, 1, 5, 4, 1, 0, 0], dtype=np.float32)
    sequence_ids = np.asarray([1, 1, 1, 2, 2, 3, 3, 3, 0, 0])
    actual = feature_converters.make_autoregressive_inputs(
        x, sequence_ids=sequence_ids, bos_id=0,
    )
    # The incorrect x[4] should not affect the output as long as the sequence_id
    # is correct.
    expected = [0, 3, 8, 0, 9.9, 0, 5, 4, 0, 0]
    np.testing.assert_array_almost_equal(actual, expected)
    self.assertEqual(actual.dtype, np.float32)


if __name__ == "__main__":
  absltest.main()
