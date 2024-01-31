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

"""Tests for asserts."""

from absl.testing import absltest
from airio._src.core import test_utils
import numpy as np

_TEXT_DATASET = [{'key1': ['val1'], 'key2': ['val2']}]
_INHOMOGENEOUS_SHAPE_DATASET = [
    {'key': np.array([[1.0, 2.0], [3.0]], dtype=object)}
]
_INT_DATASET = [{'key1': 1, 'key2': 2}]
_FLOAT_DATASET = [
    {'float_key1': np.array(0.2), 'float_key2': np.array(0.4)},
    {'float_key1': np.array(0.3), 'float_key2': np.array(0.5)},
]
_NESTED_DICT_DATASET = [{'key1': {'key2': 2, 'key3': 3}, 'key4': 4}]


class TestUtilsTest(absltest.TestCase):

  def test_assert_text_datasets_equal(self):
    test_utils.assert_datasets_equal(
        _TEXT_DATASET, {'key1': ['val1'], 'key2': ['val2']}
    )

  def test_assert_text_datasets_unequal_value(self):
    with self.assertRaises(AssertionError):
      test_utils.assert_datasets_equal(
          _TEXT_DATASET, {'key1': ['val1'], 'key2': ['val2x']}
      )

  def test_assert_text_datasets_additional_key(self):
    with self.assertRaises(AssertionError):
      test_utils.assert_datasets_equal(
          _TEXT_DATASET,
          {'key1': ['val1'], 'key2': ['val2'], 'key3': ['val3']},
      )

  def test_assert_inhomogeneous_shape_datasets_equal(self):
    test_utils.assert_datasets_equal(
        _INHOMOGENEOUS_SHAPE_DATASET,
        {'key': np.array([[1.0, 2.0], [3.0]], dtype=object)},
    )

  def test_assert_inhomogeneous_shape_datasets_not_equal(self):
    with self.assertRaises(AssertionError):
      test_utils.assert_datasets_equal(
          _INHOMOGENEOUS_SHAPE_DATASET, {'key': [[1.0], [3.0]]}
      )

  def test_assert_int_datasets_equal(self):
    test_utils.assert_datasets_equal(
        _INT_DATASET,
        [{'key1': 1, 'key2': 2}],
    )

  def test_assert_nested_dict_datasets_equal(self):
    test_utils.assert_datasets_equal(
        _NESTED_DICT_DATASET,
        [{'key1': {'key2': 2, 'key3': 3}, 'key4': 4}],
    )

  def test_assert_float_datasets_equal(self):
    test_utils.assert_datasets_equal(
        _FLOAT_DATASET,
        [
            {'float_key1': 0.2, 'float_key2': 0.4},
            {'float_key1': 0.3, 'float_key2': 0.5},
        ],
    )

  def test_assert_float_datasets_approx_equal(self):
    test_utils.assert_datasets_equal(
        _FLOAT_DATASET,
        [
            {'float_key1': 0.20000001, 'float_key2': 0.39999999},
            {'float_key1': 0.30000001, 'float_key2': 0.49999999},
        ],
    )

  def test_assert_float_datasets_not_equal(self):
    with self.assertRaises(AssertionError):
      test_utils.assert_datasets_equal(
          _FLOAT_DATASET,
          [
              {'float_key1': 0.201, 'float_key2': 0.399},
              {'float_key1': 0.301, 'float_key2': 0.499},
          ],
      )


if __name__ == '__main__':
  absltest.main()
