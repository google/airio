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

"""AirIO test utilities."""

from typing import Any, Mapping, Sequence

from absl.testing import absltest
import numpy as np


def assert_datasets_equal(
    dataset: np.ndarray,
    expected: Sequence[Mapping[str, Any]],
):
  """Tests whether the entire dataset == expected or [expected].

  Args:
    dataset: a numpy-compatible dataset (e.g. PyGrainDatasetIterator).
    expected: either a single example, or a list of examples. Each example is a
      dictionary.
  """

  if not isinstance(expected, list):
    expected = [expected]
  actual = list(dataset)
  absltest.TestCase().assertEqual(len(actual), len(expected))

  def _compare_dict(actual_dict, expected_dict):
    absltest.TestCase().assertEqual(
        set(actual_dict.keys()), set(expected_dict.keys())
    )
    for key, actual_value in actual_dict.items():
      if isinstance(actual_value, dict):
        _compare_dict(actual_value, expected_dict[key])
      elif (
          isinstance(actual_value, np.floating)
          or isinstance(actual_value, np.ndarray)
          and np.issubdtype(actual_value.dtype, np.floating)
      ):
        np.testing.assert_allclose(actual_value, expected_dict[key])
      else:
        np.testing.assert_array_equal(actual_value, expected_dict[key], key)

  for actual_ex, expected_ex in zip(actual, expected):
    _compare_dict(actual_ex, expected_ex)
