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

"""Preprocessors tests."""

from absl.testing import absltest
from airio import data_sources
from airio import dataset_providers
from airio import preprocessors
import numpy as np


class PreprocessorsTest(absltest.TestCase):

  def _get_test_src(self, num_elements=5):
    def _dataset_fn(split: str):
      del split
      return np.array(range(num_elements))

    return data_sources.FunctionDataSource(
        dataset_fn=_dataset_fn, splits=["train"]
    )

  def test_map_fn_preprocessor(self):
    def test_map_fn(ex):
      return ex + 1

    task = dataset_providers.Task(
        name="test_task",
        source=self._get_test_src(),
        preprocessors=[preprocessors.MapFnTransform(test_map_fn)],
    )
    ds = task.get_dataset(None, "train", shuffle=False)
    self.assertListEqual(list(ds), list(range(1, 6)))

  def test_random_map_fn_preprocessor(self):
    def test_random_map_fn(ex, rng):
      return ex + rng.integers(0, 10)

    task = dataset_providers.Task(
        name="test_task",
        source=self._get_test_src(),
        preprocessors=[preprocessors.RandomMapFnTransform(test_random_map_fn)],
    )
    ds = task.get_dataset(None, "train", shuffle=False, seed=42)
    self.assertListEqual(list(ds), [3, 4, 9, 11, 11])

  def test_filter_fn_preprocessor(self):
    def test_filter_fn(ex):
      return ex > 2

    task = dataset_providers.Task(
        name="test_task",
        source=self._get_test_src(),
        preprocessors=[preprocessors.FilterFnTransform(test_filter_fn)],
    )
    ds = task.get_dataset(None, "train", shuffle=False, seed=42)
    self.assertListEqual(list(ds), [3, 4])

  def test_preprocessor_empty_source(self):
    def test_filter_fn(ex):
      return ex > 2

    def test_map_fn(ex):
      return ex + 1

    def test_random_map_fn(ex, rng):
      return ex + rng.integers(0, 10)

    task = dataset_providers.Task(
        name="test_task",
        source=self._get_test_src(num_elements=0),
        preprocessors=[
            preprocessors.FilterFnTransform(test_filter_fn),
            preprocessors.MapFnTransform(test_map_fn),
            preprocessors.RandomMapFnTransform(test_random_map_fn),
        ],
    )
    with self.assertRaisesRegex(ValueError, "Invalid number of records.*"):
      _ = task.get_dataset(None, "train", shuffle=False)

  def test_preprocessor_empty_preprocessed(self):
    def test_filter_fn(ex):
      return ex > 1000

    task = dataset_providers.Task(
        name="test_task",
        source=self._get_test_src(),
        preprocessors=[preprocessors.FilterFnTransform(test_filter_fn)],
    )
    ds = task.get_dataset(None, "train", shuffle=False, seed=42)
    self.assertListEqual(list(ds), [])

  def test_preprocessor_empty_intermediates(self):
    def test_map_fn(ex):
      return ex + 1

    def test_filter_fn(ex):
      return ex > 1000

    task = dataset_providers.Task(
        name="test_task",
        source=self._get_test_src(),
        preprocessors=[
            preprocessors.FilterFnTransform(test_filter_fn),
            preprocessors.MapFnTransform(test_map_fn),
        ],
    )
    ds = task.get_dataset(None, "train", shuffle=False, seed=42)
    self.assertListEqual(list(ds), [])


if __name__ == "__main__":
  absltest.main()
