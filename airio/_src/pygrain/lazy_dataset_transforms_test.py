# Copyright 2025 The AirIO Authors.
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

"""Tests for lazy_dataset transforms."""

from typing import Sequence
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from airio._src.pygrain import lazy_dataset_transforms
import grain.python as grain
import jax.random


jax.config.update("jax_threefry_partitionable", False)


class ConcatLazyMapDatasetTest(parameterized.TestCase):

  @parameterized.parameters(
      ([3, 2],),
      ([1, 2, 3],),
      ([4, 5, 4, 1],),
  )
  def test_len(self, parent_lens: Sequence[int]):
    datasets = []
    for data_len in parent_lens:
      datasets.append(grain.MapDataset.range(data_len))

    ds = lazy_dataset_transforms.ConcatLazyMapDataset(datasets)
    self.assertLen(ds, sum(parent_lens))

  def test_getitem(self):
    parents = [
        grain.MapDataset.range(5),
        grain.MapDataset.range(5, 9),
        grain.MapDataset.range(9, 18),
        grain.MapDataset.range(18, 19),
        grain.MapDataset.range(19, 22),
    ]
    ds = lazy_dataset_transforms.ConcatLazyMapDataset(parents)
    actual = [ds[i] for i in range(len(ds))]
    self.assertSequenceEqual(actual, range(22))

  def test_iter(self):
    parents = [
        grain.MapDataset.range(5),
        grain.MapDataset.range(5, 9),
        grain.MapDataset.range(9, 18),
        grain.MapDataset.range(18, 19),
        grain.MapDataset.range(19, 22),
    ]
    ds = lazy_dataset_transforms.ConcatLazyMapDataset(parents)
    ds_iter = iter(ds)
    actual = list(ds_iter)
    self.assertSequenceEqual(actual, range(22))



class RandomMapFnLazyMapDatasetTest(absltest.TestCase):

  def test_iter_reproducible(self):
    def random_map_fn(ex, rng):
      return ex + int(jax.random.randint(rng, [], 0, 10))

    ds = grain.MapDataset.range(5)

    for _ in range(5):
      ds1 = lazy_dataset_transforms.RandomMapFnLazyMapDataset(
          ds, random_map_fn, jax.random.key(42)
      )
      self.assertListEqual(list(ds1), [5, 4, 5, 12, 13])

  def test_get_item_reproducible(self):
    def random_map_fn(ex, rng):
      return ex + int(jax.random.randint(rng, [], 0, 10))

    ds = grain.MapDataset.range(5)

    for _ in range(5):
      ds1 = lazy_dataset_transforms.RandomMapFnLazyMapDataset(
          ds, random_map_fn, jax.random.key(42)
      )
      self.assertListEqual([ds1[i] for i in range(len(ds))], [5, 4, 5, 12, 13])

  def test_len(self):
    def random_map_fn(ex, rng):
      return ex + int(jax.random.randint(rng, [], 0, 10))

    ds = grain.MapDataset.range(5)
    ds = lazy_dataset_transforms.RandomMapFnLazyMapDataset(
        ds, random_map_fn, jax.random.key(42)
    )
    self.assertLen(ds, 5)

  def test_parents_property(self):
    def random_map_fn(ex, rng):
      return ex + int(jax.random.randint(rng, [], 0, 10))

    parent_ds = grain.MapDataset.range(5)
    ds = lazy_dataset_transforms.RandomMapFnLazyMapDataset(
        parent_ds, random_map_fn, jax.random.key(42)
    )
    self.assertTupleEqual(ds.parents, (parent_ds,))



if __name__ == "__main__":
  absltest.main()
