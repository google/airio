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

"""Microbenchmarks for AirIO lazy_dataset_transforms functions."""

from airio import lazy_dataset_transforms
import google_benchmark
import grain.python as grain
import jax.random

lazy_dataset = grain.experimental.lazy_dataset


@google_benchmark.register
def length(state):
  parent_lens = [4, 5, 4, 1]
  datasets = []
  for data_len in parent_lens:
    datasets.append(lazy_dataset.RangeLazyMapDataset(data_len))
  ds = lazy_dataset_transforms.ConcatLazyMapDataset(datasets)
  while state:
    _ = len(ds)


@google_benchmark.register
def get_item(state):
  parents = [
      lazy_dataset.RangeLazyMapDataset(5),
      lazy_dataset.RangeLazyMapDataset(5, 9),
      lazy_dataset.RangeLazyMapDataset(9, 18),
      lazy_dataset.RangeLazyMapDataset(18, 19),
      lazy_dataset.RangeLazyMapDataset(19, 22),
  ]
  ds = lazy_dataset_transforms.ConcatLazyMapDataset(parents)
  while state:
    _ = [ds[i] for i in range(len(ds))]


@google_benchmark.register
def iter_items(state):
  """Analogous to the ConcatLazyMapDatasetTest with the same name."""
  parents = [
      lazy_dataset.RangeLazyMapDataset(5),
      lazy_dataset.RangeLazyMapDataset(5, 9),
      lazy_dataset.RangeLazyMapDataset(9, 18),
      lazy_dataset.RangeLazyMapDataset(18, 19),
      lazy_dataset.RangeLazyMapDataset(19, 22),
  ]
  ds = lazy_dataset_transforms.ConcatLazyMapDataset(parents)
  ds_iter = iter(ds)
  while state:
    _ = list(ds_iter)


@google_benchmark.register
def length_reproducible(state):
  """Analogous to the RandomMapFnLazyMapDatasetTest with the same name."""

  def random_map_fn(ex, rng):
    return ex + int(jax.random.randint(rng, [], 0, 10))

  ds = lazy_dataset.RangeLazyMapDataset(5)
  ds = lazy_dataset_transforms.RandomMapFnLazyMapDataset(
      ds, random_map_fn, jax.random.PRNGKey(42)
  )
  while state:
    _ = len(ds)


@google_benchmark.register
def get_item_reproducible(state):
  """Analogous to the RandomMapFnLazyMapDatasetTest with the same name."""

  def random_map_fn(ex, rng):
    return ex + int(jax.random.randint(rng, [], 0, 10))

  ds = lazy_dataset.RangeLazyMapDataset(5)
  while state:
    for _ in range(5):
      ds1 = lazy_dataset_transforms.RandomMapFnLazyMapDataset(
          ds, random_map_fn, jax.random.PRNGKey(42)
      )
      _ = [ds1[i] for i in range(len(ds))]


@google_benchmark.register
def iter_items_reproducible(state):
  """Analogous to the RandomMapFnLazyMapDatasetTest with the same name."""

  def random_map_fn(ex, rng):
    return ex + int(jax.random.randint(rng, [], 0, 10))

  ds = lazy_dataset.RangeLazyMapDataset(5)
  while state:
    for _ in range(5):
      ds1 = lazy_dataset_transforms.RandomMapFnLazyMapDataset(
          ds, random_map_fn, jax.random.PRNGKey(42)
      )
      _ = list(ds1)


if __name__ == "__main__":
  google_benchmark.main()
