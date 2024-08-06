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

import airio.pygrain as airio
import google_benchmark
import grain.python as grain
import jax.random


@google_benchmark.register
def length(state: google_benchmark.State) -> None:
  parent_lens = [4, 5, 4, 1]
  datasets = []
  for data_len in parent_lens:
    datasets.append(grain.MapDataset.range(data_len))
  ds = airio.lazy_dataset_transforms.ConcatLazyMapDataset(datasets)
  while state:
    len(ds)


@google_benchmark.register
def get_item(state: google_benchmark.State) -> None:
  parents = [
      grain.MapDataset.range(5),
      grain.MapDataset.range(5, 9),
      grain.MapDataset.range(9, 18),
      grain.MapDataset.range(18, 19),
      grain.MapDataset.range(19, 22),
  ]
  ds = airio.lazy_dataset_transforms.ConcatLazyMapDataset(parents)
  while state:
    _ = [ds[i] for i in range(len(ds))]


@google_benchmark.register
def iter_items(state: google_benchmark.State) -> None:
  """Analogous to the ConcatLazyMapDatasetTest with the same name."""
  parents = [
      grain.MapDataset.range(5),
      grain.MapDataset.range(5, 9),
      grain.MapDataset.range(9, 18),
      grain.MapDataset.range(18, 19),
      grain.MapDataset.range(19, 22),
  ]
  ds = airio.lazy_dataset_transforms.ConcatLazyMapDataset(parents)
  ds_iter = iter(ds)
  while state:
    list(ds_iter)


@google_benchmark.register
def length_reproducible(state: google_benchmark.State) -> None:
  """Analogous to the RandomMapFnLazyMapDatasetTest with the same name."""

  def random_map_fn(ex, rng):
    return ex + int(jax.random.randint(rng, [], 0, 10))

  ds = grain.MapDataset.range(5)
  ds = airio.lazy_dataset_transforms.RandomMapFnLazyMapDataset(
      ds, random_map_fn, jax.random.PRNGKey(42)
  )
  while state:
    len(ds)


@google_benchmark.register
def get_item_reproducible(state: google_benchmark.State) -> None:
  """Analogous to the RandomMapFnLazyMapDatasetTest with the same name."""

  def random_map_fn(ex, rng):
    return ex + int(jax.random.randint(rng, [], 0, 10))

  ds = grain.MapDataset.range(5)
  while state:
    for _ in range(5):
      ds1 = airio.lazy_dataset_transforms.RandomMapFnLazyMapDataset(
          ds, random_map_fn, jax.random.PRNGKey(42)
      )
      _ = [ds1[i] for i in range(len(ds))]


@google_benchmark.register
def iter_items_reproducible(state: google_benchmark.State) -> None:
  """Analogous to the RandomMapFnLazyMapDatasetTest with the same name."""

  def random_map_fn(ex, rng):
    return ex + int(jax.random.randint(rng, [], 0, 10))

  ds = grain.MapDataset.range(5)
  while state:
    for _ in range(5):
      ds1 = airio.lazy_dataset_transforms.RandomMapFnLazyMapDataset(
          ds, random_map_fn, jax.random.PRNGKey(42)
      )
      list(ds1)


if __name__ == "__main__":
  google_benchmark.main()
