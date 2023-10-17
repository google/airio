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

"""PyGrain LazyDataset transformations required for AirIO Task and Mixture.

Most, if not all, of these should be upstreamed into the grain codebase.
"""
import bisect
import dataclasses
import itertools
from typing import Any, Callable, Sequence, TypeVar
from absl import logging
import grain.python as grain
import jax
import numpy as np

lazy_dataset = grain.experimental.lazy_dataset
JaxRng = jax.Array
T = TypeVar("T")


# TODO(b/300282178): This is from grain/_src/core/transforms. Merge back there.
def even_split(
    num_examples: int, options: grain.ShardOptions
) -> tuple[int, int]:
  """Returns the interval for the shard when sharding `num_examples` evenly.

  This splits the interval [0, num_examples - 1] into `shard_count` intervals
  and returns the `shard_index`'s interval. If `drop_remainder` is True all
  intervals will have the same size.

  Args:
    num_examples: Number of examples to shard.
    options: Options for sharding the data in this process.

  Returns:
    Tuple with the start and end of the interval. The start is the first
    example that should be included in this interval and end - 1 is the last
    example to be include in the shard.
  """
  examples_per_shard = num_examples // options.shard_count
  shard_start = examples_per_shard * options.shard_index
  shard_end = examples_per_shard * (options.shard_index + 1)

  # Handle remaining examples.
  num_unused_examples = num_examples % options.shard_count

  if num_unused_examples > 0:
    if options.drop_remainder:
      logging.warning(
          "Dropping %d examples of %d examples (shard %d).",
          num_unused_examples,
          num_examples,
          options.shard_count,
      )
    else:
      shard_start += min(options.shard_index, num_unused_examples)
      shard_end += min(options.shard_index + 1, num_unused_examples)
  return shard_start, shard_end


# TODO(b/300282178): These should be upstreamed to the grain codebase.
@dataclasses.dataclass(frozen=False)
class ShardLazyMapDataset(lazy_dataset.LazyMapDataset[T]):
  """Shards a LazyMapDataset based on provided shard options."""

  parent: lazy_dataset.LazyMapDataset[T]
  shard_options: grain.ShardOptions

  def __post_init__(self):
    self._start, self._end = even_split(len(self.parent), self.shard_options)

  def __len__(self) -> int:
    return self._end - self._start  # pytype: disable=unsupported-operands

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    epoch = index // len(self)
    index_in_epoch = index % len(self)
    index = epoch * len(self.parent) + index_in_epoch + self._start  # pytype: disable=unsupported-operands
    return self.parent[index]


@dataclasses.dataclass(frozen=False)
class ConcatLazyMapDataset(lazy_dataset.LazyMapDataset[T]):
  """Concats LazyMapDatasets."""

  parents: Sequence[lazy_dataset.LazyMapDataset[T]]

  def __post_init__(self):
    self._accumulated_lens = [0] + list(
        itertools.accumulate([len(p) for p in self.parents])
    )

  def __len__(self) -> int:
    return sum(len(p) for p in self.parents)

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    parent_index = bisect.bisect_right(self._accumulated_lens, index) - 1  # pytype: disable=wrong-arg-types
    local_index = index - self._accumulated_lens[parent_index]  # pytype: disable=unsupported-operands
    return self.parents[parent_index][local_index]


class MixedLazyMapDataset(lazy_dataset.MixedLazyMapDataset[T]):
  """LazyDataset for mixtures with a few added features."""

  def __init__(
      self,
      parents: Sequence[lazy_dataset.LazyMapDataset[T]],
      proportions: Sequence[float | int] | None = None,
      stop_on_empty_dataset: bool = True,
  ):
    super().__init__(parents, proportions)
    if stop_on_empty_dataset:
      lengths = np.asarray([len(p) for p in parents])
      float_proportions = np.asarray(proportions) / sum(proportions)
      # Stop sampling once any constituent is exhausted.
      self._length = int((lengths / float_proportions).min())


@dataclasses.dataclass(frozen=False)
class RandomMapFnLazyMapDataset(lazy_dataset.LazyMapDataset[T]):
  """LazyMapDataset for random map fns with jax PRNGKey support."""

  parent: lazy_dataset.LazyMapDataset
  map_fn: Callable[[Any, JaxRng], Any]
  base_rng: JaxRng

  def __len__(self) -> int:
    return len(self.parent)

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    element = self.parent[index]
    if element is None:
      return None
    rng = jax.random.fold_in(self.base_rng, index)
    return self.map_fn(element, rng)
