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

"""PyGrain LazyDataset transformations required for AirIO Task and Mixture.

Most, if not all, of these should be upstreamed into the grain codebase.
"""

import bisect
import dataclasses
import itertools
from typing import Any, Callable, Sequence, TypeVar

import grain.python as grain
import jax


lazy_dataset = grain.experimental.lazy_dataset
JaxRng = jax.Array
T = TypeVar("T")


# TODO(b/300282178): These may be upstreamed to the grain codebase.
@dataclasses.dataclass(frozen=False)
class ConcatLazyMapDataset(lazy_dataset.LazyMapDataset[T]):
  """Concats LazyMapDatasets."""

  def __init__(
      self,
      parents: Sequence[lazy_dataset.LazyMapDataset[T]],
  ):
    super().__init__(parents)
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


@dataclasses.dataclass(frozen=False)
class RandomMapFnLazyMapDataset(lazy_dataset.LazyMapDataset[T]):
  """LazyMapDataset for random map fns with jax rng key support."""

  parent: lazy_dataset.LazyMapDataset
  map_fn: Callable[[Any, JaxRng], Any]
  base_rng: JaxRng

  def __init__(
      self,
      parent: lazy_dataset.LazyMapDataset,
      map_fn: Callable[[Any, JaxRng], Any],
      base_rng: JaxRng,
  ):
    super().__init__([parent])
    self.parent = parent
    self.map_fn = map_fn
    self.base_rng = base_rng

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
