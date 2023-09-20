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

"""AirIO preprocessor classes."""

import dataclasses
import time
from typing import Any, Callable, Union

from airio import lazy_dataset_transforms
import grain.python as grain
import jax
import numpy as np

# TODO(b/294122943): Add support for injecting runtime args, e.g. seq lens.
# TODO(b/294122943): Implement flat_map.

lazy_dataset = grain.experimental.lazy_dataset
JaxRng = Union[jax.random.PRNGKeyArray, jax.Array]


@dataclasses.dataclass
class MapFnTransform(grain.MapTransform):
  """Grain Transform to represent AirIO map preprocessors."""

  map_fn: Callable[[Any], Any]

  def map(self, element):
    """Maps a single element."""
    return self.map_fn(element)


@dataclasses.dataclass
class RandomMapFnTransform(grain.RandomMapTransform):
  """Grain Transform to represent AirIO random map preprocessors."""

  map_fn: Callable[[Any, JaxRng], Any]

  def random_map(self, element, rng: np.random.Generator):
    """Maps a single element."""
    jax_rng = jax.random.PRNGKey(rng.integers(0, 2**16 - 1))
    return self.map_fn(element, jax_rng)


@dataclasses.dataclass
class FilterFnTransform(grain.FilterTransform):
  """Grain Transform to represent AirIO filter preprocessors."""

  filter_fn: Callable[..., Any]

  def filter(self, element) -> bool:
    """Filters a single element."""
    return self.filter_fn(element)


@dataclasses.dataclass
class LazyDatasetTransform:
  """A convenience function to map Transforms to LazyDatasets."""
  transform: grain.Transformation | grain.Operation

  def __post_init__(self):
    # TODO(b/300282178): Support flat-maps and many-to-one/many transforms.
    if not isinstance(self.transform, grain.Transformation):
      raise ValueError(f"{str(self.transform)} is not supported")
    # TODO(b/300938204): Remove error for other RandomMapTransforms, once
    # these can be reproducibly processed.
    if isinstance(self.transform, grain.RandomMapTransform) and not isinstance(
        self.transform, RandomMapFnTransform
    ):
      raise ValueError(
          f"{str(self.transform)} is not safe. Use"
          " airio.preprocessors.RandomMapFnTransform instead."
      )

  def __call__(
      self, ds: lazy_dataset.LazyMapDataset, rng: JaxRng | None = None
  ):
    # pytype: disable=attribute-error
    match self.transform:
      case grain.MapTransform():
        return lazy_dataset.MapLazyMapDataset(ds, self.transform)
      case RandomMapFnTransform():
        # Special case to support reproducible stochastic transformations with
        # jax PRNGKeys.
        if rng is None:
          rng = jax.random.PRNGKey(np.int32(time.time()))
        return lazy_dataset_transforms.RandomMapFnLazyMapDataset(
            ds, map_fn=self.transform.map_fn, base_rng=rng
        )
      case grain.FilterTransform():
        return lazy_dataset.FilterLazyMapDataset(ds, self.transform)
      case grain.Batch():
        return lazy_dataset.BatchLazyMapDataset(
            ds,
            batch_size=self.transform.batch_size,
            drop_remainder=self.transform.drop_remainder,
        )
      case _:
        # Should be taken care of by post init validation.
        raise ValueError("%s is not supported" % str(self.transform))
    # pytype: enable=attribute-error
