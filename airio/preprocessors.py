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
from typing import Any, Callable

import grain.python as grain
import numpy as np

# TODO(b/294122943): Add support for injecting runtime args, e.g. seq lens.
# TODO(b/294122943): Implement flat_map.


@dataclasses.dataclass
class MapFnTransform(grain.MapTransform):
  """Grain Transform to represent AirIO map preprocessors."""

  map_fn: Callable[..., Any]

  def map(self, element):
    """Maps a single element."""
    return self.map_fn(element)


@dataclasses.dataclass
class RandomMapFnTransform(grain.RandomMapTransform):
  """Grain Transform to represent AirIO random map preprocessors."""

  map_fn: Callable[..., Any]

  def random_map(self, element, rng: np.random.Generator):
    """Maps a single element."""
    return self.map_fn(element, rng)


@dataclasses.dataclass
class FilterFnTransform(grain.FilterTransform):
  """Grain Transform to represent AirIO filter preprocessors."""

  filter_fn: Callable[..., Any]

  def filter(self, element) -> bool:
    """Filters a single element."""
    return self.filter_fn(element)
