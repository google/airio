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
import functools
import inspect
import time
from typing import Any, Callable, Mapping, Union

from airio import lazy_dataset_transforms
import grain.python as grain
import jax
import numpy as np

# TODO(b/294122943): Implement flat_map.

lazy_dataset = grain.experimental.lazy_dataset
JaxRng = jax.Array


@dataclasses.dataclass
class AirIOInjectedRuntimeArgs:
  """A set of attributes that can be injected into preprocessors at runtime."""
  sequence_lengths: Mapping[str, int]
  split: str


MapFnCallable = Union[
    Callable[[Any], Any], Callable[[Any, AirIOInjectedRuntimeArgs], Any]
]
RandomMapFnCallable = Union[
    Callable[[Any, JaxRng], Any],
    Callable[[Any, JaxRng, AirIOInjectedRuntimeArgs], Any],
]
FilterFnCallable = Union[
    Callable[[Any], bool], Callable[[Any, AirIOInjectedRuntimeArgs], bool]
]


@dataclasses.dataclass
class MapFnTransform(grain.MapTransform):
  """Grain Transform to represent AirIO map preprocessors."""

  map_fn: MapFnCallable
  runtime_args: AirIOInjectedRuntimeArgs | None = None

  def map(self, element):
    """Maps a single element."""
    return inject_runtime_args_to_fn(self.map_fn, self.runtime_args)(element)


@dataclasses.dataclass
class RandomMapFnTransform(grain.RandomMapTransform):
  """Grain Transform to represent AirIO random map preprocessors."""

  map_fn: RandomMapFnCallable
  runtime_args: AirIOInjectedRuntimeArgs | None = None

  def random_map(self, element, rng: np.random.Generator):
    """Maps a single element."""
    jax_rng = jax.random.PRNGKey(rng.integers(0, 2**16 - 1))
    return inject_runtime_args_to_fn(self.map_fn, self.runtime_args)(
        element, jax_rng
    )


@dataclasses.dataclass
class FilterFnTransform(grain.FilterTransform):
  """Grain Transform to represent AirIO filter preprocessors."""

  filter_fn: FilterFnCallable
  runtime_args: AirIOInjectedRuntimeArgs | None = None

  def filter(self, element) -> bool:
    """Filters a single element."""
    return inject_runtime_args_to_fn(self.filter_fn, self.runtime_args)(element)


FnTransforms = Union[MapFnTransform, RandomMapFnTransform, FilterFnTransform]


@dataclasses.dataclass
class PackTransform:
  """Represents configuration for a packing preprocessor.

  Attributes:
    packing_preprocessor: The packing preprocessor. This is a `Callable` that
      packs a `lazy_dataset.LazyMapDataset` based on sequence_lengths provided
      via `AirIOInjectedRuntimeArgs`, and can be extended to other types, such
      as a set of `grain.Transformation`s. A standard implementation is
      available in airio/common.
  """

  packing_preprocessor: Callable[
      [lazy_dataset.LazyMapDataset, AirIOInjectedRuntimeArgs],
      lazy_dataset.LazyMapDataset,
  ]

  def __call__(
      self,
      ds: lazy_dataset.LazyMapDataset,
      runtime_args: AirIOInjectedRuntimeArgs,
  ):
    packer = self.packing_preprocessor
    if not isinstance(ds, lazy_dataset.LazyMapDataset):
      raise ValueError(
          f"Cannot apply LazyMapDataset packing: {str(packer)} to"
          f" non-LazyMapDataset dataset: {str(ds)}"
      )
    return packer(ds, runtime_args)


# This may be extended as needed.
AirIOPreprocessor = grain.Transformation | PackTransform


@dataclasses.dataclass
class LazyDatasetTransform:
  """A convenience function to map Transforms to LazyDatasets."""
  transform: AirIOPreprocessor

  def __post_init__(self):
    # TODO(b/300282178): Support flat-maps and many-to-one/many transforms.
    if not isinstance(self.transform, AirIOPreprocessor):
      raise ValueError(f"{str(self.transform)} is not supported")
    # TODO(b/300938204): Remove error for other RandomMapTransforms, once
    # these can be reproducibly processed.
    if isinstance(self.transform, grain.RandomMapTransform) and not isinstance(
        self.transform, RandomMapFnTransform
    ):
      raise ValueError(
          f"{str(self.transform)} is not reproducible. Use"
          " airio.preprocessors.RandomMapFnTransform instead."
      )

  def __call__(
      self,
      ds: lazy_dataset.LazyMapDataset,
      rng: JaxRng | None = None,
      runtime_args: AirIOInjectedRuntimeArgs | None = None,
  ):
    # pytype: disable=attribute-error
    if isinstance(self.transform, FnTransforms):
      self.transform.runtime_args = runtime_args
      # Note: Runtime args support can be extended to general grain transforms
      # by finding and setting attrs annotated with `AirIOInjectedRuntimeArgs`.
    match self.transform:
      case grain.MapTransform():
        return lazy_dataset.MapLazyMapDataset(ds, self.transform)
      case RandomMapFnTransform():
        # Special case to support reproducible stochastic transformations with
        # jax PRNGKeys.
        if rng is None:
          rng = jax.random.PRNGKey(np.int32(time.time()))
        map_fn = inject_runtime_args_to_fn(self.transform.map_fn, runtime_args)
        return lazy_dataset_transforms.RandomMapFnLazyMapDataset(
            ds,
            map_fn=map_fn,
            base_rng=rng,
        )
      case grain.FilterTransform():
        return lazy_dataset.FilterLazyMapDataset(ds, self.transform)
      case grain.Batch():
        return lazy_dataset.BatchLazyMapDataset(
            ds,
            batch_size=self.transform.batch_size,
            drop_remainder=self.transform.drop_remainder,
        )
      case PackTransform():
        return self.transform(ds, runtime_args)
      case _:
        # Should be taken care of by post init validation.
        raise ValueError("%s is not supported" % str(self.transform))
    # pytype: enable=attribute-error


def inject_runtime_args_to_fn(
    fn: Callable[..., Any], runtime_args: AirIOInjectedRuntimeArgs
):
  """Inject a partial function with runtime_args.

  For example, for the following fn, `runtime_args` will be passed to `arg`
  using `functools.partial`. The name of the arg doesn't matter, the typing
  annotation is used to find and pass runtime args.

  def my_fn(ex, ..other_args.., arg: AirIOInjectedRuntimeArgs, ..other_args..):
    ...

  Args:
    fn: A function.
    runtime_args: A `AirIOInjectedRuntimeArgs` obj to be passed to the fn.

  Returns:
    The provided fn with `runtime_args` passed to any arg annotated with
    `AirIOInjectedRuntimeArgs` using `functools.partial`.

  Raises:
    ValueError: if there are multiple args annotated as
    `AirIOInjectedRuntimeArgs`.
  """
  all_params = inspect.signature(fn).parameters
  all_annotations = [
      (arg_name, param.annotation) for arg_name, param in all_params.items()
  ]
  runtime_args_annotations = [
      ann for ann in all_annotations if ann[1] is AirIOInjectedRuntimeArgs
  ]
  if not runtime_args_annotations:
    return fn
  if len(runtime_args_annotations) > 1:
    raise ValueError(
        "Fn has multiple args annotated with AirIOInjectedRuntimeArgs, must"
        f" have only one: {fn}, {runtime_args_annotations}"
    )
  return functools.partial(fn, **{runtime_args_annotations[0][0]: runtime_args})
