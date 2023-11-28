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
from typing import Any, Callable, Mapping

from airio import lazy_dataset_transforms
import grain.python as grain
import jax
import numpy as np

# TODO(b/294122943): Implement flat_map.

lazy_dataset = grain.experimental.lazy_dataset
LazyDataset = lazy_dataset.LazyMapDataset | lazy_dataset.LazyIterDataset
JaxRng = jax.Array


@dataclasses.dataclass
class AirIOInjectedRuntimeArgs:
  """A set of attributes that can be injected into preprocessors at runtime."""
  sequence_lengths: Mapping[str, int]
  split: str


MapFnCallable = (
    Callable[[Any], Any] | Callable[[Any, AirIOInjectedRuntimeArgs], Any]
)
RandomMapFnCallable = (
    Callable[[Any, JaxRng], Any]
    | Callable[[Any, JaxRng, AirIOInjectedRuntimeArgs], Any]
)
FilterFnCallable = (
    Callable[[Any], bool] | Callable[[Any, AirIOInjectedRuntimeArgs], bool]
)
UpdateRuntimeArgsCallable = Callable[
    [AirIOInjectedRuntimeArgs], AirIOInjectedRuntimeArgs
]


@dataclasses.dataclass
class MapFnTransform(grain.MapTransform):
  """Grain Transform to represent AirIO map preprocessors.

  Attrs:
    map_fn: A map fn to apply.
    runtime_args: This is injected by AirIO at runtime during get_dataset calls,
      and contains args that may be used for preprocessing, e.g. sequence
      lengths.
    update_runtime_args: An optional fn to update `runtime_args` for downstream
      preprocessing. This can be used when a preprocessor creates new features
      for consumption by downstream preprocessors, e.g. trimming and padding.
  """

  map_fn: MapFnCallable
  runtime_args: AirIOInjectedRuntimeArgs | None = None
  update_runtime_args: UpdateRuntimeArgsCallable | None = None

  def map(self, element):
    """Maps a single element."""
    return inject_runtime_args_to_fn(self.map_fn, self.runtime_args)(element)


@dataclasses.dataclass
class RandomMapFnTransform(grain.RandomMapTransform):
  """Grain Transform to represent AirIO random map preprocessors.

  Attrs:
    map_fn: A map fn to apply.
    runtime_args: This is injected by AirIO at runtime during get_dataset calls,
      and contains args that may be used for preprocessing, e.g. sequence
      lengths.
    update_runtime_args: An optional fn to update `runtime_args` for downstream
      preprocessing. This can be used when a preprocessor creates new features
      for consumption by downstream preprocessors, e.g. trimming and padding.
  """

  map_fn: RandomMapFnCallable
  runtime_args: AirIOInjectedRuntimeArgs | None = None
  update_runtime_args: UpdateRuntimeArgsCallable | None = None

  def random_map(self, element, rng: np.random.Generator):
    """Maps a single element."""
    jax_rng = jax.random.PRNGKey(rng.integers(0, 2**16 - 1))
    return inject_runtime_args_to_fn(self.map_fn, self.runtime_args)(
        element, jax_rng
    )


@dataclasses.dataclass
class FilterFnTransform(grain.FilterTransform):
  """Grain Transform to represent AirIO filter preprocessors.

  Attrs:
    filter_fn: A filter fn to apply.
    runtime_args: This is injected by AirIO at runtime during get_dataset calls,
      and contains args that may be used for preprocessing, e.g. sequence
      lengths.
    update_runtime_args: An optional fn to update `runtime_args` for downstream
      preprocessing. This can be used when a preprocessor creates new features
      for consumption by downstream preprocessors, e.g. trimming and padding.
  """

  filter_fn: FilterFnCallable
  runtime_args: AirIOInjectedRuntimeArgs | None = None
  update_runtime_args: UpdateRuntimeArgsCallable | None = None

  def filter(self, element) -> bool:
    """Filters a single element."""
    return inject_runtime_args_to_fn(self.filter_fn, self.runtime_args)(element)


@dataclasses.dataclass
class LazyMapTransform:
  """AirIO preprocessor class for LazyMapDataset transformations.

  Avoid using this Transform class if possible. It is important for users to set
  the `update_runtime_args` and `produces_sparse_datasets` attributes correctly
  because it is not possible to verify correctness at runtime.

  Attributes:
    transform: A `Callable` that preprocesses `lazy_dataset.LazyMapDataset`
      based on runtime args like sequence lengths provided via
      `AirIOInjectedRuntimeArgs`, and returns a `lazy_dataset.LazyMapDataset`.
    update_runtime_args: A `Callable` that updates the
      `AirIOInjectedRuntimeArgs` for use by subsequent transforms if this
      transform modifies or adds new features (e.g. segment ids after packing).
      Pass `lambda x: x` if runtime args aren't updated.
    has_none_elements: A bool to indicate whether the transform removes examples
      from the original dataset, e.g. filtering, packing, etc.
  """

  transform: Callable[
      [lazy_dataset.LazyMapDataset, AirIOInjectedRuntimeArgs],
      lazy_dataset.LazyMapDataset,
  ]
  update_runtime_args: UpdateRuntimeArgsCallable
  has_none_elements: bool

  def __call__(
      self,
      ds: lazy_dataset.LazyMapDataset,
      runtime_args: AirIOInjectedRuntimeArgs,
  ) -> lazy_dataset.LazyMapDataset:
    if not isinstance(ds, lazy_dataset.LazyMapDataset):
      raise ValueError(
          f"Cannot apply LazyMapDataset transform: {str(self.transform)} to"
          f" non-LazyMapDataset dataset: {str(ds)}"
      )
    return self.transform(ds, runtime_args)


@dataclasses.dataclass
class LazyIterTransform:
  """AirIO preprocessor class for LazyIterDataset transformations.

  Avoid using this Transform class if possible. It is important for users to set
  the `update_runtime_args` and `produces_sparse_datasets` attributes correctly
  because it is not possible to verify correctness at runtime.

  Attributes:
    transform: A `Callable` that preprocesses `lazy_dataset.LazyIterDataset`
      based on runtime args like sequence lengths provided via
      `AirIOInjectedRuntimeArgs`, and returns a `lazy_dataset.LazyIterDataset`.
    update_runtime_args: A `Callable` that updates the
      `AirIOInjectedRuntimeArgs` for use by subsequent transforms if this
      transform modifies or adds new features (e.g. segment ids after packing).
      Pass `lambda x: x` if runtime args aren't updated.
  """

  transform: Callable[
      [lazy_dataset.LazyIterDataset, AirIOInjectedRuntimeArgs],
      lazy_dataset.LazyIterDataset,
  ]
  update_runtime_args: UpdateRuntimeArgsCallable

  def __call__(
      self,
      ds: lazy_dataset.LazyMapDataset | lazy_dataset.LazyIterDataset,
      runtime_args: AirIOInjectedRuntimeArgs,
  ) -> lazy_dataset.LazyIterDataset:
    if isinstance(ds, lazy_dataset.LazyMapDataset):
      ds = ds.to_iter_dataset()
    if not isinstance(ds, lazy_dataset.LazyIterDataset):
      raise ValueError(
          f"Cannot apply LazyIterDataset transform: {str(self.transform)} to"
          f" non-LazyIterDataset dataset: {str(ds)}"
      )
    return self.transform(ds, runtime_args)


FnTransforms = MapFnTransform | RandomMapFnTransform | FilterFnTransform
LazyTransforms = LazyMapTransform | LazyIterTransform
# This may be extended as needed.
AirIOPreprocessor = grain.Transformation | LazyTransforms


@dataclasses.dataclass
class LazyDatasetTransform:
  """A convenience function to map Transforms to LazyDatasets."""
  transform: AirIOPreprocessor

  def __post_init__(self):
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

  def get_updated_runtime_args(
      self, runtime_args: AirIOInjectedRuntimeArgs
  ) -> AirIOInjectedRuntimeArgs:
    # pytype:disable=attribute-error
    if (
        hasattr(self.transform, "update_runtime_args")
        and self.transform.update_runtime_args
    ):
      return self.transform.update_runtime_args(runtime_args)
    if isinstance(self.transform, LazyTransforms):
      return self.transform.update_runtime_args(runtime_args)
    return runtime_args
    # pytype:enable=attribute-error

  def __call__(
      self,
      ds: LazyDataset,
      rng: JaxRng | None = None,
      runtime_args: AirIOInjectedRuntimeArgs | None = None,
  ):
    # pytype: disable=attribute-error
    if isinstance(self.transform, FnTransforms):
      self.transform.runtime_args = runtime_args
    match self.transform:
      case grain.MapTransform():
        return ds.map(self.transform)
      case RandomMapFnTransform():
        # Special case to support reproducible stochastic transformations with
        # jax PRNGKeys.
        # Note: LazyIterDatasets are not yet supported, but can be if needed.
        if not isinstance(ds, lazy_dataset.LazyMapDataset):
          raise ValueError(
              "RandomMapFnTransform is not yet supported for"
              " non-LazyMapDatasets. Please file a bug with the AirIO team."
          )
        if rng is None:
          rng = jax.random.PRNGKey(np.int32(time.time()))
        map_fn = inject_runtime_args_to_fn(self.transform.map_fn, runtime_args)
        return lazy_dataset_transforms.RandomMapFnLazyMapDataset(
            ds,
            map_fn=map_fn,
            base_rng=rng,
        )
      case grain.FilterTransform():
        return ds.filter(self.transform)
      case grain.Batch():
        return ds.batch(
            batch_size=self.transform.batch_size,
            drop_remainder=self.transform.drop_remainder,
        )
      case LazyMapTransform():
        return self.transform(ds, runtime_args)
      case LazyIterTransform():
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
