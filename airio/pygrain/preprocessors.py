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

"""AirIO preprocessor classes."""

import dataclasses
import time
from typing import Callable

from airio import preprocessors as preprocessors_lib
from airio.pygrain import lazy_dataset_transforms
import grain.python as grain
import jax
import numpy as np

# TODO(b/294122943): Implement flat_map.

lazy_dataset = grain.experimental.lazy_dataset
LazyDataset = lazy_dataset.LazyMapDataset | lazy_dataset.LazyIterDataset
JaxRng = jax.Array


@dataclasses.dataclass
class LazyMapTransform:
  """AirIO preprocessor class for LazyMapDataset transformations.

  Avoid using this Transform class if possible. It is important for users to set
  the `update_runtime_args` and `produces_none_elements` attributes correctly
  because it is not possible to verify correctness at runtime.

  Attributes:
    transform: A `Callable` that preprocesses `lazy_dataset.LazyMapDataset`
      based on runtime args like sequence lengths provided via
      `AirIOInjectedRuntimeArgs`, and returns a `lazy_dataset.LazyMapDataset`.
    update_runtime_args: A `Callable` that updates the
      `AirIOInjectedRuntimeArgs` for use by subsequent transforms if this
      transform modifies or adds new features (e.g. segment ids after packing).
      Pass `lambda x: x` if runtime args aren't updated.
    produces_none_elements: A bool to indicate whether the transform removes
      examples from the original dataset, e.g. filtering, packing, etc.
    requires_non_none_elements: A bool to indicate whether the transform
      requires strictly non-None elements in the original dataset, e.g.
      batching, mixing, etc.
  """

  transform: Callable[
      [
          lazy_dataset.LazyMapDataset,
          preprocessors_lib.AirIOInjectedRuntimeArgs,
          JaxRng | None,
      ],
      lazy_dataset.LazyMapDataset,
  ]
  update_runtime_args: preprocessors_lib.UpdateRuntimeArgsCallable
  produces_none_elements: bool
  requires_non_none_elements: bool

  def __call__(
      self,
      ds: lazy_dataset.LazyMapDataset,
      runtime_args: preprocessors_lib.AirIOInjectedRuntimeArgs,
      rng: JaxRng | None,
  ) -> lazy_dataset.LazyMapDataset:
    if not isinstance(ds, lazy_dataset.LazyMapDataset):
      raise ValueError(
          f"Cannot apply LazyMapDataset transform: {str(self.transform)} to"
          f" non-LazyMapDataset dataset: {str(ds)}"
      )
    return self.transform(ds, runtime_args, rng)


@dataclasses.dataclass
class LazyIterTransform:
  """AirIO preprocessor class for LazyIterDataset transformations.

  Avoid using this Transform class if possible. It is important for users to set
  the `update_runtime_args` attribute correctly because it is not possible to
  verify correctness at runtime.

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
      [
          lazy_dataset.LazyIterDataset,
          preprocessors_lib.AirIOInjectedRuntimeArgs,
          JaxRng | None,
      ],
      lazy_dataset.LazyIterDataset,
  ]
  update_runtime_args: preprocessors_lib.UpdateRuntimeArgsCallable

  def __call__(
      self,
      ds: lazy_dataset.LazyMapDataset | lazy_dataset.LazyIterDataset,
      runtime_args: preprocessors_lib.AirIOInjectedRuntimeArgs,
      rng: JaxRng | None,
  ) -> lazy_dataset.LazyIterDataset:
    if isinstance(ds, lazy_dataset.LazyMapDataset):
      ds = ds.to_iter_dataset()
    if not isinstance(ds, lazy_dataset.LazyIterDataset):
      raise ValueError(
          f"Cannot apply LazyIterDataset transform: {str(self.transform)} to"
          f" non-LazyIterDataset dataset: {str(ds)}"
      )
    return self.transform(ds, runtime_args, rng)


# These may be extended; update LazyDatasetTransform properties when adding new
# preprocessor types, specifically `produces_none_elements`,
# `can_process_iter_dataset`, and `requires_non_none_elements`.
FnTransforms = (
    preprocessors_lib.MapFnTransform
    | preprocessors_lib.RandomMapFnTransform
    | preprocessors_lib.FilterFnTransform
)
LazyTransforms = LazyMapTransform | LazyIterTransform
PyGrainAirIOPreprocessor = preprocessors_lib.AirIOPreprocessor | LazyTransforms


@dataclasses.dataclass
class LazyDatasetTransform:
  """A convenience function to map Transforms to LazyDatasets."""

  transform: PyGrainAirIOPreprocessor

  def __post_init__(self):
    if not isinstance(self.transform, PyGrainAirIOPreprocessor):
      raise ValueError(f"{str(self.transform)} is not supported")
    # TODO(b/300938204): Remove error for other RandomMapTransforms, once
    # these can be reproducibly processed.
    if isinstance(self.transform, grain.RandomMapTransform) and not isinstance(
        self.transform, preprocessors_lib.RandomMapFnTransform
    ):
      raise ValueError(
          f"{str(self.transform)} is not reproducible. Use"
          " airio.preprocessors.RandomMapFnTransform instead."
      )

  def get_updated_runtime_args(
      self, runtime_args: preprocessors_lib.AirIOInjectedRuntimeArgs
  ) -> preprocessors_lib.AirIOInjectedRuntimeArgs:
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

  @property
  def produces_none_elements(self) -> bool:
    """Returns True if the transform produces None elements, e.g. filters and LazyMap transforms.

    This is a best-effort check and may be wrong, e.g. a grain.MapTransform impl
    could produce None elements, a LazyMapTransform `produces_none_elements`
    attr could be misconfigured, etc.
    """
    if isinstance(self.transform, grain.FilterTransform):
      return True
    if isinstance(self.transform, LazyMapTransform):
      return self.transform.produces_none_elements
    return False

  @property
  def requires_non_none_elements(self) -> bool:
    if isinstance(self.transform, LazyMapTransform):
      return self.transform.requires_non_none_elements
    return isinstance(self.transform, grain.Batch)

  @property
  def can_process_iter_dataset(self) -> bool:
    return not isinstance(self.transform, LazyMapTransform)

  def __call__(
      self,
      ds: LazyDataset,
      rng: JaxRng | None = None,
      runtime_args: preprocessors_lib.AirIOInjectedRuntimeArgs | None = None,
  ):
    # pytype: disable=attribute-error
    if isinstance(self.transform, FnTransforms):
      self.transform.runtime_args = runtime_args
    match self.transform:
      case grain.MapTransform():
        return ds.map(self.transform)
      case preprocessors_lib.RandomMapFnTransform():
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
        map_fn = preprocessors_lib.inject_runtime_args_to_fn(
            self.transform.map_fn, runtime_args
        )
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
        return self.transform(ds, runtime_args, rng)
      case LazyIterTransform():
        return self.transform(ds, runtime_args, rng)
      case _:
        # Should be taken care of by post init validation.
        raise ValueError("%s is not supported" % str(self.transform))
    # pytype: enable=attribute-error
