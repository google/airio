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

"""AirIO preprocessor classes."""

import dataclasses
import time
from typing import Any
from typing import Callable
from typing import final
from typing import MutableMapping

from airio._src.core import preprocessors as core_preprocessors
from airio._src.pygrain import lazy_dataset_transforms
import grain.python as grain
import jax
import numpy as np


# TODO(b/294122943): Implement flat_map.


LazyDataset = grain.MapDataset | grain.IterDataset
JaxRng = jax.Array


@dataclasses.dataclass
class MapFnTransform(
    core_preprocessors.MapFnTransform, grain.MapTransform
):
  """Grain Transform to represent AirIO map preprocessors."""


  def map(self, element):
    """Maps a single element."""
    return core_preprocessors.inject_runtime_args_to_fn(
        self.map_fn, self.runtime_args
    )(element)


@final
class ConvertBoxesYXYXToCXCYHW(MapFnTransform):
  """Converts bounding boxes from [ymin, xmin, ymax, xmax] to [cx, cy, h, w]."""

  def __init__(self, box_field_name: str):
    """Initializes the transform.

    Args:
      box_field_name: The name of the field containing the bounding box tensor.
        The tensor is expected to have shape [..., 4].
    """
    self._box_field_name = box_field_name

  def map(self, element: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Applies the bounding box conversion to the data record.

    Args:
      element: A dict-like data record containing the data.

    Returns:
      The modified data record with the bounding box field updated in-place.

    Raises:
      ValueError: If the bounding box tensor does not have shape [..., 4].
    """
    boxes_yx_yx = element[self._box_field_name]
    if boxes_yx_yx.shape[-1] != 4:
      raise ValueError(
          f"Input tensor for field '{self._box_field_name}' must have shape "
          f"[..., 4], but got shape {boxes_yx_yx.shape}."
      )

    # Split into coordinates (numpy as np is imported at top of file)
    ymin, xmin, ymax, xmax = np.split(boxes_yx_yx, 4, axis=-1)

    # Calculate height, width, and center coordinates
    height = ymax - ymin
    width = xmax - xmin
    center_y = ymin + height / 2.0
    center_x = xmin + width / 2.0

    # Concatenate into new format [center_x, center_y, height, width]
    boxes_cx_cy_hw = np.concatenate(
        [center_x, center_y, height, width], axis=-1
    )

    # Update the field in-place
    element[self._box_field_name] = boxes_cx_cy_hw
    return element


@dataclasses.dataclass
class RandomMapFnTransform(
    core_preprocessors.RandomMapFnTransform, grain.RandomMapTransform
):
  """Grain Transform to represent AirIO random map preprocessors."""


  def random_map(self, element, rng: np.random.Generator):
    """Maps a single element."""
    jax_rng = jax.random.key(rng.integers(0, 2**16 - 1))
    return core_preprocessors.inject_runtime_args_to_fn(
        self.map_fn, self.runtime_args
    )(element, jax_rng)


@dataclasses.dataclass
class FilterFnTransform(
    core_preprocessors.FilterFnTransform, grain.FilterTransform
):
  """Grain Transform to represent AirIO filter preprocessors."""


  def filter(self, element) -> bool:
    """Filters a single element."""
    return core_preprocessors.inject_runtime_args_to_fn(
        self.filter_fn, self.runtime_args
    )(element)


@dataclasses.dataclass
class LazyMapTransform:
  """AirIO preprocessor class for LazyMapDataset transformations.

  Avoid using this Transform class if possible. It is important for users to set
  the `update_runtime_args` and `produces_none_elements` attributes correctly
  because it is not possible to verify correctness at runtime.

  Attributes:
    transform: A `Callable` that preprocesses `grain.MapDataset` based on
      runtime args like sequence lengths provided via
      `AirIOInjectedRuntimeArgs`, and returns a `grain.MapDataset`.
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
          grain.MapDataset,
          core_preprocessors.AirIOInjectedRuntimeArgs,
          JaxRng | None,
      ],
      grain.MapDataset,
  ]
  update_runtime_args: core_preprocessors.UpdateRuntimeArgsCallable
  produces_none_elements: bool
  requires_non_none_elements: bool

  def __call__(
      self,
      ds: grain.MapDataset,
      runtime_args: core_preprocessors.AirIOInjectedRuntimeArgs,
      rng: JaxRng | None,
  ) -> grain.MapDataset:
    if not isinstance(ds, grain.MapDataset):
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
    transform: A `Callable` that preprocesses `grain.IterDataset` based on
      runtime args like sequence lengths provided via
      `AirIOInjectedRuntimeArgs`, and returns a `grain.IterDataset`.
    update_runtime_args: A `Callable` that updates the
      `AirIOInjectedRuntimeArgs` for use by subsequent transforms if this
      transform modifies or adds new features (e.g. segment ids after packing).
      Pass `lambda x: x` if runtime args aren't updated.
  """

  transform: Callable[
      [
          grain.IterDataset,
          core_preprocessors.AirIOInjectedRuntimeArgs,
          JaxRng | None,
      ],
      grain.IterDataset,
  ]
  update_runtime_args: core_preprocessors.UpdateRuntimeArgsCallable

  def __call__(
      self,
      ds: grain.MapDataset | grain.IterDataset,
      runtime_args: core_preprocessors.AirIOInjectedRuntimeArgs,
      rng: JaxRng | None,
  ) -> grain.IterDataset:
    if not isinstance(ds, grain.IterDataset):
      raise ValueError(
          f"Cannot apply LazyIterDataset transform: {str(self.transform)} to"
          f" non-LazyIterDataset dataset: {str(ds)}"
      )
    return self.transform(ds, runtime_args, rng)


# These may be extended; update LazyDatasetTransform properties when adding new
# preprocessor types, specifically `produces_none_elements`,
# `can_process_iter_dataset`, and `requires_non_none_elements`.
FnTransforms = MapFnTransform | RandomMapFnTransform | FilterFnTransform
LazyTransforms = LazyMapTransform | LazyIterTransform
PyGrainAirIOPreprocessor = grain.Transformation | LazyTransforms


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
        self.transform, RandomMapFnTransform
    ):
      raise ValueError(
          f"{str(self.transform)} is not reproducible. Use"
          " airio.preprocessors.RandomMapFnTransform instead."
      )

  def get_updated_runtime_args(
      self, runtime_args: core_preprocessors.AirIOInjectedRuntimeArgs
  ) -> core_preprocessors.AirIOInjectedRuntimeArgs:
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

  @property
  def requires_iter_dataset(self) -> bool:
    return isinstance(self.transform, LazyIterTransform)

  def __call__(
      self,
      ds: LazyDataset,
      rng: JaxRng | None = None,
      runtime_args: core_preprocessors.AirIOInjectedRuntimeArgs | None = None,
  ):
    # pytype: disable=attribute-error
    if isinstance(self.transform, FnTransforms):
      self.transform.runtime_args = runtime_args
    match self.transform:
      case grain.MapTransform():
        return ds.map(self.transform)
      case RandomMapFnTransform():
        # Special case to support reproducible stochastic transformations with
        # jax rng keys.
        # Note: LazyIterDatasets are not yet supported, but can be if needed.
        if not isinstance(ds, grain.MapDataset):
          raise ValueError(
              "RandomMapFnTransform is not yet supported for"
              " non-LazyMapDatasets. Please file a bug with the AirIO team."
          )
        if rng is None:
          rng = jax.random.key(np.int32(time.time()))
        map_fn = core_preprocessors.inject_runtime_args_to_fn(
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
