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
import functools
import inspect
import typing
from typing import Any, Callable, Mapping, Protocol

from absl import logging
import jax
import numpy as np

# TODO(b/294122943): Implement flat_map.

JaxRng = jax.Array


@dataclasses.dataclass(frozen=True)
class AirIOInjectedRuntimeArgs:
  """A set of attributes that can be injected into preprocessors at runtime."""

  sequence_lengths: Mapping[str, int]
  split: str
  batch_size: int | None


  def replace(self, **kwargs):
    """Create a new AirIOInjectedRuntimeArgs instance replacing provided args."""
    return dataclasses.replace(self, **kwargs)


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
@typing.runtime_checkable
class MapFnTransform(Protocol):
  """Transform to represent AirIO map preprocessors.

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
    ...


@dataclasses.dataclass
@typing.runtime_checkable
class RandomMapFnTransform(Protocol):
  """Transform to represent AirIO random map preprocessors.

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
    ...


@dataclasses.dataclass
@typing.runtime_checkable
class FilterFnTransform(Protocol):
  """Transform to represent AirIO filter preprocessors.

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
    ...


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
  try:
    # The following may fail, e.g. for C++ functions wrapped with pybind.
    all_params = inspect.signature(fn).parameters
  except ValueError as e:
    logging.warning(
        "Failed to inspect signature of %s to inject runtime args: %s", fn, e
    )
    return fn
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
