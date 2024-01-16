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

import copy
import dataclasses
import functools
import inspect
from typing import Any, Callable, Mapping

import grain.python as grain
import jax

# TODO(b/294122943): Implement flat_map.

lazy_dataset = grain.experimental.lazy_dataset
LazyDataset = lazy_dataset.LazyMapDataset | lazy_dataset.LazyIterDataset
JaxRng = jax.Array


@dataclasses.dataclass
class AirIOInjectedRuntimeArgs:
  """A set of attributes that can be injected into preprocessors at runtime."""

  sequence_lengths: Mapping[str, int]
  split: str

  def clone(self) -> "AirIOInjectedRuntimeArgs":
    """Returns a deep copy of self."""
    return copy.deepcopy(self)



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
