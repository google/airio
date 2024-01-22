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

"""AirIO-specific dataset iterators."""

import concurrent.futures
import json
from typing import Any

from airio import dataset_iterators
from clu import asynclib
from clu.data import dataset_iterator as clu_dataset_iterator
from etils import epath
import grain.python as grain
import numpy as np


lazy_dataset = grain.experimental.lazy_dataset
LazyDataset = lazy_dataset.LazyMapDataset | lazy_dataset.LazyIterDataset


class PyGrainDatasetIteratorWrapper(dataset_iterators.AirIODatasetIterator):
  """Wrapper iterator for grain.PyGrainDatasetIterator."""

  def __init__(self, data_loader: grain.DataLoader | LazyDataset):
    super().__init__()
    self._data_loader = data_loader
    self._iterator = data_loader.__iter__()
    self._state_as_dict = isinstance(self._data_loader, LazyDataset)

    # Necessary to support peek_async().
    self._peek = None
    self._peek_future = None
    self._pool = None

  def __next__(self) -> clu_dataset_iterator.Element:
    return next(self._iterator)

  @property
  def element_spec(self) -> clu_dataset_iterator.ElementSpec:
    local_iter = iter(self._data_loader)
    first_element = next(local_iter)
    element_spec = {}
    for k, v in first_element.items():
      if isinstance(v, np.ndarray):
        element_spec[k] = clu_dataset_iterator.ArraySpec(
            dtype=v.dtype, shape=tuple(v.shape)
        )
    return element_spec

  def peek(self) -> clu_dataset_iterator.Element:
    """Returns the next element without consuming it.

    This will get the next element from the underlying iterator. The element
    is stored and return on the next call of __next__().

    Returns:
      The next element.
    """
    if self._peek is None:
      local_iter = iter(self._data_loader)
      self._peek = next(local_iter)
    return self._peek

  def peek_async(
      self,
  ) -> concurrent.futures.Future[clu_dataset_iterator.Element]:
    """Same as peek() but returns the Future of the element.

    Users can call this to warm up the iterator.

    Returns:
      Future with the next element. The element is also kept and returned on the
      next call of __next__().
    """
    if self._peek_future is None:
      if self._pool is None:
        self._pool = asynclib.Pool(max_workers=1)
      self._peek_future = self._pool(self.peek)()
    return self._peek_future

  def get_state(self) -> dict[str, Any]:
    if self._state_as_dict:
      return self._iterator.get_state()
    return json.loads(self._iterator.get_state().decode())

  def set_state(self, state: dict[str, Any]) -> None:
    if not self._state_as_dict:
      state = json.dumps(state, indent=4).encode()
    self._iterator.set_state(state)

  def save(self, filename: epath.PathLike):
    filename = epath.Path(filename)
    filename.write_text(json.dumps(self.get_state(), indent=4))

  def restore(self, filename: epath.PathLike):
    filename = epath.Path(filename)
    if not filename.exists():
      raise ValueError(f"File {filename} does not exist.")
    self.set_state(json.loads(filename.read_text()))

  def __repr__(self) -> str:
    return (
        f"PyGrainDatasetIteratorWrapper({self._data_loader!r}), "
        f"state: {self.get_state()!r}"
    )
