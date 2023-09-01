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

"""AirIO-specific dataset iterators."""

from clu.data import dataset_iterator
from etils import epath
import grain.python as grain
import numpy as np


class PyGrainDatasetIteratorWrapper(dataset_iterator.DatasetIterator):
  """Wrapper iterator for grain.PyGrainDatasetIterator."""

  def __init__(self, data_loader: grain.DataLoader):
    self._data_loader = data_loader
    self._iterator = data_loader.__iter__()

  def __next__(self) -> dataset_iterator.Element:
    return next(self._iterator)

  @property
  def element_spec(self) -> dataset_iterator.ElementSpec:
    local_iter = iter(self._data_loader)
    first_element = next(local_iter)
    element_spec = {}
    for k, v in first_element.items():
      if isinstance(v, np.ndarray):
        element_spec[k] = dataset_iterator.ArraySpec(
            dtype=v.dtype, shape=tuple(v.shape)
        )
    return element_spec

  def get_state(self) -> bytes:
    return self._iterator.get_state()

  def set_state(self, state: bytes) -> None:
    self._iterator.set_state(state)

  def save(self, filename: epath.PathLike):
    filename = epath.Path(filename)
    filename.write_text(self.get_state().decode())

  def restore(self, filename: epath.PathLike):
    filename = epath.Path(filename)
    if not filename.exists():
      raise ValueError(f"File {filename} does not exist.")
    self.set_state(filename.read_text().encode())

  def __repr__(self) -> str:
    return (
        f"PyGrainDatasetIteratorWrapper({self._data_loader!r}), "
        f"state: {self.get_state()!r}"
    )
