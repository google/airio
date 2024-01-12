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

import collections
import concurrent.futures
from typing import Any

from clu.data import dataset_iterator as clu_dataset_iterator


class AirIODatasetIterator(clu_dataset_iterator.DatasetIterator):
  """Wrapper iterator for AirIO."""

  _iterator: collections.abc.Iterator[Any] = None

  def __next__(self) -> clu_dataset_iterator.Element:
    raise NotImplementedError()

  def peek(self) -> clu_dataset_iterator.Element:
    raise NotImplementedError()

  def peek_async(
      self,
  ) -> concurrent.futures.Future[clu_dataset_iterator.Element]:
    raise NotImplementedError()

  def get_state(self) -> dict[str, Any]:
    raise NotImplementedError()

  def set_state(self, state: dict[str, Any]) -> None:
    raise NotImplementedError()

  def __repr__(self) -> str:
    return f"AirIODatasetIterator(), state: {self.get_state()!r}"

