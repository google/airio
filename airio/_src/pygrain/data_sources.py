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

"""Grain-based Data Source implementations for AirIO."""

import json
from typing import Iterable, Mapping, Union

import airio.core as airio
import grain.python as grain


class ArrayRecordDataSource(airio.data_sources.DataSource):
  """Wrapper for grain.ArrayRecordDataSource with multiple splits support."""

  def __init__(
      self,
      split_to_filepattern: Mapping[str, Union[str, Iterable[str]]],
  ):
    self._split_to_filepattern = split_to_filepattern

    self.splits = set(self._split_to_filepattern)
    self._sources = {}
    for split in self.splits:
      self._sources[split] = grain.ArrayRecordDataSource(
          self._split_to_filepattern[split],
      )

  def get_data_source(self, split: str) -> grain.ArrayRecordDataSource:
    if split not in self._sources:
      raise ValueError(f'Split {split} not found in {self.splits}.')
    return self._sources[split]

  def num_input_examples(self, split: str) -> int:
    if split not in self._sources:
      raise ValueError(f'Split {split} not found in {self.splits}.')
    return len(self._sources[split])




class JsonDataSource(airio.data_sources.DataSource):
  """Wrapper for grain.InMemoryDataSource that uses json file(s) as input data."""

  def __init__(
      self,
      split_to_filepattern: Mapping[str, Union[str, Iterable[str]]],
  ):
    """JsonDataSource constructor.

    Args:
      split_to_filepattern: a mapping of split name to file pattern(s). File
        pattern(s) can be a single string or iterable.
    """
    self._split_to_filepattern = split_to_filepattern

    self.splits = set(self._split_to_filepattern)
    self._sources = {}
    for split in self.splits:
      elements = json.load(open(self._split_to_filepattern[split]))
      self._sources[split] = grain.InMemoryDataSource(elements=elements)

  def get_data_source(self, split: str) -> grain.InMemoryDataSource:
    if split not in self._sources:
      raise ValueError(f'Split {split} not found in {self.splits}.')
    return self._sources[split]

  def num_input_examples(self, split: str) -> int:
    if split not in self._sources:
      raise ValueError(f'Split {split} not found in {self.splits}.')
    return len(self._sources[split])


