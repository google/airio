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
import typing
from typing import Iterable, Mapping, Protocol, Union
from airio._src.core import data_sources
import grain.python as grain
import numpy as np
import tensorflow_datasets as tfds


class ArrayRecordDataSource(data_sources.DataSource):
  """Wrapper for grain.ArrayRecordDataSource with multiple splits support."""

  def __init__(
      self,
      split_to_filepattern: Mapping[str, Union[str, Iterable[str]]],
  ):
    self._split_to_filepattern = split_to_filepattern

    self.splits = frozenset(self._split_to_filepattern.keys())
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




@typing.runtime_checkable
class DatasetFnCallable(Protocol):
  """Protocol for a function that returns a numpy array based on split."""

  def __call__(self, split: str) -> np.ndarray:
    ...


class FunctionDataSource(data_sources.DataSource):
  """A `DataSource` that uses a function to provide the input data."""

  def __init__(
      self,
      dataset_fn: DatasetFnCallable,
      splits: Iterable[str],
  ):
    """FunctionDataSource constructor.

    Args:
      dataset_fn: a function with the signature `dataset_fn(split)' that returns
        a numpy array.
      splits: an iterable of applicable string split names.
    """
    self._dataset_fn = dataset_fn
    self.splits = splits

  def get_data_source(self, split: str) -> np.ndarray:
    ds = self._dataset_fn(split=split)
    return ds

  def num_input_examples(self, split: str) -> int:
    if split not in self.splits:
      raise ValueError(f'Split {split} not found in {self.splits}.')
    return self._dataset_fn(split=split).size


class JsonDataSource(data_sources.DataSource):
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

    self.splits = frozenset(self._split_to_filepattern.keys())
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




class TfdsDataSource(data_sources.DataSource):
  """Wrapper for tfds.data_source with multiple splits support."""

  def __init__(
      self,
      tfds_name: str,
      tfds_data_dir: str | None = None,
      splits: Union[Iterable[str], Mapping[str, str]] | None = None,
      decoders: tfds.typing.TreeDict[tfds.decode.Decoder] | None = None,
  ):
    self._tfds_name = tfds_name
    self._tfds_data_dir = tfds_data_dir
    self._decoders = decoders

    if splits and isinstance(splits, str):
      self.splits = frozenset([splits])
    else:
      self.splits = frozenset(splits or [])

    self._sources = {}
    for split in self.splits:
      self._sources[split] = tfds.data_source(
          self._tfds_name,
          data_dir=self._tfds_data_dir,
          split=split,
          decoders=self._decoders,
      )

  def get_data_source(self, split: str):
    if split not in self._sources:
      raise ValueError(
          f'Split {split} not found in {self.splits} for {self._tfds_name}.'
      )
    return self._sources[split]

  def num_input_examples(self, split: str) -> int:
    if split not in self._sources:
      raise ValueError(
          f'Split {split} not found in {self.splits} for {self._tfds_name}.'
      )
    return len(self._sources[split])
