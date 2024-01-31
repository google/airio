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

"""Data Source implementations for AirIO."""

import typing
from typing import Iterable, Mapping, Protocol, Union

import numpy as np
import tensorflow_datasets as tfds


@typing.runtime_checkable
class DataSource(Protocol):
  """Interface for data sources wrappers with multiple splits support."""

  splits: Iterable[str] = None

  def get_data_source(self, split: str) -> ...:
    ...

  def num_input_examples(self, split: str) -> int:
    ...


@typing.runtime_checkable
class DatasetFnCallable(Protocol):
  """Protocol for a function that returns a numpy array based on split."""

  def __call__(self, split: str) -> np.ndarray:
    ...


class FunctionDataSource(DataSource):
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


class TfdsDataSource(DataSource):
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
      self.splits = {splits}
    else:
      self.splits = splits or []

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
