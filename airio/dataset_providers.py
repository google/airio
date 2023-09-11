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

"""Classes for AirIO data loading."""

import typing
from typing import Any, Iterable, List, Mapping, Protocol, Sequence, Union

from airio import data_sources
from airio import dataset_iterators
from airio import feature_converters
from clu.data import dataset_iterator as clu_dataset_iterator
import grain.python as grain
import seqio
import tensorflow_datasets as tfds


SHUFFLE_BUFFER_SIZE = 1000
DEFAULT_NUM_RECORDS_TO_INSPECT = 2
MAX_NUM_RECORDS_TO_INSPECT = 1000

# TODO(sahildua): Expose these data sources as AirIO data sources?
GrainDataSource = grain.TfdsDataSource
GrainPreprocessor = grain.Transformation | grain.Operation


@typing.runtime_checkable
class DatasetProviderBase(Protocol):
  """Abstract base for classes that provide a dataset."""

  splits: Iterable[str] = None

  def get_dataset(
      self,
      sequence_lengths: Mapping[str, int] | None = None,
      split: str = tfds.Split.TRAIN,
      feature_converter: feature_converters.PyGrainFeatureConverter
      | None = None,
      batch_size: int | None = None,
      shuffle: bool = True,
      seed: int | None = 0,
      shard_info: seqio.ShardInfo | None = None,
      num_epochs: int | None = 1,
  ) -> clu_dataset_iterator.DatasetIterator:
    """Returns the dataset iterator."""
    ...

  def num_input_examples(self, split: str) -> int | None:
    ...


class Task(DatasetProviderBase):
  """A class to manage a dataset and its related metrics."""

  name: str
  source: data_sources.DataSource

  def __init__(
      self,
      name: str,
      source: data_sources.DataSource | None = None,
      preprocessors: Sequence[GrainPreprocessor] | None = None,
  ):
    if source is None and preprocessors is None:
      raise ValueError("Either source or preprocessors must be set.")

    self.name = name
    # Default values.
    self.source = None
    self.splits = None
    self._preprocessors = None

    if source is not None:
      self.source = source
      self.splits = source.splits
    if preprocessors is not None:
      self._preprocessors = list(preprocessors)

  def set_data_source(self, source: data_sources.DataSource) -> None:
    if self.source is not None:
      raise ValueError("Source has already been set on this task.")
    self.source = source
    self.splits = source.splits

  def set_preprocessors(
      self, preprocessors: Sequence[GrainPreprocessor]
  ) -> None:
    if self._preprocessors is not None:
      raise ValueError("Preprocessors have already been set on this task.")
    self._preprocessors = list(preprocessors)

  def get_preprocessors(self) -> List[GrainPreprocessor]:
    if self._preprocessors is None:
      raise ValueError("Preprocessors have not been set on this task.")
    return list(self._preprocessors)

  def num_input_examples(self, split: str) -> int | None:
    if self.source is None:
      raise ValueError("Source has not been set on this task object.")
    return self.source.num_input_examples(split=split)

  def _get_data_source_for_split(self, split: str) -> GrainDataSource:
    if self.source is None:
      raise ValueError("Source has not been set on this task object.")
    return self.source.get_data_source(split=split)

  def _ensure_fully_defined(self) -> bool:
    if self.source is None or self._preprocessors is None:
      raise ValueError(
          "Both source and preprocessors must be set before calling"
          " get_dataset()."
      )

  # TODO(sahildua): Add logging.
  def get_dataset(
      self,
      sequence_lengths: Mapping[str, int] | None = None,
      split: str = tfds.Split.TRAIN,
      feature_converter: feature_converters.PyGrainFeatureConverter
      | None = None,
      batch_size: int | None = None,
      shuffle: bool = True,
      seed: int | None = 0,
      shard_info: seqio.ShardInfo | None = None,
      num_epochs: int | None = 1,
  ) -> clu_dataset_iterator.DatasetIterator:
    """Returns the dataset iterator as per the task configuration."""
    self._ensure_fully_defined()

    if shard_info is None:
      shard_options = grain.NoSharding()
    else:
      shard_options = grain.ShardOptions(
          shard_index=shard_info.index,
          shard_count=shard_info.num_shards,
      )

    sampler = grain.IndexSampler(
        num_records=self.num_input_examples(split=split),
        shard_options=shard_options,
        shuffle=shuffle,
        num_epochs=num_epochs,
        seed=seed,
    )

    source = self._get_data_source_for_split(split=split)

    ops = self.get_preprocessors()
    if feature_converter is not None:
      ops.extend(
          feature_converter.get_operations(
              batch_size=batch_size, task_feature_lengths=sequence_lengths
          )
      )

    return self._load_data(source=source, sampler=sampler, ops=ops)

  def _load_data(
      self,
      source: GrainDataSource,
      sampler: grain.IndexSampler,
      ops: Sequence[GrainPreprocessor],
  ) -> clu_dataset_iterator.DatasetIterator:
    """Returns a sampled data source after applying `ops`.

    A helper function for get_dataset and get_dataset_by_step.

    Args:
      source: a data source to load.
      sampler: a means of sampling from the source.
      ops: a list of transformations to apply.

    Returns an iterator of records after applying `ops`.
    """
    ds = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=ops,
    )

    return dataset_iterators.PyGrainDatasetIteratorWrapper(data_loader=ds)

  def get_dataset_by_step(
      self,
      num_records: int = DEFAULT_NUM_RECORDS_TO_INSPECT,
      sequence_lengths: Mapping[str, int] | None = None,
      split: str = tfds.Split.TRAIN,
      feature_converter: feature_converters.PyGrainFeatureConverter
      | None = None,
      batch_size: int | None = None,
      shuffle: bool = True,
      seed: int | None = 0,
  ) -> Iterable[Iterable[Mapping[str, Any]]]:
    """Returns a step-by-step transformation of a sample of records.

    Records the set of records after each transformation. Analogous to
    get_dataset(), with the recording of intermediate states.

    Args:
      num_records: the number of records to include in the sample.
      sequence_lengths: mapping of each feature key to its sequence length.
      split: the split to sample from.
      feature_converter: a feature converter.
      batch_size: the batch size.
      shuffle: whether to shuffle or not.
      seed: dataset seed.

    Returns: a list indexed by processing step. For example:
    |-----------------------------|
    | Raw data                    |
    | Preprocessing step 1        |
    | Preprocessing step 2        |
    | ...                         |
    | Final transformed data      |
    |-----------------------------|
    """
    self._ensure_fully_defined()

    # Validate num_records.
    if num_records < 1:
      num_records = DEFAULT_NUM_RECORDS_TO_INSPECT
    elif num_records > MAX_NUM_RECORDS_TO_INSPECT:
      num_records = MAX_NUM_RECORDS_TO_INSPECT

    sampler = grain.IndexSampler(
        num_records=num_records,
        shard_options=grain.NoSharding(),
        shuffle=shuffle,
        num_epochs=1,
        seed=seed,
    )

    source = self._get_data_source_for_split(split=split)

    all_ops = self.get_preprocessors()
    if feature_converter is not None:
      all_ops.extend(
          feature_converter.get_operations(
              batch_size=batch_size, task_feature_lengths=sequence_lengths
          )
      )

    # Raw data
    records_step0 = self._load_data(source=source, sampler=sampler, ops=[])
    accumulated_result = [list(records_step0)]

    if not all_ops:
      return accumulated_result

    # Apply all transformations, one by one.
    accumulated_ops = []
    for op in all_ops:
      accumulated_ops.append(op)
      records_next_step = self._load_data(
          source=source, sampler=sampler, ops=accumulated_ops
      )
      accumulated_result.append(list(records_next_step))

    return accumulated_result


def get_dataset(
    mixture_or_task: Task,
    sequence_lengths: Mapping[str, int] | None = None,
    split: str = "train",
    feature_converter: feature_converters.PyGrainFeatureConverter | None = None,
    batch_size: int | None = None,
    shuffle: bool = False,
    num_epochs: int | None = 1,
    seed: int | None = None,
) -> clu_dataset_iterator.DatasetIterator:
  """Returns the PyGrain dataset iterator."""
  return mixture_or_task.get_dataset(
      split=split,
      sequence_lengths=sequence_lengths,
      feature_converter=feature_converter,
      batch_size=batch_size,
      shuffle=shuffle,
      num_epochs=num_epochs,
      seed=seed,
  )
