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
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence, Union

from airio import data_sources
from airio import feature_converters
import grain.python as grain
import seqio
import tensorflow_datasets as tfds


SHUFFLE_BUFFER_SIZE = 1000

# TODO(sahildua): Expose these data sources as AirIO data sources?
GrainDataSource = Union[grain.TfdsDataSource, grain.SSTableDataSource]
GrainPreprocessor = grain.Transformation | grain.Operation


@typing.runtime_checkable
class DatasetProviderBase(Protocol):
  """Abstract base for classes that provide a dataset."""

  splits: Iterable[str] = None

  def get_dataset(
      self,
      sequence_length: Optional[Mapping[str, int]] = None,
      split: str = tfds.Split.TRAIN,
      feature_converter: Optional[
          feature_converters.PyGrainFeatureConverter
      ] = None,
      use_cached: bool = False,
      shuffle: bool = True,
      seed: Optional[int] = 0,
      shard_info: Optional[seqio.ShardInfo] = None,
      num_epochs: Optional[int] = 1,
  ) -> grain.PyGrainDatasetIterator:
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
      source: data_sources.DataSource,
      preprocessors: Sequence[GrainPreprocessor] | None = None,
      postprocess_fn: Callable[..., Any] | None = None,
      metric_fns: Sequence[seqio.metrics.MetricFnCallable] | None = None,
      metric_objs: Sequence[seqio.metrics.Metric] | None = None,
      shuffle_buffer_size: int | None = SHUFFLE_BUFFER_SIZE,
      source_info: seqio.SourceInfo | None = None,
  ):
    self.splits = source.splits
    self.name = name
    self.source = source
    self._source_info = source_info

    self._preprocessors = list(preprocessors) if preprocessors else []
    self._postprocess_fn = postprocess_fn
    self._metric_fns = metric_fns

    self._metric_objs = metric_objs
    self._shuffle_buffer_size = shuffle_buffer_size

  def num_input_examples(self, split: str) -> int | None:
    return self.source.num_input_examples(split=split)

  def _get_data_source_for_split(self, split: str) -> GrainDataSource:
    return self.source.get_data_source(split=split)

  # TODO(sahildua): Add logging.
  def get_dataset(
      self,
      sequence_length: Mapping[str, int] | None = None,
      split: str = tfds.Split.TRAIN,
      feature_converter: Optional[
          feature_converters.PyGrainFeatureConverter
      ] = None,
      use_cached: bool = False,
      shuffle: bool = True,
      seed: int | None = 0,
      shard_info: seqio.ShardInfo | None = None,
      num_epochs: int | None = 1,
  ) -> grain.PyGrainDatasetIterator:
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

    ops = self._preprocessors
    if feature_converter is not None:
      ops.extend(feature_converter.get_operations())

    ds = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=ops,
    )
    return ds.__iter__()


def get_dataset(
    mixture_or_task: Task,
    split: str = "train",
    feature_converter: Optional[
        feature_converters.PyGrainFeatureConverter
    ] = None,
    use_cached: bool = False,
    shuffle: bool = False,
    num_epochs: Optional[int] = 1,
    seed: Optional[int] = None,
) -> grain.PyGrainDatasetIterator:
  """Returns the PyGrain dataset iterator."""
  return mixture_or_task.get_dataset(
      split=split,
      feature_converter=feature_converter,
      use_cached=use_cached,
      shuffle=shuffle,
      num_epochs=num_epochs,
      seed=seed,
  )
