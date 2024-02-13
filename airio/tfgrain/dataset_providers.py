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

"""Classes ofr AirIO data loading with TfGrain."""

from typing import Mapping, Sequence, Union
import airio
from airio import data_sources
# Import "preprocessors" as "preprocessors_lib" to prevent naming conflicts with
# "preprocessors" attrs in this file.
from airio import preprocessors as preprocessors_lib
from clu.data import dataset_iterator as clu_dataset_iterator
import grain.tensorflow as grain
import tensorflow_datasets as tfds

AirIOPreprocessor = preprocessors_lib.AirIOPreprocessor
# TODO(sahildua): Expose these data sources as AirIO data sources?
GrainDataSource = grain.TfDataSource


class TfGrainTask(airio.dataset_providers.Task):
  """A task for loading data using TfGrain ."""

  name: str
  source: data_sources.DataSource

  def _get_data_source_for_split(self, split: str) -> GrainDataSource:
    if self.source is None:
      raise ValueError("Source has not been set on this task object.")
    return self.source.get_data_source(split=split)

  def get_dataset(
      self,
      sequence_lengths: Mapping[str, int] | None = None,
      split: str = tfds.Split.TRAIN,
      runtime_preprocessors: Sequence[AirIOPreprocessor] | None = None,
      batch_size: int | None = None,
      shuffle: bool = True,
      seed: int | None = 0,
      shard_info: airio.dataset_providers.ShardInfo | None = None,
      num_epochs: int | None = 1,
  ) -> clu_dataset_iterator.DatasetIterator:
    if shard_info is None:
      shard_options = grain.NoSharding()
    else:
      shard_options = grain.ShardOptions(
          shard_index=shard_info.index,
          shard_count=shard_info.num_shards,
      )

    sampler = grain.TfDefaultIndexSampler(
        num_records=self.num_input_examples(split=split),
        shard_options=shard_options,
        shuffle=shuffle,
        num_epochs=num_epochs,
        seed=seed,
    )

    source = self._get_data_source_for_split(split)

    ops = self.get_preprocessors()
    if runtime_preprocessors:
      ops.extend(runtime_preprocessors)
    if batch_size:
      # TODO(sahildua): add batch transform.
      pass

    # TODO(sahildua): Add runtime args.

    ds = grain.TfDataLoader(
        source=source,
        sampler=sampler,
        transformations=ops,
        iterator_options=grain.IteratorOptions(drop_grain_meta_features=True),
    )
    return iter(ds)  # pytype: disable=bad-return-type


class TfGrainMixture(airio.dataset_providers.Mixture):
  """A mixture of tasks for loading data using TfGrain."""

  def __init__(
      self,
      name: str,
      tasks: Sequence[Union[TfGrainTask, "TfGrainMixture"]],
      proportions: Sequence[float],
  ):
    for task in tasks:
      if not isinstance(task, (TfGrainTask, TfGrainMixture)):
        raise ValueError(
            f"Task '{task.name}' is not a TfGrainTask or TfGrainMixture."
        )

    super(TfGrainMixture, self).__init__(
        name=name, tasks=tasks, proportions=proportions
    )

  def get_dataset(
      self,
      sequence_lengths: Mapping[str, int] | None = None,
      split: str = tfds.Split.TRAIN,
      runtime_preprocessors: Sequence[AirIOPreprocessor] | None = None,
      batch_size: int | None = None,
      shuffle: bool = True,
      seed: int | None = 0,
      shard_info: airio.dataset_providers.ShardInfo | None = None,
      num_epochs: int | None = 1,
  ) -> clu_dataset_iterator.DatasetIterator:
    sources, proportions, transformations_per_source = [], [], []
    for task in self.leaf_tasks:
      sources.append(task.source.get_data_source(split=split))
      proportions.append(self.get_proportion(task))
      transformations_per_source.append(task.get_preprocessors())

    if shard_info is None:
      shard_options = grain.NoSharding()
    else:
      shard_options = grain.ShardOptions(
          shard_index=shard_info.index, shard_count=shard_info.num_shards
      )

    if num_epochs and num_epochs != 1:
      raise ValueError(
          "Epochs are not supported for mixtures. A mixture "
          "always repeats indefinitely over it's tasks."
      )

    # TODO(sahildua): Handle logic for tfgrain meta features and transforms.
    # TODO(sahildua): Add runtime args.

    # TODO(sahildua): Add transformations to be applied after combination of
    # all data sources, if needed.
    transformations = []

    sampler = grain.TfMixtureIndexSampler(
        [len(s) for s in sources],
        shard_options=shard_options,
        proportions=proportions,
        shuffle=shuffle,
        seed=seed,
    )
    data_loader = grain.TfMixtureDataLoader(
        sources=sources,
        sampler=sampler,
        transformations_per_source=transformations_per_source,
        transformations=transformations,
        iterator_options=grain.IteratorOptions(drop_grain_meta_features=True),
    )
    return iter(data_loader)  # pytype: disable=bad-return-type


class TfGrainTaskBuilder(airio.dataset_providers.TaskBuilder):
  """Builder class for building TfGrainTask object.

  In order to create a TfGrainTask object, build() method should be called on
  the TfGrainTaskBuilder object after setting the appropriate data source and
  preprocessors.
  """

  def build(self) -> TfGrainTask:
    """Returns a fully-defined TfGrainTask object.

    Creates a new task object using properties of the current task builder
    object as long as neither of source and preprocessors is None.

    Raises:
      ValueError: when either of the source or preprocessors is None.
    """
    if self._source is None:
      raise ValueError("Source has not been set on this task builder.")
    if self._preprocessors is None:
      raise ValueError("Preprocessors has not been set on this task builder.")

    return TfGrainTask(
        name=self._task_name,
        source=self._source,
        preprocessors=self._preprocessors,
    )

  @classmethod
  def from_task(cls, task: TfGrainTask) -> "TfGrainTaskBuilder":
    """Returns TaskBuilder for the given existing Task object.

    This method takes an existing task, copies its properties into a new
    TaskBuilder object and returns it.

    Args:
      task: Existing task object.
    """
    return TfGrainTaskBuilder(
        task_name=task.name,
        source=task.source,
        preprocessors=task.get_preprocessors(),
    )

  def __repr__(self) -> str:
    return (
        f"TfGrainTaskBuilder(task_name={self._task_name},"
        f" source={self._source}, preprocessors={self._preprocessors})"
    )
