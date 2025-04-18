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

"""Classes for AirIO data loading."""

import collections
import dataclasses
import functools
import typing
from typing import Iterable, List, Mapping, Protocol, Sequence, Union

from airio._src.core import data_sources
from airio._src.core import dataset_iterators
# Import "preprocessors" as "preprocessors_lib" to prevent naming conflicts with
# "preprocessors" attrs in this file.
from airio._src.core import preprocessors as preprocessors_lib
from airio._src.core import tokenizer
from airio._src.core import vocabularies
import grain.python as grain
import tensorflow_datasets as tfds


@dataclasses.dataclass(frozen=True)
class ShardInfo:
  """A container for specifying sharding info."""

  index: int
  num_shards: int


@typing.runtime_checkable
class DatasetProviderBase(Protocol):
  """Abstract base for classes that provide a dataset."""

  splits: Iterable[str] = None

  def get_dataset(
      self,
      sequence_lengths: Mapping[str, int] | None = None,
      split: str = tfds.Split.TRAIN,
      runtime_preprocessors: Sequence[grain.Transformation] | None = None,
      batch_size: int | None = None,
      shuffle: bool = True,
      seed: int | None = 0,
      shard_info: ShardInfo | None = None,
      num_epochs: int | None = 1,
      num_prefetch_threads: int | None = None,
      num_workers: int | None = 0,
  ) -> dataset_iterators.AirIODatasetIterator:
    """Returns the dataset iterator."""
    ...

  def num_input_examples(self, split: str) -> int | None:
    ...


@typing.runtime_checkable
class Task(DatasetProviderBase, Protocol):
  """Base class for tasks."""

  name: str
  source: data_sources.DataSource

  def __init__(
      self,
      name: str,
      source: data_sources.DataSource,
      preprocessors: Sequence[grain.Transformation] | None = None,
  ):
    self.name = name
    self.source = source
    self.splits = source.splits
    self._preprocessors = (
        list(preprocessors) if preprocessors is not None else []
    )

  def get_preprocessors(self) -> List[grain.Transformation]:
    if self._preprocessors is None:
      raise ValueError("Preprocessors have not been set on this task.")
    return list(self._preprocessors)

  def num_input_examples(self, split: str) -> int | None:
    if self.source is None:
      raise ValueError("Source has not been set on this task object.")
    return self.source.num_input_examples(split=split)


@typing.runtime_checkable
class Mixture(DatasetProviderBase, Protocol):
  """Base class for mixtures."""

  def __init__(
      self,
      name: str,
      tasks: Sequence[Union[Task, "Mixture"]],
      proportions: Sequence[float],
  ):
    if len(tasks) != len(proportions):
      raise ValueError(
          f"Mixture {name} must have same number of tasks and proportions."
          f"tasks: {tasks}, proportions: {proportions}."
      )
    all_tasks = [t for t in tasks if isinstance(t, Task)]
    all_mixtures = [m for m in tasks if isinstance(m, Mixture)]
    sub_tasks = [mix.leaf_tasks for mix in all_mixtures]
    leaf_tasks = sum(sub_tasks, all_tasks)
    duplicate_tasks = [
        t for t, c in collections.Counter(leaf_tasks).items() if c > 1
    ]
    if duplicate_tasks:
      raise ValueError(
          f"Mixture {name} has duplicate tasks: {duplicate_tasks}."
      )

    hashes = [hash(task) for task in tasks]

    self.name = name
    self._tasks_or_mixtures = dict(zip(hashes, tasks))
    self._proportions = dict(zip(hashes, proportions))

  def num_input_examples(self, split: str) -> int | None:
    return sum(
        t.num_input_examples(split)
        for t in self.tasks_or_mixtures
        if split in t.splits
    )

  def get_proportion(self, task: Task) -> float:
    """Computes the mixing proportion for the given task."""
    prop = 0.0
    task_hash = hash(task)
    if task_hash in self._proportions:
      prop += self._proportions[task_hash]

    if task not in self.leaf_tasks:
      return prop

    for sub_task in self.tasks_or_mixtures:
      if isinstance(sub_task, Mixture) and task in sub_task.leaf_tasks:
        prop += (
            self._proportions[hash(sub_task)]
            * sub_task.get_proportion(task)
            / sub_task.total_proportion
        )
    return prop

  @property
  def tasks_or_mixtures(self) -> Sequence[Union[Task, "Mixture"]]:
    """Tasks or Mixtures confiugured during Mixture init."""
    return list(self._tasks_or_mixtures.values())

  @functools.cached_property
  def leaf_tasks(self) -> Sequence[Task]:
    """Tasks contained in this Mixture."""
    all_ = self.tasks_or_mixtures
    tasks = [t for t in all_ if isinstance(t, Task)]
    mixtures = [m for m in all_ if isinstance(m, Mixture)]
    sub_tasks = [mix.leaf_tasks for mix in mixtures]
    return sum(sub_tasks, tasks)

  @property
  def total_proportion(self) -> float:
    return sum(self._proportions.values())

  @property
  def splits(self) -> Sequence[str]:
    splits = set()
    for task in self.tasks_or_mixtures:
      splits.update(task.splits)
    return tuple(splits)


@typing.runtime_checkable
class TaskBuilder(Protocol):
  """Builder class for building Task object.

  In order to create a Task object, build() method should be called on the
  TaskBuilder object after setting the appropriate data source and
  preprocessors.
  """

  def __init__(
      self,
      task_name: str,
      source: data_sources.DataSource | None = None,
      preprocessors: Sequence[grain.Transformation] | None = None,
  ):
    """Constructor for TaskBuilder.

    Args:
      task_name: Name of the task to be created.
      source: Data source for the task.
      preprocessors: List of the preprocessors for the task.
    """
    self._task_name = task_name
    self._source = source
    self._preprocessors = preprocessors

  def build(self) -> Task:
    """Returns a fully-defined Task object.

    Creates a new task object using properties of the current task builder
    object as long as neither of source and preprocessors is None.

    Raises:
      ValueError: when either of the source or preprocessors is None.
    """
    if self._source is None:
      raise ValueError("Source has not been set on this task builder.")
    if self._preprocessors is None:
      raise ValueError("Preprocessors have not been set on this task builder.")

    return Task(
        name=self._task_name,
        source=self._source,
        preprocessors=self._preprocessors,
    )

  def set_task_name(self, task_name: str) -> "TaskBuilder":
    self._task_name = task_name
    return self

  def set_data_source(self, source: data_sources.DataSource) -> "TaskBuilder":
    self._source = source
    return self

  def set_preprocessors(
      self, preprocessors: Sequence[grain.Transformation]
  ) -> "TaskBuilder":
    self._preprocessors = list(preprocessors)
    return self

  @classmethod
  def from_task(cls, task: Task) -> "TaskBuilder":
    """Returns TaskBuilder for the given existing Task object.

    This method takes an existing task, copies its properties into a new
    TaskBuilder object and returns it.

    Args:
      task: Existing task object.
    """
    return TaskBuilder(
        task_name=task.name,
        source=task.source,
        preprocessors=task.get_preprocessors(),
    )

  def __repr__(self) -> str:
    return (
        f"TaskBuilder(task_name={self._task_name}, source={self._source},"
        f" preprocessors={self._preprocessors})"
    )


def get_dataset(
    mixture_or_task: Task | Mixture,
    sequence_lengths: Mapping[str, int] | None = None,
    split: str = "train",
    runtime_preprocessors: Sequence[grain.Transformation] | None = None,
    batch_size: int | None = None,
    shuffle: bool = False,
    num_epochs: int | None = 1,
    seed: int | None = None,
    shard_info: ShardInfo | None = None,
    num_prefetch_threads: int | None = None,
    num_workers: int | None = 0,
) -> dataset_iterators.AirIODatasetIterator:
  """Returns the PyGrain dataset iterator."""
  return mixture_or_task.get_dataset(
      split=split,
      sequence_lengths=sequence_lengths,
      runtime_preprocessors=runtime_preprocessors,
      batch_size=batch_size,
      shuffle=shuffle,
      num_epochs=num_epochs,
      seed=seed,
      shard_info=shard_info,
      num_prefetch_threads=num_prefetch_threads,
      num_workers=num_workers,
  )


def get_vocabularies(
    mixture_or_task: Task | Mixture,
) -> Mapping[str, vocabularies.Vocabulary]:
  """Returns vocabularies for all features as configured in tokenizer."""
  if isinstance(mixture_or_task, Mixture):
    tasks = mixture_or_task.leaf_tasks
    if not tasks:
      return {}
    task = tasks[0]
  else:
    task = mixture_or_task

  vocabulary_map = {}
  for preproc in task.get_preprocessors():
    if isinstance(preproc, preprocessors_lib.MapFnTransform) and isinstance(
        preproc.map_fn, tokenizer.Tokenizer
    ):
      tokenizer_configs = preproc.map_fn.tokenizer_configs
      for feature_name, tokenizer_config in tokenizer_configs.items():
        vocabulary_map[feature_name] = tokenizer_config.vocab

  return vocabulary_map
