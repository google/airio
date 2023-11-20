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

import dataclasses
import functools
import typing
from typing import Any, Iterable, List, Mapping, Protocol, Sequence, Union

from airio import data_sources
from airio import dataset_iterators
from airio import feature_converters
from airio import lazy_dataset_transforms
# Import "preprocessors" as "preprocessors_lib" to prevent naming conflicts with
# "preprocessors" attrs in this file.
from airio import preprocessors as preprocessors_lib
from airio import tokenizer
from clu.data import dataset_iterator as clu_dataset_iterator
import grain.python as grain
import jax.random
from seqio import vocabularies
import tensorflow_datasets as tfds

lazy_dataset = grain.experimental.lazy_dataset
SHUFFLE_BUFFER_SIZE = 1000
DEFAULT_NUM_RECORDS_TO_INSPECT = 2
MAX_NUM_RECORDS_TO_INSPECT = 1000

# TODO(sahildua): Expose these data sources as AirIO data sources?
GrainDataSource = grain.RandomAccessDataSource
AirIOPreprocessor = preprocessors_lib.AirIOPreprocessor


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
      feature_converter: feature_converters.PyGrainFeatureConverter
      | None = None,
      batch_size: int | None = None,
      shuffle: bool = True,
      seed: int | None = 0,
      shard_info: ShardInfo | None = None,
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
      source: data_sources.DataSource,
      preprocessors: Sequence[AirIOPreprocessor] | None = None,
  ):
    self.name = name
    self.source = source
    self.splits = source.splits
    self._preprocessors = (
        list(preprocessors) if preprocessors is not None else []
    )

  def get_preprocessors(self) -> List[AirIOPreprocessor]:
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

  def _switch_to_lazy_dataset(self):
    # TODO(b/311720936): Until Task preprocessing is fully switched to
    # lazy_dataset, check and use lazy_dataset only if any preprocessor requires
    # lazy_dataset.
    for preprocessor in self.get_preprocessors():
      if not isinstance(preprocessor, grain.Transformation):
        return True
    return False

  def get_lazy_dataset(
      self,
      sequence_lengths: Mapping[str, int] | None,
      split: str,
      feature_converter: feature_converters.PyGrainFeatureConverter | None,
      batch_size: int | None,
      shuffle: bool,
      seed: int | None,
      shard_info: ShardInfo | None,
      num_epochs: int | None,
  ) -> lazy_dataset.LazyMapDataset:
    """Returns a lazy dataset for Task source and preprocessors."""
    # Step 1: Get Source.
    ds = lazy_dataset.SourceLazyMapDataset(
        self._get_data_source_for_split(split=split)
    )
    if shard_info:
      shard_options = grain.ShardOptions(
          shard_index=shard_info.index, shard_count=shard_info.num_shards
      )
      ds = lazy_dataset_transforms.ShardLazyMapDataset(ds, shard_options)

    # Step 2: Make epochs.
    if num_epochs:
      dss = [ds] * num_epochs
    else:
      # Skip repeating here, repeat the mixed dataset.
      dss = [ds]

    # Step 3: Run preprocessors and shuffle each epoch (if needed)
    preps = self._preprocessors
    if feature_converter is not None:
      preps.extend(feature_converter.get_transforms(sequence_lengths))
    if batch_size:
      preps.append(grain.Batch(batch_size=batch_size, drop_remainder=False))
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths=sequence_lengths, split=split
    )
    preprocessed_dss = []
    next_epoch_rng = jax.random.PRNGKey(seed)
    for ds in dss:
      next_epoch_rng, prep_rng = jax.random.split(next_epoch_rng)
      prep_rng, shuffle_rng = jax.random.split(prep_rng)
      for prep in preps:
        ds, runtime_args = preprocessors_lib.LazyDatasetTransform(prep)(
            ds, prep_rng, runtime_args
        )
        prep_rng, _ = jax.random.split(prep_rng)
      if shuffle:
        shuffle_seed = int(jax.random.randint(shuffle_rng, [], 0, 2**16 - 1))
        ds = lazy_dataset.ShuffleLazyMapDataset(ds, seed=shuffle_seed)
      preprocessed_dss.append(ds)
    # pylint:enable=protected-access

    # Step 4: Combine epochs if needed
    if len(preprocessed_dss) == 1:
      return preprocessed_dss[0]
    return lazy_dataset_transforms.ConcatLazyMapDataset(preprocessed_dss)

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
      shard_info: ShardInfo | None = None,
      num_epochs: int | None = 1,
  ) -> clu_dataset_iterator.DatasetIterator:
    """Returns the dataset iterator as per the task configuration."""
    # TODO(b/311720936): Until Task preprocessing is fully switched to
    # lazy_dataset, check and use lazy_dataset only if any preprocessor requires
    # lazy_dataset.
    if self._switch_to_lazy_dataset():
      ds = self.get_lazy_dataset(
          sequence_lengths=sequence_lengths,
          split=split,
          feature_converter=feature_converter,
          batch_size=batch_size,
          shuffle=shuffle,
          seed=seed,
          shard_info=shard_info,
          num_epochs=num_epochs,
      )
      # The sampler below is a no-op because sharding, shuffling and repeats are
      # done using the lazy_dataset API. It may be removed when lazy_dataset
      # releases.
      sampler = grain.IndexSampler(
          num_records=len(ds),
          shard_options=grain.NoSharding(),
          shuffle=False,
          num_epochs=1,
          seed=seed,
      )
      return self._load_data(source=ds, sampler=sampler, ops=[])
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
          feature_converter.get_transforms(
              task_feature_lengths=sequence_lengths
          )
      )
    if batch_size:
      ops.append(grain.Batch(batch_size=batch_size, drop_remainder=False))

    # Add runtime args
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths=sequence_lengths, split=split
    )
    for op in ops:
      if isinstance(op, preprocessors_lib.FnTransforms):
        op.runtime_args = runtime_args

    return self._load_data(source=source, sampler=sampler, ops=ops)

  def _load_data(
      self,
      source: GrainDataSource,
      sampler: grain.IndexSampler,
      ops: Sequence[grain.Transformation],
  ) -> clu_dataset_iterator.DatasetIterator:
    """Returns a sampled data source after applying `ops`.

    A helper function for get_dataset and get_dataset_by_step.

    Args:
      source: a data source to load.
      sampler: a means of sampling from the source.
      ops: a list of transformations to apply. Only `grain.Transformation`s are
        allowed because these are passed to the operations field in
        `grain.DataLoader`. Other types allowed by `AirIOPreprocessor` need to
        be mapped to one or more `grain.Transformation`s or handled before this
        call. Once the switch to lazy_dataset API is complete, this arg will be
        removed and all transformations will be baked into the `source` arg.

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
    # TODO(b/311720936): Add support for preprocessing using lazy_dataset.
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
          feature_converter.get_transforms(
              task_feature_lengths=sequence_lengths
          )
      )
    if batch_size:
      all_ops.append(grain.Batch(batch_size=batch_size, drop_remainder=False))

    # Raw data
    records_step0 = self._load_data(source=source, sampler=sampler, ops=[])
    accumulated_result = [list(records_step0)]

    if not all_ops:
      return accumulated_result

    # Apply all transformations, one by one.
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths=sequence_lengths, split=split
    )
    accumulated_ops = []
    for op in all_ops:
      if isinstance(op, preprocessors_lib.FnTransforms):
        op.runtime_args = runtime_args
      accumulated_ops.append(op)
      records_next_step = self._load_data(
          source=source, sampler=sampler, ops=accumulated_ops
      )
      accumulated_result.append(list(records_next_step))

    return accumulated_result


class Mixture(DatasetProviderBase):
  """A class for mixture of Tasks."""

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
    hashes = [hash(task) for task in tasks]
    if len(set(hashes)) != len(tasks):
      raise ValueError(f"Mixture {name} has duplicate tasks. tasks: {tasks}.")

    self.name = name
    self._tasks_or_mixtures = dict(zip(hashes, tasks))
    self._proportions = dict(zip(hashes, proportions))

  def get_lazy_dataset(
      self,
      sequence_lengths: Mapping[str, int] | None = None,
      split: str = tfds.Split.TRAIN,
      feature_converter: feature_converters.PyGrainFeatureConverter
      | None = None,
      batch_size: int | None = None,
      shuffle: bool = True,
      seed: int | None = 0,
      shard_info: ShardInfo | None = None,
      num_epochs: int | None = 1,
  ) -> lazy_dataset.LazyMapDataset:
    """Returns a lazy dataset for the Mixture."""
    if num_epochs is None and shuffle:
      raise ValueError(
          "Repeating indefinitely with shuffling turned on isn't supported."
      )
    datasets = []
    proportions = []
    for task in self.leaf_tasks:
      datasets.append(
          task.get_lazy_dataset(
              sequence_lengths=sequence_lengths,
              split=split,
              feature_converter=None,
              batch_size=None,
              shuffle=shuffle,
              seed=seed,
              shard_info=shard_info,
              num_epochs=num_epochs,
          )
      )
      proportions.append(self.get_proportion(task))
      # Note: We will run feature converter on and batch the mixed dataset, but
      # these can be done before mixing by setting feature_converter and
      # batch_size above and disabling them below if needed in the future.
      # Note: We may not need N epochs of a Task to populate N epochs of the
      # Mixture, but since these are lazily populated, we can skip calculating
      # the exact number of epochs required.
    ds = lazy_dataset_transforms.MixedLazyMapDataset(
        datasets, proportions, stop_on_empty_dataset=True
    )
    post_mix_preps = []
    if feature_converter:
      post_mix_preps.extend(feature_converter.get_transforms(sequence_lengths))
    if batch_size:
      post_mix_preps.append(
          grain.Batch(batch_size=batch_size, drop_remainder=False)
      )
    if post_mix_preps:
      runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
          sequence_lengths=sequence_lengths, split=split
      )
      post_mix_transforms = [
          preprocessors_lib.LazyDatasetTransform(p) for p in post_mix_preps
      ]
      for t in post_mix_transforms:
        ds, runtime_args = t(ds, runtime_args=runtime_args)
    if num_epochs is None:
      ds = lazy_dataset.RepeatLazyMapDataset(ds, num_epochs=None)
    return ds

  def get_dataset(
      self,
      sequence_lengths: Mapping[str, int] | None = None,
      split: str = tfds.Split.TRAIN,
      feature_converter: feature_converters.PyGrainFeatureConverter
      | None = None,
      batch_size: int | None = None,
      shuffle: bool = True,
      seed: int | None = 0,
      shard_info: ShardInfo | None = None,
      num_epochs: int | None = 1,
  ) -> clu_dataset_iterator.DatasetIterator:
    """Returns the dataset iterator."""
    ds = self.get_lazy_dataset(
        sequence_lengths=sequence_lengths,
        split=split,
        feature_converter=feature_converter,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        shard_info=shard_info,
        num_epochs=num_epochs,
    )
    # The sampler below is a no-op because sharding, shuffling and repeats are
    # done using the lazy_dataset API. It may be removed when lazy_dataset
    # releases.
    sampler = grain.IndexSampler(
        num_records=len(ds),
        shard_options=grain.NoSharding(),
        shuffle=False,
        num_epochs=1,
        seed=seed,
    )
    ds = grain.DataLoader(
        data_source=ds,
        sampler=sampler,
        operations=[],
    )
    return dataset_iterators.PyGrainDatasetIteratorWrapper(data_loader=ds)

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
    sub_tasks = [mix.leaf_tasks for mix in mixtures]  # pytype: disable=attribute-error
    return list(sorted(set(sum(sub_tasks, tasks)), key=lambda t: t.name))

  @property
  def total_proportion(self) -> float:
    return sum(self._proportions.values())

  @property
  def splits(self) -> Sequence[str]:
    splits = set()
    for task in self.tasks_or_mixtures:
      splits.update(task.splits)
    return tuple(splits)


class TaskBuilder:
  """Builder class for building Task object.

  In order to create a Task object, build() method should be called on the
  TaskBuilder object after setting the appropriate data source and
  preprocessors.
  """

  def __init__(
      self,
      task_name: str,
      source: data_sources.DataSource | None = None,
      preprocessors: Sequence[AirIOPreprocessor] | None = None,
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
      self, preprocessors: Sequence[AirIOPreprocessor]
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


def get_vocabularies(
    mixture_or_task: Union[Task, Mixture]
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
