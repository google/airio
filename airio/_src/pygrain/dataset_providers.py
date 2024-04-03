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

"""Classes for AirIO data loading with Grain."""

import functools
from typing import Any, Iterable, Mapping, Sequence, Union, cast

from airio._src.core import data_sources
from airio._src.core import dataset_providers as core_dataset_providers
# Import "preprocessors" as "preprocessors_lib" to prevent naming conflicts with
# "preprocessors" attrs in this file.
from airio._src.core import preprocessors as core_preprocessors_lib
from airio._src.pygrain import dataset_iterators
from airio._src.pygrain import lazy_dataset_transforms
from airio._src.pygrain import preprocessors as preprocessors_lib
from clu.data import dataset_iterator as clu_dataset_iterator
import grain.python as grain
import jax.random
import tensorflow_datasets as tfds


SHUFFLE_BUFFER_SIZE = 1000
DEFAULT_NUM_RECORDS_TO_INSPECT = 2
MAX_NUM_RECORDS_TO_INSPECT = 1000

lazy_dataset = grain.experimental.lazy_dataset
# TODO(sahildua): Expose these data sources as AirIO data sources?
GrainDataSource = grain.RandomAccessDataSource
JaxRng = jax.Array


class GrainTask(core_dataset_providers.Task):
  """A class to manage a dataset and its related metrics."""

  name: str
  source: data_sources.DataSource

  def _get_data_source_for_split(self, split: str) -> GrainDataSource:
    if self.source is None:
      raise ValueError("Source has not been set on this task object.")
    return self.source.get_data_source(split=split)

  def _switch_to_lazy_dataset(
      self,
      runtime_preprocessors: (
          Sequence[preprocessors_lib.PyGrainAirIOPreprocessor] | None
      ) = None,
  ):
    # TODO(b/311720936): Until Task preprocessing is fully switched to
    # lazy_dataset, check and use lazy_dataset only if any preprocessor requires
    # lazy_dataset.
    preps = self.get_preprocessors()
    if runtime_preprocessors:
      preps.extend(runtime_preprocessors)
    for preprocessor in preps:
      if not isinstance(preprocessor, grain.Transformation):
        return True
    return False

  def _apply_preprocessors_to_lazy_dataset(
      self,
      ds,
      preps: Sequence[preprocessors_lib.PyGrainAirIOPreprocessor],
      runtime_args: core_preprocessors_lib.AirIOInjectedRuntimeArgs,
      base_rng: JaxRng,
      has_none_elems: bool,
      num_prefetch_threads: int | None,
  ):
    prep_rng = base_rng
    for prep in preps:
      transform = preprocessors_lib.LazyDatasetTransform(prep)
      if transform.requires_iter_dataset and isinstance(
          ds, lazy_dataset.LazyMapDataset
      ):
        ds = ds.to_iter_dataset(_get_read_options(num_prefetch_threads))
      if (
          has_none_elems
          and transform.requires_non_none_elements
          and isinstance(ds, lazy_dataset.LazyMapDataset)
      ):
        if not transform.can_process_iter_dataset:
          raise ValueError(
              "There are preprocessors in this Task that produce none elements"
              " and a preprocessor which requires a non-none dataset. Consider"
              " replacing this with a preprocessor that can either handle none"
              " elements, or process a LazyIterDataset. Task name:"
              f" {self.name}, preprocessor: {prep}"
          )
        ds = ds.to_iter_dataset(_get_read_options(num_prefetch_threads))
      if transform.produces_none_elements:
        has_none_elems = True
      prep_rng, sub_rng = jax.random.split(prep_rng)
      ds = transform(ds, sub_rng, runtime_args)
      runtime_args = transform.get_updated_runtime_args(runtime_args)
    return ds, runtime_args, has_none_elems

  def get_lazy_dataset(
      self,
      sequence_lengths: Mapping[str, int] | None,
      split: str,
      runtime_preprocessors: (
          Sequence[preprocessors_lib.PyGrainAirIOPreprocessor] | None
      ),
      batch_size: int | None,
      shuffle: bool,
      seed: int | None,
      shard_info: core_dataset_providers.ShardInfo | None,
      num_epochs: int | None,
      num_prefetch_threads: int | None,
  ) -> lazy_dataset.LazyMapDataset | lazy_dataset.LazyIterDataset:
    """Returns a lazy dataset for Task source and preprocessors."""
    # Step 1: Get Source.
    ds = lazy_dataset.SourceLazyMapDataset(
        self._get_data_source_for_split(split=split)
    )
    if shard_info:
      start, end = _even_split(
          len(ds),
          shard_index=shard_info.index,
          shard_count=shard_info.num_shards,
      )
      ds = ds[start:end]

    # Step 2: Make epochs.
    if num_epochs:
      dss = [ds] * num_epochs
    else:
      # Skip repeating here, repeat the mixed dataset.
      dss = [ds]

    # Step 3: Run preprocessors and shuffle each epoch (if needed)
    preps = self._preprocessors
    updated_runtime_args = core_preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths=sequence_lengths,
        split=split,
        batch_size=batch_size,
    )
    preprocessed_dss = []
    has_none_elems = False
    next_epoch_rng = jax.random.key(seed)
    for ds in dss:
      ds_runtime_args = core_preprocessors_lib.AirIOInjectedRuntimeArgs(
          sequence_lengths=sequence_lengths,
          split=split,
          batch_size=batch_size,
      )
      next_epoch_rng, prep_rng = jax.random.split(next_epoch_rng)
      prep_rng, shuffle_rng = jax.random.split(prep_rng)
      ds, updated_runtime_args, has_none_elems = (
          self._apply_preprocessors_to_lazy_dataset(
              ds,
              preps,
              ds_runtime_args,
              prep_rng,
              has_none_elems,
              num_prefetch_threads,
          )
      )
      if shuffle:
        shuffle_seed = int(jax.random.randint(shuffle_rng, [], 0, 2**16 - 1))
        ds = lazy_dataset.ShuffleLazyMapDataset(ds, seed=shuffle_seed)
      preprocessed_dss.append(ds)
    runtime_args = updated_runtime_args
    # pylint:enable=protected-access

    # Step 4: Combine epochs if needed
    if len(preprocessed_dss) == 1:
      ds = preprocessed_dss[0]
    else:
      ds = lazy_dataset_transforms.ConcatLazyMapDataset(preprocessed_dss)

    # Step 5: Apply runtime preprocessors and batching, if needed
    runtime_preps = []
    if runtime_preprocessors:
      runtime_preps.extend(runtime_preprocessors)
    if batch_size:
      runtime_preps.append(
          grain.Batch(batch_size=batch_size, drop_remainder=False)
      )
    unused_next_epoch_rng, prep_rng = jax.random.split(next_epoch_rng)
    ds, _, _ = self._apply_preprocessors_to_lazy_dataset(
        ds,
        runtime_preps,
        runtime_args,
        prep_rng,
        has_none_elems,
        num_prefetch_threads,
    )
    return ds

  # TODO(sahildua): Add logging.
  def get_dataset(
      self,
      sequence_lengths: Mapping[str, int] | None = None,
      split: str = tfds.Split.TRAIN,
      runtime_preprocessors: (
          Sequence[preprocessors_lib.PyGrainAirIOPreprocessor] | None
      ) = None,
      batch_size: int | None = None,
      shuffle: bool = True,
      seed: int | None = 0,
      shard_info: core_dataset_providers.ShardInfo | None = None,
      num_epochs: int | None = 1,
      num_prefetch_threads: int | None = None,
      num_workers: int | None = 0,
  ) -> clu_dataset_iterator.DatasetIterator:
    """Returns the dataset iterator as per the task configuration."""
    # TODO(b/311720936): Until Task preprocessing is fully switched to
    # lazy_dataset, check and use lazy_dataset only if any preprocessor requires
    # lazy_dataset.
    if self._switch_to_lazy_dataset(runtime_preprocessors):
      ds = self.get_lazy_dataset(
          sequence_lengths=sequence_lengths,
          split=split,
          runtime_preprocessors=runtime_preprocessors,
          batch_size=batch_size,
          shuffle=shuffle,
          seed=seed,
          shard_info=shard_info,
          num_epochs=num_epochs,
          num_prefetch_threads=num_prefetch_threads,
      )
      ds = _iter_and_prefetch(
          ds, num_workers=num_workers, num_prefetch_threads=num_prefetch_threads
      )
      return dataset_iterators.PyGrainDatasetIteratorWrapper(data_loader=ds)
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
    if runtime_preprocessors:
      ops.extend(runtime_preprocessors)
    if batch_size:
      ops.append(grain.Batch(batch_size=batch_size, drop_remainder=False))

    # Add runtime args
    runtime_args = core_preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths=sequence_lengths,
        split=split,
        batch_size=batch_size,
    )
    for op in ops:
      if isinstance(op, preprocessors_lib.FnTransforms):
        op.runtime_args = runtime_args

    return self._load_data(
        source=source,
        sampler=sampler,
        ops=ops,
        num_prefetch_threads=num_prefetch_threads,
        num_workers=num_workers,
    )

  def _load_data(
      self,
      source: GrainDataSource,
      sampler: grain.IndexSampler,
      ops: Sequence[grain.Transformation],
      num_prefetch_threads: int | None = None,
      num_workers: int | None = 0,
  ) -> clu_dataset_iterator.DatasetIterator:
    """Returns a sampled data source after applying `ops`.

    A helper function for get_dataset and get_dataset_by_step.

    Args:
      source: a data source to load.
      sampler: a means of sampling from the source.
      ops: a list of transformations to apply. Only `grain.Transformation`s are
        allowed because these are passed to the operations field in
        `grain.DataLoader`. Other types allowed by
        `preprocessors_lib.PyGrainAirIOPreprocessor` need to be mapped to one or
        more `grain.Transformation`s or handled before this call. Once the
        switch to lazy_dataset API is complete, this arg will be removed and all
        transformations will be baked into the `source` arg.
      num_prefetch_threads: Number of threads reading from the DataSource in
        parallel.
      num_workers: Number of child processes launched to parallelize the
        transformations among. Zero means processing runs in the same process.
        None lets the python backend choose the value.

    Returns an iterator of records after applying `ops`.
    """
    ds = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=ops,
        worker_count=num_workers,
        read_options=_get_read_options(num_prefetch_threads),
    )
    return dataset_iterators.PyGrainDatasetIteratorWrapper(data_loader=ds)

  def get_dataset_by_step(
      self,
      num_records: int = DEFAULT_NUM_RECORDS_TO_INSPECT,
      sequence_lengths: Mapping[str, int] | None = None,
      split: str = tfds.Split.TRAIN,
      runtime_preprocessors: (
          Sequence[preprocessors_lib.PyGrainAirIOPreprocessor] | None
      ) = None,
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
      runtime_preprocessors: A list of preprocessors to apply before batching.
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
    if runtime_preprocessors:
      all_ops.extend(runtime_preprocessors)
    if batch_size:
      all_ops.append(grain.Batch(batch_size=batch_size, drop_remainder=False))

    # Raw data
    records_step0 = self._load_data(source=source, sampler=sampler, ops=[])
    accumulated_result = [list(records_step0)]

    if not all_ops:
      return accumulated_result

    # Apply all transformations, one by one.
    runtime_args = core_preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths=sequence_lengths,
        split=split,
        batch_size=batch_size,
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

  def get_updated_runtime_args(
      self,
      runtime_args: core_preprocessors_lib.AirIOInjectedRuntimeArgs,
      runtime_preprocessors: (
          Sequence[preprocessors_lib.PyGrainAirIOPreprocessor] | None
      ),
  ) -> core_preprocessors_lib.AirIOInjectedRuntimeArgs:
    """Returns updated runtime args based on preprocessors and feature converter."""
    preps = self._preprocessors
    if runtime_preprocessors:
      preps.extend(runtime_preprocessors)
    for prep in preps:
      transform = preprocessors_lib.LazyDatasetTransform(prep)
      runtime_args = transform.get_updated_runtime_args(runtime_args)
    return runtime_args

  @functools.cached_property
  def produces_none_elements(self) -> bool:
    """Returns True if any preprocessor returns none elements, excluding runtime preprocessors and batching.

    This is a best-effort check and may be wrong.
    """
    return any([
        preprocessors_lib.LazyDatasetTransform(prep).produces_none_elements
        for prep in self._preprocessors
    ])


class GrainMixture(core_dataset_providers.Mixture):
  """A class for mixture of Tasks."""

  def __init__(
      self,
      name: str,
      tasks: Sequence[Union[GrainTask, "GrainMixture"]],
      proportions: Sequence[float],
  ):
    for task in tasks:
      if not isinstance(task, (GrainTask, GrainMixture)):
        raise ValueError(
            f"Task '{task.name}' is not a GrainTask or GrainMixture."
        )

    super(GrainMixture, self).__init__(
        name=name, tasks=tasks, proportions=proportions
    )

  def get_lazy_dataset(
      self,
      sequence_lengths: Mapping[str, int] | None = None,
      split: str = tfds.Split.TRAIN,
      runtime_preprocessors: (
          Sequence[preprocessors_lib.PyGrainAirIOPreprocessor] | None
      ) = None,
      batch_size: int | None = None,
      shuffle: bool = True,
      seed: int | None = 0,
      shard_info: core_dataset_providers.ShardInfo | None = None,
      num_epochs: int | None = 1,
      num_prefetch_threads: int | None = None,
  ) -> lazy_dataset.LazyMapDataset | lazy_dataset.LazyIterDataset:
    """Returns a lazy dataset for the Mixture."""
    if num_epochs is None and shuffle:
      raise ValueError(
          "Repeating indefinitely with shuffling turned on isn't supported."
      )
    datasets = []
    proportions = []
    for task in self.leaf_tasks:
      if not isinstance(task, GrainTask):
        raise ValueError(f"Task {task} is not a GrainTask.")
      datasets.append(
          task.get_lazy_dataset(
              sequence_lengths=sequence_lengths,
              split=split,
              runtime_preprocessors=None,
              batch_size=None,
              shuffle=shuffle,
              seed=seed,
              shard_info=shard_info,
              num_epochs=num_epochs,
              num_prefetch_threads=num_prefetch_threads,
          )
      )
      proportions.append(self.get_proportion(task))
      # Note: We will run feature converter on and batch the mixed dataset, but
      # these can be done before mixing by setting feature_converter and
      # batch_size above and disabling them below if needed in the future.
      # Note: We may not need N epochs of a Task to populate N epochs of the
      # Mixture, but since these are lazily populated, we can skip calculating
      # the exact number of epochs required.
    # If any Task dataset produces a LazyIterDataset, then use a LazyIter impl
    # for mixing.
    use_mix_iter = any(
        [isinstance(ds, lazy_dataset.LazyIterDataset) for ds in datasets]
    )
    # If any Task dataset produces None elements, mixing it with a LazyMap impl
    # will deviate from desired proportions because None elements will be
    # sampled. A LazyIter impl for mixing must be used for correct behavior.
    use_mix_iter = use_mix_iter or any([
        cast(GrainTask, task).produces_none_elements for task in self.leaf_tasks
    ])

    if use_mix_iter:
      # Convert any LazyMapDatasets datasets to LazyIterDatasets.
      read_options = _get_read_options(num_prefetch_threads)
      datasets = [
          ds.to_iter_dataset(read_options)
          if isinstance(ds, lazy_dataset.LazyMapDataset)
          else ds
          for ds in datasets
      ]
      ds = lazy_dataset.MixedLazyIterDataset(datasets, proportions)
    else:
      ds = lazy_dataset.MixedLazyMapDataset(datasets, proportions)

    post_mix_preps = []
    if runtime_preprocessors:
      post_mix_preps.extend(runtime_preprocessors)
    if batch_size:
      post_mix_preps.append(
          grain.Batch(batch_size=batch_size, drop_remainder=False)
      )
    # Note: Use updated runtime args from the first Task. All updated runtime
    # args must match, or mixing won't work (compute all updated runtime args
    # and add a check here in the future if helpful).
    runtime_args = core_preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths=sequence_lengths,
        split=split,
        batch_size=batch_size,
    )
    if isinstance(self.leaf_tasks[0], GrainTask):
      runtime_args = self.leaf_tasks[0].get_updated_runtime_args(
          runtime_args, runtime_preprocessors=None
      )
    if post_mix_preps:
      post_mix_transforms = [
          preprocessors_lib.LazyDatasetTransform(p) for p in post_mix_preps
      ]
      for t in post_mix_transforms:
        ds = t(ds, runtime_args=runtime_args)
        runtime_args = t.get_updated_runtime_args(runtime_args)
    if num_epochs is None:
      ds = lazy_dataset.RepeatLazyMapDataset(ds, num_epochs=None)
    return ds

  def get_dataset(
      self,
      sequence_lengths: Mapping[str, int] | None = None,
      split: str = tfds.Split.TRAIN,
      runtime_preprocessors: (
          Sequence[preprocessors_lib.PyGrainAirIOPreprocessor] | None
      ) = None,
      batch_size: int | None = None,
      shuffle: bool = True,
      seed: int | None = 0,
      shard_info: core_dataset_providers.ShardInfo | None = None,
      num_epochs: int | None = 1,
      num_prefetch_threads: int | None = None,
      num_workers: int | None = 0,
  ) -> clu_dataset_iterator.DatasetIterator:
    """Returns the dataset iterator."""
    ds = self.get_lazy_dataset(
        sequence_lengths=sequence_lengths,
        split=split,
        runtime_preprocessors=runtime_preprocessors,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        shard_info=shard_info,
        num_epochs=num_epochs,
        num_prefetch_threads=num_prefetch_threads,
    )
    ds = _iter_and_prefetch(
        ds, num_workers=num_workers, num_prefetch_threads=num_prefetch_threads
    )
    return dataset_iterators.PyGrainDatasetIteratorWrapper(data_loader=ds)


class GrainTaskBuilder(core_dataset_providers.TaskBuilder):
  """Builder class for building GrainTask object.

  In order to create a GrainTask object, build() method should be called on the
  TaskBuilder object after setting the appropriate data source and
  preprocessors.
  """

  def build(self) -> GrainTask:
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

    return GrainTask(
        name=self._task_name,
        source=self._source,
        preprocessors=self._preprocessors,
    )

  @classmethod
  def from_task(cls, task: GrainTask) -> "GrainTaskBuilder":
    """Returns TaskBuilder for the given existing Task object.

    This method takes an existing task, copies its properties into a new
    TaskBuilder object and returns it.

    Args:
      task: Existing task object.
    """
    return GrainTaskBuilder(
        task_name=task.name,
        source=task.source,
        preprocessors=task.get_preprocessors(),
    )

  def __repr__(self) -> str:
    return (
        f"GrainTaskBuilder(task_name={self._task_name}, source={self._source},"
        f" preprocessors={self._preprocessors})"
    )


def _even_split(
    num_examples: int, shard_index: int, shard_count: int
) -> tuple[int, int]:
  """Returns the interval for the shard when sharding `num_examples` evenly.

  Args:
    num_examples: Number of examples to shard.
    shard_index: The shard index to return interval for.
    shard_count: The total number of shards.

  Returns:
    Tuple with the start and end of the interval. The start is the first
    example that should be included in this interval and end - 1 is the last
    example to be include in the shard.
  """
  examples_per_shard = num_examples // shard_count
  shard_start = examples_per_shard * shard_index
  shard_end = examples_per_shard * (shard_index + 1)

  # Handle remaining examples.
  num_unused_examples = num_examples % shard_count
  if num_unused_examples > 0:
    shard_start += min(shard_index, num_unused_examples)
    shard_end += min(shard_index + 1, num_unused_examples)
  return shard_start, shard_end


def _get_read_options(num_prefetch_threads: int | None) -> grain.ReadOptions:
  if num_prefetch_threads:
    return grain.ReadOptions(num_threads=num_prefetch_threads)
  return grain.ReadOptions()


def _iter_and_prefetch(
    ds: lazy_dataset.LazyMapDataset | lazy_dataset.LazyIterDataset,
    num_workers: int | None,
    num_prefetch_threads: int | None,
) -> lazy_dataset.LazyIterDataset:
  if not isinstance(ds, lazy_dataset.LazyIterDataset):
    ds = ds.to_iter_dataset(_get_read_options(num_prefetch_threads))
  if num_workers and num_workers > 0:
    ds = ds.prefetch(grain.MultiprocessingOptions(num_workers=num_workers))
  return ds
