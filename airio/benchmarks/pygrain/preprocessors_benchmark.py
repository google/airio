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

"""Microbenchmarks for AirIO preprocessors functions."""


import airio.pygrain as airio
import google_benchmark
import grain.python as grain
import jax
import numpy as np


_SOURCE_NUM_EXAMPLES = 5


def _lazy_map_fn(
    ds: grain.MapDataset,
    run_args: airio.AirIOInjectedRuntimeArgs,
    unused_rng: jax.Array,
) -> grain.MapDataset:
  return ds.map(lambda x: x + run_args.sequence_lengths["val"])


def _lazy_iter_fn(
    ds: grain.IterDataset,
    run_args: airio.AirIOInjectedRuntimeArgs,
    unused_rng: jax.Array,
) -> grain.IterDataset:
  return ds.map(lambda x: x + run_args.sequence_lengths["val"])


def _get_source() -> airio.FunctionDataSource:
  def _dataset_fn(split: str):
    del split
    return np.array(range(_SOURCE_NUM_EXAMPLES))

  return airio.FunctionDataSource(dataset_fn=_dataset_fn, splits=["train"])


def _get_runtime_args() -> airio.AirIOInjectedRuntimeArgs:
  return airio.AirIOInjectedRuntimeArgs(
      sequence_lengths={"val": 3},
      split="train",
      batch_size=None,
  )


def _update_runtime_args(
    run_args: airio.AirIOInjectedRuntimeArgs,
) -> airio.AirIOInjectedRuntimeArgs:
  new_seq_lens = {}
  for k, v in run_args.sequence_lengths.items():
    new_seq_lens[f"{k}_new"] = v
    new_seq_lens[k] = v + 1
  return run_args.replace(sequence_lengths=new_seq_lens)


@google_benchmark.register
def map_fn_preprocessor(state: google_benchmark.State) -> None:
  task = airio.GrainTask(
      name="test_task",
      source=_get_source(),
      preprocessors=[airio.MapFnTransform(lambda x: x + 1)],
  )
  while state:
    task.get_dataset(None, "train", shuffle=False)


@google_benchmark.register
def random_map_fn_preprocessor(state: google_benchmark.State) -> None:
  """Analogous to the PreprocessorsTest with the same name."""
  task = airio.GrainTask(
      name="test_task",
      source=_get_source(),
      preprocessors=[
          airio.RandomMapFnTransform(
              lambda x, rng: x + int(jax.random.randint(rng, [], 0, 10))
          )
      ],
  )
  while state:
    task.get_dataset(None, "train", shuffle=False, seed=42)


@google_benchmark.register
def filter_fn_preprocessor(state: google_benchmark.State) -> None:
  """Analogous to the PreprocessorsTest with the same name."""
  task = airio.GrainTask(
      name="test_task",
      source=_get_source(),
      preprocessors=[airio.FilterFnTransform(lambda x: x > 2)],
  )
  while state:
    task.get_dataset(None, "train", shuffle=False, seed=42)


@google_benchmark.register
def preprocessor_empty_preprocessed(state: google_benchmark.State) -> None:
  """Analogous to the PreprocessorsTest with the same name."""
  task = airio.GrainTask(
      name="test_task",
      source=_get_source(),
      preprocessors=[airio.FilterFnTransform(lambda x: x > 1000)],
  )
  while state:
    task.get_dataset(None, "train", shuffle=False, seed=42)


@google_benchmark.register
def preprocessor_empty_intermediates(state: google_benchmark.State) -> None:
  """Analogous to the PreprocessorsTest with the same name."""
  task = airio.GrainTask(
      name="test_task",
      source=_get_source(),
      preprocessors=[
          airio.FilterFnTransform(lambda x: x > 1000),
          airio.MapFnTransform(lambda x: x + 1),
      ],
  )
  while state:
    task.get_dataset(None, "train", shuffle=False, seed=42)


@google_benchmark.register
def map_lazydataset_transform(state: google_benchmark.State) -> None:
  """Analogous to the PreprocessorsTest with the same name."""
  transform = airio.MapFnTransform(lambda x: x + 1)
  lazy_dataset_transform = airio.preprocessors.LazyDatasetTransform(transform)
  ds = grain.MapDataset.source(list(range(_SOURCE_NUM_EXAMPLES)))
  while state:
    lazy_dataset_transform(ds)


@google_benchmark.register
def random_map_fn_lazydataset_transform(state: google_benchmark.State) -> None:
  """Analogous to the PreprocessorsTest with the same name."""
  transform = airio.RandomMapFnTransform(
      lambda x, rng: x + int(jax.random.randint(rng, [], 0, 10))
  )
  lazy_dataset_transform = airio.preprocessors.LazyDatasetTransform(transform)
  ds = grain.MapDataset.source(list(range(_SOURCE_NUM_EXAMPLES)))
  while state:
    lazy_dataset_transform(ds, rng=jax.random.key(42))


@google_benchmark.register
def filter_lazydataset_transform(state: google_benchmark.State) -> None:
  """Analogous to the PreprocessorsTest with the same name."""
  transform = airio.FilterFnTransform(lambda x: x > 2)
  lazy_dataset_transform = airio.preprocessors.LazyDatasetTransform(transform)
  ds = grain.MapDataset.source(list(range(_SOURCE_NUM_EXAMPLES)))
  while state:
    lazy_dataset_transform(ds)


@google_benchmark.register
def batch_lazydataset_transform_with_drop_remainder(
    state: google_benchmark.State,
) -> None:
  """Analogous to the PreprocessorsTest with the same name."""
  transform = grain.Batch(batch_size=2, drop_remainder=True)
  lazy_dataset_transform = airio.preprocessors.LazyDatasetTransform(transform)
  ds = grain.MapDataset.source(list(range(_SOURCE_NUM_EXAMPLES)))
  while state:
    lazy_dataset_transform(ds)


@google_benchmark.register
def batch_lazydataset_transform_without_drop_remainder(
    state: google_benchmark.State,
) -> None:
  """Analogous to the PreprocessorsTest with the same name."""
  transform = grain.Batch(batch_size=2)
  lazy_dataset_transform = airio.preprocessors.LazyDatasetTransform(transform)
  ds = grain.MapDataset.source(list(range(_SOURCE_NUM_EXAMPLES)))
  while state:
    lazy_dataset_transform(ds)


@google_benchmark.register
def map_fn_preprocessor_with_runtime_args(
    state: google_benchmark.State,
) -> None:
  """Analogous to the PreprocessorsWithInjectedArgsTest with the same name."""
  runtime_args = _get_runtime_args()
  task = airio.GrainTask(
      name="test_task",
      source=_get_source(),
      preprocessors=[
          airio.MapFnTransform(
              lambda x, rargs: x > rargs.sequence_lengths["val"], runtime_args
          )
      ],
  )
  while state:
    task.get_dataset({"val": 3}, "train", shuffle=False)


@google_benchmark.register
def random_map_fn_preprocessor_with_runtime_args(
    state: google_benchmark.State,
) -> None:
  """Analogous to the PreprocessorsWithInjectedArgsTest with the same name."""

  def test_random_map_fn(ex, rng, r_args: airio.AirIOInjectedRuntimeArgs):
    return (
        ex
        + r_args.sequence_lengths["val"]
        + int(jax.random.randint(rng, [], 0, 10))
    )

  runtime_args = _get_runtime_args()
  task = airio.GrainTask(
      name="test_task",
      source=_get_source(),
      preprocessors=[
          airio.RandomMapFnTransform(test_random_map_fn, runtime_args)
      ],
  )
  while state:
    task.get_dataset({"val": 3}, "train", shuffle=False, seed=42)


@google_benchmark.register
def filter_fn_preprocessor_with_runtime_args(
    state: google_benchmark.State,
) -> None:
  """Analogous to the PreprocessorsWithInjectedArgsTest with the same name."""
  runtime_args = _get_runtime_args()
  task = airio.GrainTask(
      name="test_task",
      source=_get_source(),
      preprocessors=[
          airio.FilterFnTransform(
              lambda x, rargs: x > rargs.sequence_lengths["val"], runtime_args
          )
      ],
  )
  while state:
    task.get_dataset({"val": 3}, "train", shuffle=False, seed=42)


@google_benchmark.register
def map_lazydataset_transform_with_runtime_args(
    state: google_benchmark.State,
) -> None:
  """Analogous to the PreprocessorsWithInjectedArgsTest with the same name."""
  runtime_args = _get_runtime_args()
  transform = airio.MapFnTransform(
      lambda x, rargs: x > rargs.sequence_lengths["val"],
  )
  lazy_dataset_transform = airio.preprocessors.LazyDatasetTransform(transform)
  ds = grain.MapDataset.source(list(range(_SOURCE_NUM_EXAMPLES)))
  while state:
    lazy_dataset_transform(ds, runtime_args=runtime_args)


@google_benchmark.register
def map_lazydataset_transform_updated_runtime_args(
    state: google_benchmark.State,
) -> None:
  """Analogous to the PreprocessorsWithInjectedArgsTest with the same name."""
  runtime_args = _get_runtime_args()
  transform = airio.MapFnTransform(
      lambda x, rargs: x > rargs.sequence_lengths["val"],
      update_runtime_args=_update_runtime_args,
  )
  lazy_dataset_transform = airio.preprocessors.LazyDatasetTransform(transform)
  ds = grain.MapDataset.source(list(range(_SOURCE_NUM_EXAMPLES)))
  while state:
    lazy_dataset_transform(ds, runtime_args=runtime_args)


@google_benchmark.register
def random_map_fn_lazydataset_transform_with_runtime_args(
    state: google_benchmark.State,
) -> None:
  """Analogous to the PreprocessorsWithInjectedArgsTest with the same name."""

  def test_random_map_fn(ex, rng, r_args: airio.AirIOInjectedRuntimeArgs):
    return (
        ex
        + r_args.sequence_lengths["val"]
        + int(jax.random.randint(rng, [], 0, 10))
    )

  runtime_args = _get_runtime_args()
  transform = airio.RandomMapFnTransform(test_random_map_fn)
  lazy_dataset_transform = airio.preprocessors.LazyDatasetTransform(transform)
  ds = grain.MapDataset.source(list(range(_SOURCE_NUM_EXAMPLES)))
  while state:
    lazy_dataset_transform(
        ds, rng=jax.random.key(42), runtime_args=runtime_args
    )


@google_benchmark.register
def random_map_fn_lazydataset_transform_updated_runtime_args(
    state: google_benchmark.State,
) -> None:
  """Analogous to the PreprocessorsWithInjectedArgsTest with the same name."""

  def test_random_map_fn(ex, rng, r_args: airio.AirIOInjectedRuntimeArgs):
    return (
        ex
        + r_args.sequence_lengths["val"]
        + int(jax.random.randint(rng, [], 0, 10))
    )

  runtime_args = _get_runtime_args()
  transform = airio.RandomMapFnTransform(
      test_random_map_fn, update_runtime_args=_update_runtime_args
  )
  lazy_dataset_transform = airio.preprocessors.LazyDatasetTransform(transform)
  ds = grain.MapDataset.source(list(range(_SOURCE_NUM_EXAMPLES)))
  while state:
    lazy_dataset_transform(
        ds, rng=jax.random.key(42), runtime_args=runtime_args
    )


@google_benchmark.register
def filter_lazydataset_transform_with_runtime_args(
    state: google_benchmark.State,
) -> None:
  """Analogous to the PreprocessorsWithInjectedArgsTest with the same name."""
  runtime_args = _get_runtime_args()
  transform = airio.FilterFnTransform(
      lambda x, rargs: x > rargs.sequence_lengths["val"]
  )
  lazy_dataset_transform = airio.preprocessors.LazyDatasetTransform(transform)
  ds = grain.MapDataset.source(list(range(_SOURCE_NUM_EXAMPLES)))
  while state:
    lazy_dataset_transform(ds, runtime_args=runtime_args)


@google_benchmark.register
def filter_lazydataset_transform_updated_runtime_args(
    state: google_benchmark.State,
) -> None:
  """Analogous to the PreprocessorsWithInjectedArgsTest with the same name."""
  runtime_args = _get_runtime_args()
  transform = airio.FilterFnTransform(
      lambda x, rargs: x > rargs.sequence_lengths["val"],
      update_runtime_args=_update_runtime_args,
  )
  lazy_dataset_transform = airio.preprocessors.LazyDatasetTransform(transform)
  ds = grain.MapDataset.source(list(range(_SOURCE_NUM_EXAMPLES)))
  while state:
    lazy_dataset_transform(ds, runtime_args=runtime_args)


@google_benchmark.register
def lazy_map_transform_with_runtime_args(state: google_benchmark.State) -> None:
  """Analogous to the PreprocessorsWithInjectedArgsTest with the same name."""
  runtime_args = _get_runtime_args()
  transform = airio.preprocessors.LazyMapTransform(
      _lazy_map_fn,
      update_runtime_args=_update_runtime_args,
      produces_none_elements=False,
      requires_non_none_elements=False,
  )
  ds = grain.MapDataset.source(range(10))
  while state:
    unused_rng = None
    transform(ds, runtime_args, unused_rng)


@google_benchmark.register
def lazy_map_transform_with_none_elements_and_runtime_args(
    state: google_benchmark.State,
) -> None:
  """Analogous to the PreprocessorsWithInjectedArgsTest with the same name."""

  class MyLazyMapDataset(grain.MapDataset):
    """Test class."""

    def __init__(self, parent, threshold):
      super().__init__(parent)
      self.threshold = threshold

    def __len__(self):
      return len(self._parent)

    def __getitem__(self, index):
      # Filters out elements less that the threshold.
      if self._parent[index] > self.threshold:
        return self._parent[index]
      return None

  def lazy_map_fn_with_nones(ds, runtime_args, unused_rng):
    return MyLazyMapDataset(ds, threshold=runtime_args.sequence_lengths["val"])

  runtime_args = _get_runtime_args()
  transform = airio.preprocessors.LazyMapTransform(
      lazy_map_fn_with_nones,
      update_runtime_args=_update_runtime_args,
      produces_none_elements=True,
      requires_non_none_elements=False,
  )
  ds = grain.MapDataset.source(range(10))
  while state:
    unused_rng = None
    transform(ds, runtime_args, unused_rng)


@google_benchmark.register
def lazy_iter_transform_with_runtime_args(
    state: google_benchmark.State,
) -> None:
  """Analogous to the PreprocessorsWithInjectedArgsTest with the same name."""
  runtime_args = _get_runtime_args()
  transform = airio.preprocessors.LazyIterTransform(
      _lazy_iter_fn,
      update_runtime_args=_update_runtime_args,
  )
  ds = grain.MapDataset.source(range(10))
  ds = ds.to_iter_dataset()
  while state:
    unused_rng = None
    transform(ds, runtime_args, unused_rng)


@google_benchmark.register
def lazy_iter_transform_on_map_dataset_with_runtime_args(
    state: google_benchmark.State,
) -> None:
  """Analogous to the PreprocessorsWithInjectedArgsTest with the same name."""
  runtime_args = _get_runtime_args()
  transform = airio.preprocessors.LazyIterTransform(
      _lazy_iter_fn,
      update_runtime_args=_update_runtime_args,
  )
  ds = grain.MapDataset.source(range(10))
  ds = ds.to_iter_dataset()
  while state:
    unused_rng = None
    transform(ds, runtime_args, unused_rng)


@google_benchmark.register
def map_transform_on_iter_dataset(state: google_benchmark.State) -> None:
  """Analogous to the PreprocessorsWithInjectedArgsTest with the same name."""
  transform = airio.MapFnTransform(lambda x: x + 1)
  ds = grain.MapDataset.source(range(10))
  ds = ds.to_iter_dataset()
  while state:
    airio.preprocessors.LazyDatasetTransform(transform)(ds)


@google_benchmark.register
def filter_transform_on_iter_dataset(state: google_benchmark.State) -> None:
  """Analogous to the PreprocessorsWithInjectedArgsTest with the same name."""
  transform = airio.FilterFnTransform(lambda x: x > 5)
  ds = grain.MapDataset.source(range(10))
  ds = ds.to_iter_dataset()
  while state:
    airio.preprocessors.LazyDatasetTransform(transform)(ds)


@google_benchmark.register
def batch_transform_on_iter_dataset_with_drop(
    state: google_benchmark.State,
) -> None:
  """Analogous to the PreprocessorsWithInjectedArgsTest with the same name."""
  batch_with_drop = grain.Batch(batch_size=3, drop_remainder=True)
  ds = grain.MapDataset.source(range(10))
  ds = ds.to_iter_dataset()
  while state:
    airio.preprocessors.LazyDatasetTransform(batch_with_drop)(ds)


@google_benchmark.register
def batch_transform_on_iter_dataset_without_drop(
    state: google_benchmark.State,
) -> None:
  """Analogous to the PreprocessorsWithInjectedArgsTest with the same name."""
  batch_without_drop = grain.Batch(batch_size=3, drop_remainder=False)
  ds = grain.MapDataset.source(range(10))
  ds = ds.to_iter_dataset()
  while state:
    airio.preprocessors.LazyDatasetTransform(batch_without_drop)(ds)


@google_benchmark.register
def lazy_iter_transform_on_iter_dataset_with_runtime_args(
    state: google_benchmark.State,
) -> None:
  """Analogous to the PreprocessorsWithInjectedArgsTest with the same name."""
  runtime_args = _get_runtime_args()
  transform = airio.preprocessors.LazyIterTransform(
      _lazy_iter_fn,
      update_runtime_args=_update_runtime_args,
  )
  ds = grain.MapDataset.source(range(10))
  ds = ds.to_iter_dataset()
  while state:
    unused_rng = None
    transform(ds, runtime_args, unused_rng)


if __name__ == "__main__":
  google_benchmark.main()
