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

"""Preprocessors tests."""

from absl.testing import absltest
from airio import data_sources
from airio import dataset_providers
from airio import preprocessors
import grain.python as grain
import jax.random
import numpy as np

lazy_dataset = grain.experimental.lazy_dataset


def lazy_map_fn(
        ds: lazy_dataset.LazyMapDataset,
        run_args: preprocessors.AirIOInjectedRuntimeArgs,
):
  return ds.map(
      lambda x: x + run_args.sequence_lengths["val"]
  )


def lazy_iter_fn(
    ds: lazy_dataset.LazyIterDataset,
    run_args: preprocessors.AirIOInjectedRuntimeArgs,
):
  return ds.map(
      lambda x: x + run_args.sequence_lengths["val"]
  )


class PreprocessorsTest(absltest.TestCase):

  def _get_test_src(self, num_elements=5):
    def _dataset_fn(split: str):
      del split
      return np.array(range(num_elements))

    return data_sources.FunctionDataSource(
        dataset_fn=_dataset_fn, splits=["train"]
    )

  def test_map_fn_preprocessor(self):
    def test_map_fn(ex):
      return ex + 1

    task = dataset_providers.Task(
        name="test_task",
        source=self._get_test_src(),
        preprocessors=[preprocessors.MapFnTransform(test_map_fn)],
    )
    ds = task.get_dataset(None, "train", shuffle=False)
    self.assertListEqual(list(ds), list(range(1, 6)))

  def test_random_map_fn_preprocessor(self):
    def test_random_map_fn(ex, rng):
      return ex + int(jax.random.randint(rng, [], 0, 10))

    task = dataset_providers.Task(
        name="test_task",
        source=self._get_test_src(),
        preprocessors=[preprocessors.RandomMapFnTransform(test_random_map_fn)],
    )
    ds = task.get_dataset(None, "train", shuffle=False, seed=42)
    self.assertListEqual(list(ds), [5, 9, 7, 3, 12])

  def test_filter_fn_preprocessor(self):
    def test_filter_fn(ex):
      return ex > 2

    task = dataset_providers.Task(
        name="test_task",
        source=self._get_test_src(),
        preprocessors=[preprocessors.FilterFnTransform(test_filter_fn)],
    )
    ds = task.get_dataset(None, "train", shuffle=False, seed=42)
    self.assertListEqual(list(ds), [3, 4])

  def test_preprocessor_empty_source(self):
    def test_filter_fn(ex):
      return ex > 2

    def test_map_fn(ex):
      return ex + 1

    def test_random_map_fn(ex, rng):
      return ex + int(jax.random.randint(rng, [], 0, 10))

    task = dataset_providers.Task(
        name="test_task",
        source=self._get_test_src(num_elements=0),
        preprocessors=[
            preprocessors.FilterFnTransform(test_filter_fn),
            preprocessors.MapFnTransform(test_map_fn),
            preprocessors.RandomMapFnTransform(test_random_map_fn),
        ],
    )
    with self.assertRaisesRegex(ValueError, "Invalid number of records.*"):
      _ = task.get_dataset(None, "train", shuffle=False)

  def test_preprocessor_empty_preprocessed(self):
    def test_filter_fn(ex):
      return ex > 1000

    task = dataset_providers.Task(
        name="test_task",
        source=self._get_test_src(),
        preprocessors=[preprocessors.FilterFnTransform(test_filter_fn)],
    )
    ds = task.get_dataset(None, "train", shuffle=False, seed=42)
    self.assertListEqual(list(ds), [])

  def test_preprocessor_empty_intermediates(self):
    def test_map_fn(ex):
      return ex + 1

    def test_filter_fn(ex):
      return ex > 1000

    task = dataset_providers.Task(
        name="test_task",
        source=self._get_test_src(),
        preprocessors=[
            preprocessors.FilterFnTransform(test_filter_fn),
            preprocessors.MapFnTransform(test_map_fn),
        ],
    )
    ds = task.get_dataset(None, "train", shuffle=False, seed=42)
    self.assertListEqual(list(ds), [])

  def test_map_lazydataset_transform(self):
    def test_map_fn(ex):
      return ex + 1

    transform = preprocessors.MapFnTransform(test_map_fn)
    lazy_dataset_transform = preprocessors.LazyDatasetTransform(transform)
    ds = lazy_dataset.SourceLazyMapDataset(list(range(5)))
    ds = lazy_dataset_transform(ds)
    self.assertListEqual(list(ds), list(range(1, 6)))

  def test_random_map_fn_lazydataset_transform(self):
    def test_random_map_fn(ex, rng):
      return ex + int(jax.random.randint(rng, [], 0, 10))

    transform = preprocessors.RandomMapFnTransform(test_random_map_fn)
    lazy_dataset_transform = preprocessors.LazyDatasetTransform(transform)
    ds = lazy_dataset.SourceLazyMapDataset(list(range(5)))
    ds = lazy_dataset_transform(ds, rng=jax.random.PRNGKey(42))
    self.assertListEqual(list(ds), [5, 4, 5, 12, 13])

  def test_random_map_lazydataset_transform_disallowed(self):

    class TestRandomMap(grain.RandomMapTransform):
      def random_map(self, element, rng: np.random.Generator):
        return element + rng.integers(0, 10)

    transform = TestRandomMap()
    with self.assertRaisesRegex(ValueError, ".*is not reproducible"):
      _ = preprocessors.LazyDatasetTransform(transform)

  def test_filter_lazydataset_transform(self):
    def test_filter_fn(ex):
      return ex > 2

    transform = preprocessors.FilterFnTransform(test_filter_fn)
    lazy_dataset_transform = preprocessors.LazyDatasetTransform(transform)
    ds = lazy_dataset.SourceLazyMapDataset(list(range(5)))
    ds = lazy_dataset_transform(ds)
    self.assertListEqual(list(ds), [3, 4])

  def test_batch_lazydataset_transform_with_drop_remainder(self):
    transform = grain.Batch(batch_size=2, drop_remainder=True)
    lazy_dataset_transform = preprocessors.LazyDatasetTransform(transform)
    ds = lazy_dataset.SourceLazyMapDataset(list(range(5)))
    ds = lazy_dataset_transform(ds)
    self.assertListEqual([t.tolist() for t in list(ds)], [[0, 1], [2, 3]])

  def test_batch_lazydataset_transform_without_drop_remainder(self):
    transform = grain.Batch(batch_size=2)
    lazy_dataset_transform = preprocessors.LazyDatasetTransform(transform)
    ds = lazy_dataset.SourceLazyMapDataset(list(range(5)))
    ds = lazy_dataset_transform(ds)
    self.assertListEqual([t.tolist() for t in list(ds)], [[0, 1], [2, 3], [4]])


class PreprocessorsWithInjectedArgsTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self._runtime_args = preprocessors.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 3},
        split="train",
    )

  def _get_test_src(self, num_elements=5):
    def _dataset_fn(split: str):
      del split
      return np.array(range(num_elements))

    return data_sources.FunctionDataSource(
        dataset_fn=_dataset_fn, splits=["train"]
    )

  def _update_runtime_args(self, run_args):
    new_seq_lens = {}
    for k, v in run_args.sequence_lengths.items():
      new_seq_lens[f"{k}_new"] = v
      new_seq_lens[k] = v + 1
    return preprocessors.AirIOInjectedRuntimeArgs(
        sequence_lengths=new_seq_lens,
        split=run_args.split,
    )

  def test_map_fn_preprocessor(self):
    def test_map_fn(ex, run_args: preprocessors.AirIOInjectedRuntimeArgs):
      return ex + run_args.sequence_lengths["val"]

    task = dataset_providers.Task(
        name="test_task",
        source=self._get_test_src(),
        preprocessors=[
            preprocessors.MapFnTransform(test_map_fn, self._runtime_args)
        ],
    )
    ds = task.get_dataset({"val": 3}, "train", shuffle=False)
    self.assertListEqual(list(ds), list(range(3, 8)))

  def test_random_map_fn_preprocessor(self):

    def test_random_map_fn(
        ex, rng, r_args: preprocessors.AirIOInjectedRuntimeArgs
    ):
      return (
          ex
          + r_args.sequence_lengths["val"]
          + int(jax.random.randint(rng, [], 0, 10))
      )

    task = dataset_providers.Task(
        name="test_task",
        source=self._get_test_src(),
        preprocessors=[
            preprocessors.RandomMapFnTransform(
                test_random_map_fn, self._runtime_args
            )
        ],
    )
    ds = task.get_dataset({"val": 3}, "train", shuffle=False, seed=42)
    self.assertListEqual(list(ds), [8, 12, 10, 6, 15])

  def test_filter_fn_preprocessor(self):
    def test_filter_fn(ex, rargs: preprocessors.AirIOInjectedRuntimeArgs):
      return ex > rargs.sequence_lengths["val"]

    task = dataset_providers.Task(
        name="test_task",
        source=self._get_test_src(),
        preprocessors=[
            preprocessors.FilterFnTransform(test_filter_fn, self._runtime_args)
        ],
    )
    ds = task.get_dataset({"val": 3}, "train", shuffle=False, seed=42)
    self.assertListEqual(list(ds), [4])

  def test_unannotated_runtime_args(self):
    def test_map_fn(ex, run_args):
      return ex + run_args.sequence_lengths["val"]

    task = dataset_providers.Task(
        name="test_task",
        source=self._get_test_src(),
        preprocessors=[
            preprocessors.MapFnTransform(test_map_fn, self._runtime_args)
        ],
    )
    with self.assertRaisesRegex(ValueError, "PyGrain encountered an error.*"):
      ds = task.get_dataset(None, "train", shuffle=False)
      _ = list(ds)

  def test_map_lazydataset_transform(self):
    def test_map_fn(ex, run_args: preprocessors.AirIOInjectedRuntimeArgs):
      return ex + run_args.sequence_lengths["val"]

    transform = preprocessors.MapFnTransform(test_map_fn)
    lazy_dataset_transform = preprocessors.LazyDatasetTransform(transform)
    ds = lazy_dataset.SourceLazyMapDataset(list(range(5)))
    ds = lazy_dataset_transform(ds, runtime_args=self._runtime_args)
    self.assertListEqual(list(ds), list(range(3, 8)))

  def test_map_lazydataset_transform_updated_runtime_args(self):
    def test_map_fn(ex, run_args: preprocessors.AirIOInjectedRuntimeArgs):
      return ex + run_args.sequence_lengths["val"]

    transform = preprocessors.MapFnTransform(
        test_map_fn, update_runtime_args=self._update_runtime_args
    )
    lazy_dataset_transform = preprocessors.LazyDatasetTransform(transform)
    ds = lazy_dataset.SourceLazyMapDataset(list(range(5)))
    ds = lazy_dataset_transform(ds, runtime_args=self._runtime_args)
    updated_runtime_args = lazy_dataset_transform.get_updated_runtime_args(
        self._runtime_args
    )
    expected_runtime_args = preprocessors.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 4, "val_new": 3},
        split="train",
    )
    self.assertListEqual(list(ds), list(range(3, 8)))
    self.assertEqual(updated_runtime_args, expected_runtime_args)

  def test_random_map_fn_lazydataset_transform(self):

    def test_random_map_fn(
        ex, rng, r_args: preprocessors.AirIOInjectedRuntimeArgs
    ):
      return (
          ex
          + r_args.sequence_lengths["val"]
          + int(jax.random.randint(rng, [], 0, 10))
      )

    transform = preprocessors.RandomMapFnTransform(test_random_map_fn)
    lazy_dataset_transform = preprocessors.LazyDatasetTransform(transform)
    ds = lazy_dataset.SourceLazyMapDataset(list(range(5)))
    ds = lazy_dataset_transform(
        ds, rng=jax.random.PRNGKey(42), runtime_args=self._runtime_args
    )
    self.assertListEqual(list(ds), [8, 7, 8, 15, 16])

  def test_random_map_fn_lazydataset_transform_updated_runtime_args(self):

    def test_random_map_fn(
        ex, rng, r_args: preprocessors.AirIOInjectedRuntimeArgs
    ):
      return (
          ex
          + r_args.sequence_lengths["val"]
          + int(jax.random.randint(rng, [], 0, 10))
      )

    transform = preprocessors.RandomMapFnTransform(
        test_random_map_fn, update_runtime_args=self._update_runtime_args
    )
    lazy_dataset_transform = preprocessors.LazyDatasetTransform(transform)
    ds = lazy_dataset.SourceLazyMapDataset(list(range(5)))
    ds = lazy_dataset_transform(
        ds, rng=jax.random.PRNGKey(42), runtime_args=self._runtime_args
    )
    updated_runtime_args = lazy_dataset_transform.get_updated_runtime_args(
        self._runtime_args
    )
    expected_runtime_args = preprocessors.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 4, "val_new": 3},
        split="train",
    )
    self.assertListEqual(list(ds), [8, 7, 8, 15, 16])
    self.assertEqual(updated_runtime_args, expected_runtime_args)

  def test_filter_lazydataset_transform(self):
    def test_filter_fn(ex, rargs: preprocessors.AirIOInjectedRuntimeArgs):
      return ex > rargs.sequence_lengths["val"]

    transform = preprocessors.FilterFnTransform(test_filter_fn)
    lazy_dataset_transform = preprocessors.LazyDatasetTransform(transform)
    ds = lazy_dataset.SourceLazyMapDataset(list(range(5)))
    ds = lazy_dataset_transform(ds, runtime_args=self._runtime_args)
    self.assertListEqual(list(ds), [4])

  def test_filter_lazydataset_transform_updated_runtime_args(self):
    def test_filter_fn(ex, rargs: preprocessors.AirIOInjectedRuntimeArgs):
      return ex > rargs.sequence_lengths["val"]

    transform = preprocessors.FilterFnTransform(
        test_filter_fn, update_runtime_args=self._update_runtime_args
    )
    lazy_dataset_transform = preprocessors.LazyDatasetTransform(transform)
    ds = lazy_dataset.SourceLazyMapDataset(list(range(5)))
    ds = lazy_dataset_transform(ds, runtime_args=self._runtime_args)
    updated_runtime_args = lazy_dataset_transform.get_updated_runtime_args(
        self._runtime_args
    )
    expected_runtime_args = preprocessors.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 4, "val_new": 3},
        split="train",
    )
    self.assertListEqual(list(ds), [4])
    self.assertEqual(updated_runtime_args, expected_runtime_args)

  def test_lazy_map_transform(self):
    run_args = preprocessors.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 3}, split="unused"
    )
    transform = preprocessors.LazyMapTransform(
        lazy_map_fn,
        update_runtime_args=self._update_runtime_args,
        has_none_elements=False,
    )
    ds = lazy_dataset.SourceLazyMapDataset(range(10))
    ds = transform(ds, run_args)
    updated_runtime_args = transform.update_runtime_args(run_args)
    expected_runtime_args = preprocessors.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 4, "val_new": 3},
        split="unused",
    )
    self.assertListEqual(list(ds), list(range(3, 13)))
    self.assertEqual(updated_runtime_args, expected_runtime_args)

  def test_lazy_map_transform_with_none_elements(self):
    class MyLazyMapDataset(lazy_dataset.LazyMapDataset):

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

    def lazy_map_fn_with_nones(ds, runtime_args):
      return MyLazyMapDataset(
          ds, threshold=runtime_args.sequence_lengths["val"]
      )

    run_args = preprocessors.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 3}, split="unused"
    )
    transform = preprocessors.LazyMapTransform(
        lazy_map_fn_with_nones,
        update_runtime_args=self._update_runtime_args,
        has_none_elements=True,
    )
    ds = lazy_dataset.SourceLazyMapDataset(range(10))
    ds = transform(ds, run_args)
    updated_runtime_args = transform.update_runtime_args(run_args)
    expected_runtime_args = preprocessors.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 4, "val_new": 3},
        split="unused",
    )
    self.assertListEqual(list(ds), list(range(4, 10)))
    self.assertEqual(updated_runtime_args, expected_runtime_args)

  def test_lazy_iter_transform(self):
    run_args = preprocessors.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 3}, split="unused"
    )
    transform = preprocessors.LazyIterTransform(
        lazy_iter_fn,
        update_runtime_args=self._update_runtime_args,
    )
    ds = lazy_dataset.SourceLazyMapDataset(range(10))
    ds = ds.to_iter_dataset()
    ds = transform(ds, run_args)
    updated_runtime_args = transform.update_runtime_args(run_args)
    expected_runtime_args = preprocessors.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 4, "val_new": 3},
        split="unused",
    )
    self.assertListEqual(list(ds), list(range(3, 13)))
    self.assertEqual(updated_runtime_args, expected_runtime_args)

  def test_lazy_iter_transform_on_map_dataset(self):
    run_args = preprocessors.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 3}, split="unused"
    )
    transform = preprocessors.LazyIterTransform(
        lazy_iter_fn,
        update_runtime_args=self._update_runtime_args,
    )
    ds = lazy_dataset.SourceLazyMapDataset(range(10))
    ds = transform(ds, run_args)
    updated_runtime_args = transform.update_runtime_args(run_args)
    expected_runtime_args = preprocessors.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 4, "val_new": 3},
        split="unused",
    )
    self.assertListEqual(list(ds), list(range(3, 13)))
    self.assertEqual(updated_runtime_args, expected_runtime_args)


if __name__ == "__main__":
  absltest.main()
