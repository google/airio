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

"""Preprocessors tests."""

from unittest import mock

from absl.testing import absltest
from airio import preprocessors as airio_preprocessors_lib
from airio.pygrain import preprocessors
import grain.python as grain
import jax.random
import numpy as np


lazy_dataset = grain.experimental.lazy_dataset


def lazy_map_fn(
    ds: lazy_dataset.LazyMapDataset,
    run_args: airio_preprocessors_lib.AirIOInjectedRuntimeArgs,
    rng: jax.Array,
):
  n = int(jax.random.randint(rng, [], 0, 10)) if rng is not None else 0
  return ds.map(lambda x: x + run_args.sequence_lengths["val"] + n)


def lazy_iter_fn(
    ds: lazy_dataset.LazyIterDataset,
    run_args: airio_preprocessors_lib.AirIOInjectedRuntimeArgs,
    rng: jax.Array,
):
  n = int(jax.random.randint(rng, [], 0, 10)) if rng is not None else 0
  return ds.map(lambda x: x + run_args.sequence_lengths["val"] + n)


class PreprocessorsTest(absltest.TestCase):

  def test_map_lazydataset_transform(self):
    def test_map_fn(ex):
      return ex + 1

    transform = airio_preprocessors_lib.MapFnTransform(test_map_fn)
    lazy_dataset_transform = preprocessors.LazyDatasetTransform(transform)
    ds = lazy_dataset.SourceLazyMapDataset(list(range(5)))
    ds = lazy_dataset_transform(ds)
    self.assertListEqual(list(ds), list(range(1, 6)))

  def test_random_map_fn_lazydataset_transform(self):
    def test_random_map_fn(ex, rng):
      return ex + int(jax.random.randint(rng, [], 0, 10))

    transform = airio_preprocessors_lib.RandomMapFnTransform(test_random_map_fn)
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

    transform = airio_preprocessors_lib.FilterFnTransform(test_filter_fn)
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
    self._runtime_args = airio_preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 3},
        split="train",
    )

  def _update_runtime_args(self, run_args):
    new_seq_lens = {}
    for k, v in run_args.sequence_lengths.items():
      new_seq_lens[f"{k}_new"] = v
      new_seq_lens[k] = v + 1
    updated = run_args.clone()
    updated.sequence_lengths = new_seq_lens
    return updated

  def test_map_lazydataset_transform(self):
    def test_map_fn(
        ex, run_args: airio_preprocessors_lib.AirIOInjectedRuntimeArgs
    ):
      return ex + run_args.sequence_lengths["val"]

    transform = airio_preprocessors_lib.MapFnTransform(test_map_fn)
    lazy_dataset_transform = preprocessors.LazyDatasetTransform(transform)
    ds = lazy_dataset.SourceLazyMapDataset(list(range(5)))
    ds = lazy_dataset_transform(ds, runtime_args=self._runtime_args)
    self.assertListEqual(list(ds), list(range(3, 8)))

  def test_map_lazydataset_transform_updated_runtime_args(self):
    def test_map_fn(
        ex, run_args: airio_preprocessors_lib.AirIOInjectedRuntimeArgs
    ):
      return ex + run_args.sequence_lengths["val"]

    transform = airio_preprocessors_lib.MapFnTransform(
        test_map_fn, update_runtime_args=self._update_runtime_args
    )
    lazy_dataset_transform = preprocessors.LazyDatasetTransform(transform)
    ds = lazy_dataset.SourceLazyMapDataset(list(range(5)))
    ds = lazy_dataset_transform(ds, runtime_args=self._runtime_args)
    updated_runtime_args = lazy_dataset_transform.get_updated_runtime_args(
        self._runtime_args
    )
    expected_runtime_args = airio_preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 4, "val_new": 3},
        split="train",
    )
    self.assertListEqual(list(ds), list(range(3, 8)))
    self.assertEqual(updated_runtime_args, expected_runtime_args)

  def test_random_map_fn_lazydataset_transform(self):
    def test_random_map_fn(
        ex, rng, r_args: airio_preprocessors_lib.AirIOInjectedRuntimeArgs
    ):
      return (
          ex
          + r_args.sequence_lengths["val"]
          + int(jax.random.randint(rng, [], 0, 10))
      )

    transform = airio_preprocessors_lib.RandomMapFnTransform(test_random_map_fn)
    lazy_dataset_transform = preprocessors.LazyDatasetTransform(transform)
    ds = lazy_dataset.SourceLazyMapDataset(list(range(5)))
    ds = lazy_dataset_transform(
        ds, rng=jax.random.PRNGKey(42), runtime_args=self._runtime_args
    )
    self.assertListEqual(list(ds), [8, 7, 8, 15, 16])

  def test_random_map_fn_lazydataset_transform_updated_runtime_args(self):
    def test_random_map_fn(
        ex, rng, r_args: airio_preprocessors_lib.AirIOInjectedRuntimeArgs
    ):
      return (
          ex
          + r_args.sequence_lengths["val"]
          + int(jax.random.randint(rng, [], 0, 10))
      )

    transform = airio_preprocessors_lib.RandomMapFnTransform(
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
    expected_runtime_args = airio_preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 4, "val_new": 3},
        split="train",
    )
    self.assertListEqual(list(ds), [8, 7, 8, 15, 16])
    self.assertEqual(updated_runtime_args, expected_runtime_args)

  def test_filter_lazydataset_transform(self):
    def test_filter_fn(
        ex, rargs: airio_preprocessors_lib.AirIOInjectedRuntimeArgs
    ):
      return ex > rargs.sequence_lengths["val"]

    transform = airio_preprocessors_lib.FilterFnTransform(test_filter_fn)
    lazy_dataset_transform = preprocessors.LazyDatasetTransform(transform)
    ds = lazy_dataset.SourceLazyMapDataset(list(range(5)))
    ds = lazy_dataset_transform(ds, runtime_args=self._runtime_args)
    self.assertListEqual(list(ds), [4])

  def test_filter_lazydataset_transform_updated_runtime_args(self):
    def test_filter_fn(
        ex, rargs: airio_preprocessors_lib.AirIOInjectedRuntimeArgs
    ):
      return ex > rargs.sequence_lengths["val"]

    transform = airio_preprocessors_lib.FilterFnTransform(
        test_filter_fn, update_runtime_args=self._update_runtime_args
    )
    lazy_dataset_transform = preprocessors.LazyDatasetTransform(transform)
    ds = lazy_dataset.SourceLazyMapDataset(list(range(5)))
    ds = lazy_dataset_transform(ds, runtime_args=self._runtime_args)
    updated_runtime_args = lazy_dataset_transform.get_updated_runtime_args(
        self._runtime_args
    )
    expected_runtime_args = airio_preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 4, "val_new": 3},
        split="train",
    )
    self.assertListEqual(list(ds), [4])
    self.assertEqual(updated_runtime_args, expected_runtime_args)

  def test_lazy_map_transform(self):
    run_args = airio_preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 3}, split="unused"
    )
    transform = preprocessors.LazyMapTransform(
        lazy_map_fn,
        update_runtime_args=self._update_runtime_args,
        produces_none_elements=False,
        requires_non_none_elements=False,
    )
    ds = lazy_dataset.SourceLazyMapDataset(range(10))
    ds = transform(ds, run_args, rng=None)
    updated_runtime_args = transform.update_runtime_args(run_args)
    expected_runtime_args = airio_preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 4, "val_new": 3},
        split="unused",
    )
    self.assertListEqual(list(ds), list(range(3, 13)))
    self.assertEqual(updated_runtime_args, expected_runtime_args)

  def test_lazy_map_transform_with_rng(self):
    run_args = airio_preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 3}, split="unused"
    )
    transform = preprocessors.LazyMapTransform(
        lazy_map_fn,
        update_runtime_args=self._update_runtime_args,
        produces_none_elements=False,
        requires_non_none_elements=False,
    )
    ds = lazy_dataset.SourceLazyMapDataset(range(10))
    ds = transform(ds, run_args, rng=jax.random.PRNGKey(42))
    updated_runtime_args = transform.update_runtime_args(run_args)
    expected_runtime_args = airio_preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 4, "val_new": 3},
        split="unused",
    )
    self.assertListEqual(list(ds), [7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
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

    def lazy_map_fn_with_nones(ds, runtime_args, unused_rng):
      return MyLazyMapDataset(
          ds, threshold=runtime_args.sequence_lengths["val"]
      )

    run_args = airio_preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 3}, split="unused"
    )
    transform = preprocessors.LazyMapTransform(
        lazy_map_fn_with_nones,
        update_runtime_args=self._update_runtime_args,
        produces_none_elements=True,
        requires_non_none_elements=False,
    )
    ds = lazy_dataset.SourceLazyMapDataset(range(10))
    ds = transform(ds, run_args, rng=None)
    updated_runtime_args = transform.update_runtime_args(run_args)
    expected_runtime_args = airio_preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 4, "val_new": 3},
        split="unused",
    )
    self.assertListEqual(list(ds), list(range(4, 10)))
    self.assertEqual(updated_runtime_args, expected_runtime_args)

  def test_lazy_iter_transform(self):
    run_args = airio_preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 3}, split="unused"
    )
    transform = preprocessors.LazyIterTransform(
        lazy_iter_fn,
        update_runtime_args=self._update_runtime_args,
    )
    ds = lazy_dataset.SourceLazyMapDataset(range(10))
    ds = ds.to_iter_dataset()
    ds = transform(ds, run_args, rng=None)
    updated_runtime_args = transform.update_runtime_args(run_args)
    expected_runtime_args = airio_preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 4, "val_new": 3},
        split="unused",
    )
    self.assertListEqual(list(ds), list(range(3, 13)))
    self.assertEqual(updated_runtime_args, expected_runtime_args)

  def test_lazy_iter_transform_with_rng(self):
    run_args = airio_preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 3}, split="unused"
    )
    transform = preprocessors.LazyIterTransform(
        lazy_iter_fn,
        update_runtime_args=self._update_runtime_args,
    )
    ds = lazy_dataset.SourceLazyMapDataset(range(10))
    ds = ds.to_iter_dataset()
    ds = transform(ds, run_args, rng=jax.random.PRNGKey(42))
    updated_runtime_args = transform.update_runtime_args(run_args)
    expected_runtime_args = airio_preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 4, "val_new": 3},
        split="unused",
    )
    self.assertListEqual(list(ds), [7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    self.assertEqual(updated_runtime_args, expected_runtime_args)

  def test_lazy_iter_transform_on_map_dataset(self):
    run_args = airio_preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 3}, split="unused"
    )
    transform = preprocessors.LazyIterTransform(
        lazy_iter_fn,
        update_runtime_args=self._update_runtime_args,
    )
    ds = lazy_dataset.SourceLazyMapDataset(range(10))
    ds = transform(ds, run_args, rng=None)
    updated_runtime_args = transform.update_runtime_args(run_args)
    expected_runtime_args = airio_preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 4, "val_new": 3},
        split="unused",
    )
    self.assertListEqual(list(ds), list(range(3, 13)))
    self.assertEqual(updated_runtime_args, expected_runtime_args)

  def test_map_transform_on_iter_dataset(self):
    transform = airio_preprocessors_lib.MapFnTransform(lambda x: x + 1)
    ds = lazy_dataset.SourceLazyMapDataset(range(10))
    ds = ds.to_iter_dataset()
    ds = preprocessors.LazyDatasetTransform(transform)(ds)
    self.assertListEqual(list(ds), list(range(1, 11)))

  def test_filter_transform_on_iter_dataset(self):
    transform = airio_preprocessors_lib.FilterFnTransform(lambda x: x > 5)
    ds = lazy_dataset.SourceLazyMapDataset(range(10))
    ds = ds.to_iter_dataset()
    ds = preprocessors.LazyDatasetTransform(transform)(ds)
    self.assertListEqual(list(ds), list(range(6, 10)))

  def test_batch_transform_on_iter_dataset(self):
    batch_with_drop = grain.Batch(batch_size=3, drop_remainder=True)
    batch_without_drop = grain.Batch(batch_size=3, drop_remainder=False)
    ds = lazy_dataset.SourceLazyMapDataset(range(10))
    ds = ds.to_iter_dataset()
    ds_with_drop = preprocessors.LazyDatasetTransform(batch_with_drop)(ds)
    ds_without_drop = preprocessors.LazyDatasetTransform(batch_without_drop)(ds)
    self.assertListEqual(
        [d.tolist() for d in ds_with_drop], [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    )
    self.assertListEqual(
        [d.tolist() for d in ds_without_drop],
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]],
    )

  def test_lazy_iter_transform_on_iter_dataset(self):
    run_args = airio_preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 3}, split="unused"
    )
    transform = preprocessors.LazyIterTransform(
        lazy_iter_fn,
        update_runtime_args=self._update_runtime_args,
    )
    ds = lazy_dataset.SourceLazyMapDataset(range(10))
    ds = ds.to_iter_dataset()
    ds = transform(ds, run_args, rng=None)
    self.assertListEqual(list(ds), list(range(3, 13)))

  def test_random_map_transform_on_iter_dataset_fails(self):
    def test_random_map_fn(ex, rng):
      return ex + int(jax.random.randint(rng, [], 0, 10))

    transform = airio_preprocessors_lib.RandomMapFnTransform(test_random_map_fn)
    ds = lazy_dataset.SourceLazyMapDataset(range(10))
    ds = ds.to_iter_dataset()
    run_args = airio_preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 3}, split="unused"
    )
    with self.assertRaisesRegex(
        ValueError, "RandomMapFnTransform is not yet supported.*"
    ):
      _ = preprocessors.LazyDatasetTransform(transform)(ds, run_args)

  def test_lazy_map_transform_on_iter_dataset_fails(self):
    transform = preprocessors.LazyMapTransform(
        lazy_map_fn,
        update_runtime_args=self._update_runtime_args,
        produces_none_elements=False,
        requires_non_none_elements=False,
    )
    ds = lazy_dataset.SourceLazyMapDataset(range(10))
    ds = ds.to_iter_dataset()
    run_args = airio_preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 3}, split="unused"
    )
    with self.assertRaisesRegex(
        ValueError, "Cannot apply LazyMapDataset transform.*"
    ):
      _ = transform(ds, run_args, rng=None)

  def test_produces_none_elements_map_fn(self):
    prep = airio_preprocessors_lib.MapFnTransform(lambda x: x + 1)
    self.assertFalse(
        preprocessors.LazyDatasetTransform(prep).produces_none_elements
    )

  def test_produces_none_elements_filter_fn(self):
    prep = airio_preprocessors_lib.FilterFnTransform(lambda x: x > 1)
    self.assertTrue(
        preprocessors.LazyDatasetTransform(prep).produces_none_elements
    )

  def test_produces_none_elements_random_map_fn(self):
    def test_random_map_fn(ex, rng):
      return ex + int(jax.random.randint(rng, [], 0, 10))

    prep = airio_preprocessors_lib.RandomMapFnTransform(test_random_map_fn)
    self.assertFalse(
        preprocessors.LazyDatasetTransform(prep).produces_none_elements
    )

  def test_produces_none_elements_lazy_map_transform_with_none(self):
    prep = preprocessors.LazyMapTransform(
        lazy_map_fn,
        update_runtime_args=self._update_runtime_args,
        produces_none_elements=True,
        requires_non_none_elements=False,
    )
    self.assertTrue(
        preprocessors.LazyDatasetTransform(prep).produces_none_elements
    )

  def test_produces_none_elements_lazy_map_transform_without_none(self):
    prep = preprocessors.LazyMapTransform(
        lazy_map_fn,
        update_runtime_args=self._update_runtime_args,
        produces_none_elements=False,
        requires_non_none_elements=False,
    )
    self.assertFalse(
        preprocessors.LazyDatasetTransform(prep).produces_none_elements
    )

  def test_produces_none_elements_lazy_iter_transform(self):
    prep = preprocessors.LazyIterTransform(
        lazy_iter_fn,
        update_runtime_args=self._update_runtime_args,
    )
    self.assertFalse(
        preprocessors.LazyDatasetTransform(prep).produces_none_elements
    )



if __name__ == "__main__":
  absltest.main()
