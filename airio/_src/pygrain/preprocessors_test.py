# Copyright 2025 The AirIO Authors.
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
from airio._src.core import preprocessors as core_preprocessors_lib
from airio._src.core import test_utils
from airio._src.pygrain import data_sources
from airio._src.pygrain import dataset_providers
from airio._src.pygrain import preprocessors
import grain.python as grain
import jax
import jax.random
import numpy as np


jax.config.update("jax_threefry_partitionable", False)


def _lazy_map_fn(
    ds: grain.MapDataset,
    run_args: core_preprocessors_lib.AirIOInjectedRuntimeArgs,
    rng: jax.Array,
):
  n = int(jax.random.randint(rng, [], 0, 10)) if rng is not None else 0
  return ds.map(lambda x: x + run_args.sequence_lengths["val"] + n)


def _lazy_iter_fn(
    ds: grain.IterDataset,
    run_args: core_preprocessors_lib.AirIOInjectedRuntimeArgs,
    rng: jax.Array,
):
  n = int(jax.random.randint(rng, [], 0, 10)) if rng is not None else 0
  return ds.map(lambda x: x + run_args.sequence_lengths["val"] + n)


def _get_test_src(num_elements=5):

  def _dataset_fn(split: str):
    del split
    return np.array(range(num_elements))

  return data_sources.FunctionDataSource(
      dataset_fn=_dataset_fn, splits=["train"]
  )


class PreprocessorsTest(absltest.TestCase):

  def test_map_fn_preprocessor(self):
    def test_map_fn(ex):
      return ex + 1

    task = dataset_providers.GrainTask(
        name="test_task",
        source=_get_test_src(),
        preprocessors=[preprocessors.MapFnTransform(test_map_fn)],
    )
    ds = task.get_dataset(None, "train", shuffle=False)
    self.assertListEqual(list(ds), list(range(1, 6)))

  def test_random_map_fn_preprocessor(self):
    def test_random_map_fn(ex, rng):
      return ex + int(jax.random.randint(rng, [], 0, 10))

    task = dataset_providers.GrainTask(
        name="test_task",
        source=_get_test_src(),
        preprocessors=[preprocessors.RandomMapFnTransform(test_random_map_fn)],
    )
    ds = task.get_dataset(None, "train", shuffle=False, seed=42)
    self.assertListEqual(list(ds), [5, 9, 7, 3, 12])

  def test_filter_fn_preprocessor(self):
    def test_filter_fn(ex):
      return ex > 2

    task = dataset_providers.GrainTask(
        name="test_task",
        source=_get_test_src(),
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

    task = dataset_providers.GrainTask(
        name="test_task",
        source=_get_test_src(num_elements=0),
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

    task = dataset_providers.GrainTask(
        name="test_task",
        source=_get_test_src(),
        preprocessors=[preprocessors.FilterFnTransform(test_filter_fn)],
    )
    ds = task.get_dataset(None, "train", shuffle=False, seed=42)
    self.assertListEqual(list(ds), [])

  def test_preprocessor_empty_intermediates(self):
    def test_map_fn(ex):
      return ex + 1

    def test_filter_fn(ex):
      return ex > 1000

    task = dataset_providers.GrainTask(
        name="test_task",
        source=_get_test_src(),
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
    ds = grain.MapDataset.source(list(range(5)))
    ds = lazy_dataset_transform(ds)
    self.assertListEqual(list(ds), list(range(1, 6)))

  def test_random_map_fn_lazydataset_transform(self):
    def test_random_map_fn(ex, rng):
      return ex + int(jax.random.randint(rng, [], 0, 10))

    transform = preprocessors.RandomMapFnTransform(test_random_map_fn)
    lazy_dataset_transform = preprocessors.LazyDatasetTransform(transform)
    ds = grain.MapDataset.source(list(range(5)))
    ds = lazy_dataset_transform(ds, rng=jax.random.key(42))
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
    ds = grain.MapDataset.source(list(range(5)))
    ds = lazy_dataset_transform(ds)
    self.assertListEqual(list(ds), [3, 4])

  def test_batch_lazydataset_transform_with_drop_remainder(self):
    transform = grain.Batch(batch_size=2, drop_remainder=True)
    lazy_dataset_transform = preprocessors.LazyDatasetTransform(transform)
    ds = grain.MapDataset.source(list(range(5)))
    ds = lazy_dataset_transform(ds)
    self.assertListEqual([t.tolist() for t in list(ds)], [[0, 1], [2, 3]])

  def test_batch_lazydataset_transform_without_drop_remainder(self):
    transform = grain.Batch(batch_size=2)
    lazy_dataset_transform = preprocessors.LazyDatasetTransform(transform)
    ds = grain.MapDataset.source(list(range(5)))
    ds = lazy_dataset_transform(ds)
    self.assertListEqual([t.tolist() for t in list(ds)], [[0, 1], [2, 3], [4]])


class PreprocessorsWithInjectedArgsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"val": 3},
        split="train",
    )

  def _update_runtime_args(self, run_args):
    new_seq_lens = {}
    for k, v in run_args.sequence_lengths.items():
      new_seq_lens[f"{k}_new"] = v
      new_seq_lens[k] = v + 1
    updated = run_args.replace(sequence_lengths=new_seq_lens)
    return updated

  def test_map_fn_preprocessor(self):
    def test_map_fn(
        ex, run_args: core_preprocessors_lib.AirIOInjectedRuntimeArgs
    ):
      return ex + run_args.sequence_lengths["val"]

    task = dataset_providers.GrainTask(
        name="test_task",
        source=_get_test_src(),
        preprocessors=[
            preprocessors.MapFnTransform(test_map_fn, self._runtime_args)
        ],
    )
    ds = task.get_dataset({"val": 3}, "train", shuffle=False)
    self.assertListEqual(list(ds), list(range(3, 8)))

  def test_random_map_fn_preprocessor(self):
    def test_random_map_fn(
        ex, rng, r_args: core_preprocessors_lib.AirIOInjectedRuntimeArgs
    ):
      return (
          ex
          + r_args.sequence_lengths["val"]
          + int(jax.random.randint(rng, [], 0, 10))
      )

    task = dataset_providers.GrainTask(
        name="test_task",
        source=_get_test_src(),
        preprocessors=[
            preprocessors.RandomMapFnTransform(
                test_random_map_fn, self._runtime_args
            )
        ],
    )
    ds = task.get_dataset({"val": 3}, "train", shuffle=False, seed=42)
    self.assertListEqual(list(ds), [8, 12, 10, 6, 15])

  def test_filter_fn_preprocessor(self):
    def test_filter_fn(
        ex, rargs: core_preprocessors_lib.AirIOInjectedRuntimeArgs
    ):
      return ex > rargs.sequence_lengths["val"]

    task = dataset_providers.GrainTask(
        name="test_task",
        source=_get_test_src(),
        preprocessors=[
            preprocessors.FilterFnTransform(test_filter_fn, self._runtime_args)
        ],
    )
    ds = task.get_dataset({"val": 3}, "train", shuffle=False, seed=42)
    self.assertListEqual(list(ds), [4])

  def test_unannotated_runtime_args(self):
    def test_map_fn(ex, run_args):
      return ex + run_args.sequence_lengths["val"]

    task = dataset_providers.GrainTask(
        name="test_task",
        source=_get_test_src(),
        preprocessors=[
            preprocessors.MapFnTransform(test_map_fn, self._runtime_args)
        ],
    )
    with self.assertRaisesRegex(
        TypeError, "missing 1 required positional argument: 'run_args'"
    ):
      ds = task.get_dataset(None, "train", shuffle=False)
      _ = list(ds)

  def test_map_lazydataset_transform(self):

    def test_map_fn(
        ex, run_args: core_preprocessors_lib.AirIOInjectedRuntimeArgs
    ):
      return ex + run_args.sequence_lengths["val"]

    transform = preprocessors.MapFnTransform(test_map_fn)
    lazy_dataset_transform = preprocessors.LazyDatasetTransform(transform)
    ds = grain.MapDataset.source(list(range(5)))
    ds = lazy_dataset_transform(ds, runtime_args=self._runtime_args)
    self.assertListEqual(list(ds), list(range(3, 8)))

  def test_map_lazydataset_transform_updated_runtime_args(self):

    def test_map_fn(
        ex, run_args: core_preprocessors_lib.AirIOInjectedRuntimeArgs
    ):
      return ex + run_args.sequence_lengths["val"]

    transform = preprocessors.MapFnTransform(
        test_map_fn, update_runtime_args=self._update_runtime_args
    )
    lazy_dataset_transform = preprocessors.LazyDatasetTransform(transform)
    ds = grain.MapDataset.source(list(range(5)))
    ds = lazy_dataset_transform(ds, runtime_args=self._runtime_args)
    updated_runtime_args = lazy_dataset_transform.get_updated_runtime_args(
        self._runtime_args
    )
    expected_runtime_args = self._runtime_args.replace(
        sequence_lengths={"val": 4, "val_new": 3},
    )
    self.assertListEqual(list(ds), list(range(3, 8)))
    self.assertEqual(updated_runtime_args, expected_runtime_args)

  def test_random_map_fn_lazydataset_transform(self):

    def test_random_map_fn(
        ex, rng, r_args: core_preprocessors_lib.AirIOInjectedRuntimeArgs
    ):
      return (
          ex
          + r_args.sequence_lengths["val"]
          + int(jax.random.randint(rng, [], 0, 10))
      )

    transform = preprocessors.RandomMapFnTransform(test_random_map_fn)
    lazy_dataset_transform = preprocessors.LazyDatasetTransform(transform)
    ds = grain.MapDataset.source(list(range(5)))
    ds = lazy_dataset_transform(
        ds, rng=jax.random.key(42), runtime_args=self._runtime_args
    )
    self.assertListEqual(list(ds), [8, 7, 8, 15, 16])

  def test_random_map_fn_lazydataset_transform_updated_runtime_args(self):

    def test_random_map_fn(
        ex, rng, r_args: core_preprocessors_lib.AirIOInjectedRuntimeArgs
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
    ds = grain.MapDataset.source(list(range(5)))
    ds = lazy_dataset_transform(
        ds, rng=jax.random.key(42), runtime_args=self._runtime_args
    )
    updated_runtime_args = lazy_dataset_transform.get_updated_runtime_args(
        self._runtime_args
    )
    expected_runtime_args = self._runtime_args.replace(
        sequence_lengths={"val": 4, "val_new": 3}
    )
    self.assertListEqual(list(ds), [8, 7, 8, 15, 16])
    self.assertEqual(updated_runtime_args, expected_runtime_args)

  def test_filter_lazydataset_transform(self):

    def test_filter_fn(
        ex, rargs: core_preprocessors_lib.AirIOInjectedRuntimeArgs
    ):
      return ex > rargs.sequence_lengths["val"]

    transform = preprocessors.FilterFnTransform(test_filter_fn)
    lazy_dataset_transform = preprocessors.LazyDatasetTransform(transform)
    ds = grain.MapDataset.source(list(range(5)))
    ds = lazy_dataset_transform(ds, runtime_args=self._runtime_args)
    self.assertListEqual(list(ds), [4])

  def test_filter_lazydataset_transform_updated_runtime_args(self):

    def test_filter_fn(
        ex, rargs: core_preprocessors_lib.AirIOInjectedRuntimeArgs
    ):
      return ex > rargs.sequence_lengths["val"]

    transform = preprocessors.FilterFnTransform(
        test_filter_fn, update_runtime_args=self._update_runtime_args
    )
    lazy_dataset_transform = preprocessors.LazyDatasetTransform(transform)
    ds = grain.MapDataset.source(list(range(5)))
    ds = lazy_dataset_transform(ds, runtime_args=self._runtime_args)
    updated_runtime_args = lazy_dataset_transform.get_updated_runtime_args(
        self._runtime_args
    )
    expected_runtime_args = self._runtime_args.replace(
        sequence_lengths={"val": 4, "val_new": 3}
    )
    self.assertListEqual(list(ds), [4])
    self.assertEqual(updated_runtime_args, expected_runtime_args)

  def test_lazy_map_transform(self):
    run_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"val": 3}
    )
    transform = preprocessors.LazyMapTransform(
        _lazy_map_fn,
        update_runtime_args=self._update_runtime_args,
        produces_none_elements=False,
        requires_non_none_elements=False,
    )
    ds = grain.MapDataset.source(range(10))
    ds = transform(ds, run_args, rng=None)
    updated_runtime_args = transform.update_runtime_args(run_args)
    expected_runtime_args = run_args.replace(
        sequence_lengths={"val": 4, "val_new": 3}
    )
    self.assertListEqual(list(ds), list(range(3, 13)))
    self.assertEqual(updated_runtime_args, expected_runtime_args)

  def test_lazy_map_transform_with_rng(self):
    run_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"val": 3}
    )
    transform = preprocessors.LazyMapTransform(
        _lazy_map_fn,
        update_runtime_args=self._update_runtime_args,
        produces_none_elements=False,
        requires_non_none_elements=False,
    )
    ds = grain.MapDataset.source(range(10))
    ds = transform(ds, run_args, rng=jax.random.key(42))
    updated_runtime_args = transform.update_runtime_args(run_args)
    expected_runtime_args = run_args.replace(
        sequence_lengths={"val": 4, "val_new": 3},
    )
    self.assertListEqual(list(ds), [7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    self.assertEqual(updated_runtime_args, expected_runtime_args)

  def test_lazy_map_transform_with_none_elements(self):

    class MyLazyMapDataset(grain.MapDataset):

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

    run_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"val": 3}
    )
    transform = preprocessors.LazyMapTransform(
        lazy_map_fn_with_nones,
        update_runtime_args=self._update_runtime_args,
        produces_none_elements=True,
        requires_non_none_elements=False,
    )
    ds = grain.MapDataset.source(range(10))
    ds = transform(ds, run_args, rng=None)
    updated_runtime_args = transform.update_runtime_args(run_args)
    expected_runtime_args = run_args.replace(
        sequence_lengths={"val": 4, "val_new": 3}
    )
    self.assertListEqual(list(ds), list(range(4, 10)))
    self.assertEqual(updated_runtime_args, expected_runtime_args)

  def test_lazy_iter_transform(self):
    run_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"val": 3}
    )
    transform = preprocessors.LazyIterTransform(
        _lazy_map_fn,
        update_runtime_args=self._update_runtime_args,
    )
    ds = grain.MapDataset.source(range(10))
    ds = ds.to_iter_dataset()
    ds = transform(ds, run_args, rng=None)
    updated_runtime_args = transform.update_runtime_args(run_args)
    expected_runtime_args = run_args.replace(
        sequence_lengths={"val": 4, "val_new": 3},
    )
    self.assertListEqual(list(ds), list(range(3, 13)))
    self.assertEqual(updated_runtime_args, expected_runtime_args)

  def test_lazy_iter_transform_with_rng(self):
    run_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"val": 3}
    )
    transform = preprocessors.LazyIterTransform(
        _lazy_map_fn,
        update_runtime_args=self._update_runtime_args,
    )
    ds = grain.MapDataset.source(range(10))
    ds = ds.to_iter_dataset()
    ds = transform(ds, run_args, rng=jax.random.key(42))
    updated_runtime_args = transform.update_runtime_args(run_args)
    expected_runtime_args = run_args.replace(
        sequence_lengths={"val": 4, "val_new": 3},
    )
    self.assertListEqual(list(ds), [7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    self.assertEqual(updated_runtime_args, expected_runtime_args)

  def test_lazy_iter_transform_on_map_dataset_fails(self):
    run_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"val": 3}
    )
    transform = preprocessors.LazyIterTransform(
        _lazy_map_fn,
        update_runtime_args=self._update_runtime_args,
    )
    self.assertTrue(
        preprocessors.LazyDatasetTransform(transform).requires_iter_dataset
    )
    ds = grain.MapDataset.source(range(10))
    with self.assertRaisesRegex(
        ValueError, "Cannot apply LazyIterDataset transform.*"
    ):
      _ = transform(ds, run_args, rng=None)

  def test_map_transform_on_iter_dataset(self):
    transform = preprocessors.MapFnTransform(lambda x: x + 1)
    ds = grain.MapDataset.source(range(10))
    ds = ds.to_iter_dataset()
    ds = preprocessors.LazyDatasetTransform(transform)(ds)
    self.assertListEqual(list(ds), list(range(1, 11)))

  def test_filter_transform_on_iter_dataset(self):
    transform = preprocessors.FilterFnTransform(lambda x: x > 5)
    ds = grain.MapDataset.source(range(10))
    ds = ds.to_iter_dataset()
    ds = preprocessors.LazyDatasetTransform(transform)(ds)
    self.assertListEqual(list(ds), list(range(6, 10)))

  def test_batch_transform_on_iter_dataset(self):
    batch_with_drop = grain.Batch(batch_size=3, drop_remainder=True)
    batch_without_drop = grain.Batch(batch_size=3, drop_remainder=False)
    ds = grain.MapDataset.source(range(10))
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
    run_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"val": 3}
    )
    transform = preprocessors.LazyIterTransform(
        _lazy_map_fn,
        update_runtime_args=self._update_runtime_args,
    )
    ds = grain.MapDataset.source(range(10))
    ds = ds.to_iter_dataset()
    ds = transform(ds, run_args, rng=None)
    self.assertListEqual(list(ds), list(range(3, 13)))

  def test_random_map_transform_on_iter_dataset_fails(self):
    def test_random_map_fn(ex, rng):
      return ex + int(jax.random.randint(rng, [], 0, 10))

    transform = preprocessors.RandomMapFnTransform(test_random_map_fn)
    ds = grain.MapDataset.source(range(10))
    ds = ds.to_iter_dataset()
    run_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"val": 3}
    )
    with self.assertRaisesRegex(
        ValueError, "RandomMapFnTransform is not yet supported.*"
    ):
      _ = preprocessors.LazyDatasetTransform(transform)(ds, run_args)

  def test_lazy_map_transform_on_iter_dataset_fails(self):
    transform = preprocessors.LazyMapTransform(
        _lazy_map_fn,
        update_runtime_args=self._update_runtime_args,
        produces_none_elements=False,
        requires_non_none_elements=False,
    )
    ds = grain.MapDataset.source(range(10))
    ds = ds.to_iter_dataset()
    run_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"val": 3}
    )
    with self.assertRaisesRegex(
        ValueError, "Cannot apply LazyMapDataset transform.*"
    ):
      _ = transform(ds, run_args, rng=None)

  def test_produces_none_elements_map_fn(self):
    prep = preprocessors.MapFnTransform(lambda x: x + 1)
    self.assertFalse(
        preprocessors.LazyDatasetTransform(prep).produces_none_elements
    )

  def test_produces_none_elements_filter_fn(self):
    prep = preprocessors.FilterFnTransform(lambda x: x > 1)
    self.assertTrue(
        preprocessors.LazyDatasetTransform(prep).produces_none_elements
    )

  def test_produces_none_elements_random_map_fn(self):
    def test_random_map_fn(ex, rng):
      return ex + int(jax.random.randint(rng, [], 0, 10))

    prep = preprocessors.RandomMapFnTransform(test_random_map_fn)
    self.assertFalse(
        preprocessors.LazyDatasetTransform(prep).produces_none_elements
    )

  def test_produces_none_elements_lazy_map_transform_with_none(self):
    prep = preprocessors.LazyMapTransform(
        _lazy_map_fn,
        update_runtime_args=self._update_runtime_args,
        produces_none_elements=True,
        requires_non_none_elements=False,
    )
    self.assertTrue(
        preprocessors.LazyDatasetTransform(prep).produces_none_elements
    )

  def test_produces_none_elements_lazy_map_transform_without_none(self):
    prep = preprocessors.LazyMapTransform(
        _lazy_map_fn,
        update_runtime_args=self._update_runtime_args,
        produces_none_elements=False,
        requires_non_none_elements=False,
    )
    self.assertFalse(
        preprocessors.LazyDatasetTransform(prep).produces_none_elements
    )

  def test_produces_none_elements_lazy_iter_transform(self):
    prep = preprocessors.LazyIterTransform(
        _lazy_map_fn,
        update_runtime_args=self._update_runtime_args,
    )
    self.assertFalse(
        preprocessors.LazyDatasetTransform(prep).produces_none_elements
    )



class ConvertBoxesYXYXToCXCYHWTest(absltest.TestCase):

  def test_conversion_single_box(self):
    transform = preprocessors.ConvertBoxesYXYXToCXCYHW(box_field_name="boxes")
    data = {"boxes": np.array([0.1, 0.2, 0.5, 0.6], dtype=np.float32)}
    # Use 'element' to match the updated signature, although the test interacts
    # with the dict directly, not the map method parameter name.
    # We pass 'data' dict.
    result = transform.map(data)

    # ymin=0.1, xmin=0.2, ymax=0.5, xmax=0.6
    # height = 0.5 - 0.1 = 0.4
    # width = 0.6 - 0.2 = 0.4
    # center_y = 0.1 + (0.4 / 2) = 0.3
    # center_x = 0.2 + (0.4 / 2) = 0.4
    # Expected: [cx, cy, h, w]
    expected = np.array([0.4, 0.3, 0.4, 0.4], dtype=np.float32)

    self.assertIn("boxes", result)
    np.testing.assert_allclose(result["boxes"], expected)

  def test_conversion_multiple_boxes(self):
    transform = preprocessors.ConvertBoxesYXYXToCXCYHW(box_field_name="boxes")
    boxes_data = np.array(
        [
            [0.1, 0.2, 0.5, 0.6],  # Box 1
            [0.0, 0.0, 1.0, 1.0],  # Box 2
        ],
        dtype=np.float32,
    )
    data = {"boxes": boxes_data}
    result = transform.map(data)

    # Box 1 expected: [0.4, 0.3, 0.4, 0.4]
    # Box 2 expected: [0.5, 0.5, 1.0, 1.0]
    expected = np.array(
        [
            [0.4, 0.3, 0.4, 0.4],
            [0.5, 0.5, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    self.assertIn("boxes", result)
    np.testing.assert_allclose(result["boxes"], expected)
    self.assertEqual(result["boxes"].shape, (2, 4))

  def test_conversion_batched_shape(self):
    # Test tensor with shape [Batch, NumBoxes, 4]
    transform = preprocessors.ConvertBoxesYXYXToCXCYHW(box_field_name="boxes")
    boxes_data = np.array(
        [
            [  # Batch 1
                [0.1, 0.2, 0.5, 0.6],
                [0.0, 0.0, 1.0, 1.0],
            ],
            [  # Batch 2
                [0.2, 0.2, 0.4, 0.4],
                [0.5, 0.5, 0.6, 0.7],
            ],
        ],
        dtype=np.float32,
    )
    data = {"boxes": boxes_data}
    result = transform.map(data)

    # Batch 1 expected:
    #   [0.4, 0.3, 0.4, 0.4]
    #   [0.5, 0.5, 1.0, 1.0]
    # Batch 2 expected:
    #   [0.3, 0.3, 0.2, 0.2]
    #   [0.6, 0.55, 0.1, 0.2]
    expected = np.array(
        [
            [
                [0.4, 0.3, 0.4, 0.4],
                [0.5, 0.5, 1.0, 1.0],
            ],
            [
                [0.3, 0.3, 0.2, 0.2],
                [0.6, 0.55, 0.1, 0.2],
            ],
        ],
        dtype=np.float32,
    )
    self.assertEqual(result["boxes"].shape, (2, 2, 4))
    np.testing.assert_allclose(result["boxes"], expected)

  def test_shape_value_error(self):
    transform = preprocessors.ConvertBoxesYXYXToCXCYHW(box_field_name="boxes")
    # Wrong shape: [..., 3] instead of [..., 4]
    data = {"boxes": np.array([0.1, 0.2, 0.5], dtype=np.float32)}
    with self.assertRaisesRegex(
        ValueError, "must have shape .*4.*, but got shape .*3"
    ):
      transform.map(data)

    # Also test batched wrong shape
    data_batch = {
        "boxes": np.array([[[0.1, 0.2, 0.5]]], dtype=np.float32)
    }
    with self.assertRaisesRegex(
        ValueError, "must have shape .*4.*, but got shape .*1, 1, 3"
    ):
      transform.map(data_batch)


if __name__ == "__main__":
  absltest.main()
