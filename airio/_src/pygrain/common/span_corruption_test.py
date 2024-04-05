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

"""Tests for span_corruption preprocessor."""

import functools
import itertools
import os
from typing import Any, Callable

from absl.testing import absltest
from airio._src.core import test_utils
from airio._src.core import tokenizer
from airio._src.pygrain import lazy_dataset_transforms
from airio._src.pygrain.common import packing
from airio._src.pygrain.common import span_corruption as asc
import grain.python as grain
import jax
import numpy as np
import seqio
from t5.data import preprocessors as t5_preps
import tensorflow as tf


lazy_dataset = grain.experimental.lazy_dataset
mock = absltest.mock


# Utils
def _get_seeds(seed: int, num_seeds: int, ds_size: int):
  # Replicates seed distribution logic in seqio map_over_dataset
  random_ds_seeds = np.arange(seed, seed + 2 * num_seeds).reshape(-1, 2)
  random_ds_seeds = tuple(tuple(s) for s in random_ds_seeds)
  seed_ds = tf.nest.map_structure(
      tf.data.experimental.RandomDataset, random_ds_seeds
  )
  range_ds = tf.data.Dataset.from_tensor_slices(range(ds_size))
  zip_ds = tf.data.Dataset.zip(range_ds, seed_ds)
  return [d[1][0] for d in zip_ds.as_numpy_iterator()]


TEST_SEEDS = itertools.chain([
    _get_seeds(94043, 1, 500),  # used by select random chunk.
    _get_seeds(94047, 1, 500),  # used by denoise.
])


class _MapWithPresetSeeds(lazy_dataset.LazyMapDataset):
  """Helper to run stochastic preprocessors with a given sequence of seeds."""

  def __init__(
      self,
      parent: lazy_dataset.LazyMapDataset,
      fn: Callable[[Any, Any], Any],  # example, seed -> example
      seeds: list[tuple[int, int]] | None = None,
  ):
    super().__init__(parent)
    self.seeds = seeds
    if not self.seeds:
      self.seeds = next(TEST_SEEDS)
    if len(self._parent) > len(self.seeds):
      raise ValueError("seeds must have equal or more elements than parent.")
    self.fn = fn

  def __len__(self):
    return len(self._parent)

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    seed = tf.cast(self.seeds[index], tf.int64)
    return self.fn(self._parent[index], seed)


def mock_random_map_fn_lazy_map_dataset(ds, map_fn, seed):
  # Use preset test seeds.
  del seed
  # Remove possible sparseness.
  ds = lazy_dataset.SourceLazyMapDataset(list(iter(ds)))
  # Apply map_fn with preset test seeds.
  return _MapWithPresetSeeds(ds, map_fn)


class SpanCorruptionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Replicate seed update logic from seqio.map_over_dataset:
    # updated_seed = seed + 2 * num_seeds
    # This update is applied once after select random chunk and once after
    # denoise, and num_seeds = 1 for both. Hence updated_seed = seed + 4
    self.enter_context(
        mock.patch.object(
            asc,
            "update_seed",
            side_effect=lambda x: x + 4,
        )
    )
    # Replace AirIO's seed distribution with preset seeds that replicate logic
    # from seqio.map_over_dataset.
    self.enter_context(
        mock.patch.object(
            lazy_dataset_transforms,
            "RandomMapFnLazyMapDataset",
            side_effect=mock_random_map_fn_lazy_map_dataset,
        )
    )
    self.enter_context(
        mock.patch.object(
            jax.random,
            "key_data",
            side_effect=lambda x: x,
        )
    )

  def test_span_corruption(self):
    # This test uses 500 tokenized examples from the C4 dataset checked into
    # test data as source, applies the span corruption preprocessor, and
    # compares it to golden data checked into test data.


    # Step 1: Create a data source over the tokenized c4 test data.
    test_data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../test_data/span_corruption",
    )
    src_filename = os.path.join(
        test_data_dir, "c4_tokenized_t5_default_vocab_500_examples.tfrecord*"
    )
    src_feature_description = {
        "targets": tf.io.FixedLenSequenceFeature(
            [], dtype=tf.int64, allow_missing=True
        )
    }
    test_src = seqio.TFExampleDataSource(
        split_to_filepattern={"train": src_filename},
        feature_description=src_feature_description,
    )
    src_ds = test_src.get_dataset("train")

    # Step 2: Apply span corruption preprocessor
    test_vocab = seqio.PassThroughVocabulary(size=32100)
    tokenizer_configs = {
        "inputs": tokenizer.TokenizerConfig(vocab=test_vocab, add_eos=True),
        "targets": tokenizer.TokenizerConfig(vocab=test_vocab, add_eos=True),
    }
    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 1024, "targets": 1024},
    )
    examples = list(src_ds.as_numpy_iterator())
    unused_seed = 1  # overridden by mocked fn to match seqio seed distribution
    airio_ds = lazy_dataset.SourceLazyMapDataset(examples)
    airio_ds = asc.span_corruption(
        airio_ds,
        seed=unused_seed,
        tokenizer_configs=tokenizer_configs,
        runtime_args=runtime_args,
    )
    airio_ds_iter = iter(airio_ds)

    # Step 3: Load golden data and compare.
    output_filename = os.path.join(
        test_data_dir,
        "c4_span_corruption_t5_default_vocab_inputs_1024_targets_1024_add_eos_true_seed_94043.tfrecord*",
    )
    output_feature_description = {
        "inputs": tf.io.FixedLenSequenceFeature(
            [], dtype=tf.int64, allow_missing=True
        ),
        "targets": tf.io.FixedLenSequenceFeature(
            [], dtype=tf.int64, allow_missing=True
        ),
    }
    out_src = seqio.TFExampleDataSource(
        split_to_filepattern={"train": output_filename},
        feature_description=output_feature_description,
    )
    expected_ds = out_src.get_dataset("train", shuffle=False)
    expected_ds_iter = expected_ds.as_numpy_iterator()
    for expected, actual in zip(expected_ds_iter, airio_ds_iter, strict=True):
      np.testing.assert_array_equal(expected["inputs"], actual["inputs"])
      np.testing.assert_array_equal(expected["targets"], actual["targets"])

  def _get_seeds(self, seed: int, num_seeds: int, ds_size: int):
    # Replicates seed distribution logic in seqio map_over_dataset
    random_ds_seeds = np.arange(seed, seed + 2 * num_seeds).reshape(-1, 2)
    random_ds_seeds = tuple(tuple(s) for s in random_ds_seeds)
    seed_ds = tf.nest.map_structure(
        tf.data.experimental.RandomDataset, random_ds_seeds
    )
    range_ds = tf.data.Dataset.from_tensor_slices(range(ds_size))
    zip_ds = tf.data.Dataset.zip(range_ds, seed_ds)
    return [d[1][0] for d in zip_ds.as_numpy_iterator()]

  def test_select_random_chunk_equivalence(self):
    # Step 1: Load tokenized test data.
    test_data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../test_data/span_corruption",
    )
    src_filename = os.path.join(
        test_data_dir, "c4_tokenized_t5_default_vocab_500_examples.tfrecord*"
    )
    src_feature_description = {
        "targets": tf.io.FixedLenSequenceFeature(
            [], dtype=tf.int64, allow_missing=True
        )
    }
    test_src = seqio.TFExampleDataSource(
        split_to_filepattern={"train": src_filename},
        feature_description=src_feature_description,
    )
    src_ds = test_src.get_dataset("train")

    # Step 2: Populate params.
    feature_key = "targets"
    max_length = 65536
    passthrough_feature_keys = []

    # Step 3: Run SeqIO path
    test_vocab = seqio.PassThroughVocabulary(size=32100)
    output_features = {
        "inputs": seqio.Feature(
            vocabulary=test_vocab,
            add_eos=True,
            required=False,
        ),
        "targets": seqio.Feature(vocabulary=test_vocab, add_eos=True),
    }
    with seqio.map_seed_manager(initial_seed=94043):
      seqio_ds = t5_preps.select_random_chunk(
          src_ds,
          output_features=output_features,
          feature_key=feature_key,
          max_length=max_length,
          passthrough_feature_keys=passthrough_feature_keys,
      )

    # Step 4: Run AirIO path.
    filter_fn = functools.partial(asc.filter_empty, feature_key=feature_key)
    map_fn = functools.partial(
        t5_preps.single_example_select_random_chunk,
        output_features=output_features,
        max_length=max_length,
        feature_key=feature_key,
        passthrough_feature_keys=passthrough_feature_keys,
    )
    examples = list(src_ds.as_numpy_iterator())
    airio_ds = lazy_dataset.SourceLazyMapDataset(examples)
    airio_ds = airio_ds.filter(filter_fn)
    # Get seed sequnce generated by SeqIO map_seed_manager + map_over_dataset to
    # test exact equivalence.
    seeds = self._get_seeds(seed=94043, num_seeds=1, ds_size=500)
    airio_ds = _MapWithPresetSeeds(airio_ds, map_fn, seeds)

    # Step 5: Verify that they are exactly the same.
    seqio_ds_iter = seqio_ds.as_numpy_iterator()
    airio_ds_iter = iter(airio_ds)
    for seqio_ex, airio_ex in zip(seqio_ds_iter, airio_ds_iter, strict=True):
      np.testing.assert_array_equal(seqio_ex["targets"], airio_ex["targets"])

  def test_split_tokens_equivalence(self):
    # Step 1: Load tokenized test data.
    test_data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../test_data/span_corruption",
    )
    src_filename = os.path.join(
        test_data_dir, "c4_tokenized_t5_default_vocab_500_examples.tfrecord*"
    )
    src_feature_description = {
        "targets": tf.io.FixedLenSequenceFeature(
            [], dtype=tf.int64, allow_missing=True
        )
    }
    test_src = seqio.TFExampleDataSource(
        split_to_filepattern={"train": src_filename},
        feature_description=src_feature_description,
    )
    src_ds = test_src.get_dataset("train")

    # Step 2: Populate params.
    feature_key = "targets"
    max_length = 65536
    passthrough_feature_keys = []
    pool_size = 128
    input_length, _ = t5_preps.random_spans_helper(
        extra_tokens_per_span_inputs=1,
        extra_tokens_per_span_targets=1,
        inputs_length=1024,
        mean_noise_span_length=3.0,
        noise_density=0.15,
    )
    self.assertEqual(input_length, 1137)  # minor check.
    # Step 3: Run SeqIO path
    test_vocab = seqio.PassThroughVocabulary(size=32100)
    output_features = {
        "inputs": seqio.Feature(
            vocabulary=test_vocab,
            add_eos=True,
            required=False,
        ),
        "targets": seqio.Feature(vocabulary=test_vocab, add_eos=True),
    }
    with seqio.map_seed_manager(initial_seed=94043):
      seqio_ds = t5_preps.select_random_chunk(
          src_ds,
          output_features=output_features,
          feature_key=feature_key,
          max_length=max_length,
          passthrough_feature_keys=passthrough_feature_keys,
      )
      seqio_ds = t5_preps.reduce_concat_tokens(
          seqio_ds, feature_key=feature_key, batch_size=pool_size
      )
      seqio_ds = t5_preps.split_tokens(
          seqio_ds,
          feature_key=feature_key,
          min_tokens_per_segment=None,
          max_tokens_per_segment=input_length,
          passthrough_feature_keys=passthrough_feature_keys,
      )

    # Step 4: Run AirIO path.
    filter_fn = functools.partial(asc.filter_empty, feature_key=feature_key)
    map_fn = functools.partial(
        t5_preps.single_example_select_random_chunk,
        output_features=output_features,
        max_length=max_length,
        feature_key=feature_key,
        passthrough_feature_keys=passthrough_feature_keys,
    )
    packer = packing.NoamPacker(feature_lengths={"targets": input_length})

    examples = list(src_ds.as_numpy_iterator())
    airio_ds = lazy_dataset.SourceLazyMapDataset(examples)
    airio_ds = airio_ds.filter(filter_fn)
    # Get seed sequnce generated by SeqIO map_seed_manager + map_over_dataset to
    # test exact equivalence.
    seeds = self._get_seeds(seed=94043, num_seeds=1, ds_size=500)
    airio_ds = _MapWithPresetSeeds(airio_ds, map_fn, seeds)
    airio_ds = packing.PackLazyMapDataset(
        airio_ds, pool_size=pool_size, packer=packer
    )

    # Step 5: Verify that they are exactly the same.
    seqio_ds_iter = seqio_ds.as_numpy_iterator()
    airio_ds_iter = iter(airio_ds)
    for seqio_ex, airio_ex in zip(seqio_ds_iter, airio_ds_iter, strict=True):
      np.testing.assert_array_equal(seqio_ex["targets"], airio_ex["targets"])


if __name__ == "__main__":
  absltest.main()
