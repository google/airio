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

"""Tests for equivalence between the airio and seqio packing impls."""

import functools
from absl.testing import absltest
import airio
import airio.pygrain.common
import grain.python as grain
import numpy as np
import seqio
import t5.data

lazy_dataset = grain.experimental.lazy_dataset


class PackingEquivalenceTest(absltest.TestCase):

  def test_noam_packing_equivalence_on_c4(self):
    # Note: The test is currently limited to 10,000 source examples because the
    # airio / pygrain impl is slow. Remove when fixed.
    src_example_limit = 10000

    # This test takes one file shard (out of 1024) of the "c4/en:2.2.0" dataset,
    # rekeys and tokenizes the examples, packs them using the noam packing impls
    # in T5 and AirIO respectively and verifies that they're exactly the same.
    # There are 356,137 examples in the source, and packing produces 179,919
    # examples.

    # Step 1: Prepare tokenized data to be used by both packing code paths.
    # SeqIO is used for convenience.
    output_features = {
        "targets": seqio.Feature(
            vocabulary=t5.data.get_default_vocabulary(), add_eos=False
        )
    }
    rekey = functools.partial(
        seqio.preprocessors.rekey, key_map={"targets": "text"}
    )
    tokenize = functools.partial(
        seqio.preprocessors.tokenize,
        output_features=output_features,
        copy_pretokenized=False,
        with_eos=False,
    )
    src = seqio.TfdsDataSource("c4/en:2.2.0")
    ds = src.get_dataset(  # 356,137 examples
        split="train",
        shuffle=False,
        seed=42,
        shard_info=seqio.ShardInfo(index=0, num_shards=1024),
    )
    ds = ds.take(src_example_limit)
    ds = rekey(ds)
    ds = tokenize(ds)

    # Step 2: Pack using t5.data
    packed_seqio_ds = ds.map(lambda x: x)  # make a copy
    packed_seqio_ds = t5.data.preprocessors.reduce_concat_tokens(
        packed_seqio_ds, feature_key="targets", batch_size=128
    )
    packed_seqio_ds = t5.data.preprocessors.split_tokens(
        packed_seqio_ds,
        feature_key="targets",
        min_tokens_per_segment=None,
        max_tokens_per_segment=1024,
        passthrough_feature_keys=[],
    )
    packed_seqio_ds_iter = packed_seqio_ds.as_numpy_iterator()

    # Step 3: Pack using AirIO
    # Populate the tokenized dataset in-memory to create a LazyMapDataset; this
    # is slow and expensive.
    examples = list(ds.as_numpy_iterator())
    packed_airio_ds = lazy_dataset.SourceLazyMapDataset(examples)
    runtime_args = airio.preprocessors.AirIOInjectedRuntimeArgs(
        sequence_lengths={"targets": 1024}, split="unused"
    )
    unused_rng = None
    packed_airio_ds = airio.pygrain.common.packing.NoamPackMapPreprocessor(
        packed_airio_ds, runtime_args, unused_rng
    )
    packed_airio_ds_iter = iter(packed_airio_ds)

    # Step 4: Verify that they are exactly the same.
    for seqio_packed, airio_packed in zip(
        packed_seqio_ds_iter, packed_airio_ds_iter, strict=True
    ):
      np.testing.assert_array_equal(
          seqio_packed["targets"], airio_packed["targets"]
      )

  def test_single_bin_true_packing_equivalence_on_wmt(self):
    # Note: The test is currently limited to 1000 source examples because the
    # airio / pygrain impl is slow. Remove when fixed.
    src_example_limit = 1000

    # This test takes one file shard (out of 128) of the
    # "wmt19_translate/de-en:1.0.0" dataset, preprocesses and tokenizes the
    # examples, packs them using the single-bin true packing impls in SeqIO and
    # AirIO respectively and verifies that they produce the same elements. There
    # are 302,268 examples in the source, and packing produces 9,647 examples.

    # Step 1: Prepare tokenized data to be used by both packing code paths.
    # SeqIO is used for convenience.
    output_features = {
        "inputs": seqio.Feature(
            vocabulary=t5.data.get_default_vocabulary(), add_eos=False
        ),
        "targets": seqio.Feature(
            vocabulary=t5.data.get_default_vocabulary(), add_eos=False
        ),
    }
    translate = functools.partial(
        t5.data.preprocessors.translate,
        source_language="en",
        target_language="de",
    )
    tokenize = functools.partial(
        seqio.preprocessors.tokenize,
        output_features=output_features,
        copy_pretokenized=False,
        with_eos=False,
    )

    src = seqio.TfdsDataSource("wmt19_translate/de-en:1.0.0")
    ds = src.get_dataset(
        split="train",
        shuffle=False,
        seed=42,
        shard_info=seqio.ShardInfo(index=0, num_shards=128),
    )
    ds = ds.take(src_example_limit)
    ds = translate(ds)
    ds = tokenize(ds)
    feature_lengths = {"inputs": 1024, "targets": 1024}

    # Step 2: Pack using seqio
    packed_seqio_ds = ds.map(lambda x: x)  # make a copy
    packed_seqio_ds = seqio.utils.trim_and_pack_dataset(
        packed_seqio_ds,
        feature_lengths=feature_lengths,
        use_custom_ops=False,
    )
    packed_seqio_ds_iter = packed_seqio_ds.as_numpy_iterator()

    # Step 3: Pack using AirIO
    # Populate the tokenized dataset in-memory to create a LazyMapDataset; this
    # is slow and expensive.
    examples = list(ds.as_numpy_iterator())
    packed_airio_ds = lazy_dataset.SourceLazyMapDataset(examples)
    runtime_args = airio.preprocessors.AirIOInjectedRuntimeArgs(
        sequence_lengths=feature_lengths, split="unused"
    )
    unused_rng = None
    pack_prep = airio.pygrain.common.packing.SingleBinTruePackMapPreprocessor
    packed_airio_ds = pack_prep(packed_airio_ds, runtime_args, unused_rng)
    updated_runtime_args = pack_prep.update_runtime_args(runtime_args)
    packed_airio_ds = packed_airio_ds.map(
        functools.partial(
            airio.pygrain.common.pad, runtime_args=updated_runtime_args
        )
    )
    packed_airio_ds_iter = iter(packed_airio_ds)

    # Step 4: Verify that they are exactly the same.
    for seqio_packed, airio_packed in zip(
        packed_seqio_ds_iter, packed_airio_ds_iter, strict=True
    ):
      # Compare keys.
      self.assertSequenceEqual(
          sorted(seqio_packed.keys()), sorted(airio_packed.keys())
      )
      for k in seqio_packed.keys():
        np.testing.assert_array_equal(seqio_packed[k], airio_packed[k])

  def _hash(self, exs: list[dict[str, np.ndarray]]) -> set[int]:
    return set(
        [hash(tuple(ex[k].tobytes() for k in sorted(ex.keys()))) for ex in exs]
    )

  def test_multi_bin_true_packing_equivalence_on_wmt(self):
    # Note: The test is currently limited to 1000 source examples because the
    # airio / pygrain impl is slow. Remove when fixed.
    src_example_limit = 1000

    # This test takes one file shard (out of 128) of the
    # "wmt19_translate/de-en:1.0.0" dataset, preprocesses and tokenizes the
    # examples, packs them using the multi-bin true packing impls in SeqIO and
    # AirIO respectively and verifies that they produce the same elements. The
    # order of packed elements may vary because AirIO yields examples as soon as
    # they are fully packed instead of waiting for the number of partially
    # packed examples to exceed the threshold. There are 302,268 examples in the
    # source, and packing produces 9,420 examples.
    #
    # Note: Yielded fully packed examples immediately may lead to differently
    # (better) packed examples, because there are more partially packed examples
    # available to fit examples into, but this is quite unlikely for large
    # num_partial_examples (which is 1000 for multi-bin true packing).

    # Step 1: Prepare tokenized data to be used by both packing code paths.
    # SeqIO is used for convenience.
    output_features = {
        "inputs": seqio.Feature(
            vocabulary=t5.data.get_default_vocabulary(), add_eos=False
        ),
        "targets": seqio.Feature(
            vocabulary=t5.data.get_default_vocabulary(), add_eos=False
        ),
    }
    translate = functools.partial(
        t5.data.preprocessors.translate,
        source_language="en",
        target_language="de",
    )
    tokenize = functools.partial(
        seqio.preprocessors.tokenize,
        output_features=output_features,
        copy_pretokenized=False,
        with_eos=False,
    )

    src = seqio.TfdsDataSource("wmt19_translate/de-en:1.0.0")
    ds = src.get_dataset(
        split="train",
        shuffle=False,
        seed=42,
        shard_info=seqio.ShardInfo(index=0, num_shards=128),
    )
    ds = ds.take(src_example_limit)
    ds = translate(ds)
    ds = tokenize(ds)
    feature_lengths = {"inputs": 1024, "targets": 1024}

    # Step 2: Pack using seqio
    packed_seqio_ds = ds.map(lambda x: x)  # make a copy
    packed_seqio_ds = seqio.utils.trim_and_pack_dataset(
        packed_seqio_ds,
        feature_lengths=feature_lengths,
        use_custom_ops=True,
    )
    packed_seqio_ds = list(packed_seqio_ds.as_numpy_iterator())

    # Step 3: Pack using AirIO
    # Populate the tokenized dataset in-memory to create a LazyMapDataset; this
    # is slow and expensive.
    examples = list(ds.as_numpy_iterator())
    packed_airio_ds = lazy_dataset.SourceLazyMapDataset(examples)
    runtime_args = airio.preprocessors.AirIOInjectedRuntimeArgs(
        sequence_lengths=feature_lengths, split="unused"
    )
    unused_rng = None
    pack_prep = airio.pygrain.common.packing.MultiBinTruePackMapPreprocessor
    packed_airio_ds = pack_prep(packed_airio_ds, runtime_args, unused_rng)
    updated_runtime_args = pack_prep.update_runtime_args(runtime_args)

    packed_airio_ds = packed_airio_ds.map(
        functools.partial(
            airio.pygrain.common.pad, runtime_args=updated_runtime_args
        )
    )
    packed_airio_ds = list(iter(packed_airio_ds))

    # Step 4: Verify that examples are exactly the same, but possibly reordered.
    # Hash examples and compare sets of hashes.
    packed_seqio_ds_hashes = self._hash(packed_seqio_ds)
    packed_airio_ds_hashes = self._hash(packed_airio_ds)
    self.assertSetEqual(packed_seqio_ds_hashes, packed_airio_ds_hashes)


if __name__ == "__main__":
  absltest.main()
