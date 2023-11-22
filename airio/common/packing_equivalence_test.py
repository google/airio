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

"""Tests for equivalence between the airio and seqio packing impls."""

import functools
from absl.testing import absltest
import airio
import airio.common
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

    # This test takes one file shard of the "c4/en:2.2.0" dataset, rekeys and
    # tokenizes the examples, packs them using the noam packing impls in T5
    # and AirIO respectively and verifies that they're exactly the same.
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
    packed_airio_ds, _ = airio.common.packing.NoamPackPreprocessor(
        packed_airio_ds, runtime_args
    )
    packed_airio_ds_iter = iter(packed_airio_ds)

    # Step 4: Verify that they are exactly the same.
    for seqio_packed, airio_packed in zip(
        packed_seqio_ds_iter, packed_airio_ds_iter, strict=True
    ):
      np.testing.assert_array_equal(
          seqio_packed["targets"], airio_packed["targets"]
      )


if __name__ == "__main__":
  absltest.main()
