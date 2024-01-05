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

"""Tests for span_corruption preprocessor."""
import os
from absl.testing import absltest
from airio.common import span_corruption
import seqio
import tensorflow as tf


class SpanCorruptionTest(absltest.TestCase):

  def test_span_corruption(self):
    # This test uses 500 tokenized examples from the C4 dataset checked into
    # test data as source, applies the span corruption preprocessor, and
    # compares it to golden data checked into test data.


    # Step 1: Create a data source over the tokenized c4 test data.
    test_data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../test_data/span_corruption",
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

    # Step 2: Load the test data and apply the span corruption preprocessor
    test_vocab = seqio.PassThroughVocabulary(size=32100)
    output_features = {
        "inputs": seqio.Feature(
            vocabulary=test_vocab,
            add_eos=True,
            required=False,
        ),
        "targets": seqio.Feature(
            vocabulary=test_vocab, add_eos=True
        ),
    }
    sequence_length = {"inputs": 1024, "targets": 1024}
    ds = test_src.get_dataset("train")
    # The seqio map_seed_manager + map_over_dataset utils implement reproducible
    # seed distribution to stochastic preprocesors, which are used extensively
    # in span corruption. The seed distribution in airio + grain will be
    # different, so we'll need a way to pass consistent seeds to both impls for
    # this test to work going forward.
    with seqio.map_seed_manager(initial_seed=94043):
      ds = span_corruption.span_corruption(
          ds, sequence_length=sequence_length, output_features=output_features
      )

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
    seqio.test_utils.assert_datasets_eq(ds, expected_ds)


if __name__ == "__main__":
  absltest.main()
