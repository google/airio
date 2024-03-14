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

import os

from absl.testing import absltest
from airio.examples import mixtures
import airio.pygrain as airio
import airio.pygrain_common as airio_common
import tensorflow_datasets as tfds

_SOURCE_NUM_EXAMPLES = 3
_SOURCE_SEQUENCE_LENGTH = 32


def _pad(
    values: list[int | bool], total_length: int, pad_value: int | bool = 0
):
  """Pads a list of values to total_length with pad_value (default: 0)."""
  if total_length <= len(values):
    return values
  padding_length = total_length - len(values)
  return values + [pad_value] * padding_length


class MixturesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test_data",
    )
    sentencepiece_vocab = airio.SentencePieceVocabulary(
        os.path.join(self.test_dir, "sentencepiece", "sentencepiece.model")
    )
    self.tokenizer_configs = {
        "inputs": airio.TokenizerConfig(vocab=sentencepiece_vocab),
        "targets": airio.TokenizerConfig(vocab=sentencepiece_vocab),
    }

  def test_mc4_mixture(self):
    with tfds.testing.mock_data(_SOURCE_NUM_EXAMPLES):
      mix = mixtures.get_mc4_mixture(tokenizer_configs=self.tokenizer_configs)
    runtime_preprocessors = airio_common.feature_converters.get_t5x_enc_dec_feature_converter_preprocessors(
        pack=False,
        use_multi_bin_packing=False,
        passthrough_feature_keys=[],
        pad_id=0,
        bos_id=0,
    )
    source_sequence_length = 1024
    sequence_lengths = {
        "inputs": source_sequence_length,
        "targets": source_sequence_length,
    }
    ds = mix.get_dataset(
        sequence_lengths,
        shuffle=False,
        seed=42,
        runtime_preprocessors=runtime_preprocessors,
        shard_info=airio.ShardInfo(index=0, num_shards=source_sequence_length),
    )
    expected_first_batch = {
        "decoder_input_tokens": _pad([0, 25, 4, 2], source_sequence_length),
        "decoder_loss_weights": _pad(
            [True, True, True], source_sequence_length, False
        ),
        "decoder_target_tokens": _pad([25, 4, 2], source_sequence_length),
        "encoder_input_tokens": _pad(
            [
                3,
                21,
                3,
                5,
                3,
                2,
                4,
                2,
                20,
                21,
                25,
            ],
            source_sequence_length,
        ),
    }
    actual_first_batch = {k: v.tolist() for k, v in next(ds).items()}
    self.assertDictEqual(actual_first_batch, expected_first_batch)


if __name__ == "__main__":
  absltest.main()
