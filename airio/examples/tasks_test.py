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

"""Tasks tests."""

import os

from absl.testing import absltest
import airio
from airio import examples
from airio.grain.common import feature_converters
from seqio import vocabularies
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


class TasksTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test_data",
    )
    sentencepiece_vocab = vocabularies.SentencePieceVocabulary(
        os.path.join(self.test_dir, "sentencepiece", "sentencepiece.model")
    )
    self.tokenizer_configs = {
        "inputs": airio.tokenizer.TokenizerConfig(vocab=sentencepiece_vocab),
        "targets": airio.tokenizer.TokenizerConfig(vocab=sentencepiece_vocab),
    }
    self.runtime_preprocessors = (
        feature_converters.get_t5x_enc_dec_feature_converter_preprocessors(
            pack=False,
            use_multi_bin_packing=False,
            passthrough_feature_keys=[],
            pad_id=0,
            bos_id=0,
        )
    )

  def test_wmt_task(self):
    with tfds.testing.mock_data(_SOURCE_NUM_EXAMPLES):
      wmt_task = examples.tasks.get_wmt_19_ende_v003_task(
          tokenizer_configs=self.tokenizer_configs
      )
    sequence_lengths = {
        "inputs": _SOURCE_SEQUENCE_LENGTH,
        "targets": _SOURCE_SEQUENCE_LENGTH,
    }
    ds = wmt_task.get_dataset(
        sequence_lengths,
        "train",
        shuffle=False,
        runtime_preprocessors=self.runtime_preprocessors,
        batch_size=2,
    )
    expected_first_batch = {
        "decoder_input_tokens": [
            _pad(
                [0, 3, 2, 4, 2, 13, 3, 5, 20, 2, 4, 2, 20, 2, 4, 2],
                _SOURCE_SEQUENCE_LENGTH,
            ),
            _pad(
                [0, 3, 4, 20, 2, 3, 5, 8, 2, 13, 8, 21, 13, 8, 2],
                _SOURCE_SEQUENCE_LENGTH,
            ),
        ],
        "decoder_loss_weights": [
            _pad([1], 15, 1) + _pad([0], 17),
            _pad([1], 14, 1) + _pad([0], 18),
        ],
        "decoder_target_tokens": [
            _pad(
                [3, 2, 4, 2, 13, 3, 5, 20, 2, 4, 2, 20, 2, 4, 2],
                _SOURCE_SEQUENCE_LENGTH,
            ),
            _pad(
                [3, 4, 20, 2, 3, 5, 8, 2, 13, 8, 21, 13, 8, 2],
                _SOURCE_SEQUENCE_LENGTH,
            ),
        ],
        "encoder_input_tokens": [
            [
                3,
                24,
                23,
                5,
                22,
                6,
                9,
                5,
                16,
                3,
                2,
                22,
                2,
                9,
                8,
                6,
                20,
                3,
                24,
                7,
                3,
                2,
                4,
                23,
                14,
                5,
                22,
                12,
                3,
                13,
                20,
                2,
            ],
            [
                3,
                24,
                23,
                5,
                22,
                6,
                9,
                5,
                16,
                3,
                2,
                22,
                2,
                9,
                8,
                6,
                20,
                3,
                24,
                7,
                3,
                2,
                4,
                23,
                14,
                5,
                22,
                12,
                3,
                2,
                3,
                8,
            ],
        ],
    }
    actual_first_batch = {k: v.tolist() for k, v in next(ds).items()}
    self.assertDictEqual(actual_first_batch, expected_first_batch)

  def test_nqo_task(self):
    with tfds.testing.mock_data(_SOURCE_NUM_EXAMPLES):
      nqo_task = examples.tasks.get_nqo_v001_task(
          tokenizer_configs=self.tokenizer_configs
      )
    sequence_lengths = {
        "inputs": _SOURCE_SEQUENCE_LENGTH,
        "targets": _SOURCE_SEQUENCE_LENGTH,
    }
    ds = nqo_task.get_dataset(
        sequence_lengths,
        "train",
        shuffle=False,
        runtime_preprocessors=self.runtime_preprocessors,
        batch_size=2,
    )
    expected_first_batch = {
        "decoder_input_tokens": [
            [
                0,
                3,
                4,
                2,
                13,
                3,
                5,
                20,
                2,
                4,
                2,
                20,
                2,
                4,
                2,
                3,
                2,
                13,
                20,
                2,
                3,
                21,
                8,
                2,
                3,
                20,
                20,
                8,
                5,
                3,
                8,
                2,
            ],
            [
                0,
                3,
                20,
                2,
                3,
                5,
                8,
                2,
                13,
                8,
                21,
                13,
                8,
                2,
                21,
                2,
                3,
                8,
                20,
                8,
                4,
                20,
                13,
                21,
                20,
                5,
                13,
                5,
                21,
                2,
                3,
                21,
            ],
        ],
        "decoder_loss_weights": [
            _pad([1], _SOURCE_SEQUENCE_LENGTH, 1),
            _pad([1], _SOURCE_SEQUENCE_LENGTH, 1),
        ],
        "decoder_target_tokens": [
            [
                3,
                4,
                2,
                13,
                3,
                5,
                20,
                2,
                4,
                2,
                20,
                2,
                4,
                2,
                3,
                2,
                13,
                20,
                2,
                3,
                21,
                8,
                2,
                3,
                20,
                20,
                8,
                5,
                3,
                8,
                2,
                3,
            ],
            [
                3,
                20,
                2,
                3,
                5,
                8,
                2,
                13,
                8,
                21,
                13,
                8,
                2,
                21,
                2,
                3,
                8,
                20,
                8,
                4,
                20,
                13,
                21,
                20,
                5,
                13,
                5,
                21,
                2,
                3,
                21,
                21,
            ],
        ],
        "encoder_input_tokens": [
            _pad(
                [
                    3,
                    22,
                    2,
                    3,
                    2,
                    4,
                    6,
                    24,
                    8,
                    7,
                    22,
                    12,
                    3,
                    2,
                    21,
                    2,
                    20,
                    2,
                    13,
                    3,
                    2,
                ],
                _SOURCE_SEQUENCE_LENGTH,
            ),
            _pad(
                [
                    3,
                    22,
                    2,
                    3,
                    2,
                    4,
                    6,
                    24,
                    8,
                    7,
                    22,
                    12,
                    3,
                    4,
                    8,
                    2,
                    3,
                    2,
                    13,
                    2,
                    4,
                    2,
                    5,
                    3,
                    2,
                    20,
                ],
                _SOURCE_SEQUENCE_LENGTH,
            ),
        ],
    }
    actual_first_batch = {k: v.tolist() for k, v in next(ds).items()}
    self.assertDictEqual(actual_first_batch, expected_first_batch)

  def test_c4_span_corruption_task(self):
    with tfds.testing.mock_data(_SOURCE_NUM_EXAMPLES):
      task = examples.tasks.get_c4_v220_span_corruption_task(
          tokenizer_configs=self.tokenizer_configs
      )
    runtime_preprocessors = (
        feature_converters.get_t5x_enc_dec_feature_converter_preprocessors(
            pack=False,
            use_multi_bin_packing=False,
            passthrough_feature_keys=[],
            pad_id=0,
            bos_id=0,
        )
    )
    source_sequence_length = 1024
    sequence_lengths = {
        "inputs": source_sequence_length,
        "targets": source_sequence_length,
    }
    ds = task.get_dataset(
        sequence_lengths,
        shuffle=False,
        seed=42,
        runtime_preprocessors=runtime_preprocessors,
        shard_info=airio.ShardInfo(index=0, num_shards=source_sequence_length),
    )

    expected_first_batch = {
        "decoder_input_tokens": _pad([0, 25, 4, 2, 1], source_sequence_length),
        "decoder_loss_weights": _pad(
            [True, True, True, True], source_sequence_length, False
        ),
        "decoder_target_tokens": _pad([25, 4, 2, 1], source_sequence_length),
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
                1,
            ],
            source_sequence_length,
        ),
    }
    actual_first_batch = {k: v.tolist() for k, v in next(ds).items()}
    self.assertDictEqual(actual_first_batch, expected_first_batch)


if __name__ == "__main__":
  absltest.main()
