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
from unittest import mock
from absl.testing import absltest
from airio._src.core import test_utils as core_test_utils
from airio._src.tfgrain import test_utils
from airio.examples.tfgrain import tasks
import airio.tfgrain as airio
import grain.tensorflow as grain
import tensorflow_datasets as tfds

_SOURCE_NUM_EXAMPLES = 3
_SOURCE_SEQUENCE_LENGTH = 32


def trim_sequence(ex, runtime_args: airio.AirIOInjectedRuntimeArgs):
  sequence_lengths = runtime_args.sequence_lengths
  for k in sequence_lengths:
    ex[k] = ex[k][:sequence_lengths[k]]
  return ex


class TasksTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "../test_data",
    )
    sentencepiece_vocab = airio.SentencePieceVocabulary(
        os.path.join(self.test_dir, "sentencepiece", "sentencepiece.model")
    )
    self.tokenizer_configs = {
        "inputs": airio.TokenizerConfig(vocab=sentencepiece_vocab),
        "targets": airio.TokenizerConfig(vocab=sentencepiece_vocab),
    }

  def test_nqo_task(self):
    # Mock the TFDS datasource.
    tfds_name = "natural_questions_open:1.0.0"
    splits = ["train", "validation"]
    tfds_builder = test_utils.get_mock_dataset_builder(tfds_name, splits)
    with mock.patch.object(
        tfds,
        "builder",
        return_value=tfds_builder,
    ):
      with mock.patch.object(
          grain,
          "TfdsDataSource",
          return_value=test_utils.get_mocked_tfds_source(
              tfds_name, _SOURCE_NUM_EXAMPLES
          ),
      ):
        nqo_task = tasks.get_nqo_v001_task(
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
          batch_size=2,
          runtime_preprocessors=[airio.MapFnTransform(trim_sequence)]
      )
      expected = [{
          "inputs": [
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
          ],
          "targets": [
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
          ],
      }]
      core_test_utils.assert_datasets_equal(expected, [next(ds)])

  def test_wmt_task(self):
    # Mock the TFDS datasource.
    builder_config = tfds.translate.wmt19.Wmt19Translate.builder_configs[
        "de-en"
    ]
    tfds_version = "1.0.0"
    tfds_name = f"wmt19_translate/{builder_config.name}:{tfds_version}"
    splits = ["train", "validation"]
    tfds_builder = test_utils.get_mock_dataset_builder(tfds_name, splits)
    with mock.patch.object(
        tfds,
        "builder",
        return_value=tfds_builder,
    ):
      with mock.patch.object(
          grain,
          "TfdsDataSource",
          return_value=test_utils.get_mocked_tfds_source(
              tfds_name, _SOURCE_NUM_EXAMPLES
          ),
      ):
        wmt_task = tasks.get_wmt_19_ende_v003_task(
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
        batch_size=2,
    )
    expected = [{
        "inputs": [
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
                3,
                4,
                3,
                13,
                4,
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
                3,
                4,
                3,
                13,
                4,
            ],
        ],
        "targets": [
            [3, 2, 4, 2, 13, 3, 5, 20, 2, 4, 2, 20, 2, 4, 2],
            [3, 2, 4, 2, 13, 3, 5, 20, 2, 4, 2, 20, 2, 4, 2],
        ],
    }]
    core_test_utils.assert_datasets_equal(expected, [next(ds)])


if __name__ == "__main__":
  absltest.main()
