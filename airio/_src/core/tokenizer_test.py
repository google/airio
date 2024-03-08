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

"""Tests for airio.tokenizer."""

import os
from unittest import mock

from absl.testing import absltest
from airio._src.core import tokenizer
from seqio import vocabularies



_TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../test_data"
)


class TokenizerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.sentencepiece_vocab = vocabularies.SentencePieceVocabulary(
        os.path.join(_TEST_DATA_DIR, "sentencepiece", "sentencepiece.model")
    )

  def test_tokenizer_config_properties(self):
    tokenizer_config = tokenizer.TokenizerConfig(vocab=self.sentencepiece_vocab)
    self.assertEqual(tokenizer_config.vocabulary, self.sentencepiece_vocab)
    self.assertEqual(tokenizer_config.add_eos, True)

  @mock.patch.object(
      dm_usage_logging, "log_event", autospec=True, return_value=[]
  )
  def test_telemetry_create_config(self, mock_log_event):
    sentencepiece_vocab = vocabularies.SentencePieceVocabulary(
        os.path.join(_TEST_DATA_DIR, "sentencepiece", "sentencepiece.model")
    )
    _ = tokenizer.TokenizerConfig(vocab=sentencepiece_vocab)
    expected = [
        mock.call(
            dm_usage_logging.Event.AIRIO,
            "airio._src.core.tokenizer.TokenizerConfig",
            tag_2="__init__",
            tag_3="",
        ),
    ]
    mock_log_event.assert_has_calls(expected, any_order=True)


if __name__ == "__main__":
  absltest.main()
