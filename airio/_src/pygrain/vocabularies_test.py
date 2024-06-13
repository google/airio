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
from airio._src.pygrain import vocabularies

from sentencepiece import sentencepiece_model_pb2


def _create_sentencepiece_vocab(
    extra_ids=0,
    reverse_extra_ids=True,
    normalizer_spec_overrides: (
        sentencepiece_model_pb2.NormalizerSpec | None
    ) = None,
) -> vocabularies.SentencePieceVocabulary:
  test_data_dir = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "../../test_data"
  )
  sentencepiece_vocab = vocabularies.SentencePieceVocabulary(
      os.path.join(test_data_dir, "sentencepiece", "sentencepiece.model"),
      extra_ids=extra_ids,
      reverse_extra_ids=reverse_extra_ids,
      normalizer_spec_overrides=normalizer_spec_overrides,
  )
  return sentencepiece_vocab


class SentencepieceVocabularyTest(absltest.TestCase):

  TEST_STRING = "this is a test"
  TEST_TOKENS = (11, 8, 6, 3, 8, 6, 3, 5, 10)
  UNK_STRING = " ⁇ "

  def test_vocab_size(self):
    vocab = _create_sentencepiece_vocab()
    self.assertEqual(26, vocab.vocab_size)

  def test_vocab_encode_decode(self):
    vocab = _create_sentencepiece_vocab()
    self.assertEqual(26, vocab.vocab_size)
    self.assertSequenceEqual(self.TEST_TOKENS, vocab.encode(self.TEST_STRING))
    self.assertEqual(self.TEST_STRING, vocab.decode(self.TEST_TOKENS))

  def test_extra_ids(self):
    vocab = _create_sentencepiece_vocab(extra_ids=10)
    self.assertEqual(36, vocab.vocab_size)
    self.assertEqual("v", vocab.decode([25]))
    test_string = "<extra_id_0> <extra_id_1> v <extra_id_9>"
    test_tokens = (35, 34, 3, 25, 26)
    self.assertEqual(test_string, vocab.decode(test_tokens))
    self.assertSequenceEqual(test_tokens, vocab.encode(test_string))

  def test_force_repeated_whitespace_preservation(self):
    test_string = "a a  a   a"  # string with repeated whitespaces

    vocab = _create_sentencepiece_vocab(
        normalizer_spec_overrides=sentencepiece_model_pb2.NormalizerSpec(
            remove_extra_whitespaces=False
        )
    )
    self.assertEqual(test_string, vocab.decode(vocab.encode(test_string)))

    vocab = _create_sentencepiece_vocab()
    self.assertEqual("a a a a", vocab.decode(vocab.encode(test_string)))

  def test_not_reversing_extra_ids(self):
    vocab = _create_sentencepiece_vocab(extra_ids=10, reverse_extra_ids=False)
    base_vocab_size = vocab.vocab_size - vocab.extra_ids

    self.assertEqual(
        "<extra_id_0> <extra_id_1>",
        vocab.decode([base_vocab_size, base_vocab_size + 1]),
    )

    reversed_vocab = _create_sentencepiece_vocab(
        extra_ids=10, reverse_extra_ids=True
    )

    self.assertNotEqual(vocab, reversed_vocab)

  def test_decode_with_unk_id(self):
    vocab = _create_sentencepiece_vocab()
    tokens = list(self.TEST_TOKENS) + [vocab.vocab_size + 1]
    self.assertEqual(self.TEST_STRING + self.UNK_STRING, vocab.decode(tokens))

  def test_decode_with_eos_id(self):
    vocab = _create_sentencepiece_vocab()
    tokens = list(self.TEST_TOKENS) + [vocab.eos_id] + list(self.TEST_TOKENS)
    self.assertEqual(self.TEST_STRING, vocab.decode(tokens))


class UnigramVocabularyTest(absltest.TestCase):

  def test_encode_converts_unigrams_to_ints_correctly(self):
    unigrams = ["this", "that", "is", "not", "a", "the", "test", "ball"]
    vocabulary = vocabularies.UnigramVocabulary(unigrams)
    self.assertEqual(vocabulary.unk_id, 9)
    with self.subTest(name="pure_python"):
      # Note that id 0 is reserved for padding.
      self.assertEqual(vocabulary.encode("that"), [2])
      self.assertEqual(vocabulary.encode("not"), [4])
      self.assertEqual(vocabulary.encode("apple"), [vocabulary.unk_id])

  def test_decode_converts_ints_to_unigrams_correctly(self):
    unigrams = ["this", "that", "is", "not", "a", "the", "test", "ball"]
    vocabulary = vocabularies.UnigramVocabulary(unigrams)
    self.assertEqual(vocabulary.decode([1]), "this")
    self.assertEqual(vocabulary.decode([3]), "is")
    self.assertEqual(vocabulary.decode([vocabulary.unk_id]), "UNK")


if __name__ == "__main__":
  absltest.main()
