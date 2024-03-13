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
import pickle
from typing import Protocol
from absl.testing import absltest
from absl.testing import parameterized
from airio._src.core import vocabularies
import numpy as np
import tensorflow as tf


def _create_sentencepiece_vocab(
    extra_ids=0,
    reverse_extra_ids=True,
) -> vocabularies.SentencePieceVocabulary:
  test_data_dir = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "../../test_data"
  )
  sentencepiece_vocab = vocabularies.SentencePieceVocabulary(
      os.path.join(test_data_dir, "sentencepiece", "sentencepiece.model"),
      extra_ids=extra_ids,
      reverse_extra_ids=reverse_extra_ids,
  )
  return sentencepiece_vocab


class VocabularyTest(absltest.TestCase):
  TEST_STR = "Testing."
  TEST_IDS = [84, 101, 115, 116, 105, 110, 103, 46]

  class AsciiTestVocab(vocabularies.Vocabulary, Protocol):

    def __init__(self, extra_ids=0, use_eos=True, use_unk=True):
      self._extra_ids = extra_ids
      self._use_eos = use_eos
      self._use_unk = use_unk

    @property
    def pad_id(self):
      return 0

    @property
    def eos_id(self):
      return 1 if self._use_eos else None

    @property
    def unk_id(self):
      return 2 if self._use_unk else None

    @property
    def base_vocab_size(self):
      return 128

    @property
    def extra_ids(self):
      return self._extra_ids

    @property
    def vocab_size(self):
      return self.base_vocab_size + self._extra_ids

    def encode(self, s):
      ...

    def decode(self, ids):
      ...

  class PyAsciiTestVocab(AsciiTestVocab):

    def encode(self, s):
      return [ord(c) for c in s]

    def _clean_ids(self, ids):
      clean_ids = list(ids)

      if self.unk_id is not None:
        vocab_size = self.base_vocab_size
        clean_ids = [self.unk_id if i >= vocab_size else i for i in clean_ids]

      if self.eos_id is not None and self.eos_id in clean_ids:
        clean_ids = clean_ids[: clean_ids.index(self.eos_id) + 1]
      return clean_ids

    def decode(self, ids):
      ids = self._clean_ids(ids)
      return "".join("<eos>" if id == 1 else chr(id) for id in ids if id > 0)

  class TfAsciiTestVocab(AsciiTestVocab):

    def encode(self, s):
      return tf.strings.unicode_decode(s, "UTF-8")

    def _clean_ids(self, ids):
      if ids.shape == (0,):
        return tf.constant(b"", dtype=tf.string)

      clean_ids = ids

      if self.unk_id is not None:
        base_vocab_size = self.base_vocab_size
        clean_ids = tf.where(
            tf.less(clean_ids, base_vocab_size), clean_ids, self.unk_id
        )

      if self.eos_id is not None:
        # Replace everything after the first eos_id with pad_id.
        after_eos = tf.cumsum(
            tf.cast(tf.equal(clean_ids, self.eos_id), tf.int32),
            exclusive=True,
            axis=-1,
        )
        clean_ids = tf.where(
            tf.cast(after_eos, tf.bool), self.pad_id, clean_ids
        )
      return clean_ids

    def decode(self, ids):
      ids = self._clean_ids(ids)
      s = tf.strings.unicode_encode(ids, "UTF-8")
      s = tf.strings.regex_replace(s, chr(0), "")
      s = tf.strings.regex_replace(s, chr(1), "<eos>")
      return s

  def _decode_tf(self, vocab, tokens):
    results = vocab.decode(tf.constant(tokens, tf.int32)).numpy()

    def _apply(fun, sequence):
      if isinstance(sequence, (list, np.ndarray)):
        return [_apply(fun, x) for x in sequence]
      else:
        return fun(sequence)

    return _apply(lambda x: x.decode("UTF-8"), results)

  def test_properties(self):
    test_vocab = self.PyAsciiTestVocab(
        use_eos=False, use_unk=True, extra_ids=10
    )
    self.assertEqual(test_vocab.extra_ids, 10)
    self.assertEqual(test_vocab.pad_id, 0)
    self.assertIsNone(test_vocab.eos_id)
    self.assertEqual(test_vocab.unk_id, 2)
    self.assertEqual(test_vocab.vocab_size, 128 + 10)

    test_vocab = self.TfAsciiTestVocab(use_eos=True, use_unk=False)
    self.assertEqual(test_vocab.extra_ids, 0)
    self.assertEqual(test_vocab.pad_id, 0)
    self.assertEqual(test_vocab.eos_id, 1)
    self.assertIsNone(test_vocab.unk_id)
    self.assertEqual(test_vocab.vocab_size, 128)

  def test_py_encode(self):
    test_vocab = self.PyAsciiTestVocab()
    self.assertSequenceEqual(test_vocab.encode(self.TEST_STR), self.TEST_IDS)
    self.assertSequenceEqual(
        tuple(test_vocab.encode(self.TEST_STR)), self.TEST_IDS
    )

  def test_tf_encode(self):
    test_vocab = self.TfAsciiTestVocab()
    self.assertSequenceEqual(
        tuple(test_vocab.encode(self.TEST_STR).numpy()), self.TEST_IDS
    )

  def test_py_decode_unk_and_eos(self):
    test_vocab = self.PyAsciiTestVocab(use_eos=True, use_unk=True)
    test_ids = [161] + self.TEST_IDS + [127, 191, 1, 0, 10]
    test_str = "\x02" + self.TEST_STR + "\x7f\x02<eos>"
    self.assertEqual(test_vocab.decode(test_ids), test_str)

  def test_tf_decode_unk_and_eos(self):
    test_vocab = self.TfAsciiTestVocab(use_eos=True, use_unk=True)
    test_ids = [161] + self.TEST_IDS + [127, 191, 1, 0, 10]
    test_str = "\x02" + self.TEST_STR + "\x7f\x02<eos>"
    self.assertEqual(self._decode_tf(test_vocab, test_ids), test_str)

  def test_py_decode_unk_only(self):
    test_vocab = self.PyAsciiTestVocab(
        use_eos=False, use_unk=True, extra_ids=35
    )
    test_ids = [161] + self.TEST_IDS + [127, 191, 1, 33, 1]
    test_str = "\x02" + self.TEST_STR + "\x7f\x02<eos>!<eos>"
    self.assertEqual(test_vocab.decode(test_ids), test_str)

  def test_tf_decode_unk_only(self):
    test_vocab = self.TfAsciiTestVocab(
        use_eos=False, use_unk=True, extra_ids=35
    )
    test_ids = [161] + self.TEST_IDS + [127, 191, 1, 33, 1]
    test_str = "\x02" + self.TEST_STR + "\x7f\x02<eos>!<eos>"
    self.assertEqual(self._decode_tf(test_vocab, test_ids), test_str)

  def test_py_decode_eos_only(self):
    test_vocab = self.PyAsciiTestVocab(use_eos=True, use_unk=False)
    test_ids = [161] + self.TEST_IDS + [127, 191, 1, 33, 1]
    test_str = "¡" + self.TEST_STR + "\x7f¿<eos>"
    self.assertEqual(test_vocab.decode(test_ids), test_str)

    test_ids = [161] + self.TEST_IDS + [127, 191]
    test_str = "¡" + self.TEST_STR + "\x7f¿"
    self.assertEqual(test_vocab.decode(test_ids), test_str)

    test_ids = [1] + self.TEST_IDS
    test_str = "<eos>"
    self.assertEqual(test_vocab.decode(test_ids), test_str)

  def test_tf_decode_eos_only(self):
    test_vocab = self.TfAsciiTestVocab(use_eos=True, use_unk=False)
    test_ids = [161] + self.TEST_IDS + [127, 191, 1, 33, 1]
    test_str = "¡" + self.TEST_STR + "\x7f¿<eos>"
    self.assertEqual(self._decode_tf(test_vocab, test_ids), test_str)

    test_ids = [161] + self.TEST_IDS + [127, 191]
    test_str = "¡" + self.TEST_STR + "\x7f¿"
    self.assertEqual(self._decode_tf(test_vocab, test_ids), test_str)

    test_ids = [1] + self.TEST_IDS
    test_str = "<eos>"
    self.assertEqual(self._decode_tf(test_vocab, test_ids), test_str)

  def test_py_decode_no_unk_or_eos(self):
    test_vocab = self.PyAsciiTestVocab(use_eos=False, use_unk=False)
    test_ids = [161] + self.TEST_IDS + [127, 191, 1, 33, 1]
    test_str = "¡" + self.TEST_STR + "\x7f¿<eos>!<eos>"
    self.assertEqual(test_vocab.decode(test_ids), test_str)

  def test_tf_decode_no_unk_or_eos(self):
    test_vocab = self.TfAsciiTestVocab(use_eos=False, use_unk=False)
    test_ids = [161] + self.TEST_IDS + [127, 191, 1, 33, 1]
    test_str = "¡" + self.TEST_STR + "\x7f¿<eos>!<eos>"
    self.assertEqual(self._decode_tf(test_vocab, test_ids), test_str)

  def test_decode_tf_batch(self):
    test_vocab = self.TfAsciiTestVocab(use_eos=True, use_unk=True)
    test_ids = (
        [161] + self.TEST_IDS + [127, 191, 1, 33, 1],
        [161] + self.TEST_IDS + [1, 191, 1, 33, 1],
    )
    test_str = (
        "\x02" + self.TEST_STR + "\x7f\x02<eos>",
        "\x02" + self.TEST_STR + "<eos>",
    )
    decoded = [
        dec.decode("UTF-8")
        for dec in test_vocab.decode(tf.constant(test_ids, tf.int32)).numpy()
    ]
    self.assertSequenceEqual(decoded, test_str)


class SentencepieceVocabularyTest(parameterized.TestCase):

  def test_equal(self):
    vocab1 = _create_sentencepiece_vocab()
    vocab2 = _create_sentencepiece_vocab()
    self.assertEqual(vocab1, vocab2)

  def test_not_equal(self):
    vocab1 = _create_sentencepiece_vocab()
    vocab2 = _create_sentencepiece_vocab(10)
    self.assertNotEqual(vocab1, vocab2)

  def test_reverse_extra_ids(self):
    vocab = _create_sentencepiece_vocab(extra_ids=10, reverse_extra_ids=False)
    reversed_vocab = _create_sentencepiece_vocab(
        extra_ids=10, reverse_extra_ids=True
    )
    self.assertNotEqual(vocab, reversed_vocab)

  def test_cache(self):
    vocab1 = _create_sentencepiece_vocab()
    vocab2 = _create_sentencepiece_vocab()
    self.assertEqual(vocab1.sp_model, vocab2.sp_model)
    self.assertEqual(vocab1.tokenizer, vocab2.tokenizer)
    vocab3 = _create_sentencepiece_vocab(extra_ids=1)
    self.assertNotEqual(vocab1.sp_model, vocab3.sp_model)
    self.assertNotEqual(vocab1.tokenizer, vocab3.tokenizer)

  def test_properties(self):
    test_vocab = _create_sentencepiece_vocab(extra_ids=10)
    self.assertEqual(test_vocab.extra_ids, 10)
    self.assertEqual(test_vocab.pad_id, 0)
    self.assertEqual(test_vocab.eos_id, 1)
    self.assertEqual(test_vocab.unk_id, 2)
    self.assertEqual(test_vocab.vocab_size, 36)

  def test_pickling(self):
    test_vocab = _create_sentencepiece_vocab(extra_ids=10)
    dumped = pickle.dumps(test_vocab)
    loaded_vocab = pickle.loads(dumped)
    self.assertEqual(test_vocab, loaded_vocab)

  def test_str(self):
    test_vocab = _create_sentencepiece_vocab(extra_ids=10)
    self.assertRegex(
        str(test_vocab),
        r"SentencePieceVocabulary\(file=[^,]*\, extra_ids=10\, spm_md5=[^)]*\)",
    )

if __name__ == "__main__":
  absltest.main()
