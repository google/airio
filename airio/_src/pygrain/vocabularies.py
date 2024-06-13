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

"""Vocabulary classes."""

from typing import Sequence

from airio._src.core import vocabularies

Encoded = Sequence[int]
Decoded = str


class SentencePieceVocabulary(
    vocabularies.SentencePieceVocabulary[Encoded, Decoded]
):
  """SentencePiece vocabulary for AirIO PyGrain."""

  def encode(self, s: Decoded) -> Encoded:
    """Encode a python string as a list of integers.

    Args:
      s: a string

    Returns:
      a list of integers (not terminated by EOS)
    """
    return self.tokenizer.EncodeAsIds(s)

  def decode(self, ids: Encoded) -> Decoded:
    """Decode a list of integers to a python string.

    Args:
      ids: a list of integers (not terminated by EOS)

    Returns:
      a string
    """
    clean_ids = list(ids)
    if self.unk_id is not None:
      vocab_size = self.base_vocab_size
      clean_ids = [self.unk_id if i >= vocab_size else i for i in clean_ids]
    if self.eos_id is not None and self.eos_id in clean_ids:
      clean_ids = clean_ids[: clean_ids.index(self.eos_id) + 1]
    # convert all the extra ids (sentinels) to UNK=2
    unk_id = self.tokenizer.unk_id()
    piece_size = self.tokenizer.GetPieceSize()
    clean_ids = [unk_id if i >= piece_size else int(i) for i in clean_ids]
    return self.tokenizer.DecodeIds(clean_ids)


class UnigramVocabulary(vocabularies.UnigramVocabulary[Encoded, Decoded]):
  """Unigram vocabulary for AirIO PyGrain."""

  def encode(self, s: Decoded) -> Encoded:
    """Encode a python string.

    Args:
      s: a string

    Returns:
      a list of a single integer
    """
    return [self._id_by_unigram.get(s, self.unk_id)]

  def decode(self, ids: Encoded) -> str:
    """Decode a list of integers.

    Args:
      ids: a list of integers

    Returns:
      a space delimited string decoded ids
    """
    return " ".join(self._unigram_by_id[id] for id in ids)
