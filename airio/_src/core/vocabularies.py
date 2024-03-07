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

"""Vocabulary classes for tokenization."""

import dataclasses
import functools
import hashlib
import threading
from typing import ClassVar, Protocol, TypeVar, Generic

from absl import logging
import tensorflow as tf

from sentencepiece import sentencepiece_model_pb2
import sentencepiece as sentencepiece_processor

PAD_ID = 0
Encoded = TypeVar("Encoded")
Decoded = TypeVar("Decoded")


class Vocabulary(Generic[Encoded, Decoded], Protocol):
  """Abstract class for vocabularies.

  Subclasses are responsible for reserving PAD_ID (commonly 0) as well as
  optionally reserving EOS_ID and UNK_ID.
  """

  @property
  def bos_id(self) -> int | None:
    ...

  @property
  def eos_id(self) -> int | None:
    ...

  @property
  def pad_id(self) -> int:
    ...

  @property
  def unk_id(self) -> int | None:
    ...

  @property
  def extra_ids(self) -> int:
    ...

  @property
  def vocab_size(self) -> int:
    """Vocabulary size, including extra ids."""
    ...

  @property
  def base_vocab_size(self) -> int:
    """Vocabulary size, excluding extra ids but including PAD/EOS/UNK."""
    ...

  def encode(self, s: Decoded) -> Encoded:
    """Tokenizes string to an int sequence, without adding EOS."""
    ...

  def decode(self, ids: Encoded) -> Decoded:
    """Detokenizes int32 iterable to a string, up through first EOS."""
    ...


class SentencePieceVocabulary(Vocabulary[Encoded, Decoded]):
  """Wrapper around sentencepiece_processor.

  Assumes the model was built using flags to reserve ID=0 for padding, ID=1 for
  EOS, and ID=2 for UNK.

  If using extra ids, you can represent them in string-form as `<extra_id_0>`,
  `<extra_id_1>`, etc. They will be indexed starting from the end of the
  vocabulary to match how the masking preprocessors are set up.

  IMPORTANT NOTE: these placeholders only work properly when they are used at
  word starts (e.g., "I like peanut butter and <extra_id_0> sandwiches." or
  "I like peanut butter and <extra_id_0>ly sandwiches" are both okay, but
  "I like peanut butter and jel<extra_id_0> sandwiches" is not.).
  """

  @dataclasses.dataclass
  class _ModelContext:
    tokenizer: sentencepiece_processor.SentencePieceProcessor
    sp_model: bytes

  _load_model_lock: ClassVar[threading.Lock] = threading.Lock()

  def __init__(
      self,
      sentencepiece_model_file: str,
      extra_ids: int = 0,
      normalizer_spec_overrides: (
          sentencepiece_model_pb2.NormalizerSpec | None
      ) = None,
      reverse_extra_ids: bool = True,
      use_fast_tokenizer: bool = False,
  ):
    """Create a SentencePieceVocabulary.

    Optionally, specify a number of extra ids to add to the end of the
    vocabulary for use as sentinels.

    Args:
      sentencepiece_model_file: path of the sentence piece model.
      extra_ids: number of extra ids to include.
      normalizer_spec_overrides: If not None, this proto will be merged into the
        model's normalizer and denormalizer specs. Thus, any options set on this
        object will override the values of those options in the loaded model.
      reverse_extra_ids: if True, extra_ids are numbered in descending order, so
        the first extra_id has the highest number. This is done for
        compatibility with span_corruption mask generation in T5.
      use_fast_tokenizer: use the tf_text fastsentencepiecetokenizer
        implementation which runs much faster.
    """
    self._sentencepiece_model_file = sentencepiece_model_file
    self._normalizer_spec_overrides = normalizer_spec_overrides
    self._reverse_extra_ids = reverse_extra_ids
    self._model: SentencePieceVocabulary._ModelContext | None = None
    self._use_fast_tokenizer = use_fast_tokenizer
    self._extra_ids = extra_ids or 0

  def __getstate__(self):
    state = self.__dict__.copy()
    # Gin config makes a deep copy of the keyword arguments of configurables.
    # When a SentencePieceVocabulary vocabulary is used as a keyword argument
    # in a Gin configurable, it must be picklable. We therefore remove
    # _model; will be initialized lazily as needed.
    del state["_model"]
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self._model = None

  def load_model(self) -> None:
    _ = self._model_context()

  def _model_context(
      self,
  ) -> _ModelContext:
    """Loads model if not yet loaded and returns the model context.

    Returns:
      The model context as a tuple of (tokenizer, sp_model).
    """
    if self._model:
      return self._model

    normalizer_spec_overrides_serialized = (
        self._normalizer_spec_overrides.SerializeToString(deterministic=True)
        if self._normalizer_spec_overrides
        else None
    )

    self._model = self._load_model(
        self._sentencepiece_model_file,
        self._extra_ids,
        normalizer_spec_overrides_serialized,
        self._reverse_extra_ids,
    )
    return self._model

  @classmethod
  @functools.lru_cache(maxsize=None)
  def _load_model(
      cls,
      sentencepiece_model_file: str,
      extra_ids: int,
      normalizer_spec_overrides_serialized: bytes | None = None,
      reverse_extra_ids: bool = True,
  ) -> _ModelContext:
    """Load SPM, Python tokenizer, and cache results to the class definition."""
    # SentencePieceProcessor::LoadFromSerializedProto is not thread-safe.
    # Without a lock, users may randomly see SIGSEGV on
    # sentencepiece::ModelInterface::pad_piece when using the vocabulary in
    # SeqIO preprocessors.
    with cls._load_model_lock:
      # Handle cases where SP can't load the file, but gfile can.
      with tf.io.gfile.GFile(sentencepiece_model_file, "rb") as f:
        sp_model = f.read()
        model = sentencepiece_model_pb2.ModelProto.FromString(sp_model)
        # Add placeholder strings for extra IDs.
        if extra_ids:
          # By default, we them in reverse order to match span corruption.
          if reverse_extra_ids:
            extra_id_tokens = reversed(range(extra_ids))
          else:
            extra_id_tokens = range(extra_ids)

          for i in extra_id_tokens:
            model.pieces.add(
                piece=f"▁<extra_id_{i}>",
                score=0.0,
                type=sentencepiece_model_pb2.ModelProto.SentencePiece.USER_DEFINED,
            )
        if normalizer_spec_overrides_serialized is not None:
          normalizer_spec_overrides = (
              sentencepiece_model_pb2.NormalizerSpec.FromString(
                  normalizer_spec_overrides_serialized
              )
          )

          model.normalizer_spec.MergeFrom(normalizer_spec_overrides)
          model.denormalizer_spec.MergeFrom(normalizer_spec_overrides)
        sp_model = model.SerializeToString()
      # Load Python tokenizer and ensure the EOS and PAD IDs are correct.
      tokenizer = sentencepiece_processor.SentencePieceProcessor()
      tokenizer.LoadFromSerializedProto(sp_model)
      if tokenizer.pad_id() != PAD_ID:
        logging.warning(
            (
                "T5 library uses PAD_ID=%s, which is different from the "
                "sentencepiece vocabulary, which defines pad_id=%s"
            ),
            PAD_ID,
            tokenizer.pad_id(),
        )

      return cls._ModelContext(tokenizer=tokenizer, sp_model=sp_model)

  @property
  def pad_id(self) -> int | None:
    return PAD_ID

  @property
  def bos_id(self) -> int | None:
    return self.tokenizer.bos_id()

  @property
  def eos_id(self) -> int | None:
    return self.tokenizer.eos_id()

  @property
  def unk_id(self) -> int | None:
    return self.tokenizer.unk_id()

  @property
  def sp_model(self) -> bytes | None:
    """Retrieve the SPM."""
    return self._model_context().sp_model

  @property
  def sentencepiece_model_file(self) -> str:
    return self._sentencepiece_model_file

  @property
  def tokenizer(self) -> sentencepiece_processor.SentencePieceProcessor:
    """Returns the Python tokenizer."""
    return self._model_context().tokenizer

  @property
  def extra_ids(self) -> int:
    return self._extra_ids

  @property
  def base_vocab_size(self):
    """Number of ids (including 0=PAD, 1=EOS, and 2=UNK).

    Returns:
      an integer, the vocabulary size
    """
    return self.tokenizer.GetPieceSize()

  @property
  def vocab_size(self):
    return self.base_vocab_size

  def __eq__(self, other):
    if not isinstance(other, SentencePieceVocabulary):
      return False
    try:
      their_md5 = hashlib.md5(other.sp_model).hexdigest()
    # If other has no sp_model attribute, we can't test for equality
    except AttributeError:
      return False
    if self.sp_model is None:
      return False
    our_md5 = hashlib.md5(self.sp_model).hexdigest()
    return our_md5 == their_md5

  def __str__(self) -> str:
    return (
        f"SentencePieceVocabulary(file={self.sentencepiece_model_file}, "
        f"extra_ids={self._extra_ids}, "
        f"spm_md5={hashlib.md5(self.sp_model).hexdigest()})"
    )

  def encode(self, s: Decoded):
    """Tokenizes string to an int sequence, without adding EOS."""
    raise NotImplementedError()

  def decode(self, ids: Encoded):
    """Detokenizes int32 iterable to a string, up through first EOS."""
    raise NotImplementedError()
