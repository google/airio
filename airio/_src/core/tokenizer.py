# Copyright 2025 The AirIO Authors.
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

"""Tokenizer-specific classes."""
import dataclasses
import typing
from typing import Generic, Mapping, Protocol, TypeVar

from airio._src.core import vocabularies

Inp = TypeVar("Inp")
Out = TypeVar("Out")


@dataclasses.dataclass(frozen=True)
class TokenizerConfig:
  """Config for tokenizer for a given feature."""

  vocab: vocabularies.Vocabulary
  add_eos: bool = True


  @property
  def vocabulary(self) -> vocabularies.Vocabulary:
    return self.vocab


@typing.runtime_checkable
@dataclasses.dataclass(frozen=True)
class Tokenizer(Generic[Inp, Out], Protocol):
  """Tokenizer class for AirIO tasks/mixtures."""

  tokenizer_configs: Mapping[str, TokenizerConfig]
  copy_pretokenized: bool = True

  def __call__(self, orig_example: Inp) -> Out:
    ...
