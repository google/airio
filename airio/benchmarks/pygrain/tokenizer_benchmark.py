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

"""Microbenchmarks for AirIO tokenizer functions."""

import os

import airio.pygrain as airio
import google_benchmark

_TEST_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../test_data"
)


@google_benchmark.register
def tokenize(state: google_benchmark.State) -> None:
  """Measure tokenization."""
  tokenizer_config = airio.TokenizerConfig(
      vocab=airio.SentencePieceVocabulary(
          os.path.join(_TEST_DIR, "sentencepiece", "sentencepiece.model")
      )
  )
  tokenizer_obj = airio.Tokenizer(
      tokenizer_configs={
          "inputs": tokenizer_config,
          "targets": tokenizer_config,
      }
  )
  orig_example = {
      "inputs": "imdb ebc   ahgjefjhfe",
      "targets": "positive",
  }
  tokenized_example = tokenizer_obj(orig_example)
  while state:
    tokenized_example.items()


@google_benchmark.register
def tokenize_do_not_copy_pretokenized(state: google_benchmark.State) -> None:
  """Measure tokenization without copying pretokenized."""
  tokenizer_config = airio.TokenizerConfig(
      vocab=airio.SentencePieceVocabulary(
          os.path.join(_TEST_DIR, "sentencepiece", "sentencepiece.model")
      )
  )
  tokenizer_obj = airio.Tokenizer(
      tokenizer_configs={
          "inputs": tokenizer_config,
          "targets": tokenizer_config,
      },
      copy_pretokenized=False,
  )
  orig_example = {
      "inputs": "imdb ebc   ahgjefjhfe",
      "targets": "positive",
  }
  tokenized_example = tokenizer_obj(orig_example)
  while state:
    tokenized_example.items()


@google_benchmark.register
def tokenize_empty(state: google_benchmark.State) -> None:
  """Measure tokenization with empty input."""
  tokenizer_config = airio.TokenizerConfig(
      vocab=airio.SentencePieceVocabulary(
          os.path.join(_TEST_DIR, "sentencepiece", "sentencepiece.model")
      )
  )
  tokenizer_obj = airio.Tokenizer(
      tokenizer_configs={
          "inputs": tokenizer_config,
          "targets": tokenizer_config,
      },
      copy_pretokenized=False,
  )
  tokenized_example = tokenizer_obj({})
  while state:
    tokenized_example.items()


if __name__ == "__main__":
  google_benchmark.main()
