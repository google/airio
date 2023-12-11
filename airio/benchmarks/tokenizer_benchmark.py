# Copyright 2023 The AirIO Authors.
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

import airio
import google_benchmark
from seqio import vocabularies


_TEST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_data"
)
_SENTENCEPIECE_VOCAB = vocabularies.SentencePieceVocabulary(
    os.path.join(_TEST_DIR, "sentencepiece", "sentencepiece.model")
)
_TOKENIZER_CONFIG = airio.tokenizer.TokenizerConfig(vocab=_SENTENCEPIECE_VOCAB)


@google_benchmark.register
def tokenize(state):
  """Analogous to the TokenizerTest with the same name."""
  orig_example = {
      "inputs": "imdb ebc   ahgjefjhfe",
      "targets": "positive",
  }
  tokenizer_configs = {
      "inputs": _TOKENIZER_CONFIG,
      "targets": _TOKENIZER_CONFIG,
  }
  tokenizer_obj = airio.tokenizer.Tokenizer(tokenizer_configs=tokenizer_configs)
  tokenized_example = tokenizer_obj(orig_example)
  while state:
    _ = tokenized_example.items()


@google_benchmark.register
def tokenize_do_not_copy_pretokenized(state):
  """Analogous to the TokenizerTest with the same name."""
  orig_example = {
      "inputs": "imdb ebc   ahgjefjhfe",
      "targets": "positive",
  }
  tokenizer_configs = {
      "inputs": _TOKENIZER_CONFIG,
      "targets": _TOKENIZER_CONFIG,
  }
  tokenizer_obj = airio.tokenizer.Tokenizer(
      tokenizer_configs=tokenizer_configs,
      copy_pretokenized=False,
  )
  tokenized_example = tokenizer_obj(orig_example)
  while state:
    _ = tokenized_example.items()


@google_benchmark.register
def tokenize_empty(state):
  """Analogous to the TokenizerTest with the same name."""
  orig_example = {
      "inputs": "imdb ebc   ahgjefjhfe",
      "targets": "positive",
  }
  tokenizer_configs = {
      "inputs": _TOKENIZER_CONFIG,
      "targets": _TOKENIZER_CONFIG,
  }
  tokenizer_obj = airio.tokenizer.Tokenizer(tokenizer_configs=tokenizer_configs)
  tokenized_example = tokenizer_obj(orig_example)
  while state:
    _ = tokenized_example.items()


if __name__ == "__main__":
  google_benchmark.main()
