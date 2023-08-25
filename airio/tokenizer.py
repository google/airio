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

"""Tokenizer-specific classes."""

import dataclasses
from typing import Mapping

import numpy as np
from seqio import vocabularies


@dataclasses.dataclass(frozen=True)
class TokenizerConfig:
  """Config for tokenizer for a given feature."""

  vocab: vocabularies.Vocabulary
  add_eos: bool = True
  dtype: np.dtype = np.int_
  rank: int = 1


def tokenize(
    orig_example,
    tokenizer_configs: Mapping[str, TokenizerConfig],
    copy_pretokenized: bool = True,
) -> Mapping[str, np.ndarray]:
  """Basic implementation of tokenization."""
  final_example = {}
  for feature_name, feature_value in orig_example.items():
    if feature_name not in tokenizer_configs:
      final_example[feature_name] = feature_value
      continue

    if copy_pretokenized:
      final_example[f"{feature_name}_pretokenized"] = feature_value

    tokenizer_config = tokenizer_configs[feature_name]
    encoded_val = tokenizer_config.vocab.encode(feature_value)
    final_example[feature_name] = np.asarray(encoded_val)

  return final_example
