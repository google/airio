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

"""Tokenizer classes."""
import dataclasses
from typing import cast, Mapping
from airio._src.core import tokenizer
import numpy as np

Inp = Mapping[str, str | bytes]
Out = Mapping[str, np.ndarray]


@dataclasses.dataclass(frozen=True)
class Tokenizer(tokenizer.Tokenizer[Inp, Out]):
  """Tokenizer class for AirIO PyGrain."""


  def __call__(self, orig_example: Inp) -> Out:
    final_example = {}
    for feature_name, feature_value in orig_example.items():
      if feature_name not in self.tokenizer_configs:
        final_example[feature_name] = feature_value
        continue

      if self.copy_pretokenized:
        final_example[f"{feature_name}_pretokenized"] = feature_value

      tokenizer_config = self.tokenizer_configs[feature_name]
      encoded_val = tokenizer_config.vocab.encode(feature_value)
      encoded_val = np.asarray(encoded_val)
      if tokenizer_config.add_eos:
        pad_width = [(0, 1)]
        # Tokenized rank is generally 1; adjust pad_width in case it's more
        pad_width += [(0, 0)] * (len(encoded_val.shape) - 1)
        encoded_val = np.pad(
            encoded_val,
            pad_width,
            constant_values=tokenizer_config.vocab.eos_id,
        )
      final_example[feature_name] = encoded_val
    final_example = cast(Out, final_example)

    return final_example
