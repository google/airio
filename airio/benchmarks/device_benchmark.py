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

"""Microbenchmarks for AirIO tpu functions."""

import functools
import os

import airio
from airio import examples
import google_benchmark
import jax
import jax.numpy as jnp
import numpy as np
from seqio import vocabularies
import tensorflow_datasets as tfds

partial = functools.partial


_SOURCE_NUM_EXAMPLES = 1000


def requires_tpu(num_devices_required):
  """Helper to skip benchmarks that require TPUs."""

  def helper1(f):
    @functools.wraps(f)
    def helper2(state):
      if jax.device_count() < num_devices_required:
        state.skip_with_error(f"requires {num_devices_required} devices")
        return
      return f(state)

    return helper2

  return helper1


def _sum_of_squares(x):
  return jnp.sum(x**2)


_sum_of_squares_dx = jax.grad(_sum_of_squares)
_sum_of_squares_dx_jit = jax.jit(_sum_of_squares_dx)


def _get_tokenizer_configs():
  test_dir = os.path.join(
      os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
      "test_data",
  )
  sentencepiece_vocab = vocabularies.SentencePieceVocabulary(
      os.path.join(test_dir, "sentencepiece", "sentencepiece.model")
  )
  return {
      "inputs": airio.tokenizer.TokenizerConfig(vocab=sentencepiece_vocab),
      "targets": airio.tokenizer.TokenizerConfig(vocab=sentencepiece_vocab),
  }


@google_benchmark.register
@requires_tpu(2)
def wmt_generated_data_benchmark(state):
  """Loads a generated WMT dataset onto TPUs and performs a simple calculation."""
  with tfds.testing.mock_data(num_examples=_SOURCE_NUM_EXAMPLES):
    wmt_task = examples.tasks.get_wmt_19_ende_v003_task(
        tokenizer_configs=_get_tokenizer_configs()
    )
  ds = wmt_task.get_dataset(split="train")

  while state:
    for element in ds:
      for _, v in element.items():
        if isinstance(v, np.ndarray):
          if v.dtype == np.int64:
            v = v.astype(np.float32)
          # Transfer to device.
          x = jax.device_put(v)
          state.pause_timing()
          # Compile.
          _sum_of_squares_dx_jit(x).block_until_ready()
          state.resume_timing()
          # Run.
          _sum_of_squares_dx_jit(x).block_until_ready()


if __name__ == "__main__":
  google_benchmark.main()
