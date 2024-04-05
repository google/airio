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

"""Microbenchmarks for AirIO tpu functions."""

import functools
import os

from airio import examples
import airio.pygrain as airio
import airio.pygrain_common as airio_common
import google_benchmark
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds

partial = functools.partial


_SOURCE_NUM_EXAMPLES = 1000


def requires_tpu(num_devices_required: int):
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


def _get_tokenizer_configs() -> dict[str, airio.TokenizerConfig]:
  test_dir = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "../../test_data",
  )
  tokenizer_config = airio.TokenizerConfig(
      vocab=airio.SentencePieceVocabulary(
          os.path.join(test_dir, "sentencepiece", "sentencepiece.model")
      )
  )
  return {
      "inputs": tokenizer_config,
      "targets": tokenizer_config,
  }


@google_benchmark.register
@requires_tpu(2)
def wmt_generated_data_benchmark(state: google_benchmark.State) -> None:
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


@google_benchmark.register
@requires_tpu(2)
def wmt_from_file_benchmark(state: google_benchmark.State) -> None:
  """Loads a WMT dataset from file onto TPUs and performs a simple calculation."""
  wmt_task = examples.tasks.get_wmt_19_ende_v003_task(
      tokenizer_configs=_get_tokenizer_configs()
  )
  ds = wmt_task.get_dataset(split="train")

  element_count = 0
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
      element_count += 1
      if element_count >= _SOURCE_NUM_EXAMPLES:
        break


@google_benchmark.register
@requires_tpu(2)
def c4_span_corruption_generated_data_benchmark(
    state: google_benchmark.State,
) -> None:
  """Loads a generated C4 dataset onto TPUs and performs a simple calculation."""
  with tfds.testing.mock_data(num_examples=_SOURCE_NUM_EXAMPLES):
    c4_task = examples.tasks.get_c4_v220_span_corruption_task(
        tokenizer_configs=_get_tokenizer_configs()
    )
  runtime_preprocessors = airio_common.feature_converters.get_t5x_enc_dec_feature_converter_preprocessors(
      pack=False,
      use_multi_bin_packing=False,
      passthrough_feature_keys=[],
      pad_id=0,
      bos_id=0,
  )
  sequence_lengths = {"inputs": 1024, "targets": 1024}
  ds = c4_task.get_dataset(
      sequence_lengths,
      split="train",
      shuffle=False,
      seed=42,
      runtime_preprocessors=runtime_preprocessors,
      shard_info=airio.ShardInfo(index=0, num_shards=1024),
  )

  while state:
    for element in ds:
      for _, v in element.items():
        if isinstance(v, np.ndarray):
          if v.dtype == np.int64 or v.dtype == bool:
            v = v.astype(np.float32)
          # Transfer to device.
          x = jax.device_put(v)
          state.pause_timing()
          # Compile.
          _sum_of_squares_dx_jit(x).block_until_ready()
          state.resume_timing()
          # Run.
          _sum_of_squares_dx_jit(x).block_until_ready()


@google_benchmark.register
@requires_tpu(2)
def c4_span_corruption_from_file_benchmark(
    state: google_benchmark.State,
) -> None:
  """Loads a C4 dataset from file onto TPUs and performs a simple calculation."""
  c4_task = examples.tasks.get_c4_v220_span_corruption_task(
      tokenizer_configs=_get_tokenizer_configs()
  )
  runtime_preprocessors = airio_common.feature_converters.get_t5x_enc_dec_feature_converter_preprocessors(
      pack=False,
      use_multi_bin_packing=False,
      passthrough_feature_keys=[],
      pad_id=0,
      bos_id=0,
  )
  sequence_lengths = {"inputs": 1024, "targets": 1024}
  ds = c4_task.get_dataset(
      sequence_lengths,
      split="train",
      shuffle=False,
      seed=42,
      runtime_preprocessors=runtime_preprocessors,
      shard_info=airio.ShardInfo(index=0, num_shards=1024),
  )

  element_count = 0
  while state:
    for element in ds:
      for _, v in element.items():
        if isinstance(v, np.ndarray):
          if v.dtype == np.int64 or v.dtype == bool:
            v = v.astype(np.float32)
          # Transfer to device.
          x = jax.device_put(v)
          state.pause_timing()
          # Compile.
          _sum_of_squares_dx_jit(x).block_until_ready()
          state.resume_timing()
          # Run.
          _sum_of_squares_dx_jit(x).block_until_ready()
      element_count += 1
      if element_count >= 100:
        break


if __name__ == "__main__":
  google_benchmark.main()
