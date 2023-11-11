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

"""Microbenchmarks for AirIO data_sources functions."""

import airio
import google_benchmark
import numpy as np

_SOURCE_NUM_EXAMPLES = 3
_SOURCE_SPLITS = {"train", "test", "unsupervised"}


def generate_function_data_source(split: str):
  if split not in _SOURCE_SPLITS:
    raise ValueError(f"Split {split} not found in {_SOURCE_SPLITS}.")
  return np.array(range(_SOURCE_NUM_EXAMPLES))


@google_benchmark.register
def function_data_source_create(state):
  while state:
    airio.data_sources.FunctionDataSource(
        dataset_fn=generate_function_data_source, splits=_SOURCE_SPLITS
    )


@google_benchmark.register
def function_data_source_get(state):
  ds = airio.data_sources.FunctionDataSource(
      dataset_fn=generate_function_data_source, splits=_SOURCE_SPLITS
  )

  while state:
    for split in _SOURCE_SPLITS:
      _ = ds.get_data_source(split)


@google_benchmark.register
def function_data_source_num_input_examples(state):
  ds = airio.data_sources.FunctionDataSource(
      dataset_fn=generate_function_data_source, splits=_SOURCE_SPLITS
  )

  while state:
    for split in _SOURCE_SPLITS:
      _ = ds.num_input_examples(split)


@google_benchmark.register
def function_data_source_splits(state):
  ds = airio.data_sources.FunctionDataSource(
      dataset_fn=generate_function_data_source, splits=_SOURCE_SPLITS
  )

  while state:
    _ = ds.splits


if __name__ == "__main__":
  google_benchmark.main()
