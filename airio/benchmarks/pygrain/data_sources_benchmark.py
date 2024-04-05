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

"""Microbenchmarks for AirIO data_sources functions."""

import os

import airio.pygrain
import google_benchmark
import numpy as np
import tensorflow_datasets as tfds

_SOURCE_NAME = "imdb_reviews"
_SOURCE_NUM_EXAMPLES = 3
_SOURCE_SPLITS = ("train", "test", "unsupervised")


def _create_array_record_data_source() -> airio.pygrain.ArrayRecordDataSource:
  """Creates a basic ArrayRecordDataSource."""
  split_to_filepattern = {
      split: os.path.join(_TEST_DATA_DIR, "classification.array_record@2")
      for split in _SOURCE_SPLITS
  }
  return airio.pygrain.ArrayRecordDataSource(
      split_to_filepattern=split_to_filepattern,
  )




def _generate_function_data_source(split: str):
  """Generates a simple dataset for testing.

  Args:
    split: must be one of ("train", "test", "unsupervised").

  Returns:
    A dataset with 3 records.
  """
  if split not in _SOURCE_SPLITS:
    raise ValueError(f"Split {split} not found in {_SOURCE_SPLITS}.")
  return np.array(range(_SOURCE_NUM_EXAMPLES))




@google_benchmark.register
def array_record_data_source_create(state: google_benchmark.State) -> None:
  """Measures creating an array record data source."""
  while state:
    _create_array_record_data_source()


@google_benchmark.register
def array_record_data_source_get(state: google_benchmark.State) -> None:
  """Measures getting an array record data source."""
  ds = _create_array_record_data_source()
  while state:
    for split in _SOURCE_SPLITS:
      ds.get_data_source(split)


@google_benchmark.register
def array_record_data_source_num_input_examples(
    state: google_benchmark.State,
) -> None:
  """Measures getting number of input examples for an array record data source."""
  ds = _create_array_record_data_source()
  while state:
    for split in _SOURCE_SPLITS:
      ds.num_input_examples(split)


@google_benchmark.register
def array_record_data_source_splits(state: google_benchmark.State) -> None:
  """Measures getting splits for an array record data source."""
  ds = _create_array_record_data_source()
  while state:
    _ = ds.splits




@google_benchmark.register
def function_data_source_create(state: google_benchmark.State) -> None:
  """Measures creating a basic function data source."""
  while state:
    airio.pygrain.FunctionDataSource(
        dataset_fn=_generate_function_data_source, splits=_SOURCE_SPLITS
    )


@google_benchmark.register
def function_data_source_get(state: google_benchmark.State) -> None:
  """Measures getting a basic function data source."""
  ds = airio.pygrain.FunctionDataSource(
      dataset_fn=_generate_function_data_source, splits=_SOURCE_SPLITS
  )
  while state:
    for split in _SOURCE_SPLITS:
      ds.get_data_source(split)


@google_benchmark.register
def function_data_source_num_input_examples(
    state: google_benchmark.State,
) -> None:
  """Measures getting number of input examples for a function data source."""
  ds = airio.pygrain.FunctionDataSource(
      dataset_fn=_generate_function_data_source, splits=_SOURCE_SPLITS
  )
  while state:
    for split in _SOURCE_SPLITS:
      ds.num_input_examples(split)


@google_benchmark.register
def function_data_source_splits(state: google_benchmark.State) -> None:
  """Measures getting splits for a function data source."""
  ds = airio.pygrain.FunctionDataSource(
      dataset_fn=_generate_function_data_source, splits=_SOURCE_SPLITS
  )
  while state:
    _ = ds.splits




@google_benchmark.register
def tfds_data_source_create(state: google_benchmark.State) -> None:
  """Measures creating a basic TFDS data source."""
  with tfds.testing.mock_data(_SOURCE_NUM_EXAMPLES):
    while state:
      airio.pygrain.TfdsDataSource(
          tfds_name=_SOURCE_NAME, splits=_SOURCE_SPLITS
      )


@google_benchmark.register
def tfds_data_source_get(state: google_benchmark.State) -> None:
  """Measures getting a basic TFDS data source."""
  with tfds.testing.mock_data(_SOURCE_NUM_EXAMPLES):
    ds = airio.pygrain.TfdsDataSource(
        tfds_name=_SOURCE_NAME, splits=_SOURCE_SPLITS
    )
  while state:
    for split in _SOURCE_SPLITS:
      ds.get_data_source(split)


@google_benchmark.register
def tfds_data_source_num_input_examples(state: google_benchmark.State) -> None:
  """Measures getting number of input examples for a TFDS data source."""
  with tfds.testing.mock_data(_SOURCE_NUM_EXAMPLES):
    ds = airio.pygrain.TfdsDataSource(
        tfds_name=_SOURCE_NAME, splits=_SOURCE_SPLITS
    )
  while state:
    for split in _SOURCE_SPLITS:
      ds.num_input_examples(split)


@google_benchmark.register
def tfds_data_source_splits(state: google_benchmark.State) -> None:
  """Measures getting splits for a TFDS data source."""
  with tfds.testing.mock_data(_SOURCE_NUM_EXAMPLES):
    ds = airio.pygrain.TfdsDataSource(
        tfds_name=_SOURCE_NAME, splits=_SOURCE_SPLITS
    )
  while state:
    _ = ds.splits


if __name__ == "__main__":
  google_benchmark.main()
