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

"""Microbenchmarks for AirIO feature_converters functions."""

import os
from typing import Dict

import airio
from airio.grain import dataset_providers
from airio.grain import feature_converters
import google_benchmark
from seqio import vocabularies
import tensorflow_datasets as tfds


_SOURCE_NAME = "imdb_reviews"
_SOURCE_NUM_EXAMPLES = 3
_SOURCE_SPLITS = ("train", "test", "unsupervised")
_TEST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_data"
)
_SENTENCEPIECE_VOCAB = vocabularies.SentencePieceVocabulary(
    os.path.join(_TEST_DIR, "sentencepiece", "sentencepiece.model")
)
_TOKENIZER_CONFIG = airio.tokenizer.TokenizerConfig(vocab=_SENTENCEPIECE_VOCAB)


def _create_feature_converter() -> (
    feature_converters.PyGrainEncDecFeatureConverter
):
  return feature_converters.PyGrainEncDecFeatureConverter(
      bos_id=1,
      pack=False,
  )


def _create_task() -> dataset_providers.GrainTask:
  """Create a simple task."""
  with tfds.testing.mock_data(_SOURCE_NUM_EXAMPLES):
    source = airio.data_sources.TfdsDataSource(
        tfds_name=_SOURCE_NAME, splits=_SOURCE_SPLITS
    )

  def _imdb_preprocessor(raw_example: Dict[str, str]) -> Dict[str, str]:
    final_example = {"inputs": "imdb " + raw_example["text"]}
    raw_label = str(raw_example["label"])
    if raw_label == "0":
      final_example["targets"] = "negative"
    elif raw_label == "1":
      final_example["targets"] = "positive"
    else:
      final_example["targets"] = "invalid"
    return final_example

  preprocessors = [
      airio.preprocessors.MapFnTransform(_imdb_preprocessor),
      airio.preprocessors.MapFnTransform(
          airio.tokenizer.Tokenizer(
              tokenizer_configs={
                  "inputs": _TOKENIZER_CONFIG,
                  "targets": _TOKENIZER_CONFIG,
              },
          )
      ),
  ]
  return dataset_providers.GrainTask(
      name="dummy_airio_task",
      source=source,
      preprocessors=preprocessors,
  )


@google_benchmark.register
def feature_converter_create(state):
  while state:
    _ = _create_feature_converter()


@google_benchmark.register
def feature_converter_get_transforms_with_feature_lengths(state):
  fc = _create_feature_converter()
  while state:
    _ = fc.get_transforms(task_feature_lengths={"inputs": 5, "targets": 4})


@google_benchmark.register
def feature_converter_get_transforms_no_trim_pad_op_when_feature_lengths_not_set(
    state,
):
  fc = _create_feature_converter()
  while state:
    _ = fc.get_transforms(
        task_feature_lengths=None,
    )


@google_benchmark.register
def feature_converter_convert_with_task_feature_lengths(state):
  task = _create_task()
  sequence_lengths = {"inputs": 5, "targets": 4}
  fc = _create_feature_converter()
  while state:
    _ = task.get_dataset(
        sequence_lengths=sequence_lengths,
        split="train",
        runtime_preprocessors=fc.get_transforms(sequence_lengths),
        shuffle=False,
    )


@google_benchmark.register
def feature_converter_convert_no_task_feature_lengths(state):
  task = _create_task()
  fc = _create_feature_converter()
  while state:
    _ = task.get_dataset(
        sequence_lengths=None,
        split="train",
        runtime_preprocessors=fc.get_transforms(task_feature_lengths=None),
        shuffle=False,
    )


if __name__ == "__main__":
  google_benchmark.main()
