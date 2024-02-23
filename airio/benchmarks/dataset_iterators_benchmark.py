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

"""Microbenchmarks for AirIO dataset_iterators functions."""

import ast
import os
import tempfile
from typing import Dict

import airio.pygrain as airio
import google_benchmark
import grain.python as grain
from seqio import vocabularies


_SOURCE_NUM_EXAMPLES = 6
_TEST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_data"
)
_SENTENCEPIECE_VOCAB = vocabularies.SentencePieceVocabulary(
    os.path.join(_TEST_DIR, "sentencepiece", "sentencepiece.model")
)
_TOKENIZER_CONFIG = airio.TokenizerConfig(vocab=_SENTENCEPIECE_VOCAB)


def parse_and_preprocess(raw_example: bytes) -> Dict[str, str]:
  raw_example = ast.literal_eval(raw_example.decode("utf-8"))

  final_example = {"inputs": raw_example["text"]}
  raw_label = str(raw_example["label"])
  if raw_label == "0":
    final_example["targets"] = "negative"
  elif raw_label == "1":
    final_example["targets"] = "positive"
  else:
    final_example["targets"] = "invalid"
  return final_example


def get_data_loader() -> grain.DataLoader:
  test_data_path = os.path.join(
      _TEST_DIR, "classification/classification.array_record@2"
  )
  return grain.DataLoader(
      data_source=grain.ArrayRecordDataSource(test_data_path),
      sampler=grain.SequentialSampler(
          num_records=_SOURCE_NUM_EXAMPLES, shard_options=grain.NoSharding()
      ),
      operations=[
          airio.MapFnTransform(parse_and_preprocess),
          airio.MapFnTransform(
              airio.Tokenizer(
                  tokenizer_configs={
                      "inputs": _TOKENIZER_CONFIG,
                      "targets": _TOKENIZER_CONFIG,
                  },
              )
          ),
      ],
  )


@google_benchmark.register
def dataset_iterator_create(state):
  data_loader = get_data_loader()
  while state:
    airio.dataset_iterators.PyGrainDatasetIteratorWrapper(data_loader)


@google_benchmark.register
def dataset_iterator_next(state):
  data_loader = get_data_loader()
  iterator_wrapper = airio.dataset_iterators.PyGrainDatasetIteratorWrapper(
      data_loader
  )
  iterator_state = iterator_wrapper.get_state()
  while state:
    _ = next(iterator_wrapper)
    state.pause_timing()
    iterator_wrapper.set_state(iterator_state)
    state.resume_timing()


@google_benchmark.register
def dataset_iterator_element_spec(state):
  data_loader = get_data_loader()
  iterator_wrapper = airio.dataset_iterators.PyGrainDatasetIteratorWrapper(
      data_loader
  )
  while state:
    _ = iterator_wrapper.element_spec


@google_benchmark.register
def dataset_iterator_peek(state):
  data_loader = get_data_loader()
  iterator_wrapper = airio.dataset_iterators.PyGrainDatasetIteratorWrapper(
      data_loader
  )
  while state:
    _ = iterator_wrapper.peek()


@google_benchmark.register
def dataset_iterator_peek_async(state):
  data_loader = get_data_loader()
  iterator_wrapper = airio.dataset_iterators.PyGrainDatasetIteratorWrapper(
      data_loader
  )
  while state:
    _ = iterator_wrapper.peek_async().result()


@google_benchmark.register
def dataset_iterator_get_state(state):
  data_loader = get_data_loader()
  iterator_wrapper = airio.dataset_iterators.PyGrainDatasetIteratorWrapper(
      data_loader
  )
  while state:
    _ = iterator_wrapper.get_state()


@google_benchmark.register
def dataset_iterator_set_state(state):
  """Measures setting state."""
  data_loader = get_data_loader()
  iterator_wrapper = airio.dataset_iterators.PyGrainDatasetIteratorWrapper(
      data_loader
  )
  iterator_state = iterator_wrapper.get_state()
  # Modify state to indicate that the first two elements have been read.
  iterator_state["last_seen_indices"]["0"] = 1
  while state:
    _ = iterator_wrapper.set_state(iterator_state)


@google_benchmark.register
def dataset_iterator_save(state):
  data_loader = get_data_loader()
  iterator_wrapper = airio.dataset_iterators.PyGrainDatasetIteratorWrapper(
      data_loader
  )
  with tempfile.NamedTemporaryFile(delete=False) as temp_state_file:
    while state:
      iterator_wrapper.save(temp_state_file.name)


@google_benchmark.register
def dataset_iterator_restore(state):
  data_loader = get_data_loader()
  iterator_wrapper = airio.dataset_iterators.PyGrainDatasetIteratorWrapper(
      data_loader
  )
  with tempfile.NamedTemporaryFile(delete=False) as temp_state_file:
    iterator_wrapper.save(temp_state_file.name)
    while state:
      iterator_wrapper.restore(temp_state_file.name)


if __name__ == "__main__":
  google_benchmark.main()
