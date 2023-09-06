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

"""Tests for airio.dataset_iterators."""

import ast
import functools
import json
import os
from typing import Dict

from absl.testing import absltest
from airio import dataset_iterators
from airio import tokenizer
from clu.data import dataset_iterator
import grain.python as grain
import numpy as np
from seqio import vocabularies


def _parse_and_preprocess(raw_example: bytes) -> Dict[str, str]:
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


class DatasetIteratorsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_data"
    )
    sentencepiece_vocab = vocabularies.SentencePieceVocabulary(
        os.path.join(self.test_dir, "sentencepiece", "sentencepiece.model")
    )
    self.tokenizer_config = tokenizer.TokenizerConfig(vocab=sentencepiece_vocab)

  def _get_dummy_data_loader(self) -> grain.DataLoader:
    test_data_path = os.path.join(self.test_dir, "classification")
    return grain.DataLoader(
        data_source=grain.ArrayRecordDataSource(
            os.path.join(test_data_path, "classification.array_record@2")
        ),
        sampler=grain.SequentialSampler(
            num_records=5, shard_options=grain.NoSharding()
        ),
        operations=[
            grain.MapOperation(map_function=_parse_and_preprocess),
            grain.MapOperation(
                functools.partial(
                    tokenizer.tokenize,
                    tokenizer_configs={
                        "inputs": self.tokenizer_config,
                        "targets": self.tokenizer_config,
                    },
                )
            ),
        ],
    )

  def test_verify_element_spec(self):
    """Verifies that element_spec can be correctly determined."""
    data_loader = self._get_dummy_data_loader()
    iterator_wrapper = dataset_iterators.PyGrainDatasetIteratorWrapper(
        data_loader
    )
    self.assertDictEqual(
        iterator_wrapper.element_spec,
        {
            "inputs": dataset_iterator.ArraySpec(dtype=np.int64, shape=(4,)),
            "targets": dataset_iterator.ArraySpec(dtype=np.int64, shape=(8,)),
        },
    )

    # Ensure that the iterator is not affected by the call to element_spec().
    # Verify that the iterator starts from the first element and includes all
    # elements after determining element_spec.
    expected_inputs = ["abc", "def", "ghi", "jkl", "mno"]
    for idx, element in enumerate(iterator_wrapper):
      self.assertEqual(
          element["inputs_pretokenized"],
          expected_inputs[idx],
      )

  def test_get_state(self):
    """Verifies that state can be correctly read."""
    data_loader = self._get_dummy_data_loader()
    iterator_wrapper = dataset_iterators.PyGrainDatasetIteratorWrapper(
        data_loader
    )
    state_bytes = iterator_wrapper.get_state()
    state_dict = json.loads(state_bytes)
    self.assertEqual(state_dict["last_seen_indices"]["0"], -1)

    # Read first element.
    _ = next(iterator_wrapper)
    new_state_bytes = iterator_wrapper.get_state()
    new_state_dict = json.loads(new_state_bytes)
    self.assertEqual(new_state_dict["last_seen_indices"]["0"], 0)

  def test_set_state(self):
    """Verifies that state can be correctly set."""
    data_loader = self._get_dummy_data_loader()
    iterator_wrapper = dataset_iterators.PyGrainDatasetIteratorWrapper(
        data_loader
    )
    state_bytes = iterator_wrapper.get_state()
    state_dict = json.loads(state_bytes)
    # Modify state to indicate that the first two elements have been read.
    state_dict["last_seen_indices"]["0"] = 1

    new_state = json.dumps(state_dict).encode()
    iterator_wrapper.set_state(new_state)
    next_element = next(iterator_wrapper)
    self.assertEqual(next_element["inputs_pretokenized"], "ghi")

  def test_save_state(self):
    """Verifies that state can be correctly saved to a file."""
    data_loader = self._get_dummy_data_loader()
    iterator_wrapper = dataset_iterators.PyGrainDatasetIteratorWrapper(
        data_loader
    )

    temp_state_file = self.create_tempfile("temp_iterator_state.json")
    iterator_wrapper.save(temp_state_file.full_path)
    state = json.loads(temp_state_file.read_text())
    self.assertEqual(state["last_seen_indices"]["0"], -1)
    self.assertEqual(state["last_worker_index"], -1)
    self.assertEqual(state["worker_count"], 0)

  def test_restore_state(self):
    """Verifies that state can be correctly restored from a file."""
    data_loader = self._get_dummy_data_loader()
    iterator_wrapper = dataset_iterators.PyGrainDatasetIteratorWrapper(
        data_loader
    )
    state = json.loads(iterator_wrapper.get_state())
    state["last_seen_indices"]["0"] = 2

    temp_state_file = self.create_tempfile("temp_iterator_state.json")
    temp_state_file.write_text(json.dumps(state))

    iterator_wrapper.restore(temp_state_file.full_path)
    state = json.loads(iterator_wrapper.get_state())
    self.assertEqual(state["last_seen_indices"]["0"], 2)
    self.assertEqual(state["last_worker_index"], -1)
    self.assertEqual(state["worker_count"], 0)

  def test_restore_state_file_not_exists(self):
    data_loader = self._get_dummy_data_loader()
    iterator_wrapper = dataset_iterators.PyGrainDatasetIteratorWrapper(
        data_loader
    )
    with self.assertRaisesRegex(
        ValueError, "File non_existent_file does not exist."
    ):
      iterator_wrapper.restore("non_existent_file")


if __name__ == "__main__":
  absltest.main()