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

"""Tests for airio.dataset_iterators."""

import ast
import json
import os
from typing import Dict
from unittest import mock

from absl.testing import absltest
from airio._src.core import tokenizer as core_tokenizer
from airio._src.pygrain import dataset_iterators
from airio._src.pygrain import preprocessors
from airio._src.pygrain import tokenizer
from clu.data import dataset_iterator as clu_dataset_iterator
import grain.python as grain
import numpy as np
from seqio import vocabularies


lazy_dataset = grain.experimental.lazy_dataset


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


def _get_expected_data() -> list[dict[str, str]]:
  return [
      {
          "inputs_pretokenized": "abc",
          "inputs": [3, 5, 2, 13],
          "targets_pretokenized": "negative",
          "targets": [3, 22, 4, 2, 18, 8, 25, 4],
      },
      {
          "inputs_pretokenized": "def",
          "inputs": [3, 21, 4, 2],
          "targets_pretokenized": "positive",
          "targets": [3, 15, 7, 6, 8, 24, 8, 25, 4],
      },
      {
          "inputs_pretokenized": "ghi",
          "inputs": [3, 2, 20, 8],
          "targets_pretokenized": "negative",
          "targets": [3, 22, 4, 2, 18, 8, 25, 4],
      },
      {
          "inputs_pretokenized": "jkl",
          "inputs": [3, 2, 9],
          "targets_pretokenized": "positive",
          "targets": [3, 15, 7, 6, 8, 24, 8, 25, 4],
      },
      {
          "inputs_pretokenized": "mno",
          "inputs": [3, 14, 22, 7],
          "targets_pretokenized": "negative",
          "targets": [3, 22, 4, 2, 18, 8, 25, 4],
      },
  ]


class DatasetIteratorsWithDataLoaderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../test_data"
    )
    sentencepiece_vocab = vocabularies.SentencePieceVocabulary(
        os.path.join(self.test_dir, "sentencepiece", "sentencepiece.model")
    )
    self.tokenizer_config = core_tokenizer.TokenizerConfig(
        vocab=sentencepiece_vocab
    )
    self.expected_data = _get_expected_data()

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
            preprocessors.MapFnTransform(_parse_and_preprocess),
            preprocessors.MapFnTransform(
                tokenizer.Tokenizer(
                    tokenizer_configs={
                        "inputs": self.tokenizer_config,
                        "targets": self.tokenizer_config,
                    },
                )
            ),
        ],
    )

  def test_iterator(self):
    data_loader = self._get_dummy_data_loader()
    it = dataset_iterators.PyGrainDatasetIteratorWrapper(data_loader)
    for actual, expected in zip(it, self.expected_data, strict=True):
      self.assertSequenceEqual(sorted(actual.keys()), sorted(expected.keys()))
      for k in actual.keys():
        np.testing.assert_array_equal(actual[k], expected[k])

  def test_take(self):
    data_loader = self._get_dummy_data_loader()
    it = dataset_iterators.PyGrainDatasetIteratorWrapper(data_loader)
    for actual, expected in zip(
        it.take(3), self.expected_data[:3], strict=True
    ):
      self.assertSequenceEqual(sorted(actual.keys()), sorted(expected.keys()))
      for k in actual.keys():
        np.testing.assert_array_equal(actual[k], expected[k])

  def test_take_less_then_more(self):
    data_loader = self._get_dummy_data_loader()
    it = dataset_iterators.PyGrainDatasetIteratorWrapper(data_loader)
    for actual, expected in zip(
        it.take(3).take(5), self.expected_data[:3], strict=True
    ):
      self.assertSequenceEqual(sorted(actual.keys()), sorted(expected.keys()))
      for k in actual.keys():
        np.testing.assert_array_equal(actual[k], expected[k])

  def test_take_more_then_less(self):
    data_loader = self._get_dummy_data_loader()
    it = dataset_iterators.PyGrainDatasetIteratorWrapper(data_loader)
    for actual, expected in zip(
        it.take(5).take(3), self.expected_data[:3], strict=True
    ):
      self.assertSequenceEqual(sorted(actual.keys()), sorted(expected.keys()))
      for k in actual.keys():
        np.testing.assert_array_equal(actual[k], expected[k])

  def test_verify_element_spec(self):
    """Verifies that element_spec can be correctly determined."""
    data_loader = self._get_dummy_data_loader()
    iterator_wrapper = dataset_iterators.PyGrainDatasetIteratorWrapper(
        data_loader
    )
    self.assertDictEqual(
        iterator_wrapper.element_spec,
        {
            "inputs": clu_dataset_iterator.ArraySpec(
                dtype=np.int64, shape=(4,)
            ),
            "targets": clu_dataset_iterator.ArraySpec(
                dtype=np.int64, shape=(8,)
            ),
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
    state_dict = iterator_wrapper.get_state()
    self.assertEqual(state_dict["last_seen_indices"]["0"], -1)

    # Read first element.
    _ = next(iterator_wrapper)
    new_state_dict = iterator_wrapper.get_state()
    self.assertEqual(new_state_dict["last_seen_indices"]["0"], 0)

  def test_set_state(self):
    """Verifies that state can be correctly set."""
    data_loader = self._get_dummy_data_loader()
    iterator_wrapper = dataset_iterators.PyGrainDatasetIteratorWrapper(
        data_loader
    )
    state_dict = iterator_wrapper.get_state()
    # Modify state to indicate that the first two elements have been read.
    state_dict["last_seen_indices"]["0"] = 1

    iterator_wrapper.set_state(state_dict)
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
    state = iterator_wrapper.get_state()
    state["last_seen_indices"]["0"] = 2

    temp_state_file = self.create_tempfile("temp_iterator_state.json")
    temp_state_file.write_text(json.dumps(state))

    iterator_wrapper.restore(temp_state_file.full_path)
    state = iterator_wrapper.get_state()
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

  def test_peek_gives_next_element(self):
    data_loader = self._get_dummy_data_loader()
    iterator_wrapper = dataset_iterators.PyGrainDatasetIteratorWrapper(
        data_loader
    )
    peek_element = iterator_wrapper.peek()
    self.assertEqual(peek_element["inputs_pretokenized"], "abc")

    next_element = next(iterator_wrapper)
    self.assertEqual(next_element["inputs_pretokenized"], "abc")

  def test_peek_async(self):
    data_loader = self._get_dummy_data_loader()
    iterator_wrapper = dataset_iterators.PyGrainDatasetIteratorWrapper(
        data_loader
    )
    future = iterator_wrapper.peek_async()
    self.assertIsNone(iterator_wrapper._peek)
    future.result()
    self.assertIsNotNone(iterator_wrapper._peek)
    peek_element = iterator_wrapper.peek()
    self.assertEqual(peek_element["inputs_pretokenized"], "abc")
    first_element = next(iterator_wrapper)
    self.assertEqual(first_element["inputs_pretokenized"], "abc")
    second_element = next(iterator_wrapper)
    self.assertEqual(second_element["inputs_pretokenized"], "def")



class DatasetIteratorsWithLazyMapDatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../test_data"
    )
    sentencepiece_vocab = vocabularies.SentencePieceVocabulary(
        os.path.join(self.test_dir, "sentencepiece", "sentencepiece.model")
    )
    self.tokenizer_config = core_tokenizer.TokenizerConfig(
        vocab=sentencepiece_vocab
    )
    self.expected_data = _get_expected_data()

  def _get_dummy_lazy_map_dataset(self) -> lazy_dataset.LazyMapDataset:
    test_data_path = os.path.join(self.test_dir, "classification")
    ds = lazy_dataset.SourceLazyMapDataset(
        grain.ArrayRecordDataSource(
            os.path.join(test_data_path, "classification.array_record@2")
        )
    )
    ds = ds.map(preprocessors.MapFnTransform(_parse_and_preprocess))
    ds = ds.map(
        preprocessors.MapFnTransform(
            tokenizer.Tokenizer(
                tokenizer_configs={
                    "inputs": self.tokenizer_config,
                    "targets": self.tokenizer_config,
                },
            )
        )
    )
    ds = ds[:5]
    return ds

  def test_take(self):
    ds = self._get_dummy_lazy_map_dataset()
    it = dataset_iterators.PyGrainDatasetIteratorWrapper(ds)
    for actual, expected in zip(
        it.take(3), self.expected_data[:3], strict=True
    ):
      self.assertSequenceEqual(sorted(actual.keys()), sorted(expected.keys()))
      for k in actual.keys():
        np.testing.assert_array_equal(actual[k], expected[k])

  def test_take_less_then_more(self):
    ds = self._get_dummy_lazy_map_dataset()
    it = dataset_iterators.PyGrainDatasetIteratorWrapper(ds)
    for actual, expected in zip(
        it.take(3).take(5), self.expected_data[:3], strict=True
    ):
      self.assertSequenceEqual(sorted(actual.keys()), sorted(expected.keys()))
      for k in actual.keys():
        np.testing.assert_array_equal(actual[k], expected[k])

  def test_take_more_then_less(self):
    ds = self._get_dummy_lazy_map_dataset()
    it = dataset_iterators.PyGrainDatasetIteratorWrapper(ds)
    for actual, expected in zip(
        it.take(5).take(3), self.expected_data[:3], strict=True
    ):
      self.assertSequenceEqual(sorted(actual.keys()), sorted(expected.keys()))
      for k in actual.keys():
        np.testing.assert_array_equal(actual[k], expected[k])

  def test_verify_element_spec(self):
    """Verifies that element_spec can be correctly determined."""
    ds = self._get_dummy_lazy_map_dataset()
    iterator_wrapper = dataset_iterators.PyGrainDatasetIteratorWrapper(ds)
    self.assertDictEqual(
        iterator_wrapper.element_spec,
        {
            "inputs": clu_dataset_iterator.ArraySpec(
                dtype=np.int64, shape=(4,)
            ),
            "targets": clu_dataset_iterator.ArraySpec(
                dtype=np.int64, shape=(8,)
            ),
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
    ds = self._get_dummy_lazy_map_dataset()
    iterator_wrapper = dataset_iterators.PyGrainDatasetIteratorWrapper(ds)
    state_dict = iterator_wrapper.get_state()
    self.assertDictEqual(state_dict, {"next_index": 0})

    # Read first element.
    _ = next(iterator_wrapper)
    new_state_dict = iterator_wrapper.get_state()
    self.assertDictEqual(new_state_dict, {"next_index": 1})

  def test_set_state(self):
    """Verifies that state can be correctly set."""
    ds = self._get_dummy_lazy_map_dataset()
    iterator_wrapper = dataset_iterators.PyGrainDatasetIteratorWrapper(ds)

    # Modify state to indicate that the first two elements have been read.
    new_state = {"next_index": 2}
    iterator_wrapper.set_state(new_state)
    next_element = next(iterator_wrapper)
    self.assertEqual(next_element["inputs_pretokenized"], "ghi")

  def test_save_state(self):
    """Verifies that state can be correctly saved to a file."""
    ds = self._get_dummy_lazy_map_dataset()
    iterator_wrapper = dataset_iterators.PyGrainDatasetIteratorWrapper(ds)

    temp_state_file = self.create_tempfile("temp_iterator_state.json")
    iterator_wrapper.save(temp_state_file.full_path)
    state = json.loads(temp_state_file.read_text())
    self.assertDictEqual(state, {"next_index": 0})

  def test_restore_state(self):
    """Verifies that state can be correctly restored from a file."""
    ds = self._get_dummy_lazy_map_dataset()
    iterator_wrapper = dataset_iterators.PyGrainDatasetIteratorWrapper(ds)
    state = {"next_index": 2}

    temp_state_file = self.create_tempfile("temp_iterator_state.json")
    temp_state_file.write_text(json.dumps(state))

    iterator_wrapper.restore(temp_state_file.full_path)
    state = iterator_wrapper.get_state()
    self.assertDictEqual(state, {"next_index": 2})

  def test_restore_state_file_not_exists(self):
    ds = self._get_dummy_lazy_map_dataset()
    iterator_wrapper = dataset_iterators.PyGrainDatasetIteratorWrapper(ds)
    with self.assertRaisesRegex(
        ValueError, "File non_existent_file does not exist."
    ):
      iterator_wrapper.restore("non_existent_file")

  def test_peek_gives_next_element(self):
    ds = self._get_dummy_lazy_map_dataset()
    iterator_wrapper = dataset_iterators.PyGrainDatasetIteratorWrapper(ds)
    peek_element = iterator_wrapper.peek()
    self.assertEqual(peek_element["inputs_pretokenized"], "abc")

    next_element = next(iterator_wrapper)
    self.assertEqual(next_element["inputs_pretokenized"], "abc")

  def test_peek_async(self):
    ds = self._get_dummy_lazy_map_dataset()
    iterator_wrapper = dataset_iterators.PyGrainDatasetIteratorWrapper(ds)
    future = iterator_wrapper.peek_async()
    self.assertIsNone(iterator_wrapper._peek)
    future.result()
    self.assertIsNotNone(iterator_wrapper._peek)
    peek_element = iterator_wrapper.peek()
    self.assertEqual(peek_element["inputs_pretokenized"], "abc")
    first_element = next(iterator_wrapper)
    self.assertEqual(first_element["inputs_pretokenized"], "abc")
    second_element = next(iterator_wrapper)
    self.assertEqual(second_element["inputs_pretokenized"], "def")


class DatasetIteratorsWithLazyIterDatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../test_data"
    )
    sentencepiece_vocab = vocabularies.SentencePieceVocabulary(
        os.path.join(self.test_dir, "sentencepiece", "sentencepiece.model")
    )
    self.tokenizer_config = core_tokenizer.TokenizerConfig(
        vocab=sentencepiece_vocab
    )
    self.expected_data = _get_expected_data()

  def _get_dummy_lazy_iter_dataset(self) -> lazy_dataset.LazyMapDataset:
    test_data_path = os.path.join(self.test_dir, "classification")
    ds = lazy_dataset.SourceLazyMapDataset(
        grain.ArrayRecordDataSource(
            os.path.join(test_data_path, "classification.array_record@2")
        )
    )
    ds = ds.map(preprocessors.MapFnTransform(_parse_and_preprocess))
    ds = ds.map(
        preprocessors.MapFnTransform(
            tokenizer.Tokenizer(
                tokenizer_configs={
                    "inputs": self.tokenizer_config,
                    "targets": self.tokenizer_config,
                },
            )
        )
    )
    ds = ds[:5]
    ds = ds.to_iter_dataset()
    return ds

  def test_iterator(self):
    ds = self._get_dummy_lazy_iter_dataset()
    it = dataset_iterators.PyGrainDatasetIteratorWrapper(ds)
    for actual, expected in zip(it, self.expected_data, strict=True):
      self.assertSequenceEqual(sorted(actual.keys()), sorted(expected.keys()))
      for k in actual.keys():
        np.testing.assert_array_equal(actual[k], expected[k])

  def test_take(self):
    ds = self._get_dummy_lazy_iter_dataset()
    it = dataset_iterators.PyGrainDatasetIteratorWrapper(ds)
    for actual, expected in zip(
        it.take(3), self.expected_data[:3], strict=True
    ):
      self.assertSequenceEqual(sorted(actual.keys()), sorted(expected.keys()))
      for k in actual.keys():
        np.testing.assert_array_equal(actual[k], expected[k])

  def test_take_less_then_more(self):
    ds = self._get_dummy_lazy_iter_dataset()
    it = dataset_iterators.PyGrainDatasetIteratorWrapper(ds)
    for actual, expected in zip(
        it.take(3).take(5), self.expected_data[:3], strict=True
    ):
      self.assertSequenceEqual(sorted(actual.keys()), sorted(expected.keys()))
      for k in actual.keys():
        np.testing.assert_array_equal(actual[k], expected[k])

  def test_take_more_then_less(self):
    ds = self._get_dummy_lazy_iter_dataset()
    it = dataset_iterators.PyGrainDatasetIteratorWrapper(ds)
    for actual, expected in zip(
        it.take(5).take(3), self.expected_data[:3], strict=True
    ):
      self.assertSequenceEqual(sorted(actual.keys()), sorted(expected.keys()))
      for k in actual.keys():
        np.testing.assert_array_equal(actual[k], expected[k])

  def test_verify_element_spec(self):
    """Verifies that element_spec can be correctly determined."""
    ds = self._get_dummy_lazy_iter_dataset()
    iterator_wrapper = dataset_iterators.PyGrainDatasetIteratorWrapper(ds)
    self.assertDictEqual(
        iterator_wrapper.element_spec,
        {
            "inputs": clu_dataset_iterator.ArraySpec(
                dtype=np.int64, shape=(4,)
            ),
            "targets": clu_dataset_iterator.ArraySpec(
                dtype=np.int64, shape=(8,)
            ),
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
    ds = self._get_dummy_lazy_iter_dataset()
    iterator_wrapper = dataset_iterators.PyGrainDatasetIteratorWrapper(ds)
    state_dict = iterator_wrapper.get_state()
    self.assertDictEqual(state_dict, {"next_index": 0})

    # Read first element.
    _ = next(iterator_wrapper)
    new_state_dict = iterator_wrapper.get_state()
    self.assertDictEqual(new_state_dict, {"next_index": 1})

  def test_set_state(self):
    """Verifies that state can be correctly set."""
    ds = self._get_dummy_lazy_iter_dataset()
    iterator_wrapper = dataset_iterators.PyGrainDatasetIteratorWrapper(ds)

    # Modify state to indicate that the first two elements have been read.
    new_state = {"next_index": 2}
    iterator_wrapper.set_state(new_state)
    next_element = next(iterator_wrapper)
    self.assertEqual(next_element["inputs_pretokenized"], "ghi")

  def test_save_state(self):
    """Verifies that state can be correctly saved to a file."""
    ds = self._get_dummy_lazy_iter_dataset()
    iterator_wrapper = dataset_iterators.PyGrainDatasetIteratorWrapper(ds)

    temp_state_file = self.create_tempfile("temp_iterator_state.json")
    iterator_wrapper.save(temp_state_file.full_path)
    state = json.loads(temp_state_file.read_text())
    self.assertDictEqual(state, {"next_index": 0})

  def test_restore_state(self):
    """Verifies that state can be correctly restored from a file."""
    ds = self._get_dummy_lazy_iter_dataset()
    iterator_wrapper = dataset_iterators.PyGrainDatasetIteratorWrapper(ds)
    state = {"next_index": 2}

    temp_state_file = self.create_tempfile("temp_iterator_state.json")
    temp_state_file.write_text(json.dumps(state))

    iterator_wrapper.restore(temp_state_file.full_path)
    state = iterator_wrapper.get_state()
    self.assertDictEqual(state, {"next_index": 2})

  def test_restore_state_file_not_exists(self):
    ds = self._get_dummy_lazy_iter_dataset()
    iterator_wrapper = dataset_iterators.PyGrainDatasetIteratorWrapper(ds)
    with self.assertRaisesRegex(
        ValueError, "File non_existent_file does not exist."
    ):
      iterator_wrapper.restore("non_existent_file")

  def test_peek_gives_next_element(self):
    ds = self._get_dummy_lazy_iter_dataset()
    iterator_wrapper = dataset_iterators.PyGrainDatasetIteratorWrapper(ds)
    peek_element = iterator_wrapper.peek()
    self.assertEqual(peek_element["inputs_pretokenized"], "abc")

    next_element = next(iterator_wrapper)
    self.assertEqual(next_element["inputs_pretokenized"], "abc")

  def test_peek_async(self):
    ds = self._get_dummy_lazy_iter_dataset()
    iterator_wrapper = dataset_iterators.PyGrainDatasetIteratorWrapper(ds)
    future = iterator_wrapper.peek_async()
    self.assertIsNone(iterator_wrapper._peek)
    future.result()
    self.assertIsNotNone(iterator_wrapper._peek)
    peek_element = iterator_wrapper.peek()
    self.assertEqual(peek_element["inputs_pretokenized"], "abc")
    first_element = next(iterator_wrapper)
    self.assertEqual(first_element["inputs_pretokenized"], "abc")
    second_element = next(iterator_wrapper)
    self.assertEqual(second_element["inputs_pretokenized"], "def")


if __name__ == "__main__":
  absltest.main()
