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

"""Tests for airio.dataset_providers."""

import functools
import os
from typing import Dict, Sequence
from unittest import mock

from absl.testing import absltest
import airio
from airio import preprocessors as preprocessors_lib
from airio import test_utils
from airio.grain import dataset_providers
import grain.python as grain
import numpy as np
from seqio import vocabularies
import tensorflow_datasets as tfds



lazy_dataset = grain.experimental.lazy_dataset
_SOURCE_NAME = "imdb_reviews"
_SOURCE_NUM_EXAMPLES = 3
_SOURCE_SPLITS = {"train", "test", "unsupervised"}


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


def _create_sentencepiece_vocab() -> vocabularies.SentencePieceVocabulary:
  test_data_dir = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "test_data"
  )
  sentencepiece_vocab = vocabularies.SentencePieceVocabulary(
      os.path.join(test_data_dir, "sentencepiece", "sentencepiece.model")
  )
  return sentencepiece_vocab


def _create_tokenizer_config() -> airio.tokenizer.TokenizerConfig:
  return airio.tokenizer.TokenizerConfig(vocab=_create_sentencepiece_vocab())


def _create_preprocessors() -> (
    Sequence[airio.dataset_providers.AirIOPreprocessor]
):
  tokenizer_config = _create_tokenizer_config()
  return [
      preprocessors_lib.MapFnTransform(_imdb_preprocessor),
      preprocessors_lib.MapFnTransform(
          airio.tokenizer.Tokenizer(
              tokenizer_configs={
                  "inputs": tokenizer_config,
                  "targets": tokenizer_config,
              },
          )
      ),
  ]


def _create_runtime_preprocessors(
    feature_lengths: Dict[str, int] | None = None,
) -> Sequence[airio.dataset_providers.AirIOPreprocessor]:
  # TODO(b/311543848): Fully remove FeatureConverter.
  return (
      airio.feature_converters.PyGrainEncDecFeatureConverter().get_transforms(
          task_feature_lengths=feature_lengths
      )
  )


def _create_source(
    source_name: str = _SOURCE_NAME,
    splits: Sequence[str] | None = None,
    num_examples: int = _SOURCE_NUM_EXAMPLES,
) -> airio.data_sources.TfdsDataSource:
  """Creates a basic TfdsDataSource."""
  if splits is None:
    splits = _SOURCE_SPLITS
  with tfds.testing.mock_data(num_examples):
    return airio.data_sources.TfdsDataSource(
        tfds_name=source_name, splits=splits
    )


def _create_fn_src(num_elements=5):
  def _dataset_fn(split: str):
    del split
    return np.arange(num_elements)

  return airio.data_sources.FunctionDataSource(
      dataset_fn=_dataset_fn, splits=["train"]
  )


def _create_task(
    source: airio.data_sources.DataSource | None,
    preprocessors: (
        Sequence[airio.dataset_providers.AirIOPreprocessor] | None
    ) = None,
    task_name: str = "dummy_airio_task",
) -> airio.dataset_providers.Task:
  """Create example AirIO task."""
  return airio.dataset_providers.Task(
      name=task_name,
      source=source,
      preprocessors=preprocessors,
  )


def _create_task_builder(
    source: airio.data_sources.DataSource | None,
    preprocessors: (
        Sequence[airio.dataset_providers.AirIOPreprocessor] | None
    ) = None,
    task_name: str = "dummy_airio_task",
) -> airio.dataset_providers.TaskBuilder:
  return airio.dataset_providers.TaskBuilder(
      task_name=task_name,
      source=source,
      preprocessors=preprocessors,
  )


class _TestFilterLazyDatasetIterator(lazy_dataset.LazyDatasetIterator):
  """Iterator that filters elements based on an int threshold."""

  def __init__(
      self,
      parent: lazy_dataset.LazyDatasetIterator,
      threshold: int,
  ):
    super().__init__()
    self._parent = parent
    self._threshold = threshold
    self._index = 0

  def __next__(self):
    while True:
      elem = next(self._parent)
      if elem > self._threshold:
        return elem

  def get_state(self):
    return {"parent": self._parent.get_state(), "threshold": self._threshold}

  def set_state(self, state):
    self._parent.set_state(state["parent"])
    self._threshold = state["threshold"]


class TestFilterLazyIterDataset(lazy_dataset.LazyIterDataset):
  """LazyIterDataset thatfilters elements based on an int threshold."""

  def __init__(
      self,
      parent: lazy_dataset.LazyIterDataset,
      threshold: int,
  ):
    super().__init__(parent)
    self._threshold = threshold

  def __iter__(self) -> _TestFilterLazyDatasetIterator:
    return _TestFilterLazyDatasetIterator(
        self._parent.__iter__(),
        threshold=self._threshold,
    )


class DatasetProviderBaseTest(absltest.TestCase):

  @mock.patch.multiple(
      airio.dataset_providers.DatasetProviderBase, __abstractmethods__=set()
  )
  def test_protocol(self):
    base = airio.dataset_providers.DatasetProviderBase
    self.assertIsNone(base.get_dataset(self, split=""))
    self.assertIsNone(base.num_input_examples(self, split=""))


class TaskTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    def test_map_fn(ex, idx):
      return {"idx": idx, "val": ex}

    def simple_to_imdb_map_fn(
        ex, rargs: preprocessors_lib.AirIOInjectedRuntimeArgs
    ):
      return {
          "inputs_pretokenized": f"{ex}",
          "inputs": np.array([ex] * rargs.sequence_lengths["inputs"]),
          "targets_pretokenized": f"{ex + 1}",
          "targets": np.array([ex + 1] * rargs.sequence_lengths["targets"]),
      }

    self._map_transform_idx_1 = preprocessors_lib.MapFnTransform(
        functools.partial(test_map_fn, idx=1)
    )
    self._simple_to_imdb_prep = preprocessors_lib.MapFnTransform(
        simple_to_imdb_map_fn
    )

  def test_create_task_with_source_only_succeeds(self):
    task = _create_task(source=_create_source(), preprocessors=None)
    self.assertIsInstance(task.source, airio.data_sources.DataSource)
    self.assertIsInstance(task.source, airio.data_sources.TfdsDataSource)
    self.assertEmpty(task.get_preprocessors())

  def test_create_task_with_source_and_empty_preprocessors_succeeds(self):
    task = _create_task(source=_create_source(), preprocessors=[])
    self.assertIsInstance(task.source, airio.data_sources.DataSource)
    self.assertIsInstance(task.source, airio.data_sources.TfdsDataSource)
    self.assertEmpty(task.get_preprocessors())

  def test_create_task(self):
    source = _create_source(splits=_SOURCE_SPLITS)
    task = _create_task(
        source=source,
        preprocessors=_create_preprocessors(),
        task_name="dummy_airio_task",
    )
    self.assertIsInstance(task.source, airio.data_sources.DataSource)
    self.assertIsInstance(task.source, airio.data_sources.TfdsDataSource)
    self.assertEqual(task.name, "dummy_airio_task")
    self.assertEqual(task.splits, _SOURCE_SPLITS)

  def test_empty_splits(self):
    source = _create_source(splits=[])
    task = _create_task(source=source, preprocessors=_create_preprocessors())
    self.assertEmpty(task.splits)

  def test_none_splits(self):
    with tfds.testing.mock_data(_SOURCE_NUM_EXAMPLES):
      source = airio.data_sources.TfdsDataSource(
          tfds_name=_SOURCE_NAME, splits=None
      )
    task = _create_task(source=source, preprocessors=_create_preprocessors())
    self.assertEmpty(task.splits)

  def test_num_input_examples(self):
    source = _create_source(
        splits=_SOURCE_SPLITS,
        num_examples=_SOURCE_NUM_EXAMPLES,
    )
    task = _create_task(source=source, preprocessors=_create_preprocessors())
    num_examples = task.num_input_examples(split="train")
    self.assertEqual(num_examples, _SOURCE_NUM_EXAMPLES)



class TaskBuilderTest(absltest.TestCase):

  def test_task_builder_from_task_copies_params_correctly(self):
    """Verify that the TaskBuilder object is created with correct params."""
    task = _create_task(
        source=_create_source(),
        task_name="dummy_airio_task",
        preprocessors=_create_preprocessors(),
    )
    task_builder = airio.dataset_providers.TaskBuilder.from_task(task)
    self.assertEqual(task_builder._task_name, "dummy_airio_task")
    self.assertEqual(task_builder._source, task.source)
    self.assertEqual(task_builder._preprocessors, task.get_preprocessors())

  def test_task_builder_build_copies_task_correctly(self):
    task_name = "dummy_airio_task"
    source = _create_source()
    preprocessors = _create_preprocessors()
    task_builder = _create_task_builder(
        source=source,
        preprocessors=preprocessors,
        task_name=task_name,
    )
    new_task = task_builder.build()
    self.assertEqual(new_task.name, task_name)
    self.assertEqual(new_task.source, source)
    self.assertEqual(new_task.get_preprocessors(), preprocessors)

  def test_task_builder_set_name_updates_name_correctly(self):
    source = _create_source()
    preprocessors = _create_preprocessors()
    task_builder = _create_task_builder(
        source=source,
        preprocessors=preprocessors,
        task_name="dummy_airio_task",
    )
    task_builder.set_task_name("new_dummy_task")
    new_task = task_builder.build()
    self.assertEqual(new_task.name, "new_dummy_task")
    # Verify rest of the properties are unchanged.
    self.assertEqual(new_task.source, source)
    self.assertEqual(new_task.get_preprocessors(), preprocessors)

  def test_task_builder_set_preprocessors_updates_preprocessors_correctly(self):
    task_name = "dummy_airio_task"
    source = _create_source()
    task_builder = _create_task_builder(
        source=source,
        preprocessors=_create_preprocessors(),
        task_name=task_name,
    )
    new_preprocessors = [preprocessors_lib.MapFnTransform(_imdb_preprocessor)]
    task_builder.set_preprocessors(new_preprocessors)
    new_task = task_builder.build()
    self.assertEqual(new_task.get_preprocessors(), new_preprocessors)
    # Verify rest of the properties are unchanged.
    self.assertEqual(new_task.name, task_name)
    self.assertEqual(new_task.source, source)

  def test_task_builder_set_data_source_updates_source_correctly(self):
    task_name = "dummy_airio_task"
    preprocessors = _create_preprocessors()
    task_builder = _create_task_builder(
        source=_create_source(),
        preprocessors=preprocessors,
        task_name=task_name,
    )
    new_splits = ["train"]
    new_source = _create_source(splits=new_splits)
    task_builder.set_data_source(new_source)
    new_task = task_builder.build()
    self.assertEqual(new_task.source, new_source)
    # Verify rest of the properties are unchanged.
    self.assertEqual(new_task.name, task_name)
    self.assertEqual(new_task.get_preprocessors(), preprocessors)

  def test_task_builder_raises_error_when_source_is_none(self):
    task_builder = _create_task_builder(
        source=None, preprocessors=_create_preprocessors()
    )
    with self.assertRaisesRegex(
        ValueError, "Source has not been set on this task builder."
    ):
      task_builder.build()

  def test_task_builder_raises_error_when_preprocessors_is_none(self):
    task_builder = _create_task_builder(
        source=_create_source(), preprocessors=None
    )
    with self.assertRaisesRegex(
        ValueError, "Preprocessors have not been set on this task builder."
    ):
      task_builder.build()

  def test_task_builder_repr(self):
    task_builder = _create_task_builder(
        source=_create_source(), task_name="dummy_airio_task"
    )
    self.assertStartsWith(
        repr(task_builder),
        "TaskBuilder(task_name=dummy_airio_task,"
        " source=<airio.data_sources.TfdsDataSource",
    )



class MixtureTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    def test_map_fn(ex, idx):
      return {"idx": idx, "val": ex}

    def simple_to_imdb_map_fn(
        ex, rargs: preprocessors_lib.AirIOInjectedRuntimeArgs
    ):
      return {
          "inputs_pretokenized": f"{ex}",
          "inputs": np.array([ex] * rargs.sequence_lengths["inputs"]),
          "targets_pretokenized": f"{ex + 1}",
          "targets": np.array([ex + 1] * rargs.sequence_lengths["targets"]),
      }

    imdb_source = _create_source(
        source_name=_SOURCE_NAME,
        splits=_SOURCE_SPLITS,
        num_examples=_SOURCE_NUM_EXAMPLES,
    )
    self._map_transform_idx_1 = preprocessors_lib.MapFnTransform(
        functools.partial(test_map_fn, idx=1)
    )
    self._map_transform_idx_2 = preprocessors_lib.MapFnTransform(
        functools.partial(test_map_fn, idx=2)
    )
    self._simple_to_imdb_prep = preprocessors_lib.MapFnTransform(
        simple_to_imdb_map_fn
    )
    self._simple_task_1 = _create_task(
        task_name="test_task1",
        source=_create_fn_src(),
        preprocessors=[self._map_transform_idx_1],
    )
    self._simple_task_2 = _create_task(
        task_name="test_task2",
        source=_create_fn_src(),
        preprocessors=[self._map_transform_idx_2],
    )
    self._imdb_task = _create_task(
        source=imdb_source, preprocessors=_create_preprocessors()
    )
    self._simple_to_imdb_task = (
        airio.dataset_providers.TaskBuilder.from_task(self._simple_task_1)
        .set_preprocessors([
            preprocessors_lib.MapFnTransform(simple_to_imdb_map_fn),
        ])
        .build()
    )



class MixturePropertiesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tasks = []
    for i in range(5):
      self.tasks.append(
          _create_task(
              source=_create_fn_src(),
              preprocessors=[],
              task_name=f"test_task_{i}",
          )
      )
    self.simple_mix = airio.dataset_providers.Mixture(
        name="test_mix_1",
        tasks=self.tasks[:3],
        proportions=[1.0, 0.5, 2.0],
    )
    self.mix_of_mix = airio.dataset_providers.Mixture(
        name="test_mix_2",
        tasks=[self.simple_mix, self.tasks[3]],
        proportions=[0.5, 0.7],
    )
    self.mix_of_mix_of_mix = airio.dataset_providers.Mixture(
        name="test_mix_3",
        tasks=[self.simple_mix, self.mix_of_mix, self.tasks[4]],
        proportions=[0.5, 0.7, 0.8],
    )

  def test_tasks_or_mixtures(self):
    self.assertListEqual(self.simple_mix.tasks_or_mixtures, self.tasks[:3])
    self.assertListEqual(
        self.mix_of_mix.tasks_or_mixtures, [self.simple_mix, self.tasks[3]]
    )
    self.assertListEqual(
        self.mix_of_mix_of_mix.tasks_or_mixtures,
        [self.simple_mix, self.mix_of_mix, self.tasks[4]],
    )

  def test_total_proportions(self):
    self.assertAlmostEqual(self.simple_mix.total_proportion, 3.5)
    self.assertAlmostEqual(self.mix_of_mix.total_proportion, 1.2)
    self.assertAlmostEqual(self.mix_of_mix_of_mix.total_proportion, 2.0)

  def test_get_proportion(self):
    self.assertAlmostEqual(self.simple_mix.get_proportion(self.tasks[0]), 1.0)
    self.assertAlmostEqual(self.simple_mix.get_proportion(self.tasks[1]), 0.5)
    self.assertAlmostEqual(self.simple_mix.get_proportion(self.tasks[2]), 2.0)
    self.assertAlmostEqual(self.simple_mix.get_proportion(self.tasks[3]), 0.0)
    self.assertAlmostEqual(self.simple_mix.get_proportion(self.tasks[4]), 0.0)

    self.assertAlmostEqual(
        self.mix_of_mix.get_proportion(self.tasks[0]), 0.5 * (1.0 / 3.5)
    )
    self.assertAlmostEqual(
        self.mix_of_mix.get_proportion(self.tasks[1]), 0.5 * (0.5 / 3.5)
    )
    self.assertAlmostEqual(
        self.mix_of_mix.get_proportion(self.tasks[2]), 0.5 * (2.0 / 3.5)
    )
    self.assertAlmostEqual(self.mix_of_mix.get_proportion(self.tasks[3]), 0.7)
    self.assertAlmostEqual(self.mix_of_mix.get_proportion(self.tasks[4]), 0.0)

    self.assertAlmostEqual(
        self.mix_of_mix_of_mix.get_proportion(self.tasks[0]),
        0.5 * (1.0 / 3.5) + 0.7 * (0.5 / 1.2) * (1.0 / 3.5),
    )
    self.assertAlmostEqual(
        self.mix_of_mix_of_mix.get_proportion(self.tasks[1]),
        0.5 * (0.5 / 3.5) + 0.7 * (0.5 / 1.2) * (0.5 / 3.5),
    )
    self.assertAlmostEqual(
        self.mix_of_mix_of_mix.get_proportion(self.tasks[2]),
        0.5 * (2.0 / 3.5) + 0.7 * (0.5 / 1.2) * (2.0 / 3.5),
    )
    self.assertAlmostEqual(
        self.mix_of_mix_of_mix.get_proportion(self.tasks[3]), 0.7 * (0.7 / 1.2)
    )
    self.assertAlmostEqual(
        self.mix_of_mix_of_mix.get_proportion(self.tasks[4]), 0.8
    )

  def test_leaf_tasks(self):
    self.assertListEqual(self.simple_mix.leaf_tasks, self.tasks[:3])
    self.assertListEqual(self.mix_of_mix.leaf_tasks, self.tasks[:4])
    self.assertListEqual(self.mix_of_mix_of_mix.leaf_tasks, self.tasks)

  def test_splits(self):
    self.assertSequenceEqual(self.simple_mix.splits, ["train"])
    self.assertSequenceEqual(self.mix_of_mix.splits, ["train"])
    self.assertSequenceEqual(self.mix_of_mix_of_mix.splits, ["train"])

  def test_num_input_examples(self):
    self.assertEqual(self.simple_mix.num_input_examples("train"), 3 * 5)
    self.assertEqual(self.mix_of_mix.num_input_examples("train"), 3 * 5 + 5)
    self.assertEqual(
        self.mix_of_mix_of_mix.num_input_examples("train"),
        3 * 5 + (3 * 5 + 5) + 5,
    )

  def test_tasks_and_proportions_mismatch(self):
    with self.assertRaisesRegex(
        ValueError,
        "Mixture invalid_mix must have same number of tasks and proportions.*",
    ):
      _ = airio.dataset_providers.Mixture(
          "invalid_mix", [self.tasks[0], self.tasks[1]], [1.0]
      )

  def test_duplicate_tasks(self):
    with self.assertRaisesRegex(
        ValueError,
        "Mixture invalid_mix has duplicate tasks.*",
    ):
      _ = airio.dataset_providers.Mixture(
          "invalid_mix", [self.tasks[0], self.tasks[0]], [1.0, 1.0]
      )


class DatasetProvidersTest(absltest.TestCase):

  def test_get_dataset_with_task(self):
    source = _create_source(
        source_name=_SOURCE_NAME,
        num_examples=_SOURCE_NUM_EXAMPLES,
    )
    task = dataset_providers.GrainTask(
        name="dummy_airio_task",
        source=source,
        preprocessors=_create_preprocessors(),
    )
    ds = airio.dataset_providers.get_dataset(task)
    expected = [
        {
            "inputs_pretokenized": "imdb ebc   ahgjefjhfe",
            "inputs": [
                3,
                8,
                14,
                21,
                2,
                3,
                4,
                2,
                13,
                3,
                5,
                20,
                2,
                4,
                2,
                20,
                2,
                4,
            ],
            "targets_pretokenized": "positive",
            "targets": [3, 15, 7, 6, 8, 24, 8, 25, 4],
        },
        {
            "inputs_pretokenized": "imdb hj aijbcidcibdg",
            "inputs": [
                3,
                8,
                14,
                21,
                2,
                3,
                20,
                2,
                3,
                5,
                8,
                2,
                13,
                8,
                21,
                13,
                8,
                2,
                21,
                2,
            ],
            "targets_pretokenized": "negative",
            "targets": [3, 22, 4, 2, 18, 8, 25, 4],
        },
        {
            "inputs_pretokenized": "imdb acdhdacfhhjb",
            "inputs": [
                3,
                8,
                14,
                21,
                2,
                3,
                5,
                13,
                21,
                20,
                21,
                5,
                13,
                2,
                20,
                20,
                2,
            ],
            "targets_pretokenized": "positive",
            "targets": [3, 15, 7, 6, 8, 24, 8, 25, 4],
        },
    ]
    test_utils.assert_datasets_equal(ds, expected)

  def test_get_dataset_with_mixture(self):
    def test_map_fn(ex, idx: int):
      return {"task_index": idx, "val": ex}

    tasks = []
    for i in range(2):
      tasks.append(
          dataset_providers.GrainTask(
              name=f"test_task_{i}",
              source=_create_fn_src(num_elements=3),
              preprocessors=[
                  preprocessors_lib.MapFnTransform(
                      functools.partial(test_map_fn, idx=i)
                  )
              ],
          )
      )
    mixture = dataset_providers.GrainMixture(
        name="test_mix",
        tasks=tasks,
        proportions=[1.0, 1.0],
    )
    ds = airio.dataset_providers.get_dataset(mixture, seed=0)
    expected = [
        {"task_index": 0, "val": 0},  # task 1, ex 0
        {"task_index": 1, "val": 0},  # task 2, ex 0
        {"task_index": 0, "val": 1},  # task 1, ex 1
        {"task_index": 1, "val": 1},  # task 2, ex 1
        {"task_index": 0, "val": 2},  # task 1, ex 2
        {"task_index": 1, "val": 2},  # task 2, ex 2
    ]
    test_utils.assert_datasets_equal(ds, expected)

  def test_get_vocabularies_returns_correct_vocabularies(self):
    source = _create_source(
        source_name=_SOURCE_NAME,
        num_examples=_SOURCE_NUM_EXAMPLES,
    )
    task = _create_task(source=source, preprocessors=_create_preprocessors())
    vocabs_map = airio.dataset_providers.get_vocabularies(task)
    expected = {
        "inputs": _create_sentencepiece_vocab(),
        "targets": _create_sentencepiece_vocab(),
    }
    self.assertEqual(vocabs_map, expected)

  def test_get_vocabularies_returns_empty_map_when_no_tokenizer(self):
    source = _create_source(
        source_name=_SOURCE_NAME,
        num_examples=_SOURCE_NUM_EXAMPLES,
    )
    task = _create_task(
        source=source,
        preprocessors=[preprocessors_lib.MapFnTransform(_imdb_preprocessor)],
    )
    vocabs_map = airio.dataset_providers.get_vocabularies(task)
    self.assertEmpty(vocabs_map)



if __name__ == "__main__":
  absltest.main()
