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

"""Tests for airio.feature_converters."""

import os
from typing import Dict, Sequence
from unittest import mock

from absl.testing import absltest
from airio import data_sources
from airio import dataset_providers
from airio import feature_converters
from airio import preprocessors
from airio import test_utils
from airio import tokenizer
import numpy as np
import seqio
from seqio import vocabularies

_SOURCE_SPLITS = {"train", "test", "unsupervised"}


class PyGrainEncDecFeatureConverterTest(absltest.TestCase):

  def _create_source(
      self,
      splits: Sequence[str] | None = None,
  ) -> data_sources.FunctionDataSource:
    """Creates a basic FunctionDataSource."""

    def _generate_dataset(split: str) -> np.ndarray:
      """Generates a dataset with values from tfds.testing.mock_data for imdb_reviews."""
      del split
      data = [
          {"text": "ebc   ahgjefjhfe", "label": "1"},
          {"text": "hj aijbcidcibdg", "label": "0"},
          {"text": "acdhdacfhhjb", "label": "1"},
      ]
      return np.array(data)

    if splits is None:
      splits = _SOURCE_SPLITS
    return data_sources.FunctionDataSource(
        dataset_fn=_generate_dataset, splits=splits
    )

  def _create_task(
      self, source: data_sources.DataSource, task_name: str = "dummy_airio_task"
  ) -> dataset_providers.Task:
    """Creates an example AirIO task."""

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

    test_data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_data"
    )
    sentencepiece_vocab = vocabularies.SentencePieceVocabulary(
        os.path.join(test_data_dir, "sentencepiece", "sentencepiece.model")
    )
    tokenizer_config = tokenizer.TokenizerConfig(vocab=sentencepiece_vocab)
    return dataset_providers.Task(
        name=task_name,
        source=source,
        preprocessors=[
            preprocessors.MapFnTransform(_imdb_preprocessor),
            preprocessors.MapFnTransform(
                tokenizer.Tokenizer(
                    tokenizer_configs={
                        "inputs": tokenizer_config,
                        "targets": tokenizer_config,
                    },
                )
            ),
        ],
    )

  def _create_feature_converter(
      self,
  ) -> feature_converters.PyGrainEncDecFeatureConverter:
    return feature_converters.PyGrainEncDecFeatureConverter(
        bos_id=1,
        pack=False,
    )

  def test_create(self):
    feature_converter = self._create_feature_converter()
    self.assertIsInstance(
        feature_converter, feature_converters.PyGrainEncDecFeatureConverter
    )

  def test_get_transforms_with_feature_lengths(self):
    feature_converter = self._create_feature_converter()
    self.assertLen(
        feature_converter.get_transforms(
            task_feature_lengths={"inputs": 5, "targets": 4}
        ),
        3,
    )

  def test_get_transforms_no_trim_pad_op_when_feature_lengths_not_set(self):
    feature_converter = self._create_feature_converter()
    transforms = feature_converter.get_transforms(
        task_feature_lengths=None,
    )
    self.assertLen(transforms, 2)

  def test_get_transforms_no_trim_pad_op_when_neither_are_set(self):
    feature_converter = self._create_feature_converter()
    transforms = feature_converter.get_transforms(
        task_feature_lengths=None,
    )
    self.assertLen(transforms, 2)

  def test_convert_with_task_feature_lengths(self):
    source = self._create_source(
        splits=_SOURCE_SPLITS,
    )
    task = self._create_task(source)
    feature_converter = self._create_feature_converter()
    ds = task.get_dataset(
        sequence_lengths={"inputs": 5, "targets": 4},
        split="train",
        feature_converter=feature_converter,
        shuffle=False,
    )
    expected = [
        {
            "encoder_input_tokens": [3, 8, 14, 21, 2],
            "decoder_target_tokens": [3, 15, 7, 6],
            "decoder_input_tokens": [3, 15, 7, 6],
            "decoder_loss_weights": [1, 1, 1, 1],
        },
        {
            "encoder_input_tokens": [3, 8, 14, 21, 2],
            "decoder_target_tokens": [3, 22, 4, 2],
            "decoder_input_tokens": [3, 22, 4, 2],
            "decoder_loss_weights": [1, 1, 1, 1],
        },
        {
            "encoder_input_tokens": [3, 8, 14, 21, 2],
            "decoder_target_tokens": [3, 15, 7, 6],
            "decoder_input_tokens": [3, 15, 7, 6],
            "decoder_loss_weights": [1, 1, 1, 1],
        },
    ]
    test_utils.assert_datasets_equal(ds, expected)

  def test_convert_no_task_feature_lengths(self):
    source = self._create_source(
        splits=_SOURCE_SPLITS,
    )
    task = self._create_task(source)
    feature_converter = self._create_feature_converter()
    ds = task.get_dataset(
        sequence_lengths=None,
        split="train",
        feature_converter=feature_converter,
        shuffle=False,
    )
    expected = [
        {
            "encoder_input_tokens": [
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
            "decoder_target_tokens": [3, 15, 7, 6, 8, 24, 8, 25, 4],
            "decoder_input_tokens": [3, 15, 7, 6, 8, 24, 8, 25, 4],
            "decoder_loss_weights": [1, 1, 1, 1, 1, 1, 1, 1, 1],
        },
        {
            "encoder_input_tokens": [
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
            "decoder_target_tokens": [3, 22, 4, 2, 18, 8, 25, 4],
            "decoder_input_tokens": [3, 22, 4, 2, 18, 8, 25, 4],
            "decoder_loss_weights": [1, 1, 1, 1, 1, 1, 1, 1],
        },
        {
            "encoder_input_tokens": [
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
            "decoder_target_tokens": [3, 15, 7, 6, 8, 24, 8, 25, 4],
            "decoder_input_tokens": [3, 15, 7, 6, 8, 24, 8, 25, 4],
            "decoder_loss_weights": [1, 1, 1, 1, 1, 1, 1, 1, 1],
        },
    ]
    test_utils.assert_datasets_equal(ds, expected)

  @mock.patch.multiple(
      feature_converters.PyGrainFeatureConverter, __abstractmethods__=set()
  )
  def test_pygrain_feature_converter_protocol(self):
    pygrain_feature_converter = feature_converters.PyGrainFeatureConverter
    self.assertIsNone(
        pygrain_feature_converter.get_transforms(
            self,
            task_feature_lengths={"inputs": 24, "targets": 12},
        )
    )


class FeatureConvertersTest(absltest.TestCase):

  def test_get_pygrain_feature_converter_sets_properties_correctly(self):
    pygrain_feature_converter = (
        feature_converters.get_pygrain_feature_converter(
            seqio.EncDecFeatureConverter(bos_id=123, pack=False)
        )
    )
    self.assertIsInstance(
        pygrain_feature_converter,
        feature_converters.PyGrainEncDecFeatureConverter,
    )
    self.assertEqual(pygrain_feature_converter._bos_id, 123)
    self.assertFalse(pygrain_feature_converter._pack)

  def test_get_pygrain_feature_converter_raises_error_for_unimplemented(self):
    with self.assertRaisesRegex(
        NotImplementedError,
        "Feature Converter .* does not have PyGrain FeatureConverter"
        " equivalent.",
    ):
      feature_converters.get_pygrain_feature_converter(
          seqio.PrefixLMFeatureConverter()
      )


if __name__ == "__main__":
  absltest.main()
