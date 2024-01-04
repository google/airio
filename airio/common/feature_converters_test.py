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

"""Tests for feature_converters."""

from absl.testing import absltest
from airio import preprocessors as preprocessors_lib
from airio.common import feature_converters
import grain.python as grain
import numpy as np

lazy_dataset = grain.experimental.lazy_dataset


class AutoregressiveInputsTest(absltest.TestCase):

  def test_autoregressive_inputs_unpacked(self):
    x = np.asarray([3, 8, 9, 5, 1, 0, 0])
    actual = feature_converters.make_autoregressive_inputs(
        x, sequence_ids=None, bos_id=0
    )
    expected = [0, 3, 8, 9, 5, 1, 0]
    np.testing.assert_array_equal(actual, expected)
    self.assertEqual(actual.dtype, np.int64)

  def test_autoregressive_inputs_unpacked_with_bos_id(self):
    x = np.asarray([3, 8, 9, 5, 1, 0, 0])
    actual = feature_converters.make_autoregressive_inputs(
        x, sequence_ids=None, bos_id=1
    )
    expected = [1, 3, 8, 9, 5, 1, 0]
    np.testing.assert_array_equal(actual, expected)
    self.assertEqual(actual.dtype, np.int64)

  def test_autoregressive_inputs_unpacked_2d(self):
    x = np.asarray([[3, 8, 1, 0, 0], [9, 5, 2, 0, 6]])
    actual = feature_converters.make_autoregressive_inputs(
        x, sequence_ids=None, bos_id=0
    )
    expected = [[0, 0, 0, 0, 0], [3, 8, 1, 0, 0]]
    np.testing.assert_array_equal(actual, expected)
    self.assertEqual(actual.dtype, np.int64)

  def test_autoregressive_inputs_packed(self):
    x = np.asarray([3, 8, 1, 9, 1, 5, 4, 1, 0, 0])
    sequence_ids = np.asarray([1, 1, 1, 2, 2, 3, 3, 3, 0, 0])
    actual = feature_converters.make_autoregressive_inputs(
        x,
        sequence_ids=sequence_ids,
        bos_id=0,
    )
    expected = [0, 3, 8, 0, 9, 0, 5, 4, 0, 0]
    np.testing.assert_array_equal(actual, expected)
    self.assertEqual(actual.dtype, np.int64)

  def test_autoregressive_inputs_packed_with_bos_id(self):
    x = np.asarray([3, 8, 1, 9, 1, 5, 4, 1, 0, 0])
    sequence_ids = np.asarray([1, 1, 1, 2, 2, 3, 3, 3, 0, 0])
    actual = feature_converters.make_autoregressive_inputs(
        x,
        sequence_ids=sequence_ids,
        bos_id=1,
    )
    expected = [1, 3, 8, 1, 9, 1, 5, 4, 1, 0]
    np.testing.assert_array_equal(actual, expected)
    self.assertEqual(actual.dtype, np.int64)

  def test_autoregressive_inputs_packed_2d(self):
    x = np.asarray([[3, 8, 1, 0, 0], [9, 5, 2, 0, 6]])
    sequence_ids = np.asarray([1, 2])
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        (
            "Only 1-D sequences are supported with packing. "
            "Got a packed 2-D sequence."
        ),
    ):
      feature_converters.make_autoregressive_inputs(
          x, sequence_ids=sequence_ids, bos_id=0
      )

  def test_autoregressive_inputs_packed_non_eos(self):
    # In the correct input format, x[4] should have been 1 (EOS).
    x = np.asarray([3, 8, 1, 9, 6, 5, 4, 1, 0, 0])
    # sequence_id is correctly formatted.
    sequence_ids = np.asarray([1, 1, 1, 2, 2, 3, 3, 3, 0, 0])
    actual = feature_converters.make_autoregressive_inputs(
        x,
        sequence_ids=sequence_ids,
        bos_id=0,
    )
    # The incorrect x[4] should not affect the output as long as the sequence_id
    # is correct.
    expected = [0, 3, 8, 0, 9, 0, 5, 4, 0, 0]
    np.testing.assert_array_equal(actual, expected)
    self.assertEqual(actual.dtype, np.int64)

  def test_autoregressive_inputs_different_dtypes(self):
    x = np.asarray([3, 8, 1, 9.9, 1, 5, 4, 1, 0, 0], dtype=np.float32)
    sequence_ids = np.asarray([1, 1, 1, 2, 2, 3, 3, 3, 0, 0])
    actual = feature_converters.make_autoregressive_inputs(
        x,
        sequence_ids=sequence_ids,
        bos_id=0,
    )
    # The incorrect x[4] should not affect the output as long as the sequence_id
    # is correct.
    expected = [0, 3, 8, 0, 9.9, 0, 5, 4, 0, 0]
    np.testing.assert_array_almost_equal(actual, expected)
    self.assertEqual(actual.dtype, np.float32)


class EncDecFeatureConverterTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._expected_unpacked_keys = [
        "encoder_input_tokens",
        "decoder_target_tokens",
        "decoder_input_tokens",
        "decoder_loss_weights",
    ]
    self._expected_packed_keys = [
        "encoder_input_tokens",
        "encoder_segment_ids",
        "encoder_positions",
        "decoder_target_tokens",
        "decoder_input_tokens",
        "decoder_loss_weights",
        "decoder_segment_ids",
        "decoder_positions",
    ]

  def _apply_preprocessors(
      self,
      ds: lazy_dataset.LazyMapDataset,
      preprocessors: preprocessors_lib.AirIOPreprocessor,
      runtime_args: preprocessors_lib.AirIOInjectedRuntimeArgs,
  ):
    for preprocessor in preprocessors:
      prep = preprocessors_lib.LazyDatasetTransform(preprocessor)
      ds = prep(ds, runtime_args=runtime_args)
      runtime_args = prep.get_updated_runtime_args(runtime_args)

    return ds, runtime_args

  def test_encoder_decoder_unpacked(self):
    x = [{"inputs": [9, 4, 3, 8, 1], "targets": [3, 9, 4, 1]}]
    ds = lazy_dataset.SourceLazyMapDataset(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 7, "targets": 5}, split="unused"
    )

    converter = feature_converters.T5XEncDecFeatureConverter(
        pack=False,
        use_multi_bin_packing=False,
        passthrough_feature_keys=[],
        pad_id=0,
        bos_id=0,
    )
    ds, _ = self._apply_preprocessors(
        ds, converter.get_preprocessors(), runtime_args
    )
    ds = list(ds)

    expected = {
        "encoder_input_tokens": [9, 4, 3, 8, 1, 0, 0],
        "decoder_target_tokens": [3, 9, 4, 1, 0],
        # mtf.transformer.autoregressive_inputs does not zero out the last eos
        # when the data is not packed. This test mimics the behavior.
        "decoder_input_tokens": [0, 3, 9, 4, 1],
        "decoder_loss_weights": [1, 1, 1, 1, 0],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(
        sorted(actual.keys()), sorted(self._expected_unpacked_keys)
    )
    for k in self._expected_unpacked_keys:
      np.testing.assert_array_equal(actual[k], expected[k])

  def test_encoder_decoder_unpacked_passthrough(self):
    x = [{
        "inputs": [9, 4, 3, 8, 1],
        "targets": [3, 9, 4, 1],
        "passthrough": [4, 2, 3],
    }]
    ds = lazy_dataset.SourceLazyMapDataset(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 7, "targets": 5, "passthrough": 3},
        split="unused",
    )

    converter = feature_converters.T5XEncDecFeatureConverter(
        pack=False,
        use_multi_bin_packing=False,
        passthrough_feature_keys=["passthrough"],
        pad_id=0,
        bos_id=0,
    )
    ds, runtime_args = self._apply_preprocessors(
        ds, converter.get_preprocessors(), runtime_args
    )
    ds = list(ds)
    expected_runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={
            "encoder_input_tokens": 7,
            "decoder_target_tokens": 5,
            "decoder_input_tokens": 5,
            "decoder_loss_weights": 5,
            "passthrough": 3,
        },
        split="unused",
    )
    self.assertEqual(runtime_args, expected_runtime_args)
    expected = {
        "encoder_input_tokens": [9, 4, 3, 8, 1, 0, 0],
        "decoder_target_tokens": [3, 9, 4, 1, 0],
        # mtf.transformer.autoregressive_inputs does not zero out the last eos
        # when the data is not packed. This test mimics the behavior.
        "decoder_input_tokens": [0, 3, 9, 4, 1],
        "decoder_loss_weights": [1, 1, 1, 1, 0],
        "passthrough": [4, 2, 3],
    }
    expected_keys = self._expected_unpacked_keys + ["passthrough"]
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(sorted(actual.keys()), sorted(expected_keys))
    for k in expected_keys:
      np.testing.assert_array_equal(actual[k], expected[k])

  def test_encoder_decoder_targets_max_length(self):
    x = [{"inputs": [9, 4, 3, 8, 1], "targets": [3, 9, 4, 5, 1]}]
    ds = lazy_dataset.SourceLazyMapDataset(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 5, "targets": 5},
        split="unused",
    )

    converter = feature_converters.T5XEncDecFeatureConverter(
        pack=False,
        use_multi_bin_packing=False,
        passthrough_feature_keys=[],
        pad_id=0,
        bos_id=0,
    )
    ds, _ = self._apply_preprocessors(
        ds, converter.get_preprocessors(), runtime_args
    )
    ds = list(ds)

    expected = {
        "encoder_input_tokens": [9, 4, 3, 8, 1],
        "decoder_target_tokens": [3, 9, 4, 5, 1],
        "decoder_input_tokens": [0, 3, 9, 4, 5],
        "decoder_loss_weights": [1, 1, 1, 1, 1],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(
        sorted(actual.keys()), sorted(self._expected_unpacked_keys)
    )
    for k in self._expected_unpacked_keys:
      np.testing.assert_array_equal(actual[k], expected[k])

  def test_encoder_decoder_extra_long_inputs_trimmed(self):
    # The conventional seqio feature converter would have failed because
    # `apply_length_check` is True by default. Since the trim preprocessor is
    # explicitly part of the feature converter, it's ok to expect untrimmed
    # sequences. This behavior can be modified if needed.
    x = [{"inputs": [9, 4, 3, 8, 4, 5, 1], "targets": [3, 9, 4, 7, 8, 1]}]
    ds = lazy_dataset.SourceLazyMapDataset(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 5, "targets": 8},
        split="unused",
    )
    converter = feature_converters.T5XEncDecFeatureConverter(
        pack=False,
        use_multi_bin_packing=False,
        passthrough_feature_keys=[],
        pad_id=0,
        bos_id=0,
    )
    ds, _ = self._apply_preprocessors(
        ds, converter.get_preprocessors(), runtime_args
    )
    ds = list(ds)

    expected = {
        "encoder_input_tokens": [9, 4, 3, 8, 4],
        "decoder_target_tokens": [3, 9, 4, 7, 8, 1, 0, 0],
        "decoder_input_tokens": [0, 3, 9, 4, 7, 8, 1, 0],
        "decoder_loss_weights": [1, 1, 1, 1, 1, 1, 0, 0],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(
        sorted(actual.keys()), sorted(self._expected_unpacked_keys)
    )
    for k in self._expected_unpacked_keys:
      np.testing.assert_array_equal(actual[k], expected[k])

  def test_encoder_decoder_packed(self):
    x = [
        {"inputs": [7, 8, 5, 1], "targets": [3, 9, 1]},
        {"inputs": [8, 4, 9, 3, 1], "targets": [4, 1]},
    ]
    ds = lazy_dataset.SourceLazyMapDataset(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 10, "targets": 7},
        split="unused",
    )

    converter = feature_converters.T5XEncDecFeatureConverter(
        pack=True,
        use_multi_bin_packing=False,
        passthrough_feature_keys=[],
        pad_id=0,
        bos_id=0,
    )
    ds, runtime_args = self._apply_preprocessors(
        ds, converter.get_preprocessors(), runtime_args
    )
    ds = list(ds)
    expected_runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={
            "encoder_input_tokens": 10,
            "decoder_target_tokens": 7,
            "decoder_input_tokens": 7,
            "decoder_loss_weights": 7,
            "encoder_segment_ids": 10,
            "decoder_segment_ids": 7,
            "encoder_positions": 10,
            "decoder_positions": 7,
        },
        split="unused",
    )
    self.assertEqual(runtime_args, expected_runtime_args)
    expected = {
        "encoder_input_tokens": [7, 8, 5, 1, 8, 4, 9, 3, 1, 0],
        "encoder_segment_ids": [1, 1, 1, 1, 2, 2, 2, 2, 2, 0],
        "encoder_positions": [0, 1, 2, 3, 0, 1, 2, 3, 4, 0],
        "decoder_target_tokens": [3, 9, 1, 4, 1, 0, 0],
        "decoder_input_tokens": [0, 3, 9, 0, 4, 0, 0],
        "decoder_loss_weights": [1, 1, 1, 1, 1, 0, 0],
        "decoder_segment_ids": [1, 1, 1, 2, 2, 0, 0],
        "decoder_positions": [0, 1, 2, 0, 1, 0, 0],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(
        sorted(actual.keys()), sorted(self._expected_packed_keys)
    )
    for k in self._expected_packed_keys:
      np.testing.assert_array_equal(actual[k], expected[k])

  def test_encoder_decoder_packed_long_sequences(self):
    x = [
        {"inputs": [7, 8, 5, 6, 9, 4, 1], "targets": [3, 9, 1]},
        {"inputs": [8, 4, 9, 3, 5, 1], "targets": [4, 1]},
    ]
    ds = lazy_dataset.SourceLazyMapDataset(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 7, "targets": 3},
        split="unused",
    )

    converter = feature_converters.T5XEncDecFeatureConverter(
        pack=True,
        use_multi_bin_packing=False,
        passthrough_feature_keys=[],
        pad_id=0,
        bos_id=0,
    )
    ds, _ = self._apply_preprocessors(
        ds, converter.get_preprocessors(), runtime_args
    )
    ds = list(ds)

    # Corner case: packing is true but task_feature_lengths are too long for
    # packing to happen. We should still get the *_segment_id, *_position
    # fields.
    expected_dataset = [
        {
            "encoder_input_tokens": [7, 8, 5, 6, 9, 4, 1],
            "encoder_segment_ids": [1, 1, 1, 1, 1, 1, 1],
            "encoder_positions": [0, 1, 2, 3, 4, 5, 6],
            "decoder_target_tokens": [3, 9, 1],
            "decoder_input_tokens": [0, 3, 9],
            "decoder_loss_weights": [1, 1, 1],
            "decoder_segment_ids": [1, 1, 1],
            "decoder_positions": [0, 1, 2],
        },
        {
            "encoder_input_tokens": [8, 4, 9, 3, 5, 1, 0],
            "encoder_segment_ids": [1, 1, 1, 1, 1, 1, 0],
            "encoder_positions": [0, 1, 2, 3, 4, 5, 0],
            "decoder_target_tokens": [4, 1, 0],
            "decoder_input_tokens": [0, 4, 0],
            "decoder_loss_weights": [1, 1, 0],
            "decoder_segment_ids": [1, 1, 0],
            "decoder_positions": [0, 1, 0],
        },
    ]
    self.assertLen(ds, 2)
    for actual, expected in zip(ds, expected_dataset):
      self.assertSequenceEqual(
          sorted(actual.keys()), sorted(self._expected_packed_keys)
      )
      for k in self._expected_packed_keys:
        np.testing.assert_array_equal(actual[k], expected[k])

  def test_encoder_decoder_extra_field_removed(self):
    x = [
        {
            "inputs": [7, 8, 5, 1],
            "targets": [3, 9, 1],
            "targets_pretokenized": "abc",
        },
        {
            "inputs": [8, 4, 9, 3, 1],
            "targets": [4, 1],
            "targets_pretokenized": "def",
        },
    ]
    ds = lazy_dataset.SourceLazyMapDataset(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 10, "targets": 7},
        split="unused",
    )

    converter = feature_converters.T5XEncDecFeatureConverter(
        pack=True,
        use_multi_bin_packing=False,
        passthrough_feature_keys=[],
        pad_id=0,
        bos_id=0,
    )
    ds, _ = self._apply_preprocessors(
        ds, converter.get_preprocessors(), runtime_args
    )
    ds = list(ds)

    expected = {
        "encoder_input_tokens": [7, 8, 5, 1, 8, 4, 9, 3, 1, 0],
        "decoder_target_tokens": [3, 9, 1, 4, 1, 0, 0],
        "decoder_input_tokens": [0, 3, 9, 0, 4, 0, 0],
        "decoder_loss_weights": [1, 1, 1, 1, 1, 0, 0],
        "encoder_segment_ids": [1, 1, 1, 1, 2, 2, 2, 2, 2, 0],
        "decoder_segment_ids": [1, 1, 1, 2, 2, 0, 0],
        "encoder_positions": [0, 1, 2, 3, 0, 1, 2, 3, 4, 0],
        "decoder_positions": [0, 1, 2, 0, 1, 0, 0],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(
        sorted(actual.keys()), sorted(self._expected_packed_keys)
    )
    for k in self._expected_packed_keys:
      np.testing.assert_array_equal(actual[k], expected[k])

  def test_encoder_decoder_packed_with_bos_id(self):
    x = [
        {"inputs": [7, 8, 5, 1], "targets": [3, 9, 1]},
        {"inputs": [8, 4, 9, 3, 1], "targets": [4, 1]},
    ]
    ds = lazy_dataset.SourceLazyMapDataset(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths={"inputs": 10, "targets": 7},
        split="unused",
    )

    converter = feature_converters.T5XEncDecFeatureConverter(
        pack=True,
        use_multi_bin_packing=False,
        passthrough_feature_keys=[],
        pad_id=0,
        bos_id=10,
    )
    ds, _ = self._apply_preprocessors(
        ds, converter.get_preprocessors(), runtime_args
    )
    ds = list(ds)

    expected = {
        "encoder_input_tokens": [7, 8, 5, 1, 8, 4, 9, 3, 1, 0],
        "encoder_segment_ids": [1, 1, 1, 1, 2, 2, 2, 2, 2, 0],
        "encoder_positions": [0, 1, 2, 3, 0, 1, 2, 3, 4, 0],
        "decoder_target_tokens": [3, 9, 1, 4, 1, 0, 0],
        "decoder_input_tokens": [10, 3, 9, 10, 4, 10, 0],
        "decoder_loss_weights": [1, 1, 1, 1, 1, 0, 0],
        "decoder_segment_ids": [1, 1, 1, 2, 2, 0, 0],
        "decoder_positions": [0, 1, 2, 0, 1, 0, 0],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(
        sorted(actual.keys()), sorted(self._expected_packed_keys)
    )
    for k in self._expected_packed_keys:
      np.testing.assert_array_equal(actual[k], expected[k])


if __name__ == "__main__":
  absltest.main()
