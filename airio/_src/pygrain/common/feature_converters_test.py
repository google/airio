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

"""Tests for feature_converters."""

from absl.testing import absltest
from airio._src.core import preprocessors as core_preprocessors_lib
from airio._src.core import test_utils
from airio._src.pygrain import preprocessors as preprocessors_lib
from airio._src.pygrain.common import feature_converters
import grain.python as grain
import numpy as np


class AutoregressiveInputsTest(absltest.TestCase):

  def test_autoregressive_inputs_unpacked(self):
    x = np.asarray([3, 8, 9, 5, 1, 0, 0])
    actual = feature_converters._make_autoregressive_inputs(
        x, sequence_ids=None, bos_id=0
    )
    expected = [0, 3, 8, 9, 5, 1, 0]
    np.testing.assert_array_equal(actual, expected)
    self.assertEqual(actual.dtype, np.int64)

  def test_autoregressive_inputs_unpacked_with_bos_id(self):
    x = np.asarray([3, 8, 9, 5, 1, 0, 0])
    actual = feature_converters._make_autoregressive_inputs(
        x, sequence_ids=None, bos_id=1
    )
    expected = [1, 3, 8, 9, 5, 1, 0]
    np.testing.assert_array_equal(actual, expected)
    self.assertEqual(actual.dtype, np.int64)

  def test_autoregressive_inputs_unpacked_2d(self):
    x = np.asarray([[3, 8, 1, 0, 0], [9, 5, 2, 0, 6]])
    actual = feature_converters._make_autoregressive_inputs(
        x, sequence_ids=None, bos_id=0
    )
    expected = [[0, 0, 0, 0, 0], [3, 8, 1, 0, 0]]
    np.testing.assert_array_equal(actual, expected)
    self.assertEqual(actual.dtype, np.int64)

  def test_autoregressive_inputs_packed(self):
    x = np.asarray([3, 8, 1, 9, 1, 5, 4, 1, 0, 0])
    sequence_ids = np.asarray([1, 1, 1, 2, 2, 3, 3, 3, 0, 0])
    actual = feature_converters._make_autoregressive_inputs(
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
    actual = feature_converters._make_autoregressive_inputs(
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
      feature_converters._make_autoregressive_inputs(
          x, sequence_ids=sequence_ids, bos_id=0
      )

  def test_autoregressive_inputs_packed_non_eos(self):
    # In the correct input format, x[4] should have been 1 (EOS).
    x = np.asarray([3, 8, 1, 9, 6, 5, 4, 1, 0, 0])
    # sequence_id is correctly formatted.
    sequence_ids = np.asarray([1, 1, 1, 2, 2, 3, 3, 3, 0, 0])
    actual = feature_converters._make_autoregressive_inputs(
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
    actual = feature_converters._make_autoregressive_inputs(
        x,
        sequence_ids=sequence_ids,
        bos_id=0,
    )
    # The incorrect x[4] should not affect the output as long as the sequence_id
    # is correct.
    expected = [0, 3, 8, 0, 9.9, 0, 5, 4, 0, 0]
    np.testing.assert_array_almost_equal(actual, expected)
    self.assertEqual(actual.dtype, np.float32)


def _apply_preprocessors(
    ds: grain.MapDataset,
    preprocessors: preprocessors_lib.PyGrainAirIOPreprocessor,
    runtime_args: core_preprocessors_lib.AirIOInjectedRuntimeArgs,
):
  for preprocessor in preprocessors:
    prep = preprocessors_lib.LazyDatasetTransform(preprocessor)
    ds = prep(ds, runtime_args=runtime_args)
    runtime_args = prep.get_updated_runtime_args(runtime_args)

  return ds, runtime_args


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

  def test_encoder_decoder_unpacked(self):
    x = [{"inputs": [9, 4, 3, 8, 1], "targets": [3, 9, 4, 1]}]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 7, "targets": 5},
    )

    preps = feature_converters.get_t5x_enc_dec_feature_converter_preprocessors(
        pack=False,
        use_multi_bin_packing=False,
        passthrough_feature_keys=[],
        pad_id=0,
        bos_id=0,
    )
    ds, _ = _apply_preprocessors(ds, preps, runtime_args)
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
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 7, "targets": 5, "passthrough": 3},
    )

    preps = feature_converters.get_t5x_enc_dec_feature_converter_preprocessors(
        pack=False,
        use_multi_bin_packing=False,
        passthrough_feature_keys=["passthrough"],
        pad_id=0,
        bos_id=0,
    )
    ds, runtime_args = _apply_preprocessors(ds, preps, runtime_args)
    ds = list(ds)
    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "encoder_input_tokens": 7,
            "decoder_target_tokens": 5,
            "decoder_input_tokens": 5,
            "decoder_loss_weights": 5,
            "passthrough": 3,
        },
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
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 5, "targets": 5},
    )

    preps = feature_converters.get_t5x_enc_dec_feature_converter_preprocessors(
        pack=False,
        use_multi_bin_packing=False,
        passthrough_feature_keys=[],
        pad_id=0,
        bos_id=0,
    )
    ds, _ = _apply_preprocessors(ds, preps, runtime_args)
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
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 5, "targets": 8},
    )
    preps = feature_converters.get_t5x_enc_dec_feature_converter_preprocessors(
        pack=False,
        use_multi_bin_packing=False,
        passthrough_feature_keys=[],
        pad_id=0,
        bos_id=0,
    )
    ds, _ = _apply_preprocessors(ds, preps, runtime_args)
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
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 10, "targets": 7},
    )

    preps = feature_converters.get_t5x_enc_dec_feature_converter_preprocessors(
        pack=True,
        use_multi_bin_packing=False,
        passthrough_feature_keys=[],
        pad_id=0,
        bos_id=0,
    )
    ds, runtime_args = _apply_preprocessors(ds, preps, runtime_args)
    ds = list(ds)
    expected_runtime_args = runtime_args.replace(
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
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 7, "targets": 3},
    )

    preps = feature_converters.get_t5x_enc_dec_feature_converter_preprocessors(
        pack=True,
        use_multi_bin_packing=False,
        passthrough_feature_keys=[],
        pad_id=0,
        bos_id=0,
    )
    ds, _ = _apply_preprocessors(ds, preps, runtime_args)
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
            "inputs_pretokenized": "def ghi",
        },
        {
            "inputs": [8, 4, 9, 3, 1],
            "targets": [4, 1],
            "targets_pretokenized": "def",
            "inputs_pretokenized": "ghi jkl",
        },
    ]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 10, "targets": 7},
    )

    preps = feature_converters.get_t5x_enc_dec_feature_converter_preprocessors(
        pack=True,
        use_multi_bin_packing=False,
        passthrough_feature_keys=[],
        pad_id=0,
        bos_id=0,
    )
    ds, _ = _apply_preprocessors(ds, preps, runtime_args)
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
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 10, "targets": 7},
    )

    preps = feature_converters.get_t5x_enc_dec_feature_converter_preprocessors(
        pack=True,
        use_multi_bin_packing=False,
        passthrough_feature_keys=[],
        pad_id=0,
        bos_id=10,
    )
    ds, _ = _apply_preprocessors(ds, preps, runtime_args)
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


class LMFeatureConverter(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._expected_unpacked_keys = [
        "decoder_target_tokens",
        "decoder_input_tokens",
        "decoder_loss_weights",
    ]
    self._expected_packed_keys = [
        "decoder_target_tokens",
        "decoder_input_tokens",
        "decoder_loss_weights",
        "decoder_segment_ids",
        "decoder_positions",
    ]

  def test_lm_unpacked(self):
    x = [{"targets": [3, 9, 1]}]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"targets": 5}
    )

    preps = feature_converters.get_t5x_lm_feature_converter_preprocessors(
        pack=False, use_multi_bin_packing=False, pad_id=0, bos_id=0
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)
    ds = list(ds)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "decoder_target_tokens": 5,
            "decoder_input_tokens": 5,
            "decoder_loss_weights": 5,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    expected = {
        "decoder_target_tokens": [3, 9, 1, 0, 0],
        "decoder_input_tokens": [0, 3, 9, 1, 0],
        "decoder_loss_weights": [1, 1, 1, 0, 0],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(
        sorted(actual.keys()), sorted(self._expected_unpacked_keys)
    )
    for k in self._expected_unpacked_keys:
      np.testing.assert_array_equal(actual[k], expected[k])

  def test_lm_only_packed(self):
    x = [{"targets": [3, 9, 1]}, {"targets": [4, 1]}]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"targets": 6}
    )

    preps = feature_converters.get_t5x_lm_feature_converter_preprocessors(
        pack=True, use_multi_bin_packing=False, pad_id=0, bos_id=0
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)
    ds = list(ds)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "decoder_target_tokens": 6,
            "decoder_input_tokens": 6,
            "decoder_loss_weights": 6,
            "decoder_segment_ids": 6,
            "decoder_positions": 6,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    expected = {
        "decoder_target_tokens": [3, 9, 1, 4, 1, 0],
        "decoder_input_tokens": [0, 3, 9, 0, 4, 0],
        "decoder_loss_weights": [1, 1, 1, 1, 1, 0],
        "decoder_positions": [0, 1, 2, 0, 1, 0],
        "decoder_segment_ids": [1, 1, 1, 2, 2, 0],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(
        sorted(actual.keys()), sorted(self._expected_packed_keys)
    )
    for k in self._expected_packed_keys:
      np.testing.assert_array_equal(actual[k], expected[k])

  def test_lm_pack_long_sequences(self):
    x = [{"targets": [3, 9, 4, 5, 1]}, {"targets": [4, 3, 2, 1]}]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"targets": 5}
    )

    preps = feature_converters.get_t5x_lm_feature_converter_preprocessors(
        pack=True, use_multi_bin_packing=False, bos_id=0, pad_id=0
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)
    ds = list(ds)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "decoder_target_tokens": 5,
            "decoder_input_tokens": 5,
            "decoder_loss_weights": 5,
            "decoder_segment_ids": 5,
            "decoder_positions": 5,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    expected_ds = [
        {
            "decoder_target_tokens": [3, 9, 4, 5, 1],
            "decoder_input_tokens": [0, 3, 9, 4, 5],
            "decoder_loss_weights": [1, 1, 1, 1, 1],
            "decoder_positions": [0, 1, 2, 3, 4],
            "decoder_segment_ids": [1, 1, 1, 1, 1],
        },
        {
            "decoder_target_tokens": [4, 3, 2, 1, 0],
            "decoder_input_tokens": [0, 4, 3, 2, 0],
            "decoder_loss_weights": [1, 1, 1, 1, 0],
            "decoder_positions": [0, 1, 2, 3, 0],
            "decoder_segment_ids": [1, 1, 1, 1, 0],
        },
    ]
    self.assertLen(ds, 2)
    for actual, expected in zip(ds, expected_ds):
      self.assertSequenceEqual(
          sorted(actual.keys()), sorted(self._expected_packed_keys)
      )
      for k in self._expected_packed_keys:
        np.testing.assert_array_equal(actual[k], expected[k])

  def test_lm_extra_field_removed(self):
    x = [
        {"targets": [3, 9, 1], "plaintext": "abc"},
        {"targets": [4, 1], "plaintext": "abc"},
    ]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"targets": 6}
    )

    preps = feature_converters.get_t5x_lm_feature_converter_preprocessors(
        pack=True, use_multi_bin_packing=False, bos_id=0, pad_id=0
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)
    ds = list(ds)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "decoder_target_tokens": 6,
            "decoder_input_tokens": 6,
            "decoder_loss_weights": 6,
            "decoder_segment_ids": 6,
            "decoder_positions": 6,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    expected = {
        "decoder_target_tokens": [3, 9, 1, 4, 1, 0],
        "decoder_input_tokens": [0, 3, 9, 0, 4, 0],
        "decoder_loss_weights": [1, 1, 1, 1, 1, 0],
        "decoder_segment_ids": [1, 1, 1, 2, 2, 0],
        "decoder_positions": [0, 1, 2, 0, 1, 0],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(
        sorted(actual.keys()), sorted(self._expected_packed_keys)
    )
    for k in self._expected_packed_keys:
      np.testing.assert_array_equal(actual[k], expected[k])

  def test_lm_only_packed_without_default_bos(self):
    x = [{"targets": [3, 9, 1]}, {"targets": [4, 1]}]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"targets": 6}
    )

    preps = feature_converters.get_t5x_lm_feature_converter_preprocessors(
        pack=True, use_multi_bin_packing=False, bos_id=10, pad_id=0
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)
    ds = list(ds)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "decoder_target_tokens": 6,
            "decoder_input_tokens": 6,
            "decoder_loss_weights": 6,
            "decoder_segment_ids": 6,
            "decoder_positions": 6,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    expected = {
        "decoder_target_tokens": [3, 9, 1, 4, 1, 0],
        "decoder_input_tokens": [10, 3, 9, 10, 4, 10],
        "decoder_loss_weights": [1, 1, 1, 1, 1, 0],
        "decoder_positions": [0, 1, 2, 0, 1, 0],
        "decoder_segment_ids": [1, 1, 1, 2, 2, 0],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(
        sorted(actual.keys()), sorted(self._expected_packed_keys)
    )
    for k in self._expected_packed_keys:
      np.testing.assert_array_equal(actual[k], expected[k])


class PrefixLMFeatureConverter(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._expected_unpacked_keys = [
        "decoder_target_tokens",
        "decoder_input_tokens",
        "decoder_loss_weights",
        "decoder_causal_attention",
    ]
    self._expected_packed_keys = [
        "decoder_target_tokens",
        "decoder_input_tokens",
        "decoder_loss_weights",
        "decoder_causal_attention",
        "decoder_segment_ids",
        "decoder_positions",
    ]

  def test_prefix_lm_unpacked(self):
    x = [{"inputs": [9, 4, 6, 1], "targets": [3, 9, 1]}]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})

    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 5, "targets": 4},
    )
    preps = (
        feature_converters.get_t5x_prefix_lm_feature_converter_preprocessors(
            pack=False,
            use_multi_bin_packing=False,
            pad_id=0,
            bos_id=0,
            loss_on_targets_only=True,
            passthrough_feature_keys=[],
        )
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "decoder_target_tokens": 9,
            "decoder_input_tokens": 9,
            "decoder_loss_weights": 9,
            "decoder_causal_attention": 9,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    expected = {
        "decoder_target_tokens": [9, 4, 6, 1, 3, 9, 1, 0, 0],
        # The last EOS token is kept if unpacked.
        "decoder_input_tokens": [0, 9, 4, 6, 1, 3, 9, 1, 0],
        "decoder_loss_weights": [0, 0, 0, 0, 1, 1, 1, 0, 0],
        "decoder_causal_attention": [1, 1, 1, 1, 1, 0, 0, 0, 0],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(
        sorted(actual.keys()), sorted(self._expected_unpacked_keys)
    )
    for k in self._expected_unpacked_keys:
      np.testing.assert_array_equal(actual[k], expected[k])

  def test_prefix_lm_unpacked_passthrough(self):
    x = [{
        "inputs": [9, 4, 6, 1],
        "targets": [3, 9, 1],
        "passthrough": [6, 5, 4, 3, 2, 1, 0],
    }]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})

    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 5, "targets": 4, "passthrough": 7},
    )
    preps = (
        feature_converters.get_t5x_prefix_lm_feature_converter_preprocessors(
            pack=False,
            use_multi_bin_packing=False,
            pad_id=0,
            bos_id=0,
            loss_on_targets_only=True,
            passthrough_feature_keys=["passthrough"],
        )
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "decoder_target_tokens": 9,
            "decoder_input_tokens": 9,
            "decoder_loss_weights": 9,
            "decoder_causal_attention": 9,
            "passthrough": 7,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    expected = {
        "decoder_target_tokens": [9, 4, 6, 1, 3, 9, 1, 0, 0],
        # The last EOS token is kept if unpacked.
        "decoder_input_tokens": [0, 9, 4, 6, 1, 3, 9, 1, 0],
        "decoder_loss_weights": [0, 0, 0, 0, 1, 1, 1, 0, 0],
        "decoder_causal_attention": [1, 1, 1, 1, 1, 0, 0, 0, 0],
        "passthrough": [6, 5, 4, 3, 2, 1, 0],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    expected_keys = self._expected_unpacked_keys + ["passthrough"]
    self.assertSequenceEqual(sorted(actual.keys()), sorted(expected_keys))
    for k in expected_keys:
      np.testing.assert_array_equal(actual[k], expected[k])

  def test_prefix_lm_unpacked_trivial_targets(self):
    x = [{"inputs": [9, 4, 6, 1], "targets": []}]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})

    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 5, "targets": 4},
    )
    preps = (
        feature_converters.get_t5x_prefix_lm_feature_converter_preprocessors(
            pack=False,
            use_multi_bin_packing=False,
            pad_id=0,
            bos_id=0,
            loss_on_targets_only=True,
            passthrough_feature_keys=[],
        )
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "decoder_target_tokens": 9,
            "decoder_input_tokens": 9,
            "decoder_loss_weights": 9,
            "decoder_causal_attention": 9,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    expected = {
        "decoder_target_tokens": [9, 4, 6, 1, 0, 0, 0, 0, 0],
        # The last EOS token is kept if unpacked.
        "decoder_input_tokens": [0, 9, 4, 6, 1, 0, 0, 0, 0],
        "decoder_loss_weights": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "decoder_causal_attention": [1, 1, 1, 1, 1, 0, 0, 0, 0],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(
        sorted(actual.keys()), sorted(self._expected_unpacked_keys)
    )
    for k in self._expected_unpacked_keys:
      np.testing.assert_array_equal(actual[k], expected[k])

  def test_prefix_lm_long_inputs_feature_length(self):
    x = [{"inputs": [9, 4, 6, 1], "targets": [3, 9, 1]}]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})

    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 10, "targets": 4},
    )
    preps = (
        feature_converters.get_t5x_prefix_lm_feature_converter_preprocessors(
            pack=False,
            use_multi_bin_packing=False,
            pad_id=0,
            bos_id=0,
            loss_on_targets_only=True,
            passthrough_feature_keys=[],
        )
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "decoder_target_tokens": 14,
            "decoder_input_tokens": 14,
            "decoder_loss_weights": 14,
            "decoder_causal_attention": 14,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    expected = {
        "decoder_target_tokens": [9, 4, 6, 1, 3, 9, 1, 0, 0, 0, 0, 0, 0, 0],
        # The last EOS token is kept if unpacked.
        "decoder_input_tokens": [0, 9, 4, 6, 1, 3, 9, 1, 0, 0, 0, 0, 0, 0],
        "decoder_loss_weights": [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        "decoder_causal_attention": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(
        sorted(actual.keys()), sorted(self._expected_unpacked_keys)
    )
    for k in self._expected_unpacked_keys:
      np.testing.assert_array_equal(actual[k], expected[k])

  def test_prefix_lm_packed(self):
    x = [
        {"inputs": [7, 8, 5, 1], "targets": [3, 9, 1]},
        {"inputs": [8, 4, 9, 3, 1], "targets": [4, 1]},
    ]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()

    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 8, "targets": 7},
    )
    preps = (
        feature_converters.get_t5x_prefix_lm_feature_converter_preprocessors(
            pack=True,
            use_multi_bin_packing=False,
            pad_id=0,
            bos_id=0,
            loss_on_targets_only=True,
            passthrough_feature_keys=[],
        )
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)
    ds = list(ds)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "decoder_target_tokens": 15,
            "decoder_input_tokens": 15,
            "decoder_loss_weights": 15,
            "decoder_segment_ids": 15,
            "decoder_positions": 15,
            "decoder_causal_attention": 15,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    expected = {
        "decoder_target_tokens": [7, 8, 5, 1, 3, 9, 1, 8, 4, 9, 3, 1, 4, 1, 0],
        "decoder_input_tokens": [0, 7, 8, 5, 1, 3, 9, 0, 8, 4, 9, 3, 1, 4, 0],
        "decoder_loss_weights": [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        "decoder_positions": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0],
        "decoder_segment_ids": [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0],
        "decoder_causal_attention": [
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
        ],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(
        sorted(actual.keys()), sorted(self._expected_packed_keys)
    )
    for k in self._expected_packed_keys:
      np.testing.assert_array_equal(actual[k], expected[k])

  def test_prefix_lm_unpacked_loss_on_inputs_and_targets(self):
    x = [{"inputs": [9, 4, 6, 1], "targets": [3, 9, 1]}]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})

    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 5, "targets": 4},
    )
    preps = (
        feature_converters.get_t5x_prefix_lm_feature_converter_preprocessors(
            pack=False,
            use_multi_bin_packing=False,
            pad_id=0,
            bos_id=0,
            loss_on_targets_only=False,
            passthrough_feature_keys=[],
        )
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "decoder_target_tokens": 9,
            "decoder_input_tokens": 9,
            "decoder_loss_weights": 9,
            "decoder_causal_attention": 9,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    expected = {
        "decoder_target_tokens": [9, 4, 6, 1, 3, 9, 1, 0, 0],
        "decoder_input_tokens": [0, 9, 4, 6, 1, 3, 9, 1, 0],
        # Loss weights on the inputs portion and padding should be zeroed out.
        "decoder_loss_weights": [1, 1, 1, 1, 1, 1, 1, 0, 0],
        "decoder_causal_attention": [1, 1, 1, 1, 1, 0, 0, 0, 0],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(
        sorted(actual.keys()), sorted(self._expected_unpacked_keys)
    )
    for k in self._expected_unpacked_keys:
      np.testing.assert_array_equal(actual[k], expected[k])

  def test_prefix_lm_packed_loss_on_inputs_and_targets(self):
    x = [
        {"inputs": [7, 8, 5, 1], "targets": [3, 9, 1]},
        {"inputs": [8, 4, 9, 3, 1], "targets": [4, 1]},
    ]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()

    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 8, "targets": 7},
    )
    preps = (
        feature_converters.get_t5x_prefix_lm_feature_converter_preprocessors(
            pack=True,
            use_multi_bin_packing=False,
            pad_id=0,
            bos_id=0,
            loss_on_targets_only=False,
            passthrough_feature_keys=[],
        )
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)
    ds = list(ds)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "decoder_target_tokens": 15,
            "decoder_input_tokens": 15,
            "decoder_loss_weights": 15,
            "decoder_segment_ids": 15,
            "decoder_positions": 15,
            "decoder_causal_attention": 15,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    expected = {
        "decoder_target_tokens": [7, 8, 5, 1, 3, 9, 1, 8, 4, 9, 3, 1, 4, 1, 0],
        "decoder_input_tokens": [0, 7, 8, 5, 1, 3, 9, 0, 8, 4, 9, 3, 1, 4, 0],
        "decoder_loss_weights": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        "decoder_positions": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0],
        "decoder_segment_ids": [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0],
        "decoder_causal_attention": [
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
        ],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(
        sorted(actual.keys()), sorted(self._expected_packed_keys)
    )
    for k in self._expected_packed_keys:
      np.testing.assert_array_equal(actual[k], expected[k])

  def test_prefix_lm_long_inputs(self):
    # TODO(b/319663351): This test should fail validation checks. Update after
    # adding checks.
    x = [
        {"inputs": [7, 8, 5, 6, 1], "targets": [3, 9, 7, 1]},
        {"inputs": [8, 4, 9, 3, 8, 1], "targets": [4, 1]},
    ]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()

    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 4, "targets": 3},
    )

    preps = (
        feature_converters.get_t5x_prefix_lm_feature_converter_preprocessors(
            pack=True,
            use_multi_bin_packing=False,
            pad_id=0,
            bos_id=0,
            loss_on_targets_only=True,
            passthrough_feature_keys=[],
        )
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "decoder_target_tokens": 7,
            "decoder_input_tokens": 7,
            "decoder_loss_weights": 7,
            "decoder_segment_ids": 7,
            "decoder_positions": 7,
            "decoder_causal_attention": 7,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    expected_ds = [
        {
            "decoder_target_tokens": [7, 8, 5, 6, 1, 3, 9],
            "decoder_input_tokens": [0, 7, 8, 5, 6, 1, 3],
            "decoder_loss_weights": [0, 0, 0, 0, 0, 1, 1],
            "decoder_segment_ids": [1, 1, 1, 1, 1, 1, 1],
            "decoder_positions": [0, 1, 2, 3, 4, 5, 6],
            "decoder_causal_attention": [1, 1, 1, 1, 1, 1, 0],
        },
        {
            "decoder_target_tokens": [8, 4, 9, 3, 8, 1, 4],
            "decoder_input_tokens": [0, 8, 4, 9, 3, 8, 1],
            "decoder_loss_weights": [0, 0, 0, 0, 0, 0, 1],
            "decoder_segment_ids": [1, 1, 1, 1, 1, 1, 1],
            "decoder_positions": [0, 1, 2, 3, 4, 5, 6],
            "decoder_causal_attention": [1, 1, 1, 1, 1, 1, 1],
        },
    ]
    for actual, expected in zip(iter(ds), expected_ds, strict=True):
      self.assertSequenceEqual(
          sorted(actual.keys()), sorted(self._expected_packed_keys)
      )
      for k in self._expected_packed_keys:
        np.testing.assert_array_equal(actual[k], expected[k])

  def test_prefix_lm_pack_long_sequences(self):
    x = [
        {"inputs": [7, 8, 5, 1], "targets": [3, 9, 1]},
        {"inputs": [8, 4, 1], "targets": [5, 1]},
    ]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()

    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 4, "targets": 3},
    )
    preps = (
        feature_converters.get_t5x_prefix_lm_feature_converter_preprocessors(
            pack=True,
            use_multi_bin_packing=False,
            pad_id=0,
            bos_id=0,
            loss_on_targets_only=True,
            passthrough_feature_keys=[],
        )
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "decoder_target_tokens": 7,
            "decoder_input_tokens": 7,
            "decoder_loss_weights": 7,
            "decoder_segment_ids": 7,
            "decoder_positions": 7,
            "decoder_causal_attention": 7,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    # The examples should not be packed because examples are not short enough.
    expected_ds = [
        {
            "decoder_target_tokens": [7, 8, 5, 1, 3, 9, 1],
            "decoder_input_tokens": [0, 7, 8, 5, 1, 3, 9],
            "decoder_loss_weights": [0, 0, 0, 0, 1, 1, 1],
            "decoder_positions": [0, 1, 2, 3, 4, 5, 6],
            "decoder_segment_ids": [1, 1, 1, 1, 1, 1, 1],
            "decoder_causal_attention": [1, 1, 1, 1, 1, 0, 0],
        },
        {
            "decoder_target_tokens": [8, 4, 1, 5, 1, 0, 0],
            "decoder_input_tokens": [0, 8, 4, 1, 5, 0, 0],
            "decoder_loss_weights": [0, 0, 0, 1, 1, 0, 0],
            "decoder_positions": [0, 1, 2, 3, 4, 0, 0],
            "decoder_segment_ids": [1, 1, 1, 1, 1, 0, 0],
            "decoder_causal_attention": [1, 1, 1, 1, 0, 0, 0],
        },
    ]
    for actual, expected in zip(iter(ds), expected_ds, strict=True):
      self.assertSequenceEqual(
          sorted(actual.keys()), sorted(self._expected_packed_keys)
      )
      for k in self._expected_packed_keys:
        np.testing.assert_array_equal(actual[k], expected[k])

  def test_convert_example(self):
    ex = {
        "targets": np.array([7, 8, 5, 1, 3, 9, 1, 0]),
        "inputs_width": np.array([4, 4, 4, 4, 4, 4, 4, 0]),
        "inputs_width_add_pos": np.array([5, 5, 5, 5, 5, 5, 5, 0]),
    }
    converted_ex = feature_converters._convert_to_t5x_prefix_lm_features(
        ex,
        loss_on_targets_only=True,
        packed=False,
        bos_id=0,
        pad_id=0,
        passthrough_feature_keys=[],
    )
    expected = {
        "decoder_target_tokens": [7, 8, 5, 1, 3, 9, 1, 0],
        "decoder_input_tokens": [0, 7, 8, 5, 1, 3, 9, 1],
        "decoder_loss_weights": [0, 0, 0, 0, 1, 1, 1, 0],
        "decoder_causal_attention": [1, 1, 1, 1, 1, 0, 0, 0],
    }
    for k in self._expected_unpacked_keys:
      np.testing.assert_array_equal(expected[k], converted_ex[k])

  def test_prefix_lm_packed_without_default_bos(self):
    x = [
        {"inputs": [7, 8, 5, 1], "targets": [3, 9, 1]},
        {"inputs": [8, 4, 9, 3, 1], "targets": [4, 1]},
    ]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()

    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 8, "targets": 7},
    )
    preps = (
        feature_converters.get_t5x_prefix_lm_feature_converter_preprocessors(
            pack=True,
            use_multi_bin_packing=False,
            pad_id=0,
            bos_id=10,
            loss_on_targets_only=True,
            passthrough_feature_keys=[],
        )
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)
    ds = list(ds)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "decoder_target_tokens": 15,
            "decoder_input_tokens": 15,
            "decoder_loss_weights": 15,
            "decoder_segment_ids": 15,
            "decoder_positions": 15,
            "decoder_causal_attention": 15,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    expected = {
        "decoder_target_tokens": [7, 8, 5, 1, 3, 9, 1, 8, 4, 9, 3, 1, 4, 1, 0],
        "decoder_input_tokens": [
            10,
            7,
            8,
            5,
            1,
            3,
            9,
            10,
            8,
            4,
            9,
            3,
            1,
            4,
            10,
        ],
        "decoder_loss_weights": [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        "decoder_positions": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0],
        "decoder_segment_ids": [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0],
        "decoder_causal_attention": [
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
        ],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(
        sorted(actual.keys()), sorted(self._expected_packed_keys)
    )
    for k in self._expected_packed_keys:
      np.testing.assert_array_equal(actual[k], expected[k])

  def test_empty_inputs(self):
    x = [
        {
            "inputs": [1],
            "targets": [1, 2],
        },
        {
            "inputs": [],
            "targets": [3, 4],
        },
        {
            "inputs": [2, 3, 4, 5, 6],
            "targets": [5, 6],
        },
    ]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()

    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 7, "targets": 7},
    )
    preps = (
        feature_converters.get_t5x_prefix_lm_feature_converter_preprocessors(
            pack=True,
            use_multi_bin_packing=False,
            pad_id=0,
            bos_id=0,
            loss_on_targets_only=True,
            passthrough_feature_keys=[],
        )
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)
    ds = list(ds)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "decoder_target_tokens": 14,
            "decoder_input_tokens": 14,
            "decoder_loss_weights": 14,
            "decoder_segment_ids": 14,
            "decoder_positions": 14,
            "decoder_causal_attention": 14,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    expected = {
        "decoder_target_tokens": [1, 1, 2, 3, 4, 2, 3, 4, 5, 6, 5, 6, 0, 0],
        "decoder_input_tokens": [0, 1, 1, 0, 3, 0, 2, 3, 4, 5, 6, 5, 0, 0],
        "decoder_loss_weights": [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        "decoder_segment_ids": [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 0, 0],
        "decoder_positions": [0, 1, 2, 0, 1, 0, 1, 2, 3, 4, 5, 6, 0, 0],
        "decoder_causal_attention": [1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(
        sorted(actual.keys()), sorted(self._expected_packed_keys)
    )
    for k in self._expected_packed_keys:
      np.testing.assert_array_equal(actual[k], expected[k])


# GOOGLE BEGIN-INTERNAL


class CmsLMFeatureConverter(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._expected_keys = [
        "article_ids",
        "input_mask",
        "observation",
        "target",
        "target_mask",
    ]

  def test_lm_unpacked(self):
    x = [{"targets": [3, 9, 1]}]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"targets": 5}
    )

    preps = feature_converters.get_cms_lm_feature_converter_preprocessors(
        pack=False,
        use_multi_bin_packing=False,
        pad_id=0,
        bos_id=0,
        target_has_suffix=False,
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)
    ds = list(ds)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "observation": 5,
            "target": 5,
            "input_mask": 5,
            "target_mask": 5,
            "article_ids": 5,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    expected = {
        "observation": [0, 3, 9, 1, 0],
        "target": [3, 9, 1, 0, 0],
        "input_mask": [1, 1, 1, 0, 0],
        "target_mask": [1, 1, 1, 0, 0],
        "article_ids": [1, 1, 1, 0, 0],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(sorted(actual.keys()), sorted(self._expected_keys))
    for k in self._expected_keys:
      np.testing.assert_array_equal(actual[k], expected[k])

  def test_lm_only_packed(self):
    x = [{"targets": [3, 9, 1]}, {"targets": [4, 1]}]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"targets": 6}
    )

    preps = feature_converters.get_cms_lm_feature_converter_preprocessors(
        pack=True,
        use_multi_bin_packing=False,
        pad_id=0,
        bos_id=0,
        target_has_suffix=False,
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)
    ds = list(ds)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "observation": 6,
            "target": 6,
            "input_mask": 6,
            "target_mask": 6,
            "article_ids": 6,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    expected = {
        "observation": [0, 3, 9, 0, 4, 0],
        "target": [3, 9, 1, 4, 1, 0],
        "input_mask": [1, 1, 1, 1, 1, 0],
        "target_mask": [1, 1, 1, 1, 1, 0],
        "article_ids": [1, 1, 1, 2, 2, 0],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(sorted(actual.keys()), sorted(self._expected_keys))
    for k in self._expected_keys:
      np.testing.assert_array_equal(actual[k], expected[k])

  def test_lm_pack_long_sequences(self):
    x = [{"targets": [3, 9, 4, 5, 1]}, {"targets": [4, 3, 2, 1]}]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"targets": 5}
    )

    preps = feature_converters.get_cms_lm_feature_converter_preprocessors(
        pack=True,
        use_multi_bin_packing=False,
        bos_id=0,
        pad_id=0,
        target_has_suffix=False,
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)
    ds = list(ds)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "observation": 5,
            "target": 5,
            "input_mask": 5,
            "target_mask": 5,
            "article_ids": 5,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    expected_ds = [
        {
            "observation": [0, 3, 9, 4, 5],
            "target": [3, 9, 4, 5, 1],
            "input_mask": [1, 1, 1, 1, 1],
            "target_mask": [1, 1, 1, 1, 1],
            "article_ids": [1, 1, 1, 1, 1],
        },
        {
            "observation": [0, 4, 3, 2, 0],
            "target": [4, 3, 2, 1, 0],
            "input_mask": [1, 1, 1, 1, 0],
            "target_mask": [1, 1, 1, 1, 0],
            "article_ids": [1, 1, 1, 1, 0],
        },
    ]
    self.assertLen(ds, 2)
    for actual, expected in zip(ds, expected_ds):
      self.assertSequenceEqual(
          sorted(actual.keys()), sorted(self._expected_keys)
      )
      for k in self._expected_keys:
        np.testing.assert_array_equal(actual[k], expected[k])

  def test_lm_extra_field_removed(self):
    x = [
        {"targets": [3, 9, 1], "plaintext": "abc"},
        {"targets": [4, 1], "plaintext": "abc"},
    ]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"targets": 6}
    )

    preps = feature_converters.get_cms_lm_feature_converter_preprocessors(
        pack=True,
        use_multi_bin_packing=False,
        bos_id=0,
        pad_id=0,
        target_has_suffix=False,
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)
    ds = list(ds)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "observation": 6,
            "target": 6,
            "input_mask": 6,
            "target_mask": 6,
            "article_ids": 6,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    expected = {
        "observation": [0, 3, 9, 0, 4, 0],
        "target": [3, 9, 1, 4, 1, 0],
        "input_mask": [1, 1, 1, 1, 1, 0],
        "target_mask": [1, 1, 1, 1, 1, 0],
        "article_ids": [1, 1, 1, 2, 2, 0],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(sorted(actual.keys()), sorted(self._expected_keys))
    for k in self._expected_keys:
      np.testing.assert_array_equal(actual[k], expected[k])

  def test_lm_only_packed_without_default_bos(self):
    x = [{"targets": [3, 9, 1]}, {"targets": [4, 1]}]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"targets": 6}
    )

    preps = feature_converters.get_cms_lm_feature_converter_preprocessors(
        pack=True,
        use_multi_bin_packing=False,
        bos_id=10,
        pad_id=0,
        target_has_suffix=False,
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)
    ds = list(ds)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "observation": 6,
            "target": 6,
            "input_mask": 6,
            "target_mask": 6,
            "article_ids": 6,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    expected = {
        "observation": [10, 3, 9, 10, 4, 10],
        "target": [3, 9, 1, 4, 1, 0],
        "input_mask": [1, 1, 1, 1, 1, 0],
        "target_mask": [1, 1, 1, 1, 1, 0],
        "article_ids": [1, 1, 1, 2, 2, 0],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(sorted(actual.keys()), sorted(self._expected_keys))
    for k in self._expected_keys:
      np.testing.assert_array_equal(actual[k], expected[k])


class CmsPrefixLMFeatureConverter(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._expected_keys = [
        "article_ids",
        "input_mask",
        "observation",
        "target",
        "target_mask",
    ]

  def test_prefix_lm_unpacked(self):
    x = [{"inputs": [9, 4, 6, 1], "targets": [3, 9, 1]}]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})

    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 5, "targets": 4},
    )
    preps = (
        feature_converters.get_cms_prefix_lm_feature_converter_preprocessors(
            pack=False,
            use_multi_bin_packing=False,
            pad_id=0,
            bos_id=0,
            loss_on_targets_only=True,
            passthrough_feature_keys=[],
            target_has_suffix=False,
        )
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "observation": 9,
            "target": 9,
            "input_mask": 9,
            "target_mask": 9,
            "article_ids": 9,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    # The last EOS token is kept if unpacked.
    expected = {
        "observation": [0, 9, 4, 6, 1, 3, 9, 1, 0],
        "target": [9, 4, 6, 1, 3, 9, 1, 0, 0],
        "input_mask": [1, 1, 1, 1, 1, 1, 1, 0, 0],
        "target_mask": [0, 0, 0, 0, 1, 1, 1, 0, 0],
        "article_ids": [1, 1, 1, 1, 1, 1, 1, 0, 0],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(sorted(actual.keys()), sorted(self._expected_keys))
    for k in self._expected_keys:
      np.testing.assert_array_equal(actual[k], expected[k])

  def test_prefix_lm_unpacked_passthrough(self):
    x = [{
        "inputs": [9, 4, 6, 1],
        "targets": [3, 9, 1],
        "passthrough": [6, 5, 4, 3, 2, 1, 0],
    }]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})

    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 5, "targets": 4, "passthrough": 7},
    )
    preps = (
        feature_converters.get_cms_prefix_lm_feature_converter_preprocessors(
            pack=False,
            use_multi_bin_packing=False,
            pad_id=0,
            bos_id=0,
            loss_on_targets_only=True,
            passthrough_feature_keys=["passthrough"],
            target_has_suffix=False,
        )
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "observation": 9,
            "target": 9,
            "input_mask": 9,
            "target_mask": 9,
            "article_ids": 9,
            "passthrough": 7,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    # The last EOS token is kept if unpacked.
    expected = {
        "observation": [0, 9, 4, 6, 1, 3, 9, 1, 0],
        "target": [9, 4, 6, 1, 3, 9, 1, 0, 0],
        "input_mask": [1, 1, 1, 1, 1, 1, 1, 0, 0],
        "target_mask": [0, 0, 0, 0, 1, 1, 1, 0, 0],
        "article_ids": [1, 1, 1, 1, 1, 1, 1, 0, 0],
        "passthrough": [6, 5, 4, 3, 2, 1, 0],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    expected_keys = self._expected_keys + ["passthrough"]
    self.assertSequenceEqual(sorted(actual.keys()), sorted(expected_keys))
    for k in expected_keys:
      np.testing.assert_array_equal(actual[k], expected[k])

  def test_prefix_lm_unpacked_trivial_targets(self):
    x = [{"inputs": [9, 4, 6, 1], "targets": []}]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})

    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 5, "targets": 4},
    )
    preps = (
        feature_converters.get_cms_prefix_lm_feature_converter_preprocessors(
            pack=False,
            use_multi_bin_packing=False,
            pad_id=0,
            bos_id=0,
            loss_on_targets_only=True,
            passthrough_feature_keys=[],
            target_has_suffix=False,
        )
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "observation": 9,
            "target": 9,
            "input_mask": 9,
            "target_mask": 9,
            "article_ids": 9,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    # The last EOS token is kept if unpacked.
    expected = {
        "observation": [0, 9, 4, 6, 1, 0, 0, 0, 0],
        "target": [9, 4, 6, 1, 0, 0, 0, 0, 0],
        "input_mask": [1, 1, 1, 1, 1, 0, 0, 0, 0],
        "target_mask": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "article_ids": [1, 1, 1, 1, 1, 0, 0, 0, 0],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(sorted(actual.keys()), sorted(self._expected_keys))
    for k in self._expected_keys:
      np.testing.assert_array_equal(actual[k], expected[k])

  def test_prefix_lm_long_inputs_feature_length(self):
    x = [{"inputs": [9, 4, 6, 1], "targets": [3, 9, 1]}]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})

    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 10, "targets": 4},
    )
    preps = (
        feature_converters.get_cms_prefix_lm_feature_converter_preprocessors(
            pack=False,
            use_multi_bin_packing=False,
            pad_id=0,
            bos_id=0,
            loss_on_targets_only=True,
            passthrough_feature_keys=[],
            target_has_suffix=False,
        )
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "observation": 14,
            "target": 14,
            "input_mask": 14,
            "target_mask": 14,
            "article_ids": 14,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    # The last EOS token is kept if unpacked.
    expected = {
        "observation": [0, 9, 4, 6, 1, 3, 9, 1, 0, 0, 0, 0, 0, 0],
        "target": [9, 4, 6, 1, 3, 9, 1, 0, 0, 0, 0, 0, 0, 0],
        "input_mask": [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        "target_mask": [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        "article_ids": [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(sorted(actual.keys()), sorted(self._expected_keys))
    for k in self._expected_keys:
      np.testing.assert_array_equal(actual[k], expected[k])

  def test_prefix_lm_packed(self):
    x = [
        {"inputs": [7, 8, 5, 1], "targets": [3, 9, 1]},
        {"inputs": [8, 4, 9, 3, 1], "targets": [4, 1]},
    ]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()

    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 8, "targets": 7},
    )
    preps = (
        feature_converters.get_cms_prefix_lm_feature_converter_preprocessors(
            pack=True,
            use_multi_bin_packing=False,
            pad_id=0,
            bos_id=0,
            loss_on_targets_only=True,
            passthrough_feature_keys=[],
            target_has_suffix=False,
        )
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)
    ds = list(ds)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "observation": 15,
            "target": 15,
            "input_mask": 15,
            "target_mask": 15,
            "article_ids": 15,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    expected = {
        "observation": [0, 7, 8, 5, 1, 3, 9, 0, 8, 4, 9, 3, 1, 4, 0],
        "target": [7, 8, 5, 1, 3, 9, 1, 8, 4, 9, 3, 1, 4, 1, 0],
        "input_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        "target_mask": [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        "article_ids": [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(sorted(actual.keys()), sorted(self._expected_keys))
    for k in self._expected_keys:
      np.testing.assert_array_equal(actual[k], expected[k])

  def test_prefix_lm_unpacked_loss_on_inputs_and_targets(self):
    x = [{"inputs": [9, 4, 6, 1], "targets": [3, 9, 1]}]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})

    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 5, "targets": 4},
    )
    preps = (
        feature_converters.get_cms_prefix_lm_feature_converter_preprocessors(
            pack=False,
            use_multi_bin_packing=False,
            pad_id=0,
            bos_id=0,
            loss_on_targets_only=False,
            passthrough_feature_keys=[],
            target_has_suffix=False,
        )
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "observation": 9,
            "target": 9,
            "input_mask": 9,
            "target_mask": 9,
            "article_ids": 9,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    # Loss weights on the inputs portion and padding should be zeroed out.
    expected = {
        "observation": [0, 9, 4, 6, 1, 3, 9, 1, 0],
        "target": [9, 4, 6, 1, 3, 9, 1, 0, 0],
        "input_mask": [1, 1, 1, 1, 1, 1, 1, 0, 0],
        "target_mask": [1, 1, 1, 1, 1, 1, 1, 0, 0],
        "article_ids": [1, 1, 1, 1, 1, 1, 1, 0, 0],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(sorted(actual.keys()), sorted(self._expected_keys))
    for k in self._expected_keys:
      np.testing.assert_array_equal(actual[k], expected[k])

  def test_prefix_lm_packed_loss_on_inputs_and_targets(self):
    x = [
        {"inputs": [7, 8, 5, 1], "targets": [3, 9, 1]},
        {"inputs": [8, 4, 9, 3, 1], "targets": [4, 1]},
    ]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()

    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 8, "targets": 7},
    )
    preps = (
        feature_converters.get_cms_prefix_lm_feature_converter_preprocessors(
            pack=True,
            use_multi_bin_packing=False,
            pad_id=0,
            bos_id=0,
            loss_on_targets_only=False,
            passthrough_feature_keys=[],
            target_has_suffix=False,
        )
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)
    ds = list(ds)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "observation": 15,
            "target": 15,
            "input_mask": 15,
            "target_mask": 15,
            "article_ids": 15,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    expected = {
        "observation": [0, 7, 8, 5, 1, 3, 9, 0, 8, 4, 9, 3, 1, 4, 0],
        "target": [7, 8, 5, 1, 3, 9, 1, 8, 4, 9, 3, 1, 4, 1, 0],
        "input_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        "target_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        "article_ids": [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(sorted(actual.keys()), sorted(self._expected_keys))
    for k in self._expected_keys:
      np.testing.assert_array_equal(actual[k], expected[k])

  def test_prefix_lm_long_inputs(self):
    # TODO(b/319663351): This test should fail validation checks. Update after
    # adding checks.
    x = [
        {"inputs": [7, 8, 5, 6, 1], "targets": [3, 9, 7, 1]},
        {"inputs": [8, 4, 9, 3, 8, 1], "targets": [4, 1]},
    ]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()

    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 4, "targets": 3},
    )

    preps = (
        feature_converters.get_cms_prefix_lm_feature_converter_preprocessors(
            pack=True,
            use_multi_bin_packing=False,
            pad_id=0,
            bos_id=0,
            loss_on_targets_only=True,
            passthrough_feature_keys=[],
            target_has_suffix=False,
        )
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "observation": 7,
            "target": 7,
            "input_mask": 7,
            "target_mask": 7,
            "article_ids": 7,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    expected_ds = [
        {
            "observation": [0, 7, 8, 5, 6, 1, 3],
            "target": [7, 8, 5, 6, 1, 3, 9],
            "input_mask": [1, 1, 1, 1, 1, 1, 1],
            "target_mask": [0, 0, 0, 0, 0, 1, 1],
            "article_ids": [1, 1, 1, 1, 1, 1, 1],
        },
        {
            "observation": [0, 8, 4, 9, 3, 8, 1],
            "target": [8, 4, 9, 3, 8, 1, 4],
            "input_mask": [1, 1, 1, 1, 1, 1, 1],
            "target_mask": [0, 0, 0, 0, 0, 0, 1],
            "article_ids": [1, 1, 1, 1, 1, 1, 1],
        },
    ]
    for actual, expected in zip(iter(ds), expected_ds, strict=True):
      self.assertSequenceEqual(
          sorted(actual.keys()), sorted(self._expected_keys)
      )
      for k in self._expected_keys:
        np.testing.assert_array_equal(actual[k], expected[k])

  def test_prefix_lm_pack_long_sequences(self):
    x = [
        {"inputs": [7, 8, 5, 1], "targets": [3, 9, 1]},
        {"inputs": [8, 4, 1], "targets": [5, 1]},
    ]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()

    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 4, "targets": 3},
    )
    preps = (
        feature_converters.get_cms_prefix_lm_feature_converter_preprocessors(
            pack=True,
            use_multi_bin_packing=False,
            pad_id=0,
            bos_id=0,
            loss_on_targets_only=True,
            passthrough_feature_keys=[],
            target_has_suffix=False,
        )
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "observation": 7,
            "target": 7,
            "input_mask": 7,
            "target_mask": 7,
            "article_ids": 7,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    # The examples should not be packed because examples are not short enough.
    expected_ds = [
        {
            "observation": [0, 7, 8, 5, 1, 3, 9],
            "target": [7, 8, 5, 1, 3, 9, 1],
            "input_mask": [1, 1, 1, 1, 1, 1, 1],
            "target_mask": [0, 0, 0, 0, 1, 1, 1],
            "article_ids": [1, 1, 1, 1, 1, 1, 1],
        },
        {
            "observation": [0, 8, 4, 1, 5, 0, 0],
            "target": [8, 4, 1, 5, 1, 0, 0],
            "input_mask": [1, 1, 1, 1, 1, 0, 0],
            "target_mask": [0, 0, 0, 1, 1, 0, 0],
            "article_ids": [1, 1, 1, 1, 1, 0, 0],
        },
    ]
    for actual, expected in zip(iter(ds), expected_ds, strict=True):
      self.assertSequenceEqual(
          sorted(actual.keys()), sorted(self._expected_keys)
      )
      for k in self._expected_keys:
        np.testing.assert_array_equal(actual[k], expected[k])

  def test_prefix_lm_packed_without_default_bos(self):
    x = [
        {"inputs": [7, 8, 5, 1], "targets": [3, 9, 1]},
        {"inputs": [8, 4, 9, 3, 1], "targets": [4, 1]},
    ]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()

    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 8, "targets": 7},
    )
    preps = (
        feature_converters.get_cms_prefix_lm_feature_converter_preprocessors(
            pack=True,
            use_multi_bin_packing=False,
            pad_id=0,
            bos_id=10,
            loss_on_targets_only=True,
            passthrough_feature_keys=[],
            target_has_suffix=False,
        )
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)
    ds = list(ds)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "observation": 15,
            "target": 15,
            "input_mask": 15,
            "target_mask": 15,
            "article_ids": 15,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    expected = {
        "observation": [10, 7, 8, 5, 1, 3, 9, 10, 8, 4, 9, 3, 1, 4, 10],
        "target": [7, 8, 5, 1, 3, 9, 1, 8, 4, 9, 3, 1, 4, 1, 0],
        "input_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        "target_mask": [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        "article_ids": [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(sorted(actual.keys()), sorted(self._expected_keys))
    for k in self._expected_keys:
      np.testing.assert_array_equal(actual[k], expected[k])

  def test_empty_inputs(self):
    x = [
        {
            "inputs": [1],
            "targets": [1, 2],
        },
        {
            "inputs": [],
            "targets": [3, 4],
        },
        {
            "inputs": [2, 3, 4, 5, 6],
            "targets": [5, 6],
        },
    ]
    ds = grain.MapDataset.source(x)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()

    runtime_args = test_utils.create_airio_injected_runtime_args(
        sequence_lengths={"inputs": 7, "targets": 7},
    )
    preps = (
        feature_converters.get_cms_prefix_lm_feature_converter_preprocessors(
            pack=True,
            use_multi_bin_packing=False,
            pad_id=0,
            bos_id=0,
            loss_on_targets_only=True,
            passthrough_feature_keys=[],
            target_has_suffix=False,
        )
    )
    ds, updated_runtime_args = _apply_preprocessors(ds, preps, runtime_args)
    ds = list(ds)

    expected_runtime_args = runtime_args.replace(
        sequence_lengths={
            "observation": 14,
            "target": 14,
            "input_mask": 14,
            "target_mask": 14,
            "article_ids": 14,
        },
    )
    self.assertEqual(updated_runtime_args, expected_runtime_args)

    expected = {
        "observation": [0, 1, 1, 0, 3, 0, 2, 3, 4, 5, 6, 5, 0, 0],
        "target": [1, 1, 2, 3, 4, 2, 3, 4, 5, 6, 5, 6, 0, 0],
        "input_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        "target_mask": [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        "article_ids": [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 0, 0],
    }
    self.assertLen(ds, 1)
    actual = ds[0]
    self.assertSequenceEqual(sorted(actual.keys()), sorted(self._expected_keys))
    for k in self._expected_keys:
      np.testing.assert_array_equal(actual[k], expected[k])


# GOOGLE END-INTERNAL


if __name__ == "__main__":
  absltest.main()
