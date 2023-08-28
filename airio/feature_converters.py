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

"""Implementation of AirIO-specific Feature Converters."""

import functools
import typing
from typing import List, Mapping, Protocol

import grain.python as grain
import numpy as np
import seqio


def _get_non_padding_positions(
    feature_value: np.ndarray, dtype: np.dtype = np.int_, pad_id: int = 0
) -> np.ndarray:
  res = feature_value != pad_id
  return res.astype(dtype)


def _create_pygrain_features(
    orig_example: Mapping[str, np.ndarray],
    task_feature_lengths: Mapping[str, int],
):
  return {k: orig_example[k][:l] for k, l in task_feature_lengths.items()}


def _convert_features_for_enc_dec(orig_example: Mapping[str, np.ndarray]):
  # TODO(sahildua): Fix "decoder_input_tokens" feature.
  return {
      "encoder_input_tokens": orig_example["inputs"],
      "decoder_target_tokens": orig_example["targets"],
      "decoder_input_tokens": orig_example["targets"],
      "decoder_loss_weights": _get_non_padding_positions(
          orig_example["targets"]
      ),
  }


def _trim_and_pad_features(
    features: Mapping[str, np.ndarray],
    sequence_lengths: Mapping[str, int],
    pad_id: int = 0,
) -> Mapping[str, np.ndarray]:
  """Trims and pads the features of an np.ndarray."""

  def _trim_and_pad_inner(key: str, val: np.ndarray):
    if key not in sequence_lengths:
      return val
    seq_len = sequence_lengths[key]
    val = val[:seq_len]
    pad_amt = seq_len - val.shape[0]
    padded_val = np.pad(
        val, (0, pad_amt), mode="constant", constant_values=pad_id
    )
    return padded_val

  return {k: _trim_and_pad_inner(k, v) for k, v in features.items()}


def _construct_pygrain_operation(fn, **args) -> grain.Operation:
  return grain.MapOperation(functools.partial(fn, **args))


@typing.runtime_checkable
class PyGrainFeatureConverter(Protocol):
  """Interface for PyGrain feature converters."""

  def get_operations(self) -> List[grain.Operation]:
    ...


class PyGrainEncDecFeatureConverter:
  """Builder for PyGrain's operations for seqio.EncDecFeatureConverter."""

  def __init__(
      self,
      *,
      batch_size: int,
      task_feature_lengths: Mapping[str, int],
      model_feature_lengths: Mapping[str, int],
      bos_id: int,
      pack: bool,
  ):
    self._batch_size = batch_size
    self._task_feature_lengths = task_feature_lengths
    self._model_feature_lengths = model_feature_lengths
    self._bos_id = bos_id
    self._pack = pack

  def get_operations(self) -> List[grain.Operation]:
    """Returns a list of PyGrain operations."""
    # TODO(sahildua): Implement packing support.
    operations = [
        _construct_pygrain_operation(
            _create_pygrain_features,
            task_feature_lengths=self._task_feature_lengths,
        ),
        _construct_pygrain_operation(_convert_features_for_enc_dec),
        _construct_pygrain_operation(
            _trim_and_pad_features,
            sequence_lengths=self._model_feature_lengths,
        ),
        grain.BatchOperation(self._batch_size, drop_remainder=True),
    ]

    return operations


def get_pygrain_feature_converter(
    feature_converter: seqio.FeatureConverter,
    *,
    batch_size: int,
    task_feature_lengths: Mapping[str, int],
) -> PyGrainFeatureConverter:
  """Wrapper for converting seqio FeatureConverter into PyGrain operations.

  This method receives a SeqIO feature converter and returns a list of PyGrain
  operations which resemble the transformations applied in the _convert_features
  method of SeqIO's FeatureConverters.

  Args:
    feature_converter: A SeqIO feature converter.
    batch_size: Batch size for applying PyGrain batching operation.
    task_feature_lengths: Mapping from feature to corresponing sequence length.

  Returns:
    A PyGrain equivalent feature converter object.
  """
  if isinstance(feature_converter, seqio.EncDecFeatureConverter):
    return PyGrainEncDecFeatureConverter(
        batch_size=batch_size,
        task_feature_lengths=task_feature_lengths,
        model_feature_lengths=feature_converter.get_model_feature_lengths(
            task_feature_lengths
        ),
        bos_id=feature_converter.bos_id,
        pack=feature_converter.pack,
    )

  raise NotImplementedError(
      f"Feature Converter '{feature_converter}' does not have Grain"
      " FeatureConverter equivalent."
  )
