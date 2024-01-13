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

from airio.grain import preprocessors
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
    task_feature_lengths: Mapping[str, int] | None,
):
  if task_feature_lengths is None:
    return orig_example
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


def _construct_pygrain_transform(fn, **args) -> grain.Transformation:
  return preprocessors.MapFnTransform(functools.partial(fn, **args))


@typing.runtime_checkable
class PyGrainFeatureConverter(Protocol):
  """Interface for PyGrain feature converters."""

  def get_transforms(
      self,
      task_feature_lengths: Mapping[str, int] | None,
  ) -> List[grain.Transformation]:
    ...


class PyGrainEncDecFeatureConverter:
  """Builder for PyGrain's transforms for seqio.EncDecFeatureConverter."""

  def __init__(
      self,
      *,
      bos_id: int = 0,
      pack: bool = False,
  ):
    self._bos_id = bos_id
    self._pack = pack

  def get_transforms(
      self,
      task_feature_lengths: Mapping[str, int] | None,
  ) -> List[grain.Transformation]:
    """Returns a list of PyGrain transforms.

    Args:
      task_feature_lengths: Mapping of feature key to corresponding sequence
        length. If None, trim/pad transform is not added.
    """
    # TODO(sahildua): Implement packing support.
    model_feature_lengths = None
    if task_feature_lengths is not None:
      model_feature_lengths = self._get_model_feature_lengths(
          task_feature_lengths
      )
    transforms = [
        _construct_pygrain_transform(
            _create_pygrain_features,
            task_feature_lengths=task_feature_lengths,
        ),
        _construct_pygrain_transform(_convert_features_for_enc_dec),
    ]
    if model_feature_lengths is not None:
      transforms.append(
          _construct_pygrain_transform(
              _trim_and_pad_features,
              sequence_lengths=model_feature_lengths,
          ),
      )
    return transforms

  def _get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]
  ) -> Mapping[str, int]:
    """Define the length relationship between input and output features."""
    encoder_length = task_feature_lengths["inputs"]
    decoder_length = task_feature_lengths["targets"]

    return {
        "encoder_input_tokens": encoder_length,
        "decoder_target_tokens": decoder_length,
        "decoder_input_tokens": decoder_length,
        "decoder_loss_weights": decoder_length,
    }


def get_pygrain_feature_converter(
    feature_converter: seqio.FeatureConverter,
) -> PyGrainFeatureConverter:
  """Wrapper to convert seqio FeatureConverter into PyGrain FeatureConverter.

  This method receives a SeqIO feature converter and returns a PyGrain-based
  feature converter. This PyGrain-based feature converter has a list of PyGrain
  operations which resemble the transformations applied in the _convert_features
  method of SeqIO FeatureConverters.

  Args:
    feature_converter: A SeqIO feature converter.

  Returns:
    A PyGrain equivalent feature converter object.
  """
  if isinstance(feature_converter, seqio.EncDecFeatureConverter):
    return PyGrainEncDecFeatureConverter(
        bos_id=feature_converter.bos_id,
        pack=feature_converter.pack,
    )

  raise NotImplementedError(
      f"Feature Converter '{feature_converter}' does not have PyGrain"
      " FeatureConverter equivalent."
  )
