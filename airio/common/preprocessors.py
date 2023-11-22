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

"""Common data preprocessors."""

from typing import Any, Dict
from airio import preprocessors as preprocessors_lib
from airio.common import constants
import numpy as np

SKIP_FEATURE = constants.SKIP_FEATURE


def trim(
    ex: Dict[str, np.ndarray],
    runtime_args: preprocessors_lib.AirIOInjectedRuntimeArgs,
):
  """Trims a dict of np.ndarrays.

  Trimming is generally followed by packing (optionally) and padding.

  Args:
    ex: a dict of np.ndarrays.
    runtime_args: An `AirIOInjectedRuntimeArgs` obj containing the
      `sequence_lengths` to trim to. If sequence length for a feature is an int,
      then the 0-th dimension of the array is trimmed. If sequence length is an
      array (rank of feature and sequence length must match), then each
      dimension is trimmed to the corresponding sequence length. If a feature
      key is missing from sequence lengths or set to
      airio.common.constants.SKIP_FEATURE, then the feature is not trimmed.

  Returns:
    Trimmed dict of np.ndarrays.

  Raises:
    ValueError if sequence_length is an array and has a rank mismatch with
    that of the feature.
  """

  def _trim(k: str, v: np.ndarray) -> np.ndarray:
    if (
        not sequence_lengths
        or k not in sequence_lengths
        or sequence_lengths[k] == SKIP_FEATURE
    ):
      return v
    # Unify lengths into an iterable so we can create a slice for each
    # dimension, even if the length is a single int.
    lengths = sequence_lengths[k]
    if isinstance(lengths, int):
      lengths = [lengths]
    else:
      if len(lengths) != len(v.shape):
        raise ValueError(
            f"Rank mismatch: sequence length for feature '{k}' is {lengths} but"
            f" shape is {v.shape}, sequence: {v}"
        )
    slices = tuple((slice(0, limit) for limit in lengths))
    return v[slices]

  sequence_lengths = runtime_args.sequence_lengths
  return {k: _trim(k, v) for k, v in ex.items()}


def pad(
    ex: Dict[str, Any],
    runtime_args: preprocessors_lib.AirIOInjectedRuntimeArgs,
    pad_id: int = 0,
):
  """Pads a dict of np.ndarrays.

  Generally preceeded by trimming and (optionally) packing.

  Args:
    ex: a dict of np.ndarrays.
    runtime_args: An `AirIOInjectedRuntimeArgs` obj containing the
      `sequence_lengths` to pad to. If sequence length for a feature is an int,
      then the 0-th dimension of the array is padded. If sequence length is an
      array (rank of feature and sequence length must match), then each
      dimension is padded to the corresponding sequence length. If a feature key
      is missing from sequence lengths or set to
      airio.common.constants.SKIP_FEATURE, then the feature is not padded.
    pad_id: int value to pad.

  Returns:
    Padded dict of np.ndarrays.

  Raises:
    ValueError if sequence_length is an array and has a rank mismatch with
    that of the feature, or if any sequence exceeds the sequence length passed
    (in this case, trim the examples first).
  """
  def _pad(k: str, v: np.ndarray) -> np.ndarray:
    if (
        not sequence_lengths
        or k not in sequence_lengths
        or sequence_lengths[k] == SKIP_FEATURE
    ):
      return v
    length_k = sequence_lengths[k]
    if isinstance(length_k, int):
      pad_amt = length_k - v.shape[0]
      if pad_amt < 0:
        raise ValueError(
            f"Length of feature '{k}' is {v.shape[0]} > sequence length"
            f" {length_k}, sequence: {v}. Please trim the sequence first."
        )
      return np.pad(
          v,
          [(0, pad_amt)] + [(0, 0)] * (len(v.shape) - 1),
          constant_values=pad_id,
      )
    if len(length_k) != len(v.shape):
      raise ValueError(
          f"Rank mismatch: sequence length for feature '{k}' is {length_k} but"
          f" shape is {v.shape}, sequence: {v}"
      )
    pad_amt = np.array([l - s for l, s in zip(length_k, v.shape)])
    if np.any(pad_amt < 0):
      raise ValueError(
          f"Shape of feature '{k}' is {v.shape} > sequence length"
          f" {length_k}; sequence: {v}. Please trim the sequence first."
      )
    pad_amt = pad_amt[..., None]
    pad_amt = np.pad(pad_amt, ((0, 0), (1, 0)))
    return np.pad(v, pad_amt, constant_values=pad_id)

  sequence_lengths = runtime_args.sequence_lengths
  return {k: _pad(k, v) for k, v in ex.items()}
