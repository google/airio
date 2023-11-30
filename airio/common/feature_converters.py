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

"""Preprocessors to replicate conventional SeqIO FeatureConverters."""

import numpy as np


def make_autoregressive_inputs(
    inputs: np.ndarray, sequence_ids: np.ndarray | None, bos_id: int
):
  """Generate inputs for an autoregressive model, by shifting the inputs.

  Modified from seqio.utils.make_autoregressive_inputs.

  For the first element of each sequence, the returned input id is bos_id
  (commonly 0).

  For a "packed" dataset, also pass the sequence_ids tensor, which aligns
  with the inputs tensor and contains different values for different
  concatenated examples.

  Example for a packed dataset:

  ```
        inputs = [3, 8, 1, 9, 1, 5, 4, 1, 0, 0]
    sequence_ids = [1, 1, 1, 2, 2, 3, 3, 3, 0, 0]
         inputs = [0, 3, 8, 0, 9, 0, 5, 4, 0, 0]
                            |     |        |
                            These positions are set to 0 if sequence_ids is not
                            None.
  ```

  Args:
    inputs: the input sequence
    sequence_ids: a 1-D sequence indicating segements belonging to separate
      examples, for packed sequences, e.g. for a padded and packed sequence =
      [ex11, ex12, ex21, ex22, ex23, pad_id, pad_id], sequence_ids would be =
      [1, 1, 2, 2, 2, pad_id, pad_id]. Set to None for unpacked datasets.
    bos_id: bos (beginning of sequence) id, commonly set to 0.

  Returns:
    a tensor with dtype tf.int32 and the same shape as inputs.
  """
  output_dtype = inputs.dtype
  if sequence_ids is not None and not np.issubdtype(
      sequence_ids.dtype, np.integer
  ):
    raise ValueError(
        "Sequence_ids should be integer-valued tensors for a packed dataset."
    )
  if sequence_ids is not None and len(inputs.shape) > 1:
    raise ValueError(
        "Only 1-D sequences are supported with packing. Got a packed"
        f" {len(inputs.shape)}-D sequence."
    )

  def shift_right_by_one(arr, fill_value):
    shifted = np.roll(arr, shift=1, axis=0)
    shifted[0] = fill_value
    return shifted

  inputs = shift_right_by_one(inputs, bos_id)
  if sequence_ids is not None:
    # Note: This means the sequence has sub-sequences of inputs. The entire
    # sequence, and hence all sub-sequences, have already been right shifted; we
    # just need to find the beginning of each sub-sequence and put a bos_id
    # there, replacing the last token of the previous input sub-sequence after
    # the right shift.

    # Find the beginning positions of examples from sequence_ids, e.g.
    # [1, 1, 2, 2, 2, 0, 0] -> [0, 1, 0, 1, 1, 0, 1]. Positions with 0 will be
    # filled by bos_id, positions with 1 will be filled by (shifted) inputs.
    not_first_in_sequence = sequence_ids == shift_right_by_one(
        sequence_ids, fill_value=0
    )
    not_first_in_sequence = not_first_in_sequence.astype(output_dtype)
    # 0s -> bos_id, 1s -> inputs
    first_ids = (1 - not_first_in_sequence) * bos_id
    input_ids = inputs * not_first_in_sequence
    # Combine.
    inputs = input_ids + first_ids

  return inputs
