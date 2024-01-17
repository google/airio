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

from collections.abc import Sequence
import dataclasses
import functools
from airio import preprocessors as preprocessors_lib
from airio.grain.common import packing
from airio.grain.common import preprocessors
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
    inputs: the input sequence.
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


def convert_to_t5x_enc_dec_features(
    ex: dict[str, np.ndarray],
    pack: bool,
    passthrough_feature_keys: Sequence[str],
    pad_id: int,
    bos_id: int,
):
  """Converts an example for a T5X encoder-decoder model.

  This replicates functionality in seqio.EncDecFeatureConverter, and should be
  preceeded by packing (optional), trimming and padding.

  "inputs" are assigned to the "encoder_input_tokens" field. "targets" are
  assigned to the "decoder_target_tokens" field. The "*_segment_ids" and
  "*_positions" fields are generated from packing. See airio.common.packing for
  details on packing. "decoder_loss_weights" is a binary mask indicating
  non-padding positions, i.e. value of 1 indicates non-padding and 0 indicates
  padding. decoder_input_tokens is produced by shifting each "targets" sequence
  (for packed sequences, this is done for each sub-sequence) to the right by
  one, placing a bos_id token at the beginning, and trimming the last token
  (generally eos) to preserve the length. This is required for autoregressive
  models. It is assumed that the loss is taken only on the decoder side.

  For example, packing and padding a dataset with the following two examples:
      [{"inputs": [7, 8, 5, 1], "targets": [3, 9, 1]},
        {"inputs": [8, 4, 9, 3, 1], "targets": [4, 1]}]
  with runtime_args.sequence_lengths = {"inputs": 10, "targets": 7}, produces:
      {
                       "inputs": [7, 8, 5, 1, 8, 4, 9, 3, 1, 0],
           "inputs_segment_ids": [1, 1, 1, 1, 2, 2, 2, 2, 2, 0],
             "inputs_positions": [0, 1, 2, 3, 0, 1, 2, 3, 4, 0],
                      "targets": [3, 9, 1, 4, 1, 0, 0],
          "targets_segment_ids": [1, 1, 1, 2, 2, 0, 0],
            "targets_positions": [0, 1, 2, 0, 1, 0, 0],
      }
  and applying `convert_to_t5x_enc_dec_features` on the example produces:
      {
       "encoder_input_tokens": [7, 8, 5, 1, 8, 4, 9, 3, 1, 0],
        "encoder_segment_ids": [1, 1, 1, 1, 2, 2, 2, 2, 2, 0],
          "encoder_positions": [0, 1, 2, 3, 0, 1, 2, 3, 4, 0],
      "decoder_target_tokens": [3, 9, 1, 4, 1, 0, 0],
       "decoder_input_tokens": [0, 3, 9, 0, 4, 0, 0],
       "decoder_loss_weights": [1, 1, 1, 1, 1, 0, 0],
        "decoder_segment_ids": [1, 1, 1, 2, 2, 0, 0],
          "decoder_positions": [0, 1, 2, 0, 1, 0, 0],
      }]

  This fn should be used together with
  `update_runtime_args_for_t5x_enc_dec_features` in airio (see docstring for
  details).

  Args:
    ex: dict[str, np.ndarray] - the example to convert.
    pack: bool - whether the input example is packed.
    passthrough_feature_keys: Sequence[str] - A list of feature names to pass
      through.
    pad_id: int - token value to use for padding. 0 is commonly used.
    bos_id: int - token value to use to indicate beginning of sequence in
      decoder input tokens. 0 is commonly used.

  Returns:
    a converted example.
  """
  def _get_non_padding_positions(
      feature_value: np.ndarray, pad_id: int
  ) -> np.ndarray:
    return feature_value != pad_id

  # targets_segment_id is present only for a packed dataset.
  decoder_input_tokens = make_autoregressive_inputs(
      ex["targets"],
      sequence_ids=ex.get("targets_segment_ids", None),
      bos_id=bos_id,
  )

  d = {
      "encoder_input_tokens": ex["inputs"],
      "decoder_target_tokens": ex["targets"],
      "decoder_input_tokens": decoder_input_tokens,
      # Loss is computed for all but the padding positions.
      "decoder_loss_weights": _get_non_padding_positions(ex["targets"], pad_id),
  }
  d.update({k: ex[k] for k in passthrough_feature_keys})

  if pack:
    d["encoder_segment_ids"] = ex["inputs_segment_ids"]
    d["decoder_segment_ids"] = ex["targets_segment_ids"]
    d["encoder_positions"] = ex["inputs_positions"]
    d["decoder_positions"] = ex["targets_positions"]
  return d


def update_runtime_args_for_t5x_enc_dec_features(
    runtime_args: preprocessors_lib.AirIOInjectedRuntimeArgs,
    pack: bool,
    passthrough_feature_keys: Sequence[str],
):
  """Updates runtime args after applying convert_to_t5x_enc_dec_features.

  Replaces "input" and "targets" with features produced by
  `convert_to_t5x_enc_dec_features`, for use by downstream preprocessors. Both
  functions should be used together as an airio preprocessor, e.g.
  update_runtime_args = functools.partial(
      update_runtime_args_for_t5x_enc_dec_features,
      pack=true,
      passthrough_feature_keys=[...],
  )
  convert_features = functools.partial(
      convert_to_t5x_enc_dec_features,
      pack=True,
      passthrough_feature_keys=[...],
      pad_id=0,
      bos_id=0,
  )
  prep = preprocessors_lib.MapFnTransform(
      convert_features,
      update_runtime_args=update_runtime_args,
  )

  Args:
    runtime_args: preprocessors_lib.AirIOInjectedRuntimeArgs - args injected by
      airio at runtime.
    pack: bool - whether the input examples are packed.
    passthrough_feature_keys: a list of feature names to pass through.

  Returns:
    updated preprocessors_lib.AirIOInjectedRuntimeArgs.
  """
  task_feature_lengths = runtime_args.sequence_lengths
  encoder_length = task_feature_lengths["inputs"]
  decoder_length = task_feature_lengths["targets"]

  model_feature_lengths = {
      "encoder_input_tokens": encoder_length,
      "decoder_target_tokens": decoder_length,
      "decoder_input_tokens": decoder_length,
      "decoder_loss_weights": decoder_length,
  }
  for k in passthrough_feature_keys:
    model_feature_lengths[k] = task_feature_lengths[k]

  if pack:
    model_feature_lengths["encoder_segment_ids"] = encoder_length
    model_feature_lengths["decoder_segment_ids"] = decoder_length
    model_feature_lengths["encoder_positions"] = encoder_length
    model_feature_lengths["decoder_positions"] = decoder_length
  updated = runtime_args.clone()
  updated.sequence_lengths = model_feature_lengths
  return updated


@dataclasses.dataclass
class T5XEncDecFeatureConverter:
  """Helper class to get AirIO preprocessors corresponding to seqio.EncDecFeatureConverter.

  Applies packing (optional), trimming, padding and
  `convert_to_t5x_enc_dec_features` in order. See
  `convert_to_t5x_enc_dec_features` docstring for details on feature conversion.

  Attrs:
    pack: bool - indicates whether the dataset should be packed. The multi-bin
      packing in airio.common.packing is functionaly equivalent to the impl in
      SeqIO, and is used here.
    use_multi_bin_packing: bool - Whether to use multi-bin or single-bin
      packing. Seting this to True is equivalent to setting
      `use_custom_packing_ops` to True in seqio.EncDecFeatureConverter. Ignored
      if `pack` is False.
    passthrough_feature_keys: Sequence[str] - A list of feature names to pass
      through.
    pad_id: int - token value to use for padding. 0 is commonly used.
    bos_id: int - token value to use to indicate beginning of sequence in
      decoder input tokens. 0 is commonly used.
  """

  pack: bool
  use_multi_bin_packing: bool
  passthrough_feature_keys: Sequence[str]
  pad_id: int
  bos_id: int

  def get_preprocessors(self) -> Sequence[preprocessors_lib.AirIOPreprocessor]:
    """Returns AirIO preprocessors corresponding to seqio.EncDecFeatureConverter."""

    update_runtime_args = functools.partial(
        update_runtime_args_for_t5x_enc_dec_features,
        pack=self.pack,
        passthrough_feature_keys=self.passthrough_feature_keys,
    )
    convert_features = functools.partial(
        convert_to_t5x_enc_dec_features,
        pack=self.pack,
        passthrough_feature_keys=self.passthrough_feature_keys,
        pad_id=self.pad_id,
        bos_id=self.bos_id,
    )
    pad = functools.partial(preprocessors.pad, pad_id=self.pad_id)
    preps = [
        preprocessors_lib.MapFnTransform(preprocessors.trim),
        preprocessors_lib.MapFnTransform(pad),
        preprocessors_lib.MapFnTransform(
            convert_features,
            update_runtime_args=update_runtime_args,
        ),
    ]
    if self.pack:
      packer = (
          packing.MultiBinTruePackIterPreprocessor
          if self.use_multi_bin_packing
          else packing.SingleBinTruePackIterPreprocessor
      )
      packer_prep = preprocessors_lib.LazyIterTransform(
          packer, update_runtime_args=packer.update_runtime_args
      )
      preps = [packer_prep] + preps
    return preps


# TODO(b/311543848): Implement LM and PrefixLM feature converters from
# seqio/feature_converters.py.
