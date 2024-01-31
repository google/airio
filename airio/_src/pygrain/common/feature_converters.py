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

"""Preprocessors to replicate conventional SeqIO FeatureConverters.

Use the following helper corresponding to the desired FeatureConverter:
 + EncDecFeatureConverter -> get_t5x_enc_dec_feature_converter_preprocessors
 + LMFeatureConverter -> get_t5x_lm_feature_converter_preprocessors
 + PrefixLMFeatureConverter -> get_t5x_prefix_lm_feature_converter_preprocessors

For example, the following code snippet in SeqIO:
```
fc = seqio.EncDecFeatureConverter(...)
ds = seqio.get_dataset(..., feature_converter=fc)
```

is equivalent to the following in AirIO:

```
preps = get_t5x_enc_dec_feature_converter_preprocessors
ds = airio.get_dataset(..., runtime_preprocessors=preps)
```
"""

from collections.abc import Sequence
import functools
from airio._src.pygrain.common import packing
from airio._src.pygrain.common import preprocessors
from airio.core import preprocessors as preprocessors_lib
from airio.pygrain import preprocessors as grain_preprocessors_lib
import numpy as np


##### Public Methods #####


# Equivalent to seqio.EncDecFeatureConverter
def get_t5x_enc_dec_feature_converter_preprocessors(
    pack: bool,
    use_multi_bin_packing: bool,
    passthrough_feature_keys: Sequence[str],
    pad_id: int,
    bos_id: int,
) -> Sequence[grain_preprocessors_lib.PyGrainAirIOPreprocessor]:
  """Returns a list of AirIO preprocessors corresponding to seqio.EncDecFeatureConverter.

  Applies packing (optional), trimming, padding and
  `_convert_to_t5x_enc_dec_features` in order. See
  `_convert_to_t5x_enc_dec_features` docstring for details on feature
  conversion.

  Args:
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

  Returns:
    a list of AirIO preprocessors.
  """
  update_runtime_args = functools.partial(
      _update_runtime_args_for_t5x_enc_dec_features,
      pack=pack,
      passthrough_feature_keys=passthrough_feature_keys,
  )
  convert_features = functools.partial(
      _convert_to_t5x_enc_dec_features,
      pack=pack,
      passthrough_feature_keys=passthrough_feature_keys,
      pad_id=pad_id,
      bos_id=bos_id,
  )
  pad = functools.partial(preprocessors.pad, pad_id=pad_id)
  preps = [
      preprocessors_lib.MapFnTransform(preprocessors.trim),
      preprocessors_lib.MapFnTransform(pad),
      preprocessors_lib.MapFnTransform(
          convert_features,
          update_runtime_args=update_runtime_args,
      ),
  ]
  if pack:
    packer = (
        packing.MultiBinTruePackIterPreprocessor
        if use_multi_bin_packing
        else packing.SingleBinTruePackIterPreprocessor
    )
    packer_prep = grain_preprocessors_lib.LazyIterTransform(
        packer, update_runtime_args=packer.update_runtime_args
    )
    preps = [packer_prep] + preps
  return preps


# Equivalent to seqio.LMFeatureConverter
def get_t5x_lm_feature_converter_preprocessors(
    pack: bool, use_multi_bin_packing: bool, pad_id: int, bos_id: int
) -> Sequence[grain_preprocessors_lib.PyGrainAirIOPreprocessor]:
  """Returns a list of AirIO preprocessors corresponding to seqio.LMFeatureConverter.

  Applies packing (optional), trimming, padding and
  `_convert_to_t5x_lm_features` in order. See
  `_convert_to_t5x_lm_features` docstring for details on feature conversion.

  Args:
    pack: bool - indicates whether the dataset should be packed. The multi-bin
      packing in airio.common.packing is functionaly equivalent to the impl in
      SeqIO, and is used here.
    use_multi_bin_packing: bool - Whether to use multi-bin or single-bin
      packing. Seting this to True is equivalent to setting
      `use_custom_packing_ops` to True in seqio.EncDecFeatureConverter. Ignored
      if `pack` is False.
    pad_id: int - token value to use for padding. 0 is commonly used.
    bos_id: int - token value to use to indicate beginning of sequence in
      decoder input tokens. 0 is commonly used.

  Returns:
    a list of AirIO preprocessors.
  """
  update_runtime_args = functools.partial(
      _update_runtime_args_for_t5x_lm_features,
      packed=pack,
  )
  convert_features = functools.partial(
      _convert_to_t5x_lm_features,
      packed=pack,
      pad_id=pad_id,
      bos_id=bos_id,
  )
  pad = functools.partial(preprocessors.pad, pad_id=pad_id)
  preps = [
      preprocessors_lib.MapFnTransform(preprocessors.trim),
      preprocessors_lib.MapFnTransform(pad),
      preprocessors_lib.MapFnTransform(
          convert_features,
          update_runtime_args=update_runtime_args,
      ),
  ]
  if pack:
    packer = (
        packing.MultiBinTruePackIterPreprocessor
        if use_multi_bin_packing
        else packing.SingleBinTruePackIterPreprocessor
    )
    packer_prep = grain_preprocessors_lib.LazyIterTransform(
        packer, update_runtime_args=packer.update_runtime_args
    )
    preps = [packer_prep] + preps
  return preps


def get_t5x_prefix_lm_feature_converter_preprocessors(
    pack: bool,
    use_multi_bin_packing: bool,
    pad_id: int,
    bos_id: int,
    loss_on_targets_only: bool,
    passthrough_feature_keys: Sequence[str],
) -> Sequence[grain_preprocessors_lib.PyGrainAirIOPreprocessor]:
  """Returns a list of AirIO preprocessors corresponding to seqio.PrefixLMFeatureConverter.

  Applies `_concat_and_add_masks_for_prefix_lm`, packing (optional), trimming,
  padding and `_convert_to_t5x_prefix_lm_features` in order. See
  `_convert_to_t5x_prefix_lm_features` docstring for details on feature
  conversion.

  Args:
    pack: bool - indicates whether the dataset should be packed. The multi-bin
      packing in airio.common.packing is functionaly equivalent to the impl in
      SeqIO, and is used here.
    use_multi_bin_packing: bool - Whether to use multi-bin or single-bin
      packing. Seting this to True is equivalent to setting
      `use_custom_packing_ops` to True in seqio.EncDecFeatureConverter. Ignored
      if `pack` is False.
    pad_id: int - token value to use for padding. 0 is commonly used.
    bos_id: int - token value to use to indicate beginning of sequence in
      decoder input tokens. 0 is commonly used.
    loss_on_targets_only: whether to compute loss on tokens which belonged to
      "targets" before concatenation.
    passthrough_feature_keys: Sequence[str] - A list of feature names to pass
      through.

  Returns:
    a list of AirIO preprocessors.
  """

  def swap_vals(arr: np.ndarray, old_val: int, new_val: int):
    return np.where(arr == old_val, np.full([arr.size], new_val), arr)

  def swap_inputs_width(ex: dict[str, np.ndarray], old_val: int, new_val: int):
    ex["inputs_width"] = swap_vals(ex["inputs_width"], old_val, new_val)
    return ex

  convert_features = functools.partial(
      _convert_to_t5x_prefix_lm_features,
      loss_on_targets_only=loss_on_targets_only,
      packed=pack,
      bos_id=bos_id,
      pad_id=pad_id,
      passthrough_feature_keys=passthrough_feature_keys,
  )
  update_runtime_args = functools.partial(
      _update_runtime_args_for_t5x_prefix_lm_features,
      packed=pack,
      passthrough_feature_keys=passthrough_feature_keys,
  )
  concat_and_add_masks = functools.partial(
      _concat_and_add_masks_for_prefix_lm,
      passthrough_feature_keys=passthrough_feature_keys,
  )
  concat_task_feature_lengths = functools.partial(
      _concat_task_feature_lengths_for_prefix_lm,
      passthrough_feature_keys=passthrough_feature_keys,
  )
  replace_0s = functools.partial(swap_inputs_width, old_val=0, new_val=-1)
  restore_0s = functools.partial(swap_inputs_width, old_val=-1, new_val=-0)
  pad = functools.partial(preprocessors.pad, pad_id=pad_id)

  preps = [
      preprocessors_lib.MapFnTransform(
          concat_and_add_masks, update_runtime_args=concat_task_feature_lengths
      ),
      preprocessors_lib.MapFnTransform(replace_0s),
  ]
  if pack:
    packer = (
        packing.MultiBinTruePackIterPreprocessor
        if use_multi_bin_packing
        else packing.SingleBinTruePackIterPreprocessor
    )
    packer_prep = grain_preprocessors_lib.LazyIterTransform(
        packer, update_runtime_args=packer.update_runtime_args
    )
    preps.append(packer_prep)
  preps.extend([
      preprocessors_lib.MapFnTransform(preprocessors.trim),
      preprocessors_lib.MapFnTransform(pad),
      preprocessors_lib.MapFnTransform(restore_0s),
      preprocessors_lib.MapFnTransform(
          convert_features, update_runtime_args=update_runtime_args
      ),
  ])
  return preps

##### Implementation and private methods #####


##### Encoder Decoder LM Feature Converter #####


def _convert_to_t5x_enc_dec_features(
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
  and applying `_convert_to_t5x_enc_dec_features` on the example produces:
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
  `_update_runtime_args_for_t5x_enc_dec_features` in airio (see docstring for
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

  # targets_segment_id is present only for a packed dataset.
  decoder_input_tokens = _make_autoregressive_inputs(
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


def _update_runtime_args_for_t5x_enc_dec_features(
    runtime_args: preprocessors_lib.AirIOInjectedRuntimeArgs,
    pack: bool,
    passthrough_feature_keys: Sequence[str],
):
  """Updates runtime args after applying _convert_to_t5x_enc_dec_features.

  Replaces "input" and "targets" with features produced by
  `_convert_to_t5x_enc_dec_features`, for use by downstream preprocessors. Both
  functions should be used together as an airio preprocessor, e.g.
  update_runtime_args = functools.partial(
      _update_runtime_args_for_t5x_enc_dec_features,
      pack=true,
      passthrough_feature_keys=[...],
  )
  convert_features = functools.partial(
      _convert_to_t5x_enc_dec_features,
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


##### LM Feature Converter #####


def _convert_to_t5x_lm_features(
    ex: dict[str, np.ndarray], packed: bool, bos_id: int, pad_id: int
) -> dict[str, np.ndarray]:
  """Feature converter for a T5X language model (decoder-only) architecture.

  The input dataset must have "targets" field only.

  One common usecase is to pre-train a decoder-only model with the standard
  language modeling objective (i.e., predict the next token given the previous
  ones) on a unlabeled text corpus which only has "targets". Then the
  pre-trained model can be fine-tuned on a supervised task, e.g., machine
  translation by concatenating "inputs" and "targets". For this use case,
  pre-train with LMFeatureConverter and fine-tune with PrefixLMFeatureConverter.

  Example: a packed dataset.

    ds = [{"targets": [3, 9, 1]}, {"targets": [4, 1]}]

    input_lengths = {"targets": 6}

    converted_ds = {
        "decoder_target_tokens": [3, 9, 1, 4, 1, 0],
         "decoder_input_tokens": [0, 3, 9, 0, 4, 0],
         "decoder_loss_weights": [1, 1, 1, 1, 1, 0],
            "decoder_positions": [0, 1, 2, 0, 1, 0],
          "decoder_segment_ids": [1, 1, 1, 2, 2, 0]
    }
  Note that two examples are packed together into one example.

  Args:
    ex: dict[str, np.ndarray] - the example to convert.
    packed: bool - whether the input example is packed.
    bos_id: int - token value to use to indicate beginning of sequence in
      decoder input tokens. 0 is commonly used.
    pad_id: int - token value to use for padding. 0 is commonly used.

  Returns:
    a converted example.
  """
  # targets_segment_id is present only for a packed dataset.
  decoder_input_tokens = _make_autoregressive_inputs(
      ex["targets"],
      sequence_ids=ex.get("targets_segment_ids", None),
      bos_id=bos_id,
  )

  d = {
      "decoder_target_tokens": ex["targets"],
      "decoder_input_tokens": decoder_input_tokens,
      "decoder_loss_weights": _get_non_padding_positions(ex["targets"], pad_id),
  }

  if packed:
    d["decoder_segment_ids"] = ex["targets_segment_ids"]
    d["decoder_positions"] = ex["targets_positions"]

  return d


def _update_runtime_args_for_t5x_lm_features(
    runtime_args: preprocessors_lib.AirIOInjectedRuntimeArgs,
    packed: bool,
):
  """Updates runtime args after applying _convert_to_t5x_lm_features.

  Replaces "input" and "targets" with features produced by
  `_convert_to_t5x_lm_features`, for use by downstream preprocessors. Both
  functions should be used together as an airio preprocessor, e.g.
  update_runtime_args = functools.partial(
      _update_runtime_args_for_t5x_lm_features,
      packed=true,
  )
  convert_features = functools.partial(
      _convert_to_t5x_lm_features,
      packed=True,
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
    packed: bool - whether the input examples are packed.

  Returns:
    updated preprocessors_lib.AirIOInjectedRuntimeArgs.
  """
  task_feature_lengths = runtime_args.sequence_lengths
  decoder_length = task_feature_lengths["targets"]
  model_feature_lengths = {
      "decoder_target_tokens": decoder_length,
      "decoder_input_tokens": decoder_length,
      "decoder_loss_weights": decoder_length,
  }
  if packed:
    model_feature_lengths["decoder_segment_ids"] = decoder_length
    model_feature_lengths["decoder_positions"] = decoder_length
  updated = runtime_args.clone()
  updated.sequence_lengths = model_feature_lengths
  return updated


##### Prefix LM Feature Converter #####


def _convert_to_t5x_prefix_lm_features(
    ex: dict[str, np.ndarray],
    loss_on_targets_only: bool,
    packed: bool,
    bos_id: int,
    pad_id: int,
    passthrough_feature_keys: Sequence[str],
) -> dict[str, np.ndarray]:
  """Feature converter for a T5X prefix language model architecture.

  The input dataset must have both "inputs" and "targets" fields. For language
  modeling objective with "targets" only dataset, use LMFeatureConverter.

  A decoder is a network which autoregressively produces an output sequence. It
  can be used for an input dataset which has a notion of "inputs" as well as
  "targets", (e.g., machine translation) by concatenating them to form the new
  targets. See Raffel et al. (2020), https://arxiv.org/abs/1910.10683, Section
  3.2.1 for a more detailed take on this topic.

  In the Prefix LM architecture discussed in Raffel et al. (2020), the tokens
  from the "inputs" portion are applied a fully visible self attention whereas
  those from "targets" are applied the causal self attention. This makes the
  contextual representation of the tokens from "inputs" bidirectional.

  In order to provide this information, this class provides an additional
  feature "decoder_causal_attention" on top of the model features returned by
  LMFeatureConverter. "decoder_causal_attention" is a binary mask where a value
  of 1 represents that the corresponding input token to the decoder belongs to
  the "inputs" before concatenation. Note that this attention mask is optional.
  For a model that does not require this feature, e.g., a fully causal masking
  on the concatenated sequence, the attention mask can be simply ignored.

  Note that "decoder_causal_attention" includes one additional position to the
  right. This is the position where the final token of the "inputs" (often an
  EOS) is read and the first "targets" token is predicted. This follows
  mesh_tensorflow/transformer/transformer.py

  Since "inputs" and "targets" are concatenated to form the new targets for the
  decoder, we might want to compute the loss only on the tokens that belong to
  "targets" before concatenation. This behavior is controlled by
  "loss_on_targets_only" attribute, which is passed to the constructor. By
  default, it is set to True. The resulting "decoder_loss_weights" therefore
  zeros out "inputs" portion as well as the padding tokens while having 1's on
  the targets token.

  Example 1: a packed dataset
  ```
  ds = [{"inputs": [7, 8, 5, 1], "targets": [3, 9, 1]},
        {"inputs": [8, 4, 9, 3, 1], "targets": [4, 1]}]

  task_feature_lengths = {"inputs": 7, "targets": 8}

  converted_ds = {
      "decoder_target_tokens": [7, 8, 5, 1, 3, 9, 1, 8, 4, 9, 3, 1, 4, 1, 0],
       "decoder_input_tokens": [0, 7, 8, 5, 1, 3, 9, 0, 8, 4, 9, 3, 1, 4, 0],
       "decoder_loss_weights": [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
          "decoder_positions": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0],
        "decoder_segment_ids": [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0],
   "decoder_causal_attention": [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
  }
  ```

  Example 2: unpacked dataset with extra long "inputs" `task_feature_lengths`
  ```
  ds = [{"inputs": [9, 4, 6, 1], "targets": [3, 9, 1]}]

  task_feature_lengths = {"inputs": 10, "targets": 4}

  converted_ds = {
         "decoder_target_tokens": [9, 4, 6, 1, 3, 9, 1, 0, 0, 0, 0, 0, 0, 0],
          "decoder_input_tokens": [0, 9, 4, 6, 1, 3, 9, 1, 0, 0, 0, 0, 0, 0],
          "decoder_loss_weights": [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
      "decoder_causal_attention": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  }

  Note that if the inputs length specified in `task_feature_lengths` is longer
  than the actual example length, the padding tokens are added after
  concatenation.
  ```

  Specifically, this function converts examples as described below:
  ```

  Suppose the original dataset is

  ds = [{"inputs": [9, 4, 6, 1], "targets": [3, 9, 1]}]

  Then the input features to this method (after padding) are

  features = {
                  "targets" = [9, 4, 6, 1, 3, 9, 1, 0, 0]
            "inputs_width" = [4, 4, 4, 4, 4, 4, 4, 0, 0]
    "inputs_width_add_pos" = [5, 5, 5, 5, 5, 5, 5, 0, 0]
  }

  where "inputs_width" is length of "inputs" tiled across length dimension and
  "inputs_width_add_pos" is the same except that it has one additional
  position.

  First the parent class's _convert_example method is used to obtain the
  standard LM features. Then we compute "decoder_causal_attention". For an
  unpacked dataset, we need to define the "positions" feature. Then,
  `positions < inputs_width_add_pos` gives the decoder_causal_attention.

      "inputs_width_add_pos" = [5, 5, 5, 5, 5, 5, 5, 0, 0]
                  "positions" = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                          <     ---------------------------
  "decoder_causal_attention" = [1, 1, 1, 1, 1, 0, 0, 0, 0]

  Then, we compute the loss weights, which requires isolating the "targets"
  position. Here we use "inputs_width" feature to filter out the "inputs"
  portion. `padding_mask` has 1's on inputs and targets and 0's on padding.
  Taking XOR filters out the targets portion.

        "inputs_width" = [4, 4, 4, 4, 4, 4, 4, 0, 0]
            "positions" = [0, 1, 2, 3, 4, 5, 6, 0, 0]
                    <     ---------------------------
                inputs = [1, 1, 1, 1, 0, 0, 0, 0, 0]
          padding_mask = [1, 1, 1, 1, 1, 1, 1, 0, 0]
                  xor    ---------------------------
  decoder_loss_weights = [0, 0, 0, 0, 1, 1, 1, 0, 0]

  Note that decoder_loss_weights is computed by the LMFeatureConverter.
  ```

  Args:
    ex: dict[str, np.ndarray] - the example to convert.
    loss_on_targets_only: whether to compute loss on tokens which belonged to
      "targets" before concatenation.
    packed: bool - whether the input example is packed.
    bos_id: int - token value to use to indicate beginning of sequence in
      decoder input tokens. 0 is commonly used.
    pad_id: int - token value to use for padding. 0 is commonly used.
    passthrough_feature_keys: Sequence[str] - A list of feature names to pass
      through.

  Returns:
    a converted example.
  """
  # First use the standard LM conversion.
  lm_features = _convert_to_t5x_lm_features(ex, packed, bos_id, pad_id)

  # Initialize the return dictionary with the lm features.
  d = dict(lm_features)

  if packed:
    positions = ex["targets_positions"]
  # Without packing, targets_positions field does not exist.
  else:
    positions = np.arange(ex["targets"].size)

  inputs_width = ex["inputs_width_add_pos"]
  # Binary mask where 1 represents a position in a non-causal attention region
  d["decoder_causal_attention"] = (positions < inputs_width).astype(
      ex["targets"].dtype
  )

  # When computing the loss weights with self.loss_on_targets_only = True, we
  # use features["inputs_width"], which encodes the number of "inputs" tokens.
  if loss_on_targets_only:
    # 1's on inputs and 0's on targets and padding.
    inputs = positions < ex["inputs_width"]

    # 1's on inputs and targets and 0's on padding.
    padding_mask = d["decoder_loss_weights"].astype(bool)

    # XOR picks targets only. See docstring for an example.
    d["decoder_loss_weights"] = np.logical_xor(inputs, padding_mask).astype(
        ex["targets"].dtype
    )

  d.update({k: ex[k] for k in passthrough_feature_keys})
  return d


def _update_runtime_args_for_t5x_prefix_lm_features(
    runtime_args: preprocessors_lib.AirIOInjectedRuntimeArgs,
    packed: bool,
    passthrough_feature_keys: Sequence[str],
):
  """Updates runtime args after applying _convert_to_t5x_prefix_lm_features.

  Replaces "input" and "targets" with features produced by
  `_convert_to_t5x_prefix_lm_features`, for use by downstream preprocessors.
  Both functions should be used together as an airio preprocessor, e.g.
  update_runtime_args = functools.partial(
      _update_runtime_args_for_t5x_prefix_lm_features,
      packed=True, passthrough_feature_keys=[]
  )
  convert_features = functools.partial(
      _convert_to_t5x_lm_features,
      packed=True,
      pad_id=0,
      bos_id=0,
      passthrough_feature_keys=[]
  )
  prep = preprocessors_lib.MapFnTransform(
      convert_features,
      update_runtime_args=update_runtime_args,
  )

  Args:
    runtime_args: preprocessors_lib.AirIOInjectedRuntimeArgs - args injected by
      airio at runtime.
    packed: bool - whether the input examples are packed.
    passthrough_feature_keys: Sequence[str] - A list of feature names to pass
      through.

  Returns:
    updated preprocessors_lib.AirIOInjectedRuntimeArgs.
  """
  task_feature_lengths = runtime_args.sequence_lengths
  decoder_length = task_feature_lengths["targets"]  # already concatenated.

  concat_args = runtime_args.clone()
  concat_args.sequence_lengths = {"targets": decoder_length}
  lm_model_feature_lengths = _update_runtime_args_for_t5x_lm_features(
      concat_args, packed=packed
  ).sequence_lengths
  model_feature_lengths = dict(lm_model_feature_lengths)
  model_feature_lengths["decoder_causal_attention"] = decoder_length
  for k in passthrough_feature_keys:
    model_feature_lengths[k] = task_feature_lengths[k]
  updated_args = runtime_args.clone()
  updated_args.sequence_lengths = model_feature_lengths
  return updated_args


def _concat_and_add_masks_for_prefix_lm(
    ex: dict[str, np.ndarray],
    passthrough_feature_keys: Sequence[str],
) -> dict[str, np.ndarray]:
  """Creates concatenated inputs and targets fields and adds masks."""
  inputs = ex["inputs"]
  targets = ex["targets"]
  # If the targets are empty, we add one padding target.
  targets = targets if targets.size else np.zeros(1, dtype=np.int32)

  # Width of the "inputs" portion in the concatenated sequence.
  width = inputs.size
  inputs_width = np.full([inputs.size + targets.size], width)

  # Width with an extra position to the right in the inputs mask. See
  # docstring for details.
  inputs_width_add_pos = np.full([inputs.size + targets.size], width + 1)

  d = {
      "targets": np.concatenate([inputs, targets], axis=-1),
      "inputs_width": inputs_width,
      "inputs_width_add_pos": inputs_width_add_pos,
  }
  d.update({k: ex[k] for k in passthrough_feature_keys})
  return d


def _concat_task_feature_lengths_for_prefix_lm(
    runtime_args: preprocessors_lib.AirIOInjectedRuntimeArgs,
    passthrough_feature_keys: Sequence[str],
) -> preprocessors_lib.AirIOInjectedRuntimeArgs:
  """Updates runtime_args after applying `_concat_and_add_masks_for_prefix_lm`.

  Both functions should be used together as an airio preprocessor, e.g.
  update_runtime_args = functools.partial(
      _concat_task_feature_lengths_for_prefix_lm
      passthrough_feature_keys=[]
  )
  concat_features = functools.partial(
      _concat_and_add_masks_for_prefix_lm,,
      passthrough_feature_keys=[]
  )
  prep = preprocessors_lib.MapFnTransform(
      concat_features,
      update_runtime_args=update_runtime_args,
  )

  Args:
    runtime_args: preprocessors_lib.AirIOInjectedRuntimeArgs - Args to update.
    passthrough_feature_keys: Sequence[str] - A list of feature names to pass
      through.

  Returns:
    updated preprocessors_lib.AirIOInjectedRuntimeArgs.
  """
  task_feature_lengths = runtime_args.sequence_lengths
  concat_length = sum(
      v
      for k, v in task_feature_lengths.items()
      if k not in passthrough_feature_keys
  )
  task_lengths = {
      "targets": concat_length,
      "inputs_width": concat_length,
      "inputs_width_add_pos": concat_length,
  }
  for k in passthrough_feature_keys:
    task_lengths[k] = task_feature_lengths[k]
  updated_args = runtime_args.clone()
  updated_args.sequence_lengths = task_lengths
  return updated_args


##### Utils #####


def _make_autoregressive_inputs(
    inputs: np.ndarray, sequence_ids: np.ndarray | None, bos_id: int
):
  """Generate inputs for an autoregressive model, by shifting the inputs.

  Modified from seqio.utils._make_autoregressive_inputs.

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


def _get_non_padding_positions(
    feature_value: np.ndarray, pad_id: int
) -> np.ndarray:
  return feature_value != pad_id
