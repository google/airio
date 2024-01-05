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

"""Preprocessors for T5 Tasks."""

import functools
import math
from typing import Any, Mapping, Optional, Protocol, Sequence

from absl import logging
import gin
import seqio
import tensorflow.compat.v2 as tf

# We disable no-value-for-parameter since the seqio.map_over_dataset leads to
# a false positive when seeds are provided.
# pylint:disable=no-value-for-parameter
AUTOTUNE = tf.data.AUTOTUNE

FeatureType = Mapping[str, tf.Tensor]


def span_corruption(dataset,
                    sequence_length,
                    output_features,
                    mean_noise_span_length=3.0,
                    noise_density=0.15,
                    input_feature_key='inputs',
                    merge_examples_to_reduce_padding=True,
                    reserved_for_packing=None,
                    passthrough_feature_keys: Optional[Sequence[str]] = None):
  """Final pretraining objective used in Raffel et al., 2019.

  Args:
    dataset: A tf.data.Dataset with dictionaries containing the key
      `input_feature_key`.
    sequence_length: dict mapping of feature key to int length for that feature.
    output_features: mapping of keys to features.
    mean_noise_span_length: the mean number of tokens per masked span per
      example.
    noise_density: what fraction of the tokens to mask.
    input_feature_key: which feature to use from the dataset as the input text
      tokens.
    merge_examples_to_reduce_padding: if True, combines multiple input examples
      to reduce padding.
    reserved_for_packing: if specified, reduces the desired inputs length by the
      specified amount to enable multiple examples to be packed together
      downstream.
    passthrough_feature_keys: a sequence of feature names that should be passed
      through to the output of this preprocessor. eg: ["tokens"]. Only
      supported if `merge_examples_to_reduce_padding` is set to False.

  Returns:
    a dataset
  """
  inputs_length = sequence_length[input_feature_key]
  if reserved_for_packing:
    inputs_length -= reserved_for_packing

  input_length, targets_length = random_spans_helper(
      extra_tokens_per_span_inputs=1,
      extra_tokens_per_span_targets=1,
      inputs_length=inputs_length,
      mean_noise_span_length=mean_noise_span_length,
      noise_density=noise_density)

  if sequence_length['targets'] < targets_length:
    raise ValueError(
        f'Expected targets length for span corruption ({targets_length}) is '
        f'greater than configured targets length '
        f"({sequence_length['targets']})")

  ds = dataset
  ds = select_random_chunk(
      ds,
      output_features=output_features,
      feature_key='targets',
      max_length=65536,
      passthrough_feature_keys=passthrough_feature_keys)
  if merge_examples_to_reduce_padding:
    if passthrough_feature_keys:
      raise ValueError('passthrough_feature_keys not supported with '
                       'merge_examples_to_reduce_padding=True. '
                       f'Got: {passthrough_feature_keys}')
    ds = reduce_concat_tokens(ds, feature_key='targets', batch_size=128)
  ds = split_tokens(
      ds,
      feature_key='targets',
      min_tokens_per_segment=None,
      max_tokens_per_segment=input_length,
      passthrough_feature_keys=passthrough_feature_keys)
  ds = denoise(
      ds,
      output_features,
      inputs_fn=noise_span_to_unique_sentinel,
      targets_fn=nonnoise_span_to_unique_sentinel,
      noise_density=noise_density,
      noise_mask_fn=functools.partial(
          random_spans_noise_mask,
          mean_noise_span_length=mean_noise_span_length),
      input_feature_key=input_feature_key,
      passthrough_feature_keys=passthrough_feature_keys)
  return ds


def single_example_select_random_chunk(
    features: FeatureType,
    seed: tf.Tensor,
    output_features: Mapping[str, seqio.Feature],
    max_length: Optional[int] = None,
    feature_key: str = 'targets',
    additional_feature_keys: Optional[Sequence[str]] = None,
    passthrough_feature_keys: Optional[Sequence[str]] = None,
    sequence_length: Optional[Mapping[str, int]] = None,
    uniform_random_start: bool = False,
    min_length: Optional[int] = None) -> FeatureType:
  """Token-preprocessor to extract one span of at most `max_length` tokens.

  If the token sequence is longer than `max_length`, then we return a random
  subsequence.  Otherwise, we return the full sequence.

  This is generally followed by split_tokens.

  Args:
    features: Single example with `feature_key` containing a tokenized sequence.
    seed: Random seed to be used.
    output_features: Mapping of keys to features.
    max_length: Typically specified in gin configs, takes priority over
      sequence_length.
    feature_key: Which feature to use from the dataset.
    additional_feature_keys: Additional features to use. The same chunk will be
      selected from these features as from the one specified in feature_key,
      so they should all have the same length.
    passthrough_feature_keys: Additional keys to pass through unchanged.
    sequence_length: Used if max_length is not specified. Typically passed in
      by the data pipeline. feature_key will be used to select the length.
    uniform_random_start: If True, will select a starting point in
      [-max_length + 1, n_tokens). If False, will select one of a set of chunks
      offset by max_length. Both of these starting points try to ensure each
      token has an equal probability of being included.
    min_length: If specified, lengths of chunks will be selected uniformly at
      random from [min_length, max_length]. Note that chunks can end up shorter
      than min_length if at the beginning or end of the sequence.

  Returns:
    The features of the selected chunk.
  """
  if passthrough_feature_keys:
    chunk_keys = set([feature_key] + (additional_feature_keys or []))
    overlap_keys = chunk_keys & set(passthrough_feature_keys)
    if overlap_keys:
      raise ValueError(
          f'chunk keys {overlap_keys} also included in passthrough keys')

  if max_length is None and sequence_length is not None:
    max_length = sequence_length[feature_key]
    if output_features[feature_key].add_eos:
      # Leave room to insert an EOS token.
      max_length -= 1
  if max_length is None:
    raise ValueError('Must specify max_length or sequence_length.')

  seeds = tf.unstack(tf.random.experimental.stateless_split(seed))
  tokens = features[feature_key]
  n_tokens = tf.shape(tokens)[0]
  if min_length is not None:
    length = tf.random.stateless_uniform([],
                                         minval=min_length,
                                         maxval=max_length,
                                         dtype=tf.int32,
                                         seed=seeds[0])
  else:
    length = max_length
  if uniform_random_start:
    start = tf.random.stateless_uniform(
        [],
        minval=-length + 1,  # pylint:disable=invalid-unary-operand-type
        maxval=n_tokens,
        dtype=tf.int32,
        seed=seeds[1])
    end = tf.minimum(start + length, n_tokens)
    start = tf.maximum(start, 0)
  else:
    num_segments = tf.cast(
        tf.math.ceil(
            tf.cast(n_tokens, tf.float32) / tf.cast(length, tf.float32)),
        tf.int32)
    start = length * tf.random.stateless_uniform(
        [], maxval=num_segments, dtype=tf.int32, seed=seeds[1])
    end = tf.minimum(start + length, n_tokens)
  chunk = {feature_key: tokens[start:end]}
  if additional_feature_keys is not None:
    for k in additional_feature_keys:
      with tf.control_dependencies([
          tf.assert_equal(
              tf.shape(tokens)[0],
              tf.shape(features[k])[0],
              message=(f'Additional feature {k} is not the same size as '
                       f'{feature_key} along axis 0 in select_random_chunk().'))
      ]):
        chunk[k] = features[k][start:end]
  if passthrough_feature_keys is not None:
    for k in passthrough_feature_keys:
      chunk[k] = features[k]
  return chunk


def select_random_chunk(dataset: tf.data.Dataset,
                        output_features: Mapping[str, seqio.Feature],
                        max_length: Optional[int] = None,
                        feature_key: str = 'targets',
                        additional_feature_keys: Optional[Sequence[str]] = None,
                        passthrough_feature_keys: Optional[
                            Sequence[str]] = None,
                        sequence_length: Optional[Mapping[str, int]] = None,
                        uniform_random_start: bool = False,
                        min_length: Optional[int] = None,
                        **unused_kwargs) -> tf.data.Dataset:
  """SeqIO wrapper for single_example_select_random_chunk()."""

  @seqio.map_over_dataset(num_seeds=1)
  def _my_fn(x, seed):
    return single_example_select_random_chunk(
        x,
        seed,
        output_features=output_features,
        max_length=max_length,
        feature_key=feature_key,
        additional_feature_keys=additional_feature_keys,
        passthrough_feature_keys=passthrough_feature_keys,
        sequence_length=sequence_length,
        uniform_random_start=uniform_random_start,
        min_length=min_length)

  # Filter empty examples.
  dataset = dataset.filter(lambda x: tf.not_equal(tf.size(x[feature_key]), 0))
  return _my_fn(dataset)


def reduce_concat_tokens(dataset,
                         feature_key='targets',
                         batch_size=128,
                         **unused_kwargs):
  """Token-preprocessor to concatenate multiple unrelated documents.

  If we want to generate examples of exactly the right length,
  (to avoid wasting space on padding), then we use this function, folowed by
  split_tokens.

  Args:
    dataset: a tf.data.Dataset with dictionaries containing the key feature_key.
    feature_key: an string
    batch_size: an integer - how many documents to concatenate into one

  Returns:
    a dataset
  """
  dataset = dataset.map(
      lambda x: {feature_key: x[feature_key]}, num_parallel_calls=AUTOTUNE)
  dataset = dataset.padded_batch(batch_size, padded_shapes={feature_key: [-1]})
  def _my_fn(x):
    tokens = tf.reshape(x[feature_key], [-1])
    # strip padding
    tokens = tf.boolean_mask(tokens, tf.cast(tokens, tf.bool))
    return {feature_key: tokens}

  return dataset.map(_my_fn, num_parallel_calls=AUTOTUNE)


def split_tokens(dataset: tf.data.Dataset,
                 min_tokens_per_segment: Optional[int] = None,
                 max_tokens_per_segment: int = gin.REQUIRED,
                 feature_key: str = 'targets',
                 additional_feature_keys: Optional[Sequence[str]] = None,
                 passthrough_feature_keys: Optional[Sequence[str]] = None,
                 **unused_kwargs) -> tf.data.Dataset:
  """Split examples into multiple examples each.

  The intended use case is to break up long examples for use in unsupervised
  transfer-learning.

  This function is generally preceded by select_random_chunk.

  If min_tokens_per_segment is provided, the segment length is chosen randomly
  per document from a log-uniform distribution.  If min_tokens_per_segment is
  None, then the segment length is max_tokens_per_segment (except for a possibly
  shorter last segment in each document).

  Args:
    dataset: a tf.data.Dataset with dictionaries containing the key feature_key.
    min_tokens_per_segment: an optional integer
    max_tokens_per_segment: an integer, the maximum number of tokens in each
      segment. Only the final segment may be shorter.
    feature_key: a string, the feature to split
    additional_feature_keys: Additional features to split. The same chunk size
      will be used, so they should be the same size as feature_key.
    passthrough_feature_keys: Features to pass through without any splitting.

  Returns:
    a dataset
  """
  if passthrough_feature_keys:
    split_keys = set([feature_key] + (additional_feature_keys or []))
    overlap_keys = split_keys & set(passthrough_feature_keys)
    if overlap_keys:
      raise ValueError(
          f'split keys {overlap_keys} also included in passthrough keys')

  @seqio.map_over_dataset(num_seeds=1)
  def _split_tokens(x, seed):
    """Split one token sequence into multiple sequences."""
    tokens = x[feature_key]
    n_tokens = tf.shape(tokens)[0]
    if min_tokens_per_segment is None:
      length = max_tokens_per_segment
    else:
      # pick a length - log-uniformly distributed
      length = tf.cast(
          tf.exp(
              tf.random.stateless_uniform(
                  [],
                  minval=math.log(min_tokens_per_segment),
                  maxval=math.log(max_tokens_per_segment),
                  seed=seed
              )
          ),
          tf.int32)

    # Pad to a multiple of length, then use tf.reshape to split up the tokens
    # into num_segments segments each of the given length.
    num_segments = tf.cast(
        tf.math.ceil(
            tf.cast(n_tokens, tf.float32) / tf.cast(length, tf.float32))
        ,
        tf.int32)
    padding = num_segments * length - tf.shape(tokens)[0]
    feature_keys_to_split = [feature_key]
    orig_lengths = {}
    outputs = {}
    if additional_feature_keys is not None:
      feature_keys_to_split.extend(additional_feature_keys)
    for k in feature_keys_to_split:
      with tf.control_dependencies([
          tf.assert_equal(
              tf.shape(tokens)[0],
              tf.shape(x[k])[0],
              message=(f'Additional feature {k} is not the same size as '
                       f'{feature_key} along axis 0 in split_tokens().')
          )
      ]):
        shape = tf.shape(x[k])[1:]
        shape_list = x[k].shape[1:]
        padded = tf.pad(
            x[k],
            tf.concat([[[0, padding]],
                       tf.zeros([len(shape_list), 2], dtype=tf.int32)],
                      axis=0))
        orig_lengths[k] = tf.concat(
            [tf.repeat(length, num_segments - 1), [length - padding]], axis=0)
        outputs[k] = tf.reshape(
            padded, tf.concat([[-1, length], shape], axis=0))

    # To avoid memory issues, don't just replicate the passthrough features
    # for every segment; use tf.data to do it so the copies don't get
    # instantiated all at once.
    outputs_ds = tf.data.Dataset.from_tensor_slices(outputs)
    orig_lengths_ds = tf.data.Dataset.from_tensor_slices(orig_lengths)
    if passthrough_feature_keys:
      passthrough = {k: v for k, v in x.items()
                     if k in passthrough_feature_keys}
      passthrough_ds = tf.data.Dataset.from_tensors(passthrough).repeat(
          tf.cast(num_segments, tf.int64))
      return tf.data.Dataset.zip((outputs_ds, orig_lengths_ds, passthrough_ds))
    else:
      return tf.data.Dataset.zip((outputs_ds, orig_lengths_ds))

  def _strip_padding_and_merge_passthrough(
      inputs, orig_lengths, passthrough=None):
    output = {}
    for k, v in inputs.items():
      output[k] = v[:orig_lengths[k]]
    if passthrough:
      for k, _ in passthrough.items():
        output[k] = passthrough[k]
    return output

  # Filter empty examples.
  dataset = dataset.filter(lambda x: tf.not_equal(tf.size(x[feature_key]), 0))

  dataset = _split_tokens(dataset).flat_map(lambda z: z)
  dataset = dataset.map(
      _strip_padding_and_merge_passthrough, num_parallel_calls=AUTOTUNE)

  return dataset


def random_spans_helper(inputs_length=gin.REQUIRED,
                        noise_density=gin.REQUIRED,
                        mean_noise_span_length=gin.REQUIRED,
                        extra_tokens_per_span_inputs=gin.REQUIRED,
                        extra_tokens_per_span_targets=gin.REQUIRED,
                        verbose=False):
  """Training parameters to avoid padding with random_spans_noise_mask.

  When training a model with random_spans_noise_mask, we would like to set the
  other training hyperparmeters in a way that avoids padding.  This function
  helps us compute these hyperparameters.

  We assume that each noise span in the input is replaced by
  extra_tokens_per_span_inputs sentinel tokens, and each non-noise span in the
  targets is replaced by extra_tokens_per_span_targets sentinel tokens.

  This function tells us the required number of tokens in the raw example (for
  split_tokens()) as well as the length of the encoded targets.

  Note that this function assumes the inputs and targets will have EOS appended
  and includes that in the reported length.

  Args:
    inputs_length: an integer - desired length of the tokenized inputs sequence
    noise_density: a float
    mean_noise_span_length: a float
    extra_tokens_per_span_inputs: an integer
    extra_tokens_per_span_targets: an integer
    verbose: a bool indicating whether to log sequence lengths
  Returns:
    tokens_length: length of original text in tokens
    targets_length: an integer - length in tokens of encoded targets sequence
  """
  def _tokens_length_to_inputs_length_targets_length(tokens_length):
    num_noise_tokens = int(round(tokens_length * noise_density))
    num_nonnoise_tokens = tokens_length - num_noise_tokens
    num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
    # inputs contain all nonnoise tokens, sentinels for all noise spans
    # and one EOS token.
    return (
        num_nonnoise_tokens +
        num_noise_spans * extra_tokens_per_span_inputs + 1,
        num_noise_tokens +
        num_noise_spans * extra_tokens_per_span_targets + 1)

  tokens_length = inputs_length - 1
  while (_tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0]
         <= inputs_length):
    tokens_length += 1
  inputs_length, targets_length = (
      _tokens_length_to_inputs_length_targets_length(tokens_length))
  # minor hack to get the targets length to be equal to inputs length
  # which is more likely to have been set to a nice round number.
  if noise_density == 0.5 and targets_length > inputs_length:
    tokens_length -= 1
    targets_length -= 1
  if verbose:
    logging.info(
        'tokens_length=%s inputs_length=%s targets_length=%s '
        'noise_density=%s mean_noise_span_length=%s ',
        tokens_length, inputs_length, targets_length,
        noise_density, mean_noise_span_length)
  return tokens_length, targets_length


class DenoiseNoiseMaskFn(Protocol):

  def __call__(self, num_tokens: tf.Tensor, noise_density: float,
               seeds: tf.Tensor) -> tf.Tensor:
    """Computes the boolean makes. Seeds should have shape [2, 2]."""


class DenoiseInputsFn(Protocol):

  def __call__(self, tokens: tf.Tensor, noise_mask: tf.Tensor, vocabulary,
               seeds: tf.Tensor) -> tf.Tensor:
    """Computes the input tokens. Seeds should have shape [2, 2]."""


class DenoiseTargetsFn(Protocol):

  def __call__(self, tokens: tf.Tensor, noise_mask: tf.Tensor, vocabulary,
               seeds: tf.Tensor) -> tf.Tensor:
    """Computes the target tokens. Seeds should have shape [2, 2]."""


def single_example_denoise(features: FeatureType,
                           seed: tf.Tensor,
                           *,
                           output_features: Mapping[str, Any],
                           noise_density: float,
                           noise_mask_fn: DenoiseNoiseMaskFn,
                           inputs_fn: DenoiseInputsFn,
                           targets_fn: Optional[DenoiseTargetsFn] = None,
                           passthrough_feature_keys: Optional[
                               Sequence[str]] = None,
                           input_feature_key: str = 'inputs') -> FeatureType:
  """Preprocessing function for self-supervised denoising tasks.

  This function takes a dataset containing "targets" sequences,
  and turns each sequence into a dictionary containing:
  {
     "inputs": noisy version of the original sequence
     "targets": the full original sequence or missing parts of original sequence
  }

  In particular, for each sequence, we choose a boolean noise_mask identifying
  which tokens in the sequence to corrupt, as defined by the given
  noise_mask_fn.

  Given the sequence and the noise mask, we generate the inputs and targets
  using the given inputs_fn and targets_fn respectively.

  The self-supervised tasks vary along these axes:
    - noise_density: What fraction of the tokens to select as noise
    - noise_mask_fn: What pattern should the noise mask follow
         (iid, regular segments, etc.)
    - inputs_fn: How to apply the noise
         (drop noise tokens, replace with sentinels, etc.)
    - targets_fn: How to represent the output
         (full sequence, only non-noise tokens, etc.)

  Note: Some functionality has been deleted, which we may or may not want to
  restore at a later date.  The code for this functionality can be found in
  the deleted code for this CL.  In particular:
    - mixture of masking and random replacement
    - task labels prepended to the inputs

  Args:
    features: Flat dictionary of features.
    seed: Random seed to use.
    output_features: a dict mapping feature name to t5.data.Feature.
    noise_density: a float
    noise_mask_fn: a function from (length, noise_density) -> boolean mask
    inputs_fn: a function from (tokens, noise_mask, vocabulary) -> tokens
    targets_fn: a function from (tokens, noise_mask, vocabulary) -> tokens
    passthrough_feature_keys: names of additional features to include in output
    input_feature_key: name of feature to use as inputs

  Returns:
    A preprocessed features.
  """
  if passthrough_feature_keys and (input_feature_key in passthrough_feature_keys
                                   or 'targets' in passthrough_feature_keys):
    raise ValueError(
        f"passthrough keys cannot contain '{input_feature_key}' or 'targets'")

  seeds = tf.unstack(tf.random.experimental.stateless_split(seed, 6))
  tokens = features['targets']
  vocabulary = output_features['targets'].vocabulary
  if (input_feature_key in output_features and
      vocabulary != output_features[input_feature_key].vocabulary):
    raise ValueError(
        'denoise creates inputs based on tokenized targets but was applied '
        'to a task that uses different vocabularies for inputs and targets.')
  noise_mask = noise_mask_fn(tf.size(tokens), noise_density, seeds=seeds[:2])
  inputs = inputs_fn(tokens, noise_mask, vocabulary, seeds=seeds[2:4])
  if targets_fn:
    targets = targets_fn(tokens, noise_mask, vocabulary, seeds=seeds[4:6])
  else:
    targets = tokens
  return {
      input_feature_key: inputs,
      'targets': targets,
      **{
          k: features[k]
          for k in features
          if passthrough_feature_keys and k in passthrough_feature_keys
      }
  }


def denoise(dataset,
            output_features,
            noise_density=gin.REQUIRED,
            noise_mask_fn=gin.REQUIRED,
            inputs_fn=gin.REQUIRED,
            targets_fn=None,
            passthrough_feature_keys: Optional[Sequence[str]] = None,
            input_feature_key='inputs',
            **unused_kwargs):
  """SeqIO wrapper for single_example_denoise()."""

  @seqio.map_over_dataset(num_seeds=1)
  def my_fn(features, seed):
    return single_example_denoise(
        features,
        seed,
        output_features=output_features,
        noise_density=noise_density,
        noise_mask_fn=noise_mask_fn,
        inputs_fn=inputs_fn,
        targets_fn=targets_fn,
        passthrough_feature_keys=passthrough_feature_keys,
        input_feature_key=input_feature_key)

  return my_fn(dataset)


def random_spans_noise_mask(length,
                            noise_density,
                            seeds,
                            mean_noise_span_length=3.0,
                            random_roll=False):
  """Noise mask consisting of random spans of noise tokens.

  The number of noise tokens and the number of noise spans and non-noise spans
  are determined deterministically as follows:

    num_noise_tokens = round(length * noise_density)
    num_nonnoise_spans = num_noise_spans = round(
       num_noise_tokens / mean_noise_span_length)

  Spans alternate between non-noise and noise, beginning with non-noise.
  Subject to the above restrictions, all masks are equally likely.

  Args:
    length: an int32 scalar (length of the incoming token sequence)
    noise_density: a float - approximate density of output mask
    seeds: an int32 Tensor, shaped (2, 2)
    mean_noise_span_length: a number
    random_roll: bool, whether to roll the mask by a random integer offset in
      [0, length). Set random_roll to True to get a more uniform distribution
      of masked positions. Specifically, when random_roll is False (default) and
      a single span is enough to satisfy the noise density requirement, this
      fuction masks only the last few positions.

  Returns:
    a boolean tensor with shape [length]
  """

  if noise_density == 0.0:
    return tf.zeros(length, tf.bool)

  orig_length = length
  # increase length to avoid degeneracy
  length = tf.maximum(length, 2)
  def to_int(x):
    return tf.cast(x, tf.int32)
  def to_float(x):
    return tf.cast(x, tf.float32)
  num_noise_tokens = to_int(tf.round(to_float(length) * noise_density))
  # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
  num_noise_tokens = tf.minimum(tf.maximum(num_noise_tokens, 1), length - 1)
  num_noise_spans = to_int(
      tf.round(to_float(num_noise_tokens) / mean_noise_span_length))
  # avoid degeneracy by ensuring positive number of noise spans
  num_noise_spans = tf.maximum(num_noise_spans, 1)
  num_nonnoise_tokens = length - num_noise_tokens
  # pick the lengths of the noise spans and the non-noise spans
  def _random_segmentation(num_items, num_segments, seed):
    """Partition a sequence of items randomly into non-empty segments.

    Args:
      num_items: an integer scalar > 0
      num_segments: an integer scalar in [1, num_items]
      seed: an integer seed
    Returns:
      a Tensor with shape [num_segments] containing positive integers that add
      up to num_items
    """
    first_in_segment = tf.pad(
        seqio.stateless_shuffle(
            to_int(tf.range(num_items - 1) < num_segments - 1),
            seed),
        [[1, 0]])
    segment_id = tf.cumsum(first_in_segment)
    segment_length = tf.math.segment_sum(tf.ones_like(segment_id), segment_id)
    return segment_length
  noise_span_lengths = _random_segmentation(
      num_noise_tokens, num_noise_spans, seeds[0])
  nonnoise_span_lengths = _random_segmentation(
      num_nonnoise_tokens, num_noise_spans, seeds[1])
  interleaved_span_lengths = tf.reshape(
      tf.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
      [num_noise_spans * 2])
  span_starts = tf.cumsum(interleaved_span_lengths)[:-1]
  span_start_indicator = tf.math.unsorted_segment_sum(
      tf.ones_like(span_starts), span_starts, length)
  span_num = tf.cumsum(span_start_indicator)
  is_noise = tf.equal(span_num % 2, 1)

  mask = is_noise[:orig_length]

  if random_roll:
    roll_seed = (seeds[0][0]+seeds[1][1], seeds[0][1]-seeds[1][0])  # new seed.
    # Roll the mask by a random offset e.g. for offset=2: [1,2,3,4] => [3,4,1,2]
    offset = tf.random.stateless_uniform(
        [1], seed=roll_seed, dtype=tf.int32, minval=0, maxval=length)[0]
    mask = tf.roll(mask, shift=offset, axis=0)

  return mask


def sentinel_id(vocabulary, return_value=None):
  """Token ID to use as a sentinel.

  By default, we use the last token in the vocabulary.

  Args:
    vocabulary: a t5.data.vocabularies.Vocabulary
    return_value: an optional integer
  Returns:
    an integer
  """
  if return_value is not None:
    return return_value
  return vocabulary.vocab_size - 1


def noise_span_to_unique_sentinel(tokens, noise_mask, vocabulary, seeds):
  """Replace each run of consecutive noise tokens with a different sentinel.

  The idea here is to be able to align the dropped spans in the inputs
  with the markers in the targets.

  We want to generate training examples like
  "We hold X to be Y that" -> "X these truths Y self evident Z"

  Sentinels assigned in decreasing order within the sequence starting at
  vocabulary.size - 1.  That is, we appropriate the last tokens in the
  vocabulary for additional use as sentinels.

  TODO(noam): we may want to try enlarging the vocabulary and leaving room
  for the sentinels instead.  However, this requires enlarging the embedding
  tables in the model, so that is a bigger change.

  Args:
    tokens: a 1d integer Tensor
    noise_mask: a boolean Tensor with the same shape as tokens
    vocabulary: a vocabulary.Vocabulary
    seeds: an unused int32 Tensor
  Returns:
    a Tensor with the same shape and dtype as tokens
  """
  del seeds

  prev_token_is_noise = tf.pad(noise_mask[:-1], [[1, 0]])

  first_noise_tokens = tf.logical_and(
      noise_mask, tf.logical_not(prev_token_is_noise))
  subsequent_noise_tokens = tf.logical_and(noise_mask, prev_token_is_noise)

  sentinel = sentinel_id(vocabulary) + 1 - tf.cumsum(
      tf.cast(first_noise_tokens, tokens.dtype))

  tokens = tf.where(first_noise_tokens, sentinel, tokens)
  return tf.boolean_mask(tokens, tf.logical_not(subsequent_noise_tokens))


def nonnoise_span_to_unique_sentinel(tokens, noise_mask, vocabulary, seeds):
  return noise_span_to_unique_sentinel(
      tokens, tf.logical_not(noise_mask), vocabulary, seeds)
