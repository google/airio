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
from typing import Optional, Sequence

from airio import lazy_dataset_transforms
from airio import preprocessors as preprocessors_lib
from airio.common import packing
import grain.python as grain
import jax
from t5.data import preprocessors as t5_preps

lazy_dataset = grain.experimental.lazy_dataset


def filter_empty(ex, feature_key):
  return ex[feature_key].any()


def update_seed(seed):
  seed, _ = jax.random.split(seed)
  return seed


def span_corruption(
    dataset: lazy_dataset.LazyMapDataset,
    seed,
    output_features,
    runtime_args: preprocessors_lib.AirIOInjectedRuntimeArgs,
    mean_noise_span_length=3.0,
    noise_density=0.15,
    input_feature_key='inputs',
    merge_examples_to_reduce_padding=True,
    reserved_for_packing=None,
    passthrough_feature_keys: Optional[Sequence[str]] = None,
):
  """Final pretraining objective used in Raffel et al., 2019.

  Args:
    dataset: A tf.data.Dataset with dictionaries containing the key
      `input_feature_key`.
    seed: An initial seed to use for stateless random operations.
    output_features: mapping of keys to features.
    runtime_args: A AirIOInjectedRuntimeArgs obj containing sequence lengths.
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
      through to the output of this preprocessor. eg: ["tokens"]. Only supported
      if `merge_examples_to_reduce_padding` is set to False.

  Returns:
    a dataset
  """
  if not merge_examples_to_reduce_padding:
    raise NotImplementedError(
        'merge_examples_to_reduce_padding=False is not supported.'
    )
  sequence_length = runtime_args.sequence_lengths
  inputs_length = sequence_length[input_feature_key]
  if reserved_for_packing:
    inputs_length -= reserved_for_packing

  input_length, targets_length = t5_preps.random_spans_helper(
      extra_tokens_per_span_inputs=1,
      extra_tokens_per_span_targets=1,
      inputs_length=inputs_length,
      mean_noise_span_length=mean_noise_span_length,
      noise_density=noise_density,
  )

  if sequence_length['targets'] < targets_length:
    raise ValueError(
        f'Expected targets length for span corruption ({targets_length}) is '
        'greater than configured targets length '
        f'({sequence_length["targets"]})'
    )

  feature_key = 'targets'
  max_length = 65536
  pack_pool_size = 128

  filter_fn = functools.partial(filter_empty, feature_key=feature_key)
  select_random_chunk_fn = functools.partial(
      t5_preps.single_example_select_random_chunk,
      output_features=output_features,
      max_length=max_length,
      feature_key=feature_key,
      passthrough_feature_keys=passthrough_feature_keys,
  )
  packer = packing.NoamPacker(feature_lengths={feature_key: input_length})
  denoise_fn = functools.partial(
      t5_preps.single_example_denoise,
      output_features=output_features,
      inputs_fn=t5_preps.noise_span_to_unique_sentinel,
      targets_fn=t5_preps.nonnoise_span_to_unique_sentinel,
      noise_density=noise_density,
      noise_mask_fn=functools.partial(
          t5_preps.random_spans_noise_mask,
          mean_noise_span_length=mean_noise_span_length,
      ),
      input_feature_key=input_feature_key,
      passthrough_feature_keys=passthrough_feature_keys,
  )
  ds = dataset
  ds = ds.filter(filter_fn)
  ds = lazy_dataset_transforms.RandomMapFnLazyMapDataset(
      ds, select_random_chunk_fn, seed
  )
  ds = packing.PackLazyMapDataset(ds, pool_size=pack_pool_size, packer=packer)
  seed = update_seed(seed)
  ds = lazy_dataset_transforms.RandomMapFnLazyMapDataset(ds, denoise_fn, seed)
  return ds
