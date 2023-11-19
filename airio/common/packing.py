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

"""Packing implementation."""

import collections
import copy
import dataclasses
import math
from typing import Any, Sequence, TypeVar, Tuple
from airio import preprocessors as preprocessors_lib
import airio.common.constants
import grain.python as grain
import jax
from jaxtyping import PyTree  # pylint: disable=g-importing-member
import numpy as np

lazy_dataset = grain.experimental.lazy_dataset
T = TypeVar("T")
SKIP_FEATURE = airio.common.constants.SKIP_FEATURE


@dataclasses.dataclass
class AirIOPackDatasetPreprocessor:
  """Packs a dataset based on sequence lengths provided via AirIOInjectedRuntimeArgs.

  Replcates algorithms used traditionally in SeqIO. See PackLazyMapDataset for
  more details. The output is a sparse dataset. Be careful when mixing after
  packing, because sparse datasets mixed using a LazyMapDataset impl will not
  match desired mixing proportions. Use a LazyIterDataset impl instead.

  Attributes:
    pool_size: The number of examples that are pooled together and packed. A
      larger pool_size leads to potentially better packing but requires more
      memory.
    num_partial_examples: The number of partially packed examples to maintain
      during packing. Larger num_partial_examples leads to potentially better
      packing but requires more memory.
    strict_packing: If turned on, examples are only packed together if combined
      lengths are less than or equal to given sequence lengths. If not, examples
      are combined if there is any available space, even if combined lengths
      exceed given sequence lengths. If turned on, users need to make sure to
      trim the dataset as a subsequent step. Turning off strict_packing leads to
      lesser padding, but should be used carefully and only on datasets with one
      feature. Otherwise, features in an example may become misaligned.
    add_marker_features: If turned on, marker features are added for each packed
      feature to help distinguish features belonging to different examples. For
      "feature", marker features added are "feature_segment_ids" and
      "feature_positions". For instance, if [1, 2, 3], [4, 5], and [6, 7, 8] are
      packed, the packed sequence is [1, 2, 3, 4, 5, 6, 7, 8], segment_ids are
      [1, 1, 1, 2, 2, 3, 3, 3], and positions are [0, 1, 2, 0, 1, 0, 1, 2].
  """

  pool_size: int
  num_partial_examples: int
  strict_packing: bool
  add_marker_features: bool

  def __call__(
      self,
      ds: lazy_dataset.LazyMapDataset,
      runtime_args: preprocessors_lib.AirIOInjectedRuntimeArgs,
  ) -> Tuple[
      lazy_dataset.LazyMapDataset, preprocessors_lib.AirIOInjectedRuntimeArgs
  ]:
    return PackLazyMapDataset(
        ds,
        feature_lengths=runtime_args.sequence_lengths,
        pool_size=self.pool_size,
        num_partial_examples=self.num_partial_examples,
        strict_packing=self.strict_packing,
        add_marker_features=self.add_marker_features,
    ), self.update_runtime_args(runtime_args)

  def update_runtime_args(
      self,
      runtime_args: preprocessors_lib.AirIOInjectedRuntimeArgs,
  ) -> preprocessors_lib.AirIOInjectedRuntimeArgs:
    """Updates runtime args with new sequence lengths for subsequent preprocessors."""
    seq_lens = runtime_args.sequence_lengths
    new_seq_lens = {}
    for feature in seq_lens:
      new_seq_lens[feature] = seq_lens[feature]
      if not seq_lens[feature] or seq_lens[feature] == SKIP_FEATURE:
        continue
      if self.add_marker_features:
        new_seq_lens[f"{feature}_segment_ids"] = seq_lens[feature]
        new_seq_lens[f"{feature}_positions"] = seq_lens[feature]
    return preprocessors_lib.AirIOInjectedRuntimeArgs(
        sequence_lengths=new_seq_lens,
        split=runtime_args.split,
    )


@dataclasses.dataclass(frozen=False)
class PackLazyMapDataset(lazy_dataset.LazyMapDataset[T]):
  """Packs a dataset. Produces a sparse dataset.

  Replicates algorithms used traditionally in SeqIO / Tensor2Tensor. Summary:
  + Groups of examples from the dataset are pooled / batched together.
  + Examples are packed within each pool.
  + The packed pools are "unbatched" to produce a dataset of packed examples.
      The number of packed example in each pool could range from 1 (all packed
      into one example) to pool_size (none packed).

  A sparse dataset is produced since the last step is a flat-map. Be careful
  when mixing after packing, because sparse datasets mixed using a
  LazyMapDataset impl will not match desired mixing proportions. Use a
  LazyIterDataset implementation instead.

  Attributes:
    parent: The dataset to pack.
    feature_lengths: Maximum feature lengths to pack to.
    pool_size: The number of examples that are pooled together and packed. A
      larger pool_size leads to potentially better packing but requires more
      memory.
    num_partial_examples: The number of partially packed examples to maintain
      during packing. Larger num_partial_examples leads to potentially better
      packing but requires more memory.
    strict_packing: If turned on, examples are only packed together if combined
      lengths are less than or equal to given sequence lengths. If not, examples
      are combined if there is any available space, even if combined lengths
      exceed given sequence lengths. If turned on, users need to make sure to
      trim the dataset as a subsequent step. Turning off strict_packing leads to
      lesser padding, but should be used carefully and only on datasets with one
      feature. Otherwise, features in an example may become misaligned.
    add_marker_features: If turned on, marker features are added for each packed
      feature to help distinguish features belonging to different examples. For
      "feature", marker features added are "feature_segment_ids" and
      "feature_positions". For instance, if [1, 2, 3], [4, 5], and [6, 7, 8] are
      packed, the packed sequence is [1, 2, 3, 4, 5, 6, 7, 8], segment_ids are
      [1, 1, 1, 2, 2, 3, 3, 3], and positions are [0, 1, 2, 0, 1, 0, 1, 2].
  """

  def __init__(
      self,
      parent: lazy_dataset.LazyMapDataset,
      feature_lengths: PyTree[int],
      pool_size: int,
      num_partial_examples: int,
      strict_packing: bool,
      add_marker_features: bool,
  ):
    super().__init__([parent])
    self._packed_ds = PoolLazyMapDataset(parent, pool_size)
    pack_flatmap = PackPoolFlatMap(
        feature_lengths,
        pool_size,
        num_partial_examples,
        strict_packing,
        add_marker_features,
    )
    self._packed_ds = lazy_dataset.FlatMapLazyMapDataset(
        self._packed_ds, pack_flatmap
    )

  @property
  def parent(self):
    return self._parent

  def __len__(self):
    return len(self.parent)

  def __getitem__(self, index: slice):
    if isinstance(index, slice):
      return self.slice(index)
    return self._packed_ds[index]


class PoolLazyMapDataset(lazy_dataset.LazyMapDataset[T]):
  """Pools consecutive examples from a LazyMapDataset.

  Example:
    ```
    ds = lazy_dataset.SourceLazyMapDataset(range(10))
    ds = PoolLazyMapDataset(ds, 3)
    list(ds)
    > [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    ```
  """

  def __init__(
      self,
      parent: lazy_dataset.LazyMapDataset,
      pool_size: int,
  ):
    super().__init__(parent)
    self._pool_size = pool_size
    self._length = math.ceil(len(self._parent) / self._pool_size)

  def __len__(self):
    return self._length

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    start = index * self._pool_size
    stop = min(len(self._parent), (index + 1) * self._pool_size)
    return [self._parent[i] for i in range(start, stop)]


class PackPoolFlatMap(grain.experimental.FlatMapTransform):
  """Packs and yields a pool of examples.

  Iterates through the pool of examples, and uses `MultiBinPacker` to fit them
  and yield packed examples. See `MultiBinPacker` for more details. A sparse
  dataset is produced since this is a flat-map. Be careful when mixing after
  packing, because sparse datasets mixed using a LazyMapDataset impl will not
  match desired mixing proportions. Use a LazyIterDataset impl instead.

  Attrs:
    feature_lengths: Maximum feature lengths to pack to.
    pool_size: The number of examples that are pooled together and packed. A
      larger pool_size leads to potentially better packing but requires more
      memory.
    num_partial_examples: The number of partially packed examples to maintain
      during packing. Larger num_partial_examples leads to potentially better
      packing but requires more memory.
    strict_packing: If turned on, examples are only packed together if combined
      lengths are less than or equal to given sequence lengths. If not, examples
      are combined if there is any available space, even if combined lengths
      exceed given sequence lengths. If turned on, users need to make sure to
      trim the dataset as a subsequent step. Turning off strict_packing leads to
      lesser padding, but should be used carefully and only on datasets with one
      feature. Otherwise, features in an example may become misaligned.
    add_marker_features: If turned on, marker features are added for each packed
      feature to help distinguish features belonging to different examples. For
      "feature", marker features added are "feature_segment_ids" and
      "feature_positions". For instance, if [1, 2, 3], [4, 5], and [6, 7, 8] are
      packed, the packed sequence is [1, 2, 3, 4, 5, 6, 7, 8], segment_ids are
      [1, 1, 1, 2, 2, 3, 3, 3], and positions are [0, 1, 2, 0, 1, 0, 1, 2].
  """

  def __init__(
      self,
      feature_lengths: PyTree[int],
      pool_size: int,
      num_partial_examples: int,
      strict_packing: bool,
      add_marker_features: bool,
  ):
    self.max_fan_out = pool_size
    self.feature_lengths = feature_lengths
    self.num_partial_examples = num_partial_examples
    self.strict_packing = strict_packing
    self.add_marker_features = add_marker_features

  def flat_map(self, example_pool):
    packer = MultiBinPacker(
        feature_lengths=self.feature_lengths,
        num_partial_examples=self.num_partial_examples,
        strict_packing=self.strict_packing,
        add_marker_features=self.add_marker_features,
    )
    for ex in example_pool:
      packed_examples = packer.fit_example(ex)
      for packed_example in packed_examples:
        yield packed_example
    while packer.has_partially_packed_examples():
      yield packer.get_packed_example()


class MultiBinPacker:
  """Container and utils to pack examples. Supports partially packed examples.

  This can be used to pack a fixed pool of examples (LazyMapDataset impl) or an
  iterative stream of examples (LazyIterDataset impl). A typical workflow would
  be:
    ```
    packer = MultiBinPacker(feature_lengths, num_partial_examples)
    for example in dataset:
      packed_examples = packer.fit_example(example)
      if packed_example: yield fully_packed
    while packer.has_partially_packed_examples():
      yield packer.get_packed_example()
    ```
  Under the hood, it closely resembles packing implementations traditionally
  used in SeqIO / Tensor2Tensor, although exact match should not be expected.
    + Maintains a queue of upto `num_partial_examples` partially packed
    examples. Whenever there are more than that, dequeues extra examples from
    the beginning.
    + For each incoming example, tries to fit it into existing partially packed
    examples (from left to right) and adds to first partially packed example
    that fits.
    + If no fit is found, appends a new partially packed example to the end of
    the queue.
    + A partially packed example is yielded and removed from the queue as soon
    as it's fully packed, rather than at the end (less memory usage).
    PackSequencesKOp may be different.

  Attrs:
    feature_lengths: Maximum feature lengths to pack to.
    num_partial_examples: The number of partially packed examples to maintain
      during packing. Larger num_partial_examples leads to potentially better
      packing but requires more memory.
    strict_packing: If turned on, examples are only packed together if combined
      lengths are less than or equal to given sequence lengths. If not, examples
      are combined if there is any available space, even if combined lengths
      exceed given sequence lengths. If turned on, users need to make sure to
      trim the dataset as a subsequent step. Turning off strict_packing leads to
      lesser padding, but should be used carefully and only on datasets with one
      feature. Otherwise, features in an example may become misaligned.
    add_marker_features: If turned on, marker features are added for each packed
      feature to help distinguish features belonging to different examples. For
      "feature", marker features added are "feature_segment_ids" and
      "feature_positions". For instance, if [1, 2, 3], [4, 5], and [6, 7, 8] are
      packed, the packed sequence is [1, 2, 3, 4, 5, 6, 7, 8], segment_ids are
      [1, 1, 1, 2, 2, 3, 3, 3], and positions are [0, 1, 2, 0, 1, 0, 1, 2].
  """

  feature_lengths: PyTree[int]
  num_partial_examples: int
  strict_packing: bool
  add_marker_features: bool
  _partially_packed_examples: collections.deque["PartiallyPackedExample"]
  _flat_feature_lengths: Sequence[int]

  def __init__(
      self,
      feature_lengths: PyTree[int],
      num_partial_examples: int,
      strict_packing: bool,
      add_marker_features: bool,
  ):
    self.feature_lengths = feature_lengths
    self.num_partial_examples = num_partial_examples
    self.strict_packing = strict_packing
    self.add_marker_features = add_marker_features
    self._partially_packed_examples = collections.deque[
        PartiallyPackedExample
    ]()
    self._flat_feature_lengths = flatten(self.feature_lengths)

  def has_partially_packed_examples(self):
    return len(self._partially_packed_examples)

  def get_packed_example(self):
    return self._partially_packed_examples.popleft().pack_and_unflatten(
        length_struct=self.feature_lengths,
        add_marker_features=self.add_marker_features,
    )

  def fit_example(self, ex):
    """Fits example into existing partially packed examples or creates new.

    Args:
      ex: An example to pack.

    Returns:
      A list of packed examples if a fully packed example was produced, or if
      partially packed examples exceeded the maximum allowed number.
    """
    packed_examples = []
    # First, release any extra partially packed examples.
    while len(self._partially_packed_examples) > self.num_partial_examples:
      packed_examples.append(self.get_packed_example())

    # Flatten and trim example to max packed length to make packing feasible.
    flat_ex = flatten(ex)
    flat_ex = trim_flattened(flat_ex, self._flat_feature_lengths)

    # Add if example fits an existing partially packed example; check if
    # resulting partially packed example becomes fully packed
    fits = False
    fully_packed: PartiallyPackedExample = None
    for partially_packed in self._partially_packed_examples:
      if partially_packed.example_fits(flat_ex, self.strict_packing):
        fits = True
        partially_packed.add_example(flat_ex)
        if partially_packed.is_fully_packed():
          fully_packed = partially_packed
        break

    # Add to result if example fit and became fully packed
    if fully_packed:
      assert fits
      packed = fully_packed.pack_and_unflatten(
          length_struct=self.feature_lengths,
          add_marker_features=self.add_marker_features,
      )
      self._partially_packed_examples.remove(fully_packed)
      packed_examples.append(packed)

    # If not, create new partially packed example; add to result if fully packed
    if not fits:
      partially_packed = PartiallyPackedExample(
          copy.copy(self._flat_feature_lengths)
      )
      partially_packed.add_example(flat_ex)
      if partially_packed.is_fully_packed():
        packed_examples.append(
            partially_packed.pack_and_unflatten(
                length_struct=self.feature_lengths,
                add_marker_features=self.add_marker_features,
            )
        )
      else:
        self._partially_packed_examples.append(partially_packed)

    return packed_examples


@dataclasses.dataclass
class PartiallyPackedExample:
  """Container and utils for partially packed examples.

  Operates on flattened examples and lengths using jax tree_flatten to simplify
  structures. Unflattens when producing packed examples.

  available_flat_lengths: A flattened (using jax.tree_util.tree_flatten) list
    of maximum feature lengths to pack to.
  """

  # A list of examples that can be packed together.
  _partially_packed_flat_example_list: list[Any]
  # Combined lengths of the partially packed example.
  _combined_flat_lengths: list[int]
  # Remaining space available along
  _available_flat_lengths: list[int]

  def __init__(
      self,
      available_flat_lengths: list[int],
  ):
    self._partially_packed_flat_example_list = []
    self._available_flat_lengths = available_flat_lengths
    self._combined_flat_lengths = [
        SKIP_FEATURE if l == SKIP_FEATURE else 0 for l in available_flat_lengths
    ]

  def add_example(self, flat_ex: Any):
    self._partially_packed_flat_example_list.append(flat_ex)
    for i in range(len(self._available_flat_lengths)):
      if self._available_flat_lengths[i] == SKIP_FEATURE:
        # Feature should not be packed.
        continue
      length = len(flat_ex[i])
      self._available_flat_lengths[i] = max(
          0, self._available_flat_lengths[i] - length
      )
      self._combined_flat_lengths[i] += length

  def example_fits(self, flat_ex: Any, strict_packing: bool = False):
    fits_any_feature = False
    for i in range(len(self._available_flat_lengths)):
      if self._available_flat_lengths[i] == SKIP_FEATURE:
        # Feature should not be packed.
        continue
      if self._available_flat_lengths[i] < len(flat_ex[i]) and strict_packing:
        return False
      if self._available_flat_lengths[i]:
        fits_any_feature = True
    return fits_any_feature

  def is_fully_packed(self):
    for length in self._available_flat_lengths:
      if length and length != SKIP_FEATURE:
        return False
    return True

  def pack_and_unflatten(
      self,
      length_struct: PyTree[int],
      add_marker_features: bool,
  ):
    """Produces a packed, unflattened example."""
    flat_elements = self._partially_packed_flat_example_list
    flat_packed_element = []
    for feature in range(len(self._combined_flat_lengths)):
      if self._combined_flat_lengths[feature] == SKIP_FEATURE:
        # Feature should not be packed.
        flat_packed_element.append(
            [flat_elements[i][feature] for i in range(len(flat_elements))]
        )
        continue
      sequence_length = self._combined_flat_lengths[feature]
      remaining_dims = flat_elements[0][feature].shape[1:]
      shape = [sequence_length, *remaining_dims]
      dtype = flat_elements[0][feature].dtype
      values = np.zeros(shape, dtype=dtype)
      segmentations = np.zeros(shape=[sequence_length], dtype=np.int32)
      positions = np.zeros(shape=[sequence_length], dtype=np.int32)

      start = 0
      for i in range(len(flat_elements)):
        length = min(len(flat_elements[i][feature]), sequence_length - start)
        end = start + length
        values[start:end] = flat_elements[i][feature][:length]
        segmentations[start:end] = i + 1
        positions[start:end] = np.arange(length)
        start += length
      if add_marker_features:
        flat_packed_element.append((values, segmentations, positions))
      else:
        flat_packed_element.append(values)
    packed_element = unflatten_as(length_struct, flat_packed_element)
    # Special treatment for dictionaries.
    if isinstance(packed_element, dict):
      for key in list(packed_element):
        value = packed_element[key]
        if isinstance(value, tuple) and len(value) == 3:
          packed_element[key] = value[0]
          packed_element[f"{key}_segment_ids"] = value[1]
          packed_element[f"{key}_positions"] = value[2]
    return packed_element


def trim_flattened(flat_ex, flat_lengths):
  for i in range(len(flat_lengths)):
    if flat_lengths[i] == SKIP_FEATURE:
      # Feature should not be trimmed.
      continue
    flat_ex[i] = flat_ex[i][: flat_lengths[i], ...]
  return flat_ex


def flatten(structure):
  return jax.tree_util.tree_flatten(structure)[0]


def unflatten_as(structure, flat_sequence):
  return jax.tree_util.tree_unflatten(
      jax.tree_util.tree_structure(structure), flat_sequence
  )
