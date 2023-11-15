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
from typing import Any, Sequence, TypeVar
import grain.python as grain
import jax
from jaxtyping import PyTree  # pylint: disable=g-importing-member
import numpy as np

lazy_dataset = grain.experimental.lazy_dataset
T = TypeVar("T")


@dataclasses.dataclass(frozen=False)
class PackLazyMapDataset(lazy_dataset.LazyMapDataset[T]):
  """Packs a dataset. Produces a sparse dataset."""

  def __init__(self, parent, feature_lengths, pool_size, num_partial_examples):
    super().__init__([parent])
    self._packed_ds = PoolLazyMapDataset(parent, pool_size)
    pack_flatmap = PackPoolFlatMap(
        feature_lengths, pool_size, num_partial_examples
    )
    self._packed_ds = lazy_dataset.FlatMapLazyMapDataset(
        self._packed_ds, pack_flatmap
    )

  @property
  def parent(self):
    return self._parent

  def __len__(self) -> int:
    return len(self.parent)

  def __getitem__(self, index: slice) -> lazy_dataset.LazyMapDataset:
    if isinstance(index, slice):
      return self.slice(index)
    return self._packed_ds[index]


class PoolLazyMapDataset(lazy_dataset.LazyMapDataset[T]):
  """Pools consecutive examples from a LazyMapDataset."""

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
  """Packs and yields a pool of examples."""

  def __init__(self, feature_lengths, pool_size, num_partial_examples):
    self.max_fan_out = pool_size
    self.feature_lengths = feature_lengths
    self.num_partial_examples = num_partial_examples

  def flat_map(self, example_pool):
    packer = MultiBinPacker(
        feature_lengths=self.feature_lengths,
        num_partial_examples=self.num_partial_examples,
    )
    for ex in example_pool:
      packed_examples = packer.fit_example(ex)
      for packed_example in packed_examples:
        yield packed_example
    while packer.has_partially_packed_examples():
      yield packer.get_packed_example()


class MultiBinPacker:
  """Container and utils to pack examples. Supports partially packed examples.

  TODO(b/310685401): More documentation.

  A typical workflow would be:
    packer = MultiBinPacker(feature_lengths, num_partial_examples)
    for example in dataset:
      packed_examples = packer.fit_example(example)
      if packed_example: yield fully_packed
    while packer.has_partially_packed_examples():
      yield packer.get_packed_example()
  """

  feature_lengths: PyTree[int | None]
  num_partial_examples: int
  _partially_packed_examples: collections.deque["PartiallyPackedExample"]
  _flat_feature_lengths: Sequence[int | None]

  def __init__(self, feature_lengths, num_partial_examples):
    self.feature_lengths = feature_lengths
    self.num_partial_examples = num_partial_examples
    self._partially_packed_examples = collections.deque[
        PartiallyPackedExample
    ]()
    self._flat_feature_lengths = flatten(self.feature_lengths)

  def has_partially_packed_examples(self):
    return len(self._partially_packed_examples)

  def get_packed_example(self):
    return self._partially_packed_examples.popleft().pack_and_unflatten(
        flat_feature_lengths=self._flat_feature_lengths,
        length_struct=self.feature_lengths,
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
      if partially_packed.example_fits(flat_ex):
        fits = True
        partially_packed.add_example(flat_ex)
        if partially_packed.is_fully_packed():
          fully_packed = partially_packed
        break

    # Add to result if example fit and became fully packed
    if fully_packed:
      assert fits
      packed = fully_packed.pack_and_unflatten(
          flat_feature_lengths=self._flat_feature_lengths,
          length_struct=self.feature_lengths,
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
                flat_feature_lengths=self._flat_feature_lengths,
                length_struct=self.feature_lengths,
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
  """

  # A list of examples that can be packed together.
  _partially_packed_flat_example_list: list[Any]
  # Remaining space available along
  _available_flat_lengths: list[int | None]

  def __init__(
      self,
      available_flat_lengths: list[int | None],
  ):
    self._partially_packed_flat_example_list = []
    self._available_flat_lengths = available_flat_lengths

  def add_example(self, flat_ex: Any):
    self._partially_packed_flat_example_list.append(flat_ex)
    for i in range(len(self._available_flat_lengths)):
      if self._available_flat_lengths[i] is None:
        # Feature should not be packed.
        continue
      length = len(flat_ex[i])
      self._available_flat_lengths[i] -= length

  def example_fits(self, flat_ex: Any):
    for i in range(len(self._available_flat_lengths)):
      if self._available_flat_lengths[i] is None:
        # Feature should not be packed.
        continue
      if self._available_flat_lengths[i] < len(flat_ex[i]):
        return False
    return True

  def is_fully_packed(self):
    for length in self._available_flat_lengths:
      if length:
        return False
    return True

  def pack_and_unflatten(
      self,
      flat_feature_lengths: Sequence[int | None],
      length_struct: PyTree[int | None],
  ):
    """Produces a packed, unflattened example."""
    flat_elements = self._partially_packed_flat_example_list
    flat_packed_element = []
    for feature in range(len(flat_feature_lengths)):
      if flat_feature_lengths[feature] is None:
        # Feature should not be packed.
        flat_packed_element.append(
            [flat_elements[i][feature] for i in range(len(flat_elements))]
        )
        continue
      sequence_length = flat_feature_lengths[feature]
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
      flat_packed_element.append((values, segmentations, positions))
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
    flat_ex[i] = flat_ex[i][:flat_lengths[i], ...]
  return flat_ex


def flatten(structure):
  return jax.tree_util.tree_flatten(structure)[0]


def unflatten_as(structure, flat_sequence):
  return jax.tree_util.tree_unflatten(
      jax.tree_util.tree_structure(structure), flat_sequence
  )
