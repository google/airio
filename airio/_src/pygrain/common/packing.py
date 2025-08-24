# Copyright 2025 The AirIO Authors.
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

import base64
import collections
import copy
import dataclasses
import math
from typing import Any, Callable, Sequence, TypeVar

from airio._src.core import preprocessors as core_preprocessors
from airio._src.pygrain.common import constants
import grain.python as grain
import jax
from jaxtyping import PyTree  # pylint: disable=g-importing-member
import numpy as np
import typing_extensions


T = TypeVar("T")
SKIP_FEATURE = constants.SKIP_FEATURE


class PackerProtocol(typing_extensions.Protocol):
  """Interface for a packer.

  A typical workflow is:
  ```
  packer = PackerImpl(...)
  examples = ...iterator or sequence of examples...
  for example in examples:
    packed_examples = packer.fit_example(example)
    for packed_example in packed_examples:
      yield packed_example
  while packer.has_partially_packed_examples():
    yield packer.get_packed_example()
  ```
  """

  @property
  def feature_lengths(self) -> PyTree[int]:
    """The maximum feature lengths to pack to."""
    ...

  @feature_lengths.setter
  def feature_lengths(self, feature_lengths: PyTree[int]):
    """A setter for feature_lengths.

    This is required because `feature_lengths` may not be known when
    initializing the packer. Implementations should enforce that these are set
    before packing.

    Args:
      feature_lengths: the maximum feature lengths to pack to.
    """
    ...

  def fit_example(self, ex: PyTree[np.ndarray]) -> Sequence[PyTree[np.ndarray]]:
    """Fit an example, return fully packed examples if any."""
    ...

  def has_partially_packed_examples(self) -> bool:
    """Returns True if there are partially packed examples."""
    ...

  def get_packed_example(self) -> PyTree[np.ndarray]:
    """Gets a packed example from the partially packed buffer, if present."""
    ...

  def get_packed_feature_lengths(self) -> PyTree[int]:
    """Returns feature lengths for packed examples.

    This should add marker features if any, e.g. segment_ids and positions.
    """
    ...

  def as_dict(self) -> dict[str, Any]:
    """Returns a dict representation of the packer."""

  @classmethod
  def from_dict(cls, state: dict[str, Any]):
    """Loads the packer from a dict representation."""


@dataclasses.dataclass
class AirIOPackDatasetMapPreprocessor:
  """Packs a `LazyMapDataset`.

  Creates pools of examples based on `pool_size` and applies the `packer` to
  each pool to produce packed examples. Examples are packed to sequence lengths
  provided via AirIOInjectedRuntimeArgs. See PackLazyMapDataset for more
  details. The output is a sparse dataset.

  Be careful when mixing after packing, because sparse datasets mixed using a
  LazyMapDataset impl will not match desired mixing proportions. Convert to
  iter or use a LazyIterDataset impl instead (note that a LazyIterDataset
  cannot be globally shuffled).

  Attributes:
    pool_size: The number of examples that are pooled together and packed. A
      larger pool_size leads to potentially better packing but requires more
      memory. A function to compute pool size based on packed feature lengths is
      also allowed.
    packer: A `PackerProtocol` impl to pack examples.
  """
  pool_size: int | Callable[[PyTree[int]], int]
  packer: PackerProtocol

  def __call__(
      self,
      ds: grain.MapDataset,
      runtime_args: core_preprocessors.AirIOInjectedRuntimeArgs,
      unused_rng: jax.Array,
  ) -> grain.MapDataset:
    self.packer.feature_lengths = runtime_args.sequence_lengths
    if not isinstance(self.pool_size, int):
      self.pool_size = self.pool_size(runtime_args.sequence_lengths)
    return PackLazyMapDataset(ds, pool_size=self.pool_size, packer=self.packer)

  def update_runtime_args(
      self,
      runtime_args: core_preprocessors.AirIOInjectedRuntimeArgs,
  ) -> core_preprocessors.AirIOInjectedRuntimeArgs:
    """Updates runtime args with new sequence lengths for subsequent preprocessors."""
    updated = dataclasses.replace(
        runtime_args, sequence_lengths=self.packer.get_packed_feature_lengths()
    )
    return updated


@dataclasses.dataclass
class AirIOPackDatasetIterPreprocessor:
  """Packs a `LazyIterDataset`.

  Iterates through the dataset and uses the `packer` to produce packed examples.
  Examples are packed to sequence lengths provided via AirIOInjectedRuntimeArgs.
  See PackLazyIterDataset for more details.

  Unlike LazyMapDataset packing impls, datasets packed with this impl can be
  mixed correctly because they don't produce empty elements. However, the packed
  dataset cannot be globally shuffled, hence this is unsuitable for certain
  packing algorithms like noam-packing.

  Attributes:
    packer: A `PackerProtocol` impl to pack examples.
  """
  packer: PackerProtocol

  def __call__(
      self,
      ds: grain.IterDataset,
      runtime_args: core_preprocessors.AirIOInjectedRuntimeArgs,
      unused_rng: jax.Array,
  ) -> grain.IterDataset:
    self.packer.feature_lengths = runtime_args.sequence_lengths
    return PackLazyIterDataset(ds, packer=self.packer)

  def update_runtime_args(
      self,
      runtime_args: core_preprocessors.AirIOInjectedRuntimeArgs,
  ) -> core_preprocessors.AirIOInjectedRuntimeArgs:
    """Updates runtime args with new sequence lengths for subsequent preprocessors."""
    updated = dataclasses.replace(
        runtime_args, sequence_lengths=self.packer.get_packed_feature_lengths()
    )
    return updated


@dataclasses.dataclass(frozen=False)
class PackLazyMapDataset(grain.MapDataset[T]):
  """Packs a dataset. Produces a sparse dataset.

  Replicates algorithms traditionally used in SeqIO / Tensor2Tensor. Summary:
  + Groups of examples from the dataset are pooled / batched together.
  + Examples are packed within each pool.
  + The packed pools are "unbatched" to produce a dataset of packed examples.
      The number of packed example in each pool could range from 1 (all packed
      into one example) to pool_size (none packed).

  A sparse dataset is produced since the last step is a flat-map. Be careful
  when mixing after packing, because sparse datasets mixed using a
  LazyMapDataset impl will not match desired mixing proportions. Convert to
  iter or use a LazyIterDataset impl instead (note that a LazyIterDataset
  cannot be globally shuffled).

  Attributes:
    parent: The dataset to pack.
    pool_size: The number of examples that are pooled together and packed. A
      larger pool_size leads to potentially better packing but requires more
      memory.
    packer: A `PackerProtocol` impl to pack examples.
  """

  def __init__(
      self, parent: grain.MapDataset, pool_size: int, packer: PackerProtocol
  ):
    super().__init__([parent])
    self._packed_ds = PoolLazyMapDataset(parent, pool_size)
    pack_flatmap = PackPoolFlatMap(pool_size, packer)
    self._packed_ds = grain.experimental.FlatMapMapDataset(
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


class PoolLazyMapDataset(grain.MapDataset[T]):
  """Pools consecutive examples from a LazyMapDataset.

  Example:
    ```
    ds = grain.MapDataset.source(range(10))
    ds = PoolLazyMapDataset(ds, 3)
    list(ds)
    > [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    ```
  """

  def __init__(
      self,
      parent: grain.MapDataset,
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

  Iterates through examples in the pool, and uses the provided `packer` to fit
  and yield packed examples. See packer impls for details on packing algorithms.
  A sparse dataset is produced since this is a flat-map. Be careful when mixing
  after packing, because sparse datasets mixed using a LazyMapDataset impl will
  not match desired mixing proportions. Convert to iter or use a LazyIterDataset
  impl instead (note that a LazyIterDataset cannot be globally shuffled).

  Attrs:
    pool_size: The number of examples that are pooled together and packed. A
      larger pool_size leads to potentially better packing but requires more
      memory.
    packer: A `PackerProtocol` impl to pack examples.
  """

  def __init__(
      self,
      pool_size: int,
      packer: PackerProtocol,
  ):
    self.max_fan_out = pool_size
    self.packer = packer

  def flat_map(self, example_pool):
    # Do not use the shared copy because `flat_map` may be called on multiple
    # `example_pool`s in parallel.
    packer = copy.deepcopy(self.packer)
    yielded = 0
    for ex in example_pool:
      packed_examples = packer.fit_example(ex)
      for packed_example in packed_examples:
        if yielded >= self.max_fan_out:
          # TODO(b/321990328): Raise an error here because it indicates that
          # packed lengths are not properly configured, and let users explicitly
          # override.
          return
        yielded += 1
        yield packed_example
    while packer.has_partially_packed_examples():
      if yielded >= self.max_fan_out:
        return
      yielded += 1
      yield packer.get_packed_example()


class _PackLazyDatasetIterator(grain.DatasetIterator):
  """See PackLazyIterDataset for details.

  Warning: This object is not threadsafe!
  """

  def __init__(
      self,
      parent: grain.DatasetIterator,
      packer: PackerProtocol,
  ):
    super().__init__()
    self._parent_iter = parent
    self._packer = packer
    self._packer_type = type(packer)
    self._packed_examples = collections.deque[PyTree[np.ndarray]]()

  # pytype:disable=attribute-error
  def __next__(self):
    if self._packed_examples:
      return self._packed_examples.popleft()
    try:
      while not self._packed_examples:
        ex = next(self._parent_iter)
        for ex in self._packer.fit_example(ex):
          self._packed_examples.append(ex)
      return self._packed_examples.popleft()
    except StopIteration as e:
      if self._packed_examples:
        return self._packed_examples.popleft()
      if self._packer.has_partially_packed_examples():
        return self._packer.get_packed_example()
      raise e
  # pytype:enable=attribute-error

  def get_state(self):
    return {
        "parent": self._parent_iter.get_state(),
        "packer": self._packer.as_dict(),
        "packed_examples": [save_np_tree(ex) for ex in self._packed_examples],
    }

  def set_state(self, state):
    self._parent_iter.set_state(state["parent"])
    self._packer = self._packer_type.from_dict(state["packer"])
    self._packed_examples = collections.deque[PyTree[np.ndarray]](
        [load_np_tree(t) for t in state["packed_examples"]]
    )


@dataclasses.dataclass(frozen=False)
class PackLazyIterDataset(grain.IterDataset[T]):
  """Packs a `LazyIterDataset`.

  Iterates through parent dataset and applies the packer to examples. Yields
  packed examples returned by the packer.

  Unlike LazyMapDataset packing impls, datasets packed with this impl can be
  mixed correctly because they don't produce empty elements. However, the packed
  dataset cannot be globally shuffled, hence this is unsuitable for certain
  packing algorithms like noam-packing.

  Attributes:
    parent: The dataset to pack.
    packer: A `PackerProtocol` impl to pack examples.
  """

  def __init__(self, parent: grain.IterDataset, packer: PackerProtocol):
    super().__init__([parent])
    self._packer = packer

  def __iter__(self) -> grain.DatasetIterator:
    return _PackLazyDatasetIterator(
        iter(self._parent), self._packer
    )  # pytype: disable=wrong-arg-types


class MultiBinPacker:
  """Container and utils to pack examples. Supports partially packed examples.

  This can be used to pack a fixed pool of examples (LazyMapDataset impl) or an
  iterative stream of examples (LazyIterDataset impl). A typical workflow would
  be:
    ```
    packer = MultiBinPacker(feature_lengths, num_partial_examples)
    for example in dataset:
      packed_examples = packer.fit_example(example)
      if packed_examples:
        for packed in packed_examples:
          yield packed
    while packer.has_partially_packed_examples():
      yield packer.get_packed_example()
    ```
  Under the hood, it closely resembles packing implementations traditionally
  used in SeqIO / Tensor2Tensor, although an exact match should not be expected.
    + Maintains a queue of up to `num_partial_examples` partially packed
    examples. Whenever there are more, extra examples are dequeued from the
    beginning.
    + For each incoming example, tries to fit it into existing partially packed
    examples (from left to right) and adds to first partially packed example
    that fits.
    + If no fit is found, appends a new partially packed example to the end of
    the queue.
    + A partially packed example is yielded and removed from the queue as soon
    as it's fully packed, rather than at the end (less memory usage).

  Attrs:
    num_partial_examples: The number of partially packed examples to maintain
      during packing. Larger num_partial_examples leads to potentially better
      packing but requires more memory.
    feature_lengths: Maximum feature lengths to pack to. Set to None (default)
      if not known when initializing the MultiBinPacker, and can be set later.
      Must be set before packing examples.
  """

  def __init__(
      self,
      num_partial_examples: int,
      feature_lengths: PyTree[int] | None = None,
  ):
    self.num_partial_examples = num_partial_examples
    self._feature_lengths = feature_lengths
    self._partially_packed_examples = collections.deque[
        PartiallyPackedExample
    ]()
    self._flat_feature_lengths = flatten(feature_lengths)

  @property
  def feature_lengths(self):
    return self._feature_lengths

  @property
  def flat_feature_lengths(self):
    return self._flat_feature_lengths

  @feature_lengths.setter
  def feature_lengths(self, feature_lengths):
    self._feature_lengths = feature_lengths
    self._flat_feature_lengths = flatten(feature_lengths)

  def as_dict(self):
    return {
        "num_partial_examples": copy.deepcopy(self.num_partial_examples),
        "feature_lengths": copy.deepcopy(self._feature_lengths),
        "partially_packed_examples": [
            p.as_dict() for p in self._partially_packed_examples
        ],
    }

  @classmethod
  def from_dict(cls, state):
    obj = cls(state["num_partial_examples"])
    if state["feature_lengths"]:
      obj.feature_lengths = state["feature_lengths"]
    obj._partially_packed_examples = collections.deque[PartiallyPackedExample]([
        PartiallyPackedExample.from_dict(p)
        for p in state["partially_packed_examples"]
    ])
    return obj

  def has_partially_packed_examples(self):
    return len(self._partially_packed_examples)

  def get_packed_example(self):
    if not self._partially_packed_examples:
      raise ValueError("No packed examples available.")
    return unflatten_packed_example(  # pytype: disable=wrong-arg-types
        self._partially_packed_examples.popleft().pack(),
        length_struct=self.feature_lengths,
    )

  def get_packed_feature_lengths(self):
    if not self.feature_lengths:
      raise ValueError("feature_lengths must be set before packing.")
    seq_lens = self.feature_lengths
    new_seq_lens = {}
    for feature in seq_lens:
      new_seq_lens[feature] = seq_lens[feature]
      if not seq_lens[feature] or seq_lens[feature] == SKIP_FEATURE:
        continue
      new_seq_lens[f"{feature}_segment_ids"] = seq_lens[feature]
      new_seq_lens[f"{feature}_positions"] = seq_lens[feature]
    return new_seq_lens

  def fit_example(self, ex: PyTree[np.ndarray]) -> Sequence[PyTree[np.ndarray]]:
    """Fits example into existing partially packed examples or creates new.

    Args:
      ex: An example to pack.

    Returns:
      A list of packed examples if a fully packed example was produced, or if
      partially packed examples exceeded the maximum allowed number.
    """
    if not self.feature_lengths:
      raise ValueError("feature_lengths must be set before packing.")
    packed_examples = []
    # First, release any extra partially packed examples.
    while len(self._partially_packed_examples) > self.num_partial_examples:
      packed_examples.append(self.get_packed_example())

    # Flatten and trim example to max packed length to make packing feasible.
    flat_ex = flatten(ex)
    flat_ex = trim_flattened(flat_ex, self.flat_feature_lengths)

    # Add if example fits an existing partially packed example; check if
    # resulting partially packed example becomes fully packed
    fits = False
    fully_packed: PartiallyPackedExample = None
    fully_packed_idx = None
    for idx, partially_packed in enumerate(self._partially_packed_examples):
      if partially_packed.example_fits(flat_ex):
        fits = True
        partially_packed.add_example(flat_ex)
        if partially_packed.is_fully_packed():
          fully_packed = partially_packed
          fully_packed_idx = idx
        break

    # Add to result if example fit and became fully packed
    if fully_packed:
      assert fits
      packed = unflatten_packed_example(  # pytype: disable=wrong-arg-types
          fully_packed.pack(),
          length_struct=self.feature_lengths,
      )
      del self._partially_packed_examples[fully_packed_idx]
      # self._partially_packed_examples.remove(fully_packed)
      packed_examples.append(packed)

    # If not, create new partially packed example; add to result if fully packed
    if not fits:
      partially_packed = PartiallyPackedExample(
          copy.copy(self.flat_feature_lengths)
      )
      partially_packed.add_example(flat_ex)
      if partially_packed.is_fully_packed():
        packed_examples.append(
            unflatten_packed_example(
                partially_packed.pack(),
                length_struct=self.feature_lengths,
            )
        )
      else:
        self._partially_packed_examples.append(partially_packed)
    return packed_examples


class NoamPacker:
  """An implementation of the popular Noam Packing algorithm.

  This can be used to pack a fixed pool of examples (LazyMapDataset impl) or an
  iterative stream of examples (LazyIterDataset impl). A typical workflow would
  be:
    ```
    packer = NoamPacker(feature_lengths)
    for example in dataset:
      packed_examples = packer.fit_example(example)
      if packed_examples:
        for packed in packed_examples:
          yield packed
    while packer.has_partially_packed_examples():
      yield packer.get_packed_example()
    ```

  This implements the following logic invoked in
  `t5.data.preprocessors.span_corruption`:
  ```
  from t5.data import preprocessors
  ds = preprocessors.reduce_concat_tokens(ds)
      ds, feature_key="feature", batch_size=128
  )
  ds = preprocessors.split_tokens(
      ds,
      feature_key="feature",
      min_tokens_per_segment=None,
      max_tokens_per_segment=feature_length,
      passthrough_feature_keys=passthrough_feature_keys,
  )
  ```
  Noam packing appends sequences from consecutive examples until they exceed
  desired packing lengths, then slices them, producing packed examples with zero
  padding.  The remainder of a sliced example forms the beginning of the next
  packed sequence. There may not be enough examples to fill the desired packed
  length at the end of a pool / stream of examples, so there may be occasional
  examples with padding.

  A few things to keep in mind:
    + Noam-packing should be used when examples have a single feature or
    multiple correlated features (e.g. token ids, token weights). Otherwise,
    sequences may get sliced to different lengths, causing misalignment.
    + Noam-packing does not produce segment_ids and positions features, thus
    treating the packed example as a single example without boundaries. Support
    for marker features can be added.
    + The append and slice approach leads to examples being possibly sliced and
    distributed over multiple packed examples. Hence, shuffling the packed
    dataset is highly recommended.
    + When using this packer on a pool of examples (LazyMapDataset), it is
    possible (although rare) that the packed dataset has more examples than the
    unpacked dataset. In this case, packed examples may be capped because the
    maximum possible size of a LazyMapDataset must be statically known.
  """

  def __init__(
      self,
      feature_lengths: PyTree[int] | None = None,
  ):
    self._feature_lengths = feature_lengths
    self._flat_feature_lengths = flatten(feature_lengths)
    self._partially_packed_example: PartiallyPackedExample = None
    if feature_lengths:
      self._partially_packed_example = PartiallyPackedExample(
          copy.copy(self._flat_feature_lengths)
      )

  @property
  def feature_lengths(self):
    return self._feature_lengths

  @property
  def flat_feature_lengths(self):
    return self._flat_feature_lengths

  @feature_lengths.setter
  def feature_lengths(self, feature_lengths):
    self._feature_lengths = feature_lengths
    self._flat_feature_lengths = flatten(feature_lengths)
    self._partially_packed_example = PartiallyPackedExample(
        copy.copy(self._flat_feature_lengths)
    )

  def as_dict(self):
    return {
        "feature_lengths": copy.deepcopy(self._feature_lengths),
        "partially_packed_example": self._partially_packed_example.as_dict(),
    }

  @classmethod
  def from_dict(cls, state):
    obj = cls()
    if state["feature_lengths"]:
      obj.feature_lengths = state["feature_lengths"]
      obj._partially_packed_example = PartiallyPackedExample.from_dict(
          state["partially_packed_example"]
      )
    return obj

  # pytype:disable=attribute-error
  def has_partially_packed_examples(self):
    return not self._partially_packed_example.is_empty()

  def get_packed_example(self):
    if self._partially_packed_example.is_empty():
      raise ValueError("No packed examples available.")
    return self._slice_and_keep_remainder()

  def get_packed_feature_lengths(self):
    if not self.feature_lengths:
      raise ValueError("feature_lengths must be set before packing.")
    return self.feature_lengths

  def fit_example(self, ex: PyTree[np.ndarray]) -> Sequence[PyTree[np.ndarray]]:
    """Fits example into existing partially packed example.

    Args:
      ex: An example to pack.

    Returns:
      A list of packed examples if fully packed examples were produced.
    """
    if not self.feature_lengths:
      raise ValueError("feature_lengths must be set before packing.")
    assert not self._partially_packed_example.is_fully_packed()

    packed_examples = []

    flat_ex = flatten(ex)
    self._partially_packed_example.add_example(flat_ex)
    # Keep slicing packed examples from the partially packed example until there
    # is space available for a new example. Multiple slices may be needed if
    # sequences are significantly longer than packed lengths.
    while self._partially_packed_example.is_fully_packed():
      packed_examples.append(self._slice_and_keep_remainder())

    return packed_examples

  def _slice_and_keep_remainder(self):
    """Packs and slices at required lengths; adds remaining to new partially packed example."""
    packed = self._partially_packed_example.pack()
    # remove marker features.
    packed = remove_packed_marker_features_flattened(
        packed, self.flat_feature_lengths
    )
    trimmed = trim_flattened(copy.copy(packed), self.flat_feature_lengths)
    remainder = remainder_after_trim_flattened(
        copy.copy(packed), self.flat_feature_lengths
    )
    self._partially_packed_example = PartiallyPackedExample(
        copy.copy(self.flat_feature_lengths)
    )
    if not is_empty_flattened(remainder, self.flat_feature_lengths):
      self._partially_packed_example.add_example(remainder)
    return unflatten_packed_example(trimmed, length_struct=self.feature_lengths)
  # pytype:enable=attribute-error


@dataclasses.dataclass
class PartiallyPackedExample:
  """Container and utils for partially packed examples.

  Operates on flattened examples and lengths using jax tree_flatten to simplify
  structures. Unflattens when producing packed examples.

  available_flat_lengths: A flattened (using jax.tree_util.tree_flatten) list
    of maximum feature lengths to pack to.
  """

  # A list of examples that can be packed together.
  _partially_packed_flat_example_list: list[PyTree[np.ndarray]]
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

  def as_dict(self):
    d = {
        "partially_packed_flat_example_list": (
            save_np_tree(self._partially_packed_flat_example_list)
        ),
        "combined_flat_lengths": copy.deepcopy(self._combined_flat_lengths),
        "available_flat_lengths": copy.deepcopy(self._available_flat_lengths),
    }
    return d

  @classmethod
  def from_dict(cls, state):
    obj = cls(state["available_flat_lengths"])
    obj._partially_packed_flat_example_list = load_np_tree(
        state["partially_packed_flat_example_list"]
    )
    obj._combined_flat_lengths = state["combined_flat_lengths"]
    return obj

  def add_example(self, flat_ex: Any):
    """Adds example to partially packed example list."""
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

  def example_fits(self, flat_ex: Any):
    for i in range(len(self._available_flat_lengths)):
      if self._available_flat_lengths[i] == SKIP_FEATURE:
        # Feature should not be packed.
        continue
      if self._available_flat_lengths[i] < len(flat_ex[i]):
        return False
    return True

  def is_fully_packed(self):
    for length in self._available_flat_lengths:
      if length and length != SKIP_FEATURE:
        return False
    return True

  def is_empty(self):
    for length in self._combined_flat_lengths:
      if length and length != SKIP_FEATURE:
        return False
    return True

  def pack(self):
    """Produces a packed, flattened example."""
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
      flat_packed_element.append((values, segmentations, positions))
    return flat_packed_element


def trim_flattened(flat_ex: list[np.ndarray], flat_lengths: list[int]):
  for i in range(len(flat_lengths)):
    if flat_lengths[i] == SKIP_FEATURE:
      # Feature should not be trimmed.
      continue
    flat_ex[i] = flat_ex[i][: flat_lengths[i], ...]
  return flat_ex


def remainder_after_trim_flattened(
    flat_ex: list[np.ndarray], flat_lengths: list[int]
):
  for i in range(len(flat_lengths)):
    if flat_lengths[i] == SKIP_FEATURE:
      # Feature should not be trimmed. Keep last item from the sequence.
      flat_ex[i] = flat_ex[i][-1]
      continue
    flat_ex[i] = flat_ex[i][flat_lengths[i]:, ...]
  return flat_ex


def is_empty_flattened(flat_ex, flat_lengths):
  for i in range(len(flat_lengths)):
    if flat_lengths[i] == SKIP_FEATURE:
      # Feature should be ignored.
      continue
    if flat_ex[i].size:
      return False
  return True


def remove_packed_marker_features_flattened(packed, flat_lengths):
  packed_without_markers = []
  for i in range(len(flat_lengths)):
    if flat_lengths[i] == SKIP_FEATURE:
      # Feature does not have markers.
      packed_without_markers.append(packed[i])
      continue
    packed_without_markers.append(packed[i][0])
  return packed_without_markers


def flatten(structure):
  return jax.tree_util.tree_flatten(structure)[0]


def unflatten_as(structure, flat_sequence):
  return jax.tree_util.tree_unflatten(
      jax.tree_util.tree_structure(structure), flat_sequence
  )


def unflatten_packed_example(
    packed_flat_example: Sequence[np.ndarray], length_struct: PyTree[int]
) -> PyTree[np.ndarray]:
  """Unflattens packed example."""
  packed_element = unflatten_as(length_struct, packed_flat_example)
  # Special treatment for dictionaries.
  if isinstance(packed_element, dict):
    for key in list(packed_element):
      value = packed_element[key]
      if isinstance(value, tuple) and len(value) == 3:
        packed_element[key] = value[0]
        packed_element[f"{key}_segment_ids"] = value[1]
        packed_element[f"{key}_positions"] = value[2]
  return packed_element


def save_np_tree(tree: PyTree[np.ndarray]):
  return {
      "data": jax.tree.map(
          lambda x: base64.b64encode(x.tobytes()).decode(), tree
      ),
      "dtypes": jax.tree.map(lambda x: x.dtype.descr[0][1], tree),
      "shapes": jax.tree.map(lambda x: x.shape, tree),
  }


def load_np_tree(tree: dict[str, PyTree[Any]]):
  return jax.tree.map(
      lambda x, y, z: np.frombuffer(
          base64.b64decode(x), dtype=np.dtype(y)
      ).reshape(z),
      tree["data"],
      tree["dtypes"],
      tree["shapes"],
  )


NoamPackMapPreprocessor = AirIOPackDatasetMapPreprocessor(
    packer=NoamPacker(), pool_size=128
)

SingleBinTruePackMapPreprocessor = AirIOPackDatasetMapPreprocessor(
    packer=MultiBinPacker(num_partial_examples=1),
    pool_size=lambda feature_lengths: max(128, max(flatten(feature_lengths))),
)

MultiBinTruePackMapPreprocessor = AirIOPackDatasetMapPreprocessor(
    packer=MultiBinPacker(num_partial_examples=1000),
    pool_size=lambda feature_lengths: max(128, max(flatten(feature_lengths))),
)

NoamPackIterPreprocessor = AirIOPackDatasetIterPreprocessor(packer=NoamPacker())

SingleBinTruePackIterPreprocessor = AirIOPackDatasetIterPreprocessor(
    packer=MultiBinPacker(num_partial_examples=1),
)

MultiBinTruePackIterPreprocessor = AirIOPackDatasetIterPreprocessor(
    packer=MultiBinPacker(num_partial_examples=1000),
)
