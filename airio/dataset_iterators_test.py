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

"""Tests for airio.dataset_iterators."""

from unittest import mock

from absl.testing import absltest
from airio import dataset_iterators


class AirIODatasetIteratorsTest(absltest.TestCase):

  @mock.patch.multiple(
      dataset_iterators.AirIODatasetIterator, __abstractmethods__=set()
  )
  def test_abstract_class(self):
    iterator = dataset_iterators.AirIODatasetIterator
    with self.assertRaises(NotImplementedError):
      iterator.peek(self)
      iterator.peek_async(self)
      iterator.get_state(self)
      iterator.set_state(self, {})
      iterator.__repr__(self)


if __name__ == "__main__":
  absltest.main()
