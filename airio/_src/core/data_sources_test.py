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

"""Tests for airio.data_sources."""

from unittest import mock

from absl.testing import absltest
from airio._src.core import data_sources

import multiprocessing


_SOURCE_NAME = "imdb_reviews"
_SOURCE_NUM_EXAMPLES = 3
_SOURCE_SPLITS = frozenset(["train", "test", "unsupervised"])


class DataSourceTest(absltest.TestCase):

  @mock.patch.multiple(data_sources.DataSource, __abstractmethods__=set())
  def test_protocol(self):
    source = data_sources.DataSource
    self.assertIsNone(source.get_data_source(self, split=""))
    self.assertIsNone(source.num_input_examples(self, split=""))


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
