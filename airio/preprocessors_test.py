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

"""Preprocessors tests."""

from absl.testing import absltest
from airio import data_sources
from airio import dataset_providers
from airio import preprocessors
import numpy as np


class PreprocessorsTest(absltest.TestCase):

  def test_map_fn_preprocessor(self):
    def _dataset_fn(split: str):
      del split
      return np.array(range(5))

    src = data_sources.FunctionDataSource(
        dataset_fn=_dataset_fn, splits=["train"]
    )

    def test_map_fn(ex):
      return ex + 1

    task = dataset_providers.Task(
        name="test_task",
        source=src,
        preprocessors=[preprocessors.MapFnTransform(test_map_fn)],
    )
    ds = task.get_dataset(None, "train", shuffle=False)
    self.assertListEqual(list(ds), list(range(1, 6)))


if __name__ == "__main__":
  absltest.main()
