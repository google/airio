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

"""Tests for AirIO example Tasks that may perform file reads, etc."""

from absl.testing import absltest
import airio
from airio.examples import tasks
from airio.grain.common import feature_converters


class TaskEquivalenceTest(absltest.TestCase):

  def test_c4_span_corruption_task(self):
    # TODO(b/314188951): Make this test hermetic by reading source and outputs
    # from test_data.
    # Prints an example of the airio C4 span corruption task.
    task = tasks.get_c4_v220_span_corruption_task()
    feature_converter = feature_converters.T5XEncDecFeatureConverter(
        pack=False,
        use_multi_bin_packing=False,
        passthrough_feature_keys=[],
        pad_id=0,
        bos_id=0,
    )
    sequence_lengths = {"inputs": 1024, "targets": 1024}
    ds = task.get_dataset(
        sequence_lengths,
        shuffle=False,
        seed=94043,
        runtime_preprocessors=feature_converter.get_preprocessors(),
        shard_info=airio.ShardInfo(index=0, num_shards=1024),
    )
    print("Element Spec:", ds.element_spec)
    for d in ds:
      print("Example:", d)
      break


if __name__ == "__main__":
  absltest.main()
