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

"""Preprocessors tests."""

import inspect
from unittest import mock

from absl.testing import absltest
from airio._src.core import preprocessors
import jax.random



class PreprocessorsTest(absltest.TestCase):

  @mock.patch.multiple(
      preprocessors.MapFnTransform, __abstractmethods__=set()
  )
  def test_protocol_map_fn_transform(self):
    transform = preprocessors.MapFnTransform
    self.assertIsNone(transform.map(self, element=""))

  @mock.patch.multiple(
      preprocessors.RandomMapFnTransform, __abstractmethods__=set()
  )
  def test_protocol_random_map_fn_transform(self):
    transform = preprocessors.RandomMapFnTransform
    self.assertIsNone(
        transform.random_map(self, element="", rng=jax.random.PRNGKey(42))
    )

  @mock.patch.multiple(
      preprocessors.FilterFnTransform, __abstractmethods__=set()
  )
  def test_protocol_filter_fn_transform(self):
    transform = preprocessors.FilterFnTransform
    self.assertIsNone(transform.filter(self, element=""))


class PreprocessorsWithInjectedArgsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._runtime_args = preprocessors.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 3},
        split="train",
    )

  def test_create_runtime_args_succeeds(self):
    runtime_args = preprocessors.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 3},
        split="train",
    )
    self.assertIsInstance(runtime_args, preprocessors.AirIOInjectedRuntimeArgs)

  def test_inject_runtime_args_to_fn_injects_args(self):
    def test_map_fn(ex, args: preprocessors.AirIOInjectedRuntimeArgs):
      del args
      return ex + 1

    runtime_args = preprocessors.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 3},
        split="train",
    )
    result = preprocessors.inject_runtime_args_to_fn(test_map_fn, runtime_args)
    inspect_result = inspect.signature(result).parameters
    result_args = inspect_result["args"]
    expected_injected_args = (
        "AirIOInjectedRuntimeArgs(sequence_lengths={'val': 3}, split='train')"
    )
    self.assertTrue(str(result_args).endswith(expected_injected_args))

  def test_inject_runtime_args_to_fn_without_runtime_args_returns_same(self):
    def test_map_fn(ex):
      return ex + 1

    expected_parameters = test_map_fn.__code__.co_varnames
    runtime_args = preprocessors.AirIOInjectedRuntimeArgs(
        sequence_lengths={"val": 3},
        split="train",
    )
    result = preprocessors.inject_runtime_args_to_fn(test_map_fn, runtime_args)
    result_parameters = result.__code__.co_varnames
    self.assertEqual(result_parameters, expected_parameters)



if __name__ == "__main__":
  absltest.main()
