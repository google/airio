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

"""Data Source implementations for AirIO."""

import typing
from typing import Iterable, Protocol


@typing.runtime_checkable
class DataSource(Protocol):
  """Interface for data sources wrappers with multiple splits support."""

  splits: Iterable[str] = None

  def get_data_source(self, split: str) -> ...:
    ...

  def num_input_examples(self, split: str) -> int:
    ...
