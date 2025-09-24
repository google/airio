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

"""Import to top-level API."""

# pylint:disable=unused-import,g-importing-member

from ._src.core import data_sources
from ._src.core import dataset_iterators
from ._src.core import dataset_providers
from ._src.core import preprocessors
from ._src.core import tokenizer

# Individual members and functions.
from ._src.core.data_sources import DataSource
from ._src.core.dataset_iterators import AirIODatasetIterator
from ._src.core.dataset_providers import DatasetProviderBase
from ._src.core.dataset_providers import get_dataset
from ._src.core.dataset_providers import get_vocabularies
from ._src.core.dataset_providers import Mixture
from ._src.core.dataset_providers import ShardInfo
from ._src.core.dataset_providers import Task
from ._src.core.dataset_providers import TaskBuilder
from ._src.core.preprocessors import AirIOInjectedRuntimeArgs
from ._src.core.preprocessors import FilterFnTransform
from ._src.core.preprocessors import MapFnTransform
from ._src.core.preprocessors import RandomMapFnTransform
from ._src.core.tokenizer import TokenizerConfig
from ._src.core.vocabularies import Vocabulary

# Version number.
from .version import __version__
