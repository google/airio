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

"""Install AirIO."""

import os
import sys
import setuptools

# To enable importing version.py directly, we add its path to sys.path.
version_path = os.path.join(os.path.dirname(__file__), 'airio')
sys.path.append(version_path)
from version import __version__  # pylint: disable=g-import-not-at-top

# Get the long description from the README file.
with open('README.md') as fp:
  _LONG_DESCRIPTION = fp.read()

_jax_version = '0.4.16'
_jaxlib_version = '0.4.16'

setuptools.setup(
    name='airio',
    version=__version__,
    description=(
        'AirIO: Task-based datasets, preprocessing, and evaluation for sequence'
        ' models.'
    ),
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Google Inc.',
    author_email='no-reply@google.com',
    url='http://github.com/google/airio',
    license='Apache 2.0',
    packages=[
        # 'airio',
        'airio.core',
        'airio.pygrain',
        'airio.pygrain_common',
        'airio.tfgrain',
    ],
    package_dir={
        # 'airio': 'airio',
        'airio.core': 'airio/_src/core/',
        'airio.pygrain': 'airio/_src/pygrain/',
        'airio.pygrain.common': 'airio/_src/pygrain/common/',
        'airio.tfgrain': 'airio/_src/tfgrain/',
    },
    include_package_data=True,
    scripts=[],
    install_requires=[
        'absl-py',
        'clu @ git+https://github.com/google/CommonLoopUtils#egg=clu',
        'grain-nightly==0.0.6',
        f'jax >= {_jax_version}',
        f'jaxlib >= {_jaxlib_version}',
        (
            'jestimator @'
            ' git+https://github.com/google-research/jestimator#egg=jestimator'
        ),
        'numpy',
        'seqio @ git+https://github.com/google/seqio#egg=seqio',
        # Ping to a specific version to avoid endless backtracking during
        # pip dependency resolution.
        'tfds-nightly==4.9.2.dev202308090034',
    ],
    extras_require={
        'test': ['pytest'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='sequence preprocessing nlp machinelearning',
)
