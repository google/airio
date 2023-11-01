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

"""Example AirIO Tasks for reference and tests."""

import functools
import logging
from typing import Dict

from absl import logging
from airio import data_sources
from airio import dataset_providers
from airio import preprocessors
from airio import tokenizer
import babel
from seqio import vocabularies
import tensorflow_datasets as tfds

_DEFAULT_EXTRA_IDS = 100
_DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"
_DEFAULT_VOCAB = vocabularies.SentencePieceVocabulary(
    _DEFAULT_SPM_PATH, _DEFAULT_EXTRA_IDS
)


def get_wmt_19_ende_v003_task(
    tokenizer_configs: Dict[str, tokenizer.TokenizerConfig] | None = None,
) -> dataset_providers.Task:
  """Returns an AirIO Task for WMT 19 en/de dataset."""
  # source
  builder_config = tfds.translate.wmt19.Wmt19Translate.builder_configs["de-en"]
  tfds_version = "1.0.0"
  tfds_name = f"wmt19_translate/{builder_config.name}:{tfds_version}"

  # tokenizer
  if tokenizer_configs is None:
    tokenizer_configs = {
        "inputs": tokenizer.TokenizerConfig(vocab=_DEFAULT_VOCAB),
        "targets": tokenizer.TokenizerConfig(vocab=_DEFAULT_VOCAB),
    }

  return dataset_providers.Task(
      "wmt19_ende_v003",
      source=data_sources.TfdsDataSource(
          tfds_name=tfds_name, splits=["train", "validation"]
      ),
      preprocessors=[
          preprocessors.MapFnTransform(
              functools.partial(
                  translate,
                  source_language=builder_config.language_pair[1],
                  target_language=builder_config.language_pair[0],
              )
          ),
          preprocessors.MapFnTransform(
              tokenizer.Tokenizer(
                  tokenizer_configs=tokenizer_configs,
              )
          ),
      ],
  )


def get_nqo_v001_task(
    tokenizer_configs: Dict[str, tokenizer.TokenizerConfig] | None = None,
) -> dataset_providers.Task:
  """Create example AirIO task."""

  # source
  tfds_version = "1.0.0"
  tfds_name = f"natural_questions_open:{tfds_version}"

  # tokenizer
  if tokenizer_configs is None:
    tokenizer_configs = {
        "inputs": tokenizer.TokenizerConfig(vocab=_DEFAULT_VOCAB),
        "targets": tokenizer.TokenizerConfig(vocab=_DEFAULT_VOCAB),
    }

  return dataset_providers.Task(
      name="dummy_airio_task",
      source=data_sources.TfdsDataSource(
          tfds_name=tfds_name, splits=["train", "validation"]
      ),
      preprocessors=[
          preprocessors.MapFnTransform(question),
          preprocessors.MapFnTransform(
              tokenizer.Tokenizer(
                  tokenizer_configs=tokenizer_configs,
                  copy_pretokenized=False,
              )
          ),
      ],
  )


# Preprocessors for WMT Task.
def translate(
    ex: Dict[bytes | str, bytes | str],
    source_language: str,
    target_language: str,
) -> Dict[bytes | str, bytes | str]:
  """Convert a translation dataset to a text2text pair.

  For example, say the dataset returns examples of this format:
    {'de': 'Das ist gut.', 'en': 'That is good.'}
  If source_language = 'de', target_language = 'en', then the outputs will have
  the format:
    {'inputs': 'translate German to English: Das ist gut.',
     'targets': 'That is good.'}

  Args:
    ex: an example to process.
    source_language: source language code (e.g. 'en') to translate from.
    target_language: target language code (e.g. 'de') to translate to.

  Returns:
    A preprocessed example with the format listed above.
  """
  # Language codes like zh-cn are not supported; use only the first 2 chars
  for language in (source_language, target_language):
    if language != language[:2]:
      logging.warning(
          "Extended language code %s not supported. Falling back on %s.",
          language,
          language[:2],
      )
  lang_id_to_string = {
      source_language: babel.Locale(source_language[:2]).english_name,
      target_language: babel.Locale(target_language[:2]).english_name,
  }
  if isinstance(ex[source_language], bytes):
    src_str = (
        f"translate {lang_id_to_string[source_language]} to"
        f" {lang_id_to_string[target_language]}: ".encode()
        + ex[source_language]
    )
  else:
    src_str = (
        f"translate {lang_id_to_string[source_language]} to"
        f" {lang_id_to_string[target_language]}: "
        + ex[source_language]
    )
  return {
      "inputs": src_str,
      "targets": ex[target_language],
  }


# Preprocessors for NQO Task.
def question(ex: Dict[str, str]) -> Dict[str, str]:
  """Convert a natural question dataset to a text2text pair.

  For example, say the dataset returns examples of this format:
    {'answer': 'Romi Van Renterghem.',
     'question': 'who is the girl in more than you know'}
  The outputs will have the format:
    {'inputs': 'nq question: who is the girl in more than you know',
     'targets': 'Romi Van Renterghem.'}

  Args:
    ex: an example to process.

  Returns:
    A preprocessed example with the format listed above.
  """
  final_example = {
      "inputs": "nq question: " + ex["question"],
      "targets": ", ".join(ex["answer"]),
  }
  return final_example
