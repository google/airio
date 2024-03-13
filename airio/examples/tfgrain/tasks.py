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

"""Airio TFGrain task examples."""

import functools
import airio.tfgrain as airio
import babel
import tensorflow as tf
import tensorflow_datasets as tfds

_DEFAULT_EXTRA_IDS = 100
_DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"
_DEFAULT_VOCAB = airio.SentencePieceVocabulary(
    _DEFAULT_SPM_PATH, _DEFAULT_EXTRA_IDS
)


def get_nqo_v001_task(
    tokenizer_configs: dict[str, airio.TokenizerConfig] | None = None,
) -> airio.TfGrainTask:
  """Create example AirIO task."""

  # source
  tfds_version = "1.0.0"
  tfds_name = f"natural_questions_open:{tfds_version}"

  # tokenizer
  if tokenizer_configs is None:
    tokenizer_configs = {
        "inputs": airio.TokenizerConfig(vocab=_DEFAULT_VOCAB),
        "targets": airio.TokenizerConfig(vocab=_DEFAULT_VOCAB),
    }

  return airio.TfGrainTask(
      name="natural_questions_open_v001",
      source=airio.TfdsDataSource(
          tfds_name=tfds_name, splits=["train", "validation"]
      ),
      preprocessors=[
          airio.MapFnTransform(question),
          airio.MapFnTransform(
              airio.Tokenizer(
                  tokenizer_configs=tokenizer_configs,
                  copy_pretokenized=False,
              )
          ),
      ],
  )


def get_wmt_19_ende_v003_task(
    tokenizer_configs: dict[str, airio.TokenizerConfig] | None = None,
) -> airio.TfGrainTask:
  """Returns an AirIO Task for WMT 19 en/de dataset."""
  # source
  builder_config = tfds.translate.wmt19.Wmt19Translate.builder_configs["de-en"]
  tfds_version = "1.0.0"
  tfds_name = f"wmt19_translate/{builder_config.name}:{tfds_version}"

  # tokenizer
  if tokenizer_configs is None:
    tokenizer_configs = {
        "inputs": airio.TokenizerConfig(vocab=_DEFAULT_VOCAB),
        "targets": airio.TokenizerConfig(vocab=_DEFAULT_VOCAB),
    }

  return airio.TfGrainTask(
      "wmt19_ende_v003",
      source=airio.TfdsDataSource(
          tfds_name=tfds_name, splits=["train", "validation"]
      ),
      preprocessors=[
          airio.MapFnTransform(
              functools.partial(
                  translate,
                  source_language=builder_config.language_pair[1],
                  target_language=builder_config.language_pair[0],
              )
          ),
          airio.MapFnTransform(
              airio.Tokenizer(
                  tokenizer_configs=tokenizer_configs,
                  copy_pretokenized=False,
              )
          ),
      ],
  )


# Preprocessors for NQO Task.
def question(ex: dict[str, str]) -> dict[str, str]:
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
      "targets": tf.strings.reduce_join(ex["answer"], separator=", "),
  }
  return final_example


# Preprocessors for WMT Task.
def translate(x, source_language, target_language):
  """Convert a translation dataset to a text2text pair.

  For example, say the dataset returns examples of this format:
    {'de': 'Das ist gut.', 'en': 'That is good.'}
  If source_language = 'de', target_language = 'en', then the outputs will have
  the format:
    {'inputs': 'translate German to English: Das ist gut.',
     'targets': 'That is good.'}

  Args:
    x: an example to process.
    source_language: source language code (e.g. 'en') to translate from.
    target_language: target language code (e.g. 'de') to translate to.

  Returns:
    A preprocessed example with the format listed above.
  """
  # Language codes like zh-cn are not supported; use only the first 2 chars
  lang_id_to_string = {
      source_language: babel.Locale(source_language[:2]).english_name,
      target_language: babel.Locale(target_language[:2]).english_name,
  }
  src_str = "translate {}".format(lang_id_to_string[source_language])
  tgt_str = " to {}: ".format(lang_id_to_string[target_language])
  return {
      "inputs": tf.strings.join([src_str, tgt_str, x[source_language]]),
      "targets": x[target_language],
  }
