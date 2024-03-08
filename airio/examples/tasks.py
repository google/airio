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

"""Example AirIO Tasks for reference and tests."""

import functools
import logging

from absl import logging
import airio.pygrain as airio
import airio.pygrain_common as airio_common
import babel
from seqio import preprocessors as seqio_preprocessors
import tensorflow_datasets as tfds

_DEFAULT_EXTRA_IDS = 100
_DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"
_DEFAULT_VOCAB = airio.SentencePieceVocabulary(
    _DEFAULT_SPM_PATH, _DEFAULT_EXTRA_IDS
)


def get_wmt_19_ende_v003_task(
    tokenizer_configs: dict[str, airio.TokenizerConfig] | None = None,
) -> airio.GrainTask:
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

  return airio.GrainTask(
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


def get_nqo_v001_task(
    tokenizer_configs: dict[str, airio.TokenizerConfig] | None = None,
) -> airio.GrainTask:
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

  return airio.GrainTask(
      name="dummy_airio_task",
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


# Preprocessors for WMT Task.
def translate(
    ex: dict[bytes | str, bytes | str],
    source_language: str,
    target_language: str,
) -> dict[bytes | str, bytes | str]:
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
      "targets": ", ".join(ex["answer"]),
  }
  return final_example


def append_eos_after_trim(
    ex,
    runtime_args: airio.AirIOInjectedRuntimeArgs,
    tokenizer_configs,
):
  """Wrapper over seqio append_eos_after_trim preprocessor."""
  sequence_lengths = runtime_args.sequence_lengths
  return seqio_preprocessors.append_eos_after_trim_impl(
      ex, output_features=tokenizer_configs, sequence_length=sequence_lengths
  )


def get_c4_v220_span_corruption_task(
    tokenizer_configs: dict[str, airio.TokenizerConfig] | None = None,
):
  """AirIO Task for C4 span corruption."""
  rekey_fn = functools.partial(
      airio_common.preprocessors.rekey,
      key_map={"inputs": None, "targets": "text"},
  )
  vocab = _DEFAULT_VOCAB
  if tokenizer_configs is None:
    tokenizer_configs = {
        "inputs": airio.TokenizerConfig(vocab=vocab, add_eos=True),
        "targets": airio.TokenizerConfig(vocab=vocab, add_eos=True),
    }
  append_eos_after_trim_fn = functools.partial(
      append_eos_after_trim, tokenizer_configs=tokenizer_configs
  )
  return airio.GrainTask(
      "c4_v220_span_corruption",
      source=airio.TfdsDataSource(
          tfds_name="c4/en:2.2.0", splits=["train", "validation"]
      ),
      preprocessors=[
          airio.MapFnTransform(rekey_fn),
          airio.MapFnTransform(
              airio.Tokenizer(
                  tokenizer_configs=tokenizer_configs,
              )
          ),
          airio_common.span_corruption.create_span_corruption_transform(
              tokenizer_configs
          ),
          airio.MapFnTransform(append_eos_after_trim_fn),
      ],
  )
