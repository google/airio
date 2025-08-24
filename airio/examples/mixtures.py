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

"""Example AirIO mixtures."""

import functools

import airio.pygrain as airio
import airio.pygrain_common as airio_common

_DEFAULT_EXTRA_IDS = 100
_DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"
_DEFAULT_VOCAB = airio.SentencePieceVocabulary(
    _DEFAULT_SPM_PATH, _DEFAULT_EXTRA_IDS
)
# This is a truncated list. See tfds.text.c4.MC4_LANGUAGES for the full list.
MC4_LANGUAGES = [
    "af",
    "be",
    "de",
    "en",
    "fr",
    "ga",
    "haw",
    "id",
    "ja",
    "ku",
    "lo",
    "mi",
    "ne",
    "pa",
    "ro",
    "st",
    "th",
    "uz",
    "vi",
    "xh",
    "yi",
    "zh",
]


def get_mc4_mixture(
    tokenizer_configs: dict[str, airio.TokenizerConfig] | None = None,
):
  """AirIO Mixture for multilingual C4 span corruption."""
  rekey_fn = functools.partial(
      airio_common.preprocessors.rekey,
      key_map={
          "inputs": None,
          "targets": "text"
      })
  vocab = _DEFAULT_VOCAB
  if tokenizer_configs is None:
    tokenizer_configs = {
        "inputs": airio.TokenizerConfig(vocab=vocab, add_eos=True),
        "targets": airio.TokenizerConfig(vocab=vocab, add_eos=True),
    }
  tasks = []
  for lang in MC4_LANGUAGES:
    task = airio.GrainTask(
        f"mc4.{lang}",
        source=airio.TfdsDataSource(
            tfds_name="c4/multilingual:3.1.0",
            splits={"train": lang, "validation": f"{lang}-validation"},
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
        ],
    )
    tasks.append(task)
  proportions = [1.0] * len(tasks)
  return airio.GrainMixture("mc4", tasks, proportions)
