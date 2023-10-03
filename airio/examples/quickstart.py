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

"""Tester file to debug task/mixture interfaces."""

from typing import Dict

from absl import app
from airio import data_sources
from airio import dataset_providers
from airio import preprocessors
from airio import tokenizer
from seqio import vocabularies


DEFAULT_SPM_PATH = "gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model"
DEFAULT_VOCAB = vocabularies.SentencePieceVocabulary(DEFAULT_SPM_PATH)


def create_task() -> dataset_providers.Task:
  """Create example AirIO task."""

  def _imdb_preprocessor(raw_example: Dict[str, bytes]) -> Dict[str, str]:
    final_example = {"inputs": "imdb " + raw_example["text"].decode("utf-8")}
    raw_label = str(raw_example["label"])
    if raw_label == "0":
      final_example["targets"] = "negative"
    elif raw_label == "1":
      final_example["targets"] = "positive"
    else:
      final_example["targets"] = "invalid"
    return final_example

  return dataset_providers.Task(
      name="dummy_airio_task",
      source=data_sources.TfdsDataSource(
          tfds_name="imdb_reviews/plain_text:1.0.0", splits=["train"]
      ),
      preprocessors=[
          preprocessors.MapFnTransform(_imdb_preprocessor),
          preprocessors.MapFnTransform(
              tokenizer.Tokenizer(
                  tokenizer_configs={
                      "inputs": tokenizer.TokenizerConfig(vocab=DEFAULT_VOCAB),
                      "targets": tokenizer.TokenizerConfig(vocab=DEFAULT_VOCAB),
                  },
              )
          ),
      ],
  )


def main(_) -> None:
  task = create_task()
  print(f"Task name: {task.name}\n")

  ds = task.get_dataset(split="train")

  cnt = 0
  for element in ds:
    print(element)
    cnt += 1
    if cnt == 10:
      break


if __name__ == "__main__":
  app.run(main)
