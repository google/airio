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

"""Example script that shows record inspection after each processing step."""

from typing import Dict

from absl import app
import airio.pygrain as airio


DEFAULT_SPM_PATH = (
    "gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model"
)
DEFAULT_VOCAB = airio.SentencePieceVocabulary(DEFAULT_SPM_PATH)
RECORDS_TO_INSPECT = 2


def create_task() -> airio.GrainTask:
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

  return airio.GrainTask(
      name="dummy_airio_task",
      source=airio.TfdsDataSource(
          tfds_name="imdb_reviews/plain_text:1.0.0", splits=["train"]
      ),
      preprocessors=[
          airio.MapFnTransform(_imdb_preprocessor),
          airio.MapFnTransform(
              airio.Tokenizer(
                  tokenizer_configs={
                      "inputs": airio.TokenizerConfig(vocab=DEFAULT_VOCAB),
                      "targets": airio.TokenizerConfig(vocab=DEFAULT_VOCAB),
                  },
              )
          ),
      ],
  )


def main(_) -> None:
  task = create_task()
  print("INSPECT:")
  steps = task.get_dataset_by_step(
      num_records=RECORDS_TO_INSPECT, split="train"
  )
  step_cnt = 0
  for step in steps:
    print(f"Step {step_cnt}:")
    element_cnt = 0
    for element in step:
      print(element.keys())
      element_cnt += 1
      if element_cnt >= RECORDS_TO_INSPECT:
        break
    step_cnt += 1


if __name__ == "__main__":
  app.run(main)
