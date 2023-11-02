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

"""Example script that shows training with WMT."""

from collections.abc import Sequence
import dataclasses
import functools
import logging
import tempfile

from absl import app
import airio
from airio import examples
from clu.data import dataset_iterator as clu_dataset_iterator
from seqio import vocabularies
from t5x import adafactor
from t5x import gin_utils
from t5x import models
from t5x import partitioning
from t5x import train as train_lib
from t5x import trainer
from t5x import utils
from t5x.examples.t5 import network

_DEFAULT_EXTRA_IDS = 100
_DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"
_DEFAULT_VOCAB = vocabularies.SentencePieceVocabulary(
    _DEFAULT_SPM_PATH, _DEFAULT_EXTRA_IDS
)
_SOURCE_SEQUENCE_LENGTH = 32


def create_dataset(
    task: airio.dataset_providers.Task,
) -> clu_dataset_iterator.DatasetIterator:
  sequence_lengths = {
      "inputs": _SOURCE_SEQUENCE_LENGTH,
      "targets": _SOURCE_SEQUENCE_LENGTH,
  }
  return task.get_dataset(
      sequence_lengths,
      "train",
      shuffle=False,
      feature_converter=airio.feature_converters.PyGrainEncDecFeatureConverter(),
      batch_size=2,
  )


def get_t5_model(**config_overrides) -> models.EncoderDecoderModel:
  """Returns a small T5 1.1 model to use for testing."""
  tiny_config = network.T5Config(
      vocab_size=32128,
      dtype="bfloat16",
      emb_dim=8,
      num_heads=4,
      num_encoder_layers=2,
      num_decoder_layers=2,
      head_dim=3,
      mlp_dim=16,
      mlp_activations=("gelu", "linear"),
      dropout_rate=0.0,
      logits_via_embedding=False,
  )

  tiny_config = dataclasses.replace(tiny_config, **config_overrides)
  return models.EncoderDecoderModel(
      module=network.Transformer(tiny_config),
      input_vocabulary=_DEFAULT_VOCAB,
      output_vocabulary=_DEFAULT_VOCAB,
      optimizer_def=adafactor.Adafactor(
          decay_rate=0.8,
          step_offset=0,
          logical_factor_rules=adafactor.standard_logical_factor_rules(),
      ),
  )


def create_train_fn(task: airio.dataset_providers.Task):
  """Returns a function for training."""
  train_dataset_cfg = utils.DatasetConfig(
      mixture_or_task_name=task,
      task_feature_lengths={"inputs": 32, "targets": 32},
      split="train",
      batch_size=8,
      shuffle=False,
      pack=False,
      use_cached=False,
      seed=0,
  )
  eval_dataset_cfg = utils.DatasetConfig(
      mixture_or_task_name=task,
      task_feature_lengths={"inputs": 32, "targets": 32},
      split="validation",
      batch_size=8,
      shuffle=False,
      pack=False,
      use_cached=False,
      seed=0,
  )
  partitioner = partitioning.PjitPartitioner(num_partitions=4)
  trainer_cls = functools.partial(
      trainer.Trainer,
      learning_rate_fn=utils.create_learning_rate_scheduler(
          factors="constant * rsqrt_decay",
          base_learning_rate=1.0,
          warmup_steps=1000,
      ),
      num_microbatches=None,
  )
  restore_cfg = None
  ckpt_cfg = utils.CheckpointConfig(
      save=utils.SaveCheckpointConfig(
          dtype="float32",
          period=4,
          checkpoint_steps=[0, 1, 2, 3, 4, 80, 97, 100],
      ),
      restore=restore_cfg,
  )
  return functools.partial(
      train_lib.train,
      model=get_t5_model(),
      train_dataset_cfg=train_dataset_cfg,
      train_eval_dataset_cfg=eval_dataset_cfg,
      infer_eval_dataset_cfg=None,
      checkpoint_cfg=ckpt_cfg,
      partitioner=partitioner,
      trainer_cls=trainer_cls,
      total_steps=3,
      eval_steps=2,
      eval_period=1000,
      random_seed=0,
      summarize_config_fn=gin_utils.summarize_gin_config,
      use_orbax=False,
      gc_period=4,
  )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  wmt_task = examples.tasks.get_wmt_19_ende_v003_task()

  train_fn = create_train_fn(wmt_task)
  workdir = tempfile.mkdtemp()
  step, _ = train_fn(model_dir=workdir)
  logging.info("Completed %s training steps.", step)


if __name__ == "__main__":
  app.run(main)
