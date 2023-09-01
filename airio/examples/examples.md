# AirIO Examples

The files in this directory illustrate basic usage of AirIO.

* Quickstart: [quickstart.py](quickstart.py)
* Inspect: [inspect.py](inspect.py)

## Quickstart

The quickstart example demonstrates creation of a basic `Task` with two
preprocessing steps. It performs the following actions:

1. Load the [IMDB reviews][imdb_reviews] dataset.
2. Map the raw data to a format suitable for training.
3. Tokenize the text using SeqIO's
[`SentencePieceVocabulary`][seqio_vocabularies].

The task's `get_dataset()` method is called, to demonstrate the contents of
each record after processing.

## Inspect

The inspection example demonstrates creation of a basic `Task` with two
preprocessing steps. It performs the following actions:

1. Load the [IMDB reviews][imdb_reviews] dataset.
2. Map the raw data to a format suitable for training.
3. Tokenize the text using SeqIO's
[`SentencePieceVocabulary`][seqio_vocabularies].

The task's `get_dataset_by_step()` method is called, to demonstrate the contents
of each record after each individual processing step.

By default, the number of records to inspect at each step is set to 2. Any value
between 1 - 1000 can be used.


[imdb_reviews]: https://www.tensorflow.org/datasets/catalog/imdb_reviews
[seqio_vocabularies]: https://github.com/google/seqio/blob/main/seqio/vocabularies.py