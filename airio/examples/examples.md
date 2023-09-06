# AirIO Examples

The files in this directory illustrate basic usage of AirIO.

* Quickstart: [quickstart.py](quickstart.py), [quickstart.ipynb](quickstart.ipynb)
* Inspect: [inspect.py](inspect.py), [inspect.ipynb](inspect.ipynb)

## Quickstart

The quickstart example demonstrates creation of a basic `Task` with two
preprocessing steps. It performs the following actions:

1. Load the [IMDB reviews][imdb_reviews] dataset.
2. Map the raw data to a format suitable for training.
3. Tokenize the text using SeqIO's
[`SentencePieceVocabulary`][seqio_vocabularies].

The task's `get_dataset()` method is called, to demonstrate the contents of
each record after fully processing all steps.


## Inspect

The inspection example demonstrates creation of a basic `Task` with two
preprocessing steps. It performs the following actions:

1. Load the [IMDB reviews][imdb_reviews] dataset.
2. Map the raw data to a format suitable for training.
3. Tokenize the text using SeqIO's
[`SentencePieceVocabulary`][seqio_vocabularies].

The task's `get_dataset_by_step()` method is called, to demonstrate the contents
of each record after each individual processing step.

### Number of records

By default, the number of records to inspect at each step is set to 2. Any value
between 1 - 1000 can be used. To use a different value, adjust the `num_records`
parameter, e.g. to view 3 records at each step:

```
steps = task.get_dataset_by_step(num_records=3)
```



[imdb_reviews]: https://www.tensorflow.org/datasets/catalog/imdb_reviews
[inspect_ipynb]: inspect.ipynb
[inspect_py]: inspect.py
[quickstart_ipynb]: quickstart.ipynb
[quickstart_py]: quickstart.py
[seqio_vocabularies]: https://github.com/google/seqio/blob/main/seqio/vocabularies.py