# AirIO Examples

The files in this directory illustrate basic usage of AirIO.

* Quickstart: [quickstart.py][quickstart_py], [quickstart.ipynb][quickstart_ipynb]
* Inspect: [inspect.py][inspect_py], [inspect.ipynb][inspect_ipynb]
* Train on WMT: [train_wmt.py][train_wmt_py], [train_wmt.ipynb][train_wmt_ipynb]

## Quickstart

The quickstart example demonstrates creation of a basic `Task` with two
preprocessing steps. It performs the following actions:

1. Load the [IMDB reviews][imdb_reviews] dataset.
2. Map the raw data to a format suitable for training.
3. Tokenize the text using AirIO's
[`SentencePieceVocabulary`][airio_vocabularies].

The task's `get_dataset()` method is called, to demonstrate the contents of
each record after fully processing all steps.


## Inspect

The inspection example demonstrates creation of a basic `Task` with two
preprocessing steps. It performs the following actions:

1. Load the [IMDB reviews][imdb_reviews] dataset.
2. Map the raw data to a format suitable for training.
3. Tokenize the text using AirIO's
[`SentencePieceVocabulary`][airio_vocabularies].

The task's `get_dataset_by_step()` method is called, to demonstrate the contents
of each record after each individual processing step.

### Number of records

By default, the number of records to inspect at each step is set to 2. Any value
between 1 - 1000 can be used. To use a different value, adjust the `num_records`
parameter, e.g. to view 3 records at each step:

```
steps = task.get_dataset_by_step(num_records=3)
```


## Train on WMT

The training example demonstrates training with T5X on the [WMT][wmt] data set.


[imdb_reviews]: https://www.tensorflow.org/datasets/catalog/imdb_reviews
[inspect_ipynb]: https://github.com/google/airio/tree/main/airio/examples/inspect.ipynb
[inspect_py]: https://github.com/google/airio/tree/main/airio/examples/inspect.py
[quickstart_ipynb]: https://github.com/google/airio/tree/main/airio/examples/quickstart.ipynb
[quickstart_py]: https://github.com/google/airio/tree/main/airio/examples/quickstart.py
[train_wmt_ipynb]: https://github.com/google/airio/tree/main/airio/examples/train_wmt.ipynb
[train_wmt_py]: https://github.com/google/airio/tree/main/airio/examples/train_wmt.py
[airio_vocabularies]: https://github.com/google/airio/blob/main/airio/_src/pygrain/vocabularies.py
[wmt]: https://www.tensorflow.org/datasets/catalog/wmt19_translate