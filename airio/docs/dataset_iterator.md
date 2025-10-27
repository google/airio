# Dataset Iterator

## PyGrainDatasetIteratorWrapper

`GrainTask.get_dataset` returns a `PyGrainDatasetIteratorWrapper`. It supports
iteration (`next(ds_iter)`), checkpointing (`save` and `restore` methods), and a
few convenience methods like `peek`, `peek_async` and `element_spec` to make it
easier for higher-level training libraries to integrate with AirIO.
