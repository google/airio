# Performance

`GrainTask.get_dataset` provides two options to tune performance:

+   `num_workers`: Use this to control the number of child processes created by
    PyGrain to parallelize the processing of the input pipeline. Grain
    recommends empirically tuning this value to your use case. A simple
    heuristic is to sweep over the values [2, 10, 25, 50, 100] for the
    worker_count and choose the value that performs best.
+   `num_prefetch_threads`: Use this to control the number of threads reading
    from the DataSource in parallel. Similar to `num_workers`, it is best to
    empirically tune this value to your use case.
