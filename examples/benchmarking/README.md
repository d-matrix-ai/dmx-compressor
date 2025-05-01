## Runtime, error, and accuracy measurements for DMX models

All the core functionality is in benchmark.py. There are 3 evaluation functions
1. `measure_model_error(model_maker,active_modes,reference_mode)`
2. `measure_model_accuracy(model_maker,active_modes)`
3. `measure_model_runtime(model_maker,active_modes)`

Their first argument is a `model_maker` callable that returns the tuple: `(model,model_runner,model_evaluator,device)`.  `model_runner` is a callable that takes a single argument (the model) and runs it on a sample input. `model_evaluator` is a callable that evaluates the model accuracy and return a dictionary with accuracy metrics. See `benchmark_clip.py` and `benchmark_llama.py` for usage examples.

The evaluation functions do not return any values. They only print tables to stdout with the various measure metrics. These tables are compatible with Github's markdown and they can be pasted on Github issues or Readmes. 
