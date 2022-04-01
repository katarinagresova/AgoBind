Place for notebooks, experiments and exploration.

Note: training and evaluation is encapsulated in `run.py` script and notebooks only specify parameters and call `run.py` with it. This is due to `comet.ml` limitations where logging of `train_loss` was not working properly when everything was directly in notebook.