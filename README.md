# Setup

## Environment

Use the `environment.yaml` file to create a conda environment.

## Training and Testing Models

Every combination of a model and a dataset needs a separate YAML file similar to `config.yaml` to run. This file defines most of the specifications regarding the training and testing. The example `config.yaml` file provided makes all the specifications self-explanatory.

### Running one model

Run the `run.py` file to just run training and testing for one model, whose config file (renamed to `config.yaml`) is placed in the same directory as the code.

### Running multiple models

Create `.yaml` files for each model and put all such files in a directory called `Configs`, alongside the directory containing this repository. Run the `autorun.py` file to sequentially run the training and testing for each model.
