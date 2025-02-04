# Exploring the Continuum from Static Explanations to Freeform Chat Interfaces

Repository for the backend of the paper ``Exploring-the-Continuum-from-Static-Explanations-to-Freeform-Chat-Interfaces``.
Currently working for the UCI adult census dataset as described in the paper.
Additionally,
the [experiment frontend](https://github.com/dimitrymindlin/continuum_frontend)
is needed to run the experiment UI.

## Table of Contents
- [Starting the Experiment locally](#starting-the-experiment-locally)
- [Starting the Experiment in docker](#starting-the-experiment-in-docker)
- [Analysing the experiment results](#analysing-the-experiment)
- [Running on your own models and datasets](#running-on-your-own-models-and-datasets)
- [Main changes compared to TalkToModel](#main-changes-compared-to-talktomodel)
- [Citation](#citation)

## Starting the Experiment locally

- create and activate a virtual environment with python 3.9 
  - e.g. ``conda create -n dialogue-xai python=3.9``
- install requirements
- run flask_app.py
  - When running the first time, all explanations are precomputed and stored in the cache folder.
  - While running the app will display a link to a frontend, this is the old talk-to-model frontend and is currently not
  working since we did not implement the intent recognition yet.
- start [frontend](https://github.com/dimitrymindlin/continuum_frontend) and use provided link to start experiment.

## Starting the Experiment in docker

 - make build
 - make run
 - start [frontend](https://github.com/dimitrymindlin/continuum_frontend) with docker compose

## Analysing the experiment

- ``experiment_analysis/create_analysis_data.py`` is a script to create the analysis_files from logging information in the database.
- We share the preprocessed data in ``experiment_analysis/data_chat`` for the current paper and ``experiment_analysis/data_static_interactive`` for the data from our [previous study](https://github.com/dimitrymindlin/Measuring-User-Understanding-in-Dialogue-based-XAI-Systems) on the static and interactive conditions.
- The main analysis was performed in Prism and I am working on a way to share it... Let me know if you need the analysis scripts sooner.

## Running on your own models and datasets

### Model and Dataset
The `data` folder contains the data and train scripts. For example, `adult.csv` is used in `adult_train.py` to train a 
random forest model and save the model, model settings and column information in a separate folder `adult` for the 
explanations later on. When introducing a new dataset, make a new train script and make sure to save the column mappings
and settings in a separate folder.

### Configuration
To run the experiments on your own dataset, create a new config fle in configs folder. Take the `adult-config.gin` as 
an example and adjust the settings to your needs. Then set the `global_config.gin` to your new config file.

## Main changes compared to Previous Work

- Implemeted NLU through LLM intent recognition. Prompts and pipeline can be found in `parsing/llm_intent_recognition`.
- Implemented Dialogue Manager that either directly maps the intent to the explanation or uses a dialogue policy and tracks already used explanations. Can be found in `explain/dialogue_manager`.

## Citation

```bibtex
{
...
}
```

## Contact

You can reach out to dimitry.mindlin@uni-bielefeld.de with any questions or issues you're running into.
