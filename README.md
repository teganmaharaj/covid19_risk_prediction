# Supervised COVID-19 risk prediction

As part of a project for creating a COVID-19 risk-management app, we have created a supervised learning dataset for predicting individuals' level of risk of infection, as well as their source of infection, from features of individuals (e.g. pre-existing medical conditions) and features of encounters between individuals. The dataset is output by [a city-level simulator](https://https://github.com/pg2455/covid_p2p_simulation) (a stochastic agent-based model). 

**The goal** of providing this dataset is to find machine learning models (or any method!) which can do a good job of predicting risk and sources of infection from the provided features. The features are constrained by many concerns about privacy and security, making ordinary contact-tracing impracticable; this is why we need to train the predictors on simulated data. The simulated data is parsed to 'look like' the real data that would eventually be gathered by the app. The best risk estimator(s) will be used in an app to provide personalized recommendations and interventions. There is potential for these targetted interventions to reduce the spread of COVID-19 much more effectively than generic social distancing or other measures.

This repo contains pytorch dataloaders and a Transformer model; you can start from these and replace the Transformer with your own model, or use them as inspiration for development with another framework. Upload your results to the table by making a PR (details below). 

**IMPORTANT:** Do not train/tune on the test set, optimize for any of the metrics, or otherwise attempt to "cheat" at the task. This is not a contest. This project has real-world applications; under-estimating risk due to poor generalization/over-fitting could be dangerous.  We will keep a private test set to check for this, but is extremely important to use all machine learning best-practices, and it is everyone's individual responsibility to to so to the best of their ability.


## Quick Start/ Overview

1. Clone or fork this repo
2. [Download the data](https://covid-p2p-simulation.s3.ca-central-1.amazonaws.com/covi-1k-04-27.zip)
3. Extract the data to a folder called data inside the repo : `unzip covi-1k-04-27.zip data`
4. Install dependencies (see below) and `mkdir exp`
5. Run the transformer on CPU to make sure everything is working `python train.py exp/MY-CTT-EXPERIMENT-0 --inherit base_config/CTT-0 --config.device cpu` 
6. Replace the transformer model with your own and start experimenting!
7. Upload your results to the results table below by making a PR to this repo

## More information

### Dataset details

Extract the provided zip file into `\data`.
```
unzip data.zip data
```

### Dependencies

Besides `pytorch` and the usual ML stack, you will need `speedrun` which you can install as: 
```
 pip install git+https://github.com/inferno-pytorch/speedrun.git@dev`. 
```

For logging with wandb, you'll also need to `pip install wandb`. 

### Train the transformer model

Make an experimental directory: `mkdir exp`

Run the training script, logging to `exp/`:
```
python train.py exp/MY-CTT-EXPERIMENT-0 --inherit base_config/CTT-0
```
This will start training on a GPU, if available. If you want to use a CPU instead, append  `--config.device cpu` to the above command. 

### Train your own model

Replace the models.py with your own if you want to use this code as a scaffold. Feel free to use only the data loaders and metrics and write your own main loop etc., but we may be slower to evaluate your PR the more different it is from this code.

### Task Details

For a full write-up of this task, see [this document](TODO).

**Input:** TODO

**Targets:** TODO

**Metrics:** 

* **P**: Precision is of the top 1% of highest-risk people, what % are correctly identified as being infected
* **P-U**: Precision-Untested is of the top 1% of highest-risk people, excluding those who have a positive test, what % are correctly identified as being infected
* **P-A**: Precision-Asymptomatic is of the top 1% of highest-risk people, excluding those who have a positive test and those who have symptoms, what % are correctly identified as being infected
* **R**: Recall is what % of those infected are correctly identified as being infected
* **R-U**: Recall-Untested is what % of those infected are correctly identified as being infected, among people who have not been tested
* **R-A** Recall-Asymptomatic is what % of those infected are correctly identified as being infected, among people who are asymptomatic
* **MSE**: is Mean Squared Error, between the target risk and the prediction. (Possibly N/A for non-ML methods)
* **MRR**: is Mean Reciprocal Rank TODO


## Results 

Model Name | Brief description | ML? | P | P-U | P-A | R | R-U | R-A | MSE | MRR
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
[Naive Contact Tracing](TODO) | Simple risk calculation based on number of contacts | No | - | -| - | - | - | - | - | -
[Transformer](https://github.com/nasimrahaman/ctt) | Uses attention over last 14 days of encounters | Yes | - | - | - | - | - | - | - | -


### Reporting Results

To report results in the leaderboard submit a pull request from your repo to the master branch of this repo:
* Place your row at the appropriate height so that the table is sorted by performance on the first metric (Precision)
* You must fill all fields in the leaderboard row (except metrics which do not apply to your metuod:
    - Model name (which is a link to your repo)
    - One-line description of your model
    - Whether your method employs machine learning (yes/no)
    - Metrics (all Precision and  and MRR)
* Make sure your repo has a brief description of your model in the README.md
* The repo making the PR should contain all of your code, which must be open-source (not private)
* Tag @teganmaharaj and @nasimrahaman as reviewers of your PR


## Links to other parts of this risk-management project

The next stage for a successful risk predictor is to be integrated into the loop of the simulator to test different intervention strategies based on the predicted risk. There is also the potential to jointly train both the simulator and risk predictor. Projects linked below explore both of these possibilities.

* [Simulator](https://github.com/pg2455/covid_p2p_simulation): Generates the underlying data, using an epidemiolgically-informed agent-based model written in Simpy 
* [Data parser](TODO): Takes the logs from the Simulator and generates the supervised learning dataset, masking/discretizing some values in accordance with privacy concerns.
* [Transformer model](https://github.com/nasimrahaman/ctt): Full version of the model the example code in here is based on
* [GraphNN model](TODO): Implementation of a graph neural network for joint risk prediction and simulator optimizatio 
* [coVAEd model](TODO): Variational auto-encoder approach for joint risk prediction and model optimization
* [Decentralized Loopy Belief](TODO): Training of a graphical model of disease propagation via message-passing
