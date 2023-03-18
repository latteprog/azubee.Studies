# Introduction

This project contains the source files (and data) for the evaluation of the studies done for Azubee in the context of Interaction Design Engineering.

# Getting Started

This section is going to provide you a short introduction in how to setup the repository on your local system.

## Installation Process
All relevant project files are obtained from this repository. However, it is required to have the following software installed and configured.

### [Python](https://www.python.org/downloads/)

As SciPy is used for performing the evaluation, Python is a mandatory install.


## Setup

1. We recommend to use a virtual environment to ensure consistency, e.g.
```
conda create -n azubee-evaluation python=3.8
```
2. Activate environment
```
conda activate azubee-evaluation
```
3. Install the dependencies
```
conda install -c conda-forge --file requirements.txt
```

If you need to remove the environment, you can do it with the following command.
```
conda deactivate
conda env remove -n azubee-evaluation
```

## Evaluation

1. Transform the raw data placed within **data** directory
```
python data_preparation.py
```
2. Run pre study
```
python pre_study.py
```
3. Run main study
```
python main_study.py
```
