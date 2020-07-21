# Spoken Language Identification using CRNN

This repository is an implementation of the paper, "[Language Identification Using Deep
Convolutional Recurrent Neural Networks](https://arxiv.org/pdf/1708.04811.pdf)" by Bartz et al. written using PyTorch.

# Requirements
* You will require the tools ```wget```, ```curl```, and ```sox```
* Install all python dependancies by running ```pip install -r requirements.txt```( recommended to work inside a venv )

# Usage
* Run the scripts in data_prep_scripts and set up the dataset
* Run train.py as ```python train.py <optional command line arguments>```. This script will save the best model and the final model in a folder called model_saves
* Run test.py as ```python test.py <optional command line arguments>```. This script will request you to enter the path to the model which you want to test on. Once entered it will give you results.





