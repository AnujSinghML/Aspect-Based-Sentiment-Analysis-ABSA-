# ABSA
This repository contains the implementation of Aspect-Based Sentiment Analysis (ABSA) using NLP and a Predictor Toolkit for sentiment prediction.

Table of Contents:
Installation
Usage
 Training the Model
 Running the Predictor Toolkit

Installation
Clone this repository and install the required packages using pip.
Ensure you have the datasets (Restaurants_Train_v2.csv and restaurants-trial.csv) in the root directory.

Usage:
Training the Model:
Place your datasets (Restaurants_Train_v2.csv and restaurants-trial.csv) in the root directory.
Run the ABSA_NLP.py script to train the model. This will save the best model and results in the specified directories.(may take lot of time depending on your system's performance)

Running the Predictor Toolkit:
Ensure the best model is saved in the best_model directory (generated from the training step).
Run the Predictor_Toolkit.py script to predict sentiment for given aspects and sentences.

Follow the prompts to input sentences and aspects for sentiment prediction.
