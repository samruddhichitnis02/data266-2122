# DATA 266 Homework 1

## Overview

This project explores autoregressive models, neural network based diabetes classification, and GPU accelerated matrix multiplication using CUDA.

The main goal of this project is to understand how machine learning models are trained and evaluated, how different frameworks can implement the same neural network, and how GPU parallelism can improve computational performance for large matrix operations.

## Part 1: Autoregressive Models

An autoregressive model predicts the next value in a sequence by using previously observed values.

For example, a language model predicts the next word or token based on the words that came before it. A time series model can predict tomorrow's temperature using temperatures from previous days. Similarly, a financial forecasting model can use historical prices to estimate a future price.

Common real world examples include:

1. Chatbots that generate one word at a time

2. Speech recognition systems that predict the next sound or word

3. Weather forecasting systems that use previous weather observations

4. Stock market and demand forecasting models

5. Predictive text and autocomplete systems

Autoregressive models are useful when the order of observations matters. They generate predictions step by step, using earlier information as context for later predictions.

## Part 2: Diabetes Neural Network Experiment

The diabetes dataset is used to build a binary classification model. The dataset contains eight input features and one binary target label.

The neural network experiment was implemented in both PyTorch and TensorFlow so that the same model design and evaluation process could be compared across two deep learning frameworks.

## Data Preprocessing

The dataset was loaded without a header row. The first eight columns were treated as input features, and the last column was treated as the target label.

The data processing steps were:

1. Load the CSV dataset

2. Separate the features from the target

3. Split the data into training, validation, and testing subsets

4. Standardize the features using the training data

5. Apply the same scaler to the validation and testing data

The dataset was divided into 70 percent training data, 15 percent validation data, and 15 percent testing data.

The data split was fixed using the required random seed so that every model comparison used the same samples.

The scaler was fitted only on the training set. This prevents information from the validation or testing sets from leaking into the training process.

## Data Visualization

The dataset was explored using:

1. A correlation matrix

2. Feature distribution histograms

The correlation matrix shows the relationships between the input features and the target. The feature distribution plots show how the values of each input feature are spread across the dataset.

## Baseline Neural Network

The baseline model uses the following architecture:

```text
Eight input features
64 neuron hidden layer
32 neuron hidden layer
One output neuron