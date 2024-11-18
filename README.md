# Optimal Locations for Hybrid Renewable Energy Station (HRES) Recommendation Analysis

## Overview

This repository contains two major projects related to identifying Hybrid Renewable Energy Station (HRES) locations and building a cloud-based machine learning pipeline for optimal station placement.

1. **Identifying Optimal Locations for HRES in Australia**: Utilises machine learning to analyse NASA datasets, comparing 41 different algorithms via LazyPredict, and recommending optimal HRES locations.
2. **Cloud-Based ML Pipeline for Identifying HRES**: An end-to-end pipeline built using AWS SageMaker for data preprocessing, model training, evaluation, and deployment. The pipeline includes hyperparameter tuning, model evaluation, and automated workflows.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Report](#report)
- [Installation](#installation)
- [Performance](#performance)

## Introduction

Hybrid Renewable Energy Stations (HRES) play a crucial role in integrating renewable energy sources to meet energy demands sustainably. This project is divided into two phases:
1. The **first phase** focuses on utilising machine learning to analyse environmental datasets and recommend optimal locations for HRES in Australia. LazyPredict is used to compare the performance of 41 machine learning models, providing insights into the best locations for HRES based on environmental factors like solar radiation and wind speed.
2. The **second phase** focuses on implementing a cloud-based machine learning pipeline using AWS SageMaker. This pipeline automates the process of data preprocessing, model training, evaluation, and deployment, ensuring an efficient and scalable solution for HRES analysis.    
    
## Features

- **41-Model Comparison**: Utilises LazyPredict to evaluate 41 machine learning models and identify the best-performing one for HRES location identification.
- **Cloud-Based ML Pipeline**: Implements a fully automated machine learning pipeline using AWS SageMaker, from data preprocessing to model deployment.
- **Evaluation and Hyperparameter Tuning**: The SageMaker pipeline includes hyperparameter optimisation to enhance model accuracy.

## Report

For the detailed report on the analysis and recommendations, please refer to the Google Docs link below:

[HRES Identification and Location Report](https://drive.google.com/file/d/1ZeGBPC8Dy49Ev_AhLAcX8QueEQ7QCtcM/view?usp=sharing)

## Installation

### Prerequisites

Ensure you have Python 3.10 or above installed, and an active AWS account for running the SageMaker pipelines.

## Performance

### First Project: Identifying Optimal Locations for HRES using LightGBM

**Train Dataset Performance Metrics**:
- **MAE (Mean Absolute Error)**: `0.1940`
- **MSE (Mean Squared Error)**: `0.0640`
- **RMSE (Root Mean Squared Error)**: `0.2530`
- **MSLE (Mean Squared Log Error)**: `0.0017`
- **RMSLE (Root Mean Squared Log Error)**: `0.0413`
- **R² Score**: `0.9778`

**Test Dataset Performance Metrics**:
- **MAE (Mean Absolute Error)**: `0.2814`
- **MSE (Mean Squared Error)**: `0.1356`
- **RMSE (Root Mean Squared Error)**: `0.3682`
- **MSLE (Mean Squared Log Error)**: `0.0034`
- **RMSLE (Root Mean Squared Log Error)**: `0.0580`
- **R² Score**: `0.9542`

**Differences between Train and Test Dataset Metrics**:
- **MAE Difference**: `-0.0874`
- **MSE Difference**: `-0.0716`
- **RMSE Difference**: `-0.1152`
- **MSLE Difference**: `-0.0017`
- **RMSLE Difference**: `-0.0168`
- **R² Score Difference**: `0.0237`

### Second Project: Cloud-Based ML Pipeline for HRES using XGBoost

The performance metrics for the SageMaker project using XGBoost are currently unavailable. However, the XGBoost model was used with hyperparameter tuning in AWS SageMaker to optimise the results.

More detailed performance metrics and evaluation for this part of the project will be included in future updates.

