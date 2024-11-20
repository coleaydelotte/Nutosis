#  Machine Learning Analysis of Central Park Squirrel Disease Outbreak

Machine learning analysis of disease outbreak among Central Park squirrels using census and weather data (October 2018).

## Overview

This project analyzes a disease outbreak among Central Park's squirrel population using data from the Squirrel Census combined with NYC weather data. A Gradient Boosting model was developed to identify potentially diseased squirrels, achieving 96.86% accuracy in detection.

## Key Findings

- Disease rates by fur color:
  - Black squirrels: 70% (highest risk)
  - Cinnamon squirrels: 50% (moderate risk) 
  - Gray squirrels: 2% (lowest risk)

- Identified disease clusters in specific park regions
- Weather showed minimal correlation with disease presence
- Early morning hours showed slightly increased sighting rates

## Model Performance

Training metrics after GridSearchCV optimization:
- Accuracy: 96.86%
- Precision: 88.26%
- Recall: 82.26%
- F1 Score: 85.16%

Test set metrics:
- Accuracy: 93.6%
- Precision: 72.5% 
- Recall: 50.9%
- AUC Score: 0.87

## Technologies Used

- Python 3
- pandas
- scikit-learn 
- matplotlib
- seaborn
- XGBoost

## Data Sources

- Squirrel Census data (3,023 sightings)
- NYC Weather data from Kaggle

## Setup

1. Install required packages:
```bash
pip install pandas scikit-learn matplotlib seaborn xgboost
```

2. Load the pickled model and preprocessor:
```python
import pickle

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
    
with open('preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)
```

## Features

- TF-IDF vectorization for text data
- One-hot encoding for categorical variables 
- StandardScaler for numerical features
- Custom transformers for data cleaning

## Model Architecture

Gradient Boosting Classifier with optimized parameters:
- max_depth: 5
- learning_rate: 0.1
- min_samples_leaf: 1
- min_samples_split: 2
- n_estimators: 200

## Limitations

- Moderate recall rate (50.9% on test set)
- Missing values in behavioral data
- Limited temporal scope (October only)
- Model requires all features for prediction

