# Netflix Customer Churn Prediction

## Overview

This project predicts whether a Netflix customer is likely to churn based on their subscription details, viewing behavior, and account activity. The goal is to help identify at-risk customers and understand the key factors influencing churn through machine learning and explainable AI techniques.

## Live Demo

Try the application here:

https://netflix-churn-project-4pwpzmpdxipsbp3y9ppiby.streamlit.app/

## Features

* Predict customer churn probability
* Interactive Streamlit web application
* Explain predictions using SHAP (SHapley Additive Explanations)
* Compare Logistic Regression and Random Forest models
* Visualize model performance with confusion matrix
* Feature importance analysis using SHAP

## Dataset Features

The model uses the following customer attributes:

* Age
* Gender
* Subscription Type
* Watch Hours
* Days Since Last Login
* Region
* Device Type
* Monthly Fee
* Payment Method
* Number of Profiles
* Average Watch Time Per Day
* Favorite Genre

## Project Workflow

1. Data Cleaning and Preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Model Training
5. Model Evaluation
6. Explainable AI using SHAP
7. Streamlit Deployment

## Models Used

### Logistic Regression

* Baseline classification model
* Fast and interpretable

### Random Forest Classifier

* Ensemble learning model
* Higher predictive performance
* Selected as the final model

## Model Performance

| Model               | Accuracy  |
| ------------------- | --------- |
| Logistic Regression | Evaluated |
| Random Forest       | ~94%      |

## Explainable AI

The project uses SHAP (SHapley Additive Explanations) to explain individual predictions and identify the most influential factors contributing to customer churn.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* SHAP
* Streamlit
* Matplotlib
* Seaborn
* Joblib

## Project Structure

```text
netflix-churn-project/
│
├── app.py
├── pro1.py
├── netflix_customer_churn.csv
├── churn_model.pkl
├── preprocessor.pkl
├── feature_names.pkl
├── shap_summary.png
├── confusion_matrix.png
├── requirements.txt
└── README.md
```
