# Netflix Customer Churn Prediction

## Overview

This project predicts whether a Netflix customer is likely to churn based on subscription details, viewing behavior, and account activity.

The project combines **SQL-based business analytics**, **machine learning**, and **explainable AI** to identify at-risk customers and understand the key drivers behind customer churn.

---

## Live Demo

Try the application here:

https://netflix-churn-project-4pwpzmpdxipsbp3y9ppiby.streamlit.app/

---

## Key Features

### Machine Learning

* Predict customer churn probability
* Random Forest classification model
* Customer-level churn risk assessment
* SHAP explainability for prediction interpretation

### SQL Analytics Showcase

The application includes a dedicated SQL Analytics section demonstrating real-world business queries:

* Churn Rate by Subscription Type
* Churn by Age Group
* Customer Segmentation using Window Functions
* High-Risk Customer Identification
* Revenue Analysis
* Most Engaged Customers
* Payment Method Analysis

Each SQL query is displayed alongside the resulting analysis and business insight.

### Explainable AI

* SHAP feature importance visualization
* Customer-specific churn explanations
* Waterfall plots for prediction transparency

---

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

---

## Project Workflow

### Data Analytics Pipeline

1. Data Cleaning and Preprocessing
2. Exploratory Data Analysis (EDA)
3. SQL-Based Business Analysis
4. Customer Segmentation
5. Feature Engineering
6. Machine Learning Model Training
7. Explainable AI using SHAP
8. Streamlit Deployment

---

## SQL Concepts Demonstrated

### Aggregations

```sql
AVG()
SUM()
COUNT()
GROUP BY
```

### Conditional Logic

```sql
CASE WHEN
```

### Window Functions

```sql
NTILE()
```

### Business Analytics

* Churn Analysis
* Revenue Analysis
* Customer Segmentation
* Customer Retention Analytics
* Customer Behavior Analysis

---

## Models Used

### Logistic Regression

* Baseline classification model
* Fast and interpretable

### Random Forest Classifier

* Ensemble learning model
* Higher predictive performance
* Selected as the final model

---

## Model Performance

| Model               | Accuracy  |
| ------------------- | --------- |
| Logistic Regression | Evaluated |
| Random Forest       | ~94%      |

---

## Explainable AI

The project uses SHAP to explain individual predictions and identify the most influential factors contributing to customer churn.

Key benefits:

* Transparent predictions
* Feature importance analysis
* Customer-level churn explanations

---

## Technologies Used

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### SQL & Analytics

* SQL
* Business Analytics
* Customer Segmentation

### Machine Learning

* Scikit-learn
* Random Forest
* Logistic Regression

### Explainability

* SHAP

### Visualization

* Matplotlib
* Seaborn

### Deployment

* Streamlit

---

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
```
