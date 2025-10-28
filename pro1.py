import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("netflix_customer_churn.csv")
# print(df.head(6))
# print(df.info())
# print(df.columns)
# print(df.isna().sum())
# print(df['churned'].value_counts())
y = df['churned']
X = df.drop('churned', axis=1)
categorical_cols = X.select_dtypes(include=['object']).columns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'   # keep numeric cols as it is
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
model=LogisticRegression()
model.fit(X_train_processed,y_train)
y_pred = model.predict(X_test_processed)
from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train_processed, y_train)
y_pred_rf = rf_model.predict(X_test_processed)

from sklearn.metrics import accuracy_score, classification_report
print("✅ Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf) * 100, "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
import joblib
joblib.dump(rf_model, "churn_model.pkl")

feature_names = preprocessor.get_feature_names_out()

# Create dataframe of feature importances
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print(importance_df.head(10)) 
joblib.dump(preprocessor, "preprocessor.pkl")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
disp.plot()
plt.title("Confusion Matrix - Random Forest")
plt.show()

