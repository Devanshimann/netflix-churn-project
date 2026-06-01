import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import shap

df = pd.read_csv("netflix_customer_churn.csv")

y = df['churned']
X = df.drop(['churned', 'customer_id'], axis=1)

categorical_cols = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed  = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()


lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_processed, y_train)
y_pred_lr = lr_model.predict(X_test_processed)

acc_lr = accuracy_score(y_test, y_pred_lr)

print(f"LR Accuracy: {acc_lr:.4f}")
print(classification_report(y_test, y_pred_lr))
n_estimators = 200

rf_model = RandomForestClassifier(
    n_estimators=n_estimators,
    random_state=42
)

rf_model.fit(X_train_processed, y_train)

y_pred_rf = rf_model.predict(X_test_processed)

acc_rf = accuracy_score(y_test, y_pred_rf)

print(f"RF Accuracy: {acc_rf * 100:.2f}%")
print(classification_report(y_test, y_pred_rf))


joblib.dump(rf_model, "churn_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")

X_test_dense = X_test_processed
if hasattr(X_test_dense, "toarray"):
    X_test_dense = X_test_dense.toarray()

explainer = shap.TreeExplainer(rf_model)

shap_sample = X_test_dense[:100]
shap_values = explainer.shap_values(shap_sample, check_additivity=False)

if isinstance(shap_values, list):
    shap_values_churn = shap_values[1]
else:
    shap_values_churn = shap_values[:, :, 1]
# Save everything the app needs

joblib.dump(shap_values_churn, "shap_values.pkl")
joblib.dump(X_test_dense, "X_test_dense.pkl")
joblib.dump(feature_names.tolist(), "feature_names.pkl")

print("SHAP values saved.")

shap.summary_plot(
    shap_values_churn,
    features=shap_sample,
    feature_names=feature_names,
    show=False
)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print("shap_summary.png saved — add this to your README!")
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
disp.plot()
plt.title("Confusion Matrix - Random Forest")
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
