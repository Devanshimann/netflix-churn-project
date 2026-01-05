import streamlit as st
import pandas as pd
import joblib

# ---------------- LOAD FILES ----------------
model = joblib.load("churn_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Load original dataset ONLY to get valid categories & dtypes
df = pd.read_csv("netflix_customer_churn.csv")

# ---------------- UI ----------------
st.set_page_config(page_title="Netflix Churn Predictor", layout="centered")

st.title("📺 Netflix Customer Churn Prediction")
st.write("Enter customer details to predict churn probability")

# ---------------- USER INPUTS ----------------
age = st.number_input("Age", min_value=1, max_value=100, value=25)

gender = st.selectbox(
    "Gender",
    sorted(df["gender"].dropna().unique())
)

subscription_type = st.selectbox(
    "Subscription Type",
    sorted(df["subscription_type"].dropna().unique())
)

watch_hours = st.number_input(
    "Total Watch Hours",
    min_value=0.0,
    value=100.0
)

last_login_days = st.number_input(
    "Days Since Last Login",
    min_value=0,
    value=5
)

region = st.selectbox(
    "Region",
    sorted(df["region"].dropna().unique())
)

device = st.selectbox(
    "Device",
    sorted(df["device"].dropna().unique())
)

monthly_fee = st.number_input(
    "Monthly Fee",
    min_value=0.0,
    value=500.0
)

payment_method = st.selectbox(
    "Payment Method",
    sorted(df["payment_method"].dropna().unique())
)

number_of_profiles = st.number_input(
    "Number of Profiles",
    min_value=1,
    max_value=10,
    value=2
)

avg_watch_time_per_day = st.number_input(
    "Average Watch Time Per Day (hours)",
    min_value=0.0,
    value=2.5
)

favorite_genre = st.selectbox(
    "Favorite Genre",
    sorted(df["favorite_genre"].dropna().unique())
)

# ---------------- CREATE INPUT DATAFRAME ----------------
input_df = pd.DataFrame({
    "customer_id": [0],  # dummy value to match training schema
    "age": [age],
    "gender": [gender],
    "subscription_type": [subscription_type],
    "watch_hours": [watch_hours],
    "last_login_days": [last_login_days],
    "region": [region],
    "device": [device],
    "monthly_fee": [monthly_fee],
    "payment_method": [payment_method],
    "number_of_profiles": [number_of_profiles],
    "avg_watch_time_per_day": [avg_watch_time_per_day],
    "favorite_genre": [favorite_genre]
})

# Force exact dtype match with training data
X_train_schema = df.drop("churned", axis=1)
input_df = input_df.astype(X_train_schema.dtypes.to_dict())

# ---------------- PREDICTION ----------------
if st.button("Predict Churn"):
    processed_input = preprocessor.transform(input_df)
    prediction = model.predict(processed_input)[0]
    probability = model.predict_proba(processed_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to CHURN\n\nProbability: {probability:.2%}")
    else:
        st.success(f"✅ Customer is NOT likely to churn\n\nProbability: {probability:.2%}")
