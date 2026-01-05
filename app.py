import streamlit as st
import pandas as pd
import joblib


model = joblib.load("churn_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.title("📺 Netflix Customer Churn Prediction")
st.write("Fill customer details to predict whether they will churn")


age = st.number_input("Age", min_value=1, max_value=100)
gender = st.selectbox("Gender", ["Male", "Female"])
subscription_type = st.selectbox(
    "Subscription Type", ["Basic", "Standard", "Premium"]
)
watch_hours = st.number_input("Total Watch Hours", min_value=0.0)
last_login_days = st.number_input("Days Since Last Login", min_value=0)

region = st.selectbox(
    "Region",
    ['Africa', 'Europe', 'Asia', 'Oceania', 'South America', 'North America']
)

device = st.selectbox(
    "Device", ["Mobile", "Laptop", "Smart TV", "Tablet"]
)

monthly_fee = st.number_input("Monthly Fee", min_value=0.0)

payment_method = st.selectbox(
    "Payment Method", ["Credit Card", "Debit Card", "UPI", "Net Banking"]
)

number_of_profiles = st.number_input(
    "Number of Profiles", min_value=1, max_value=10
)

avg_watch_time_per_day = st.number_input(
    "Average Watch Time Per Day (hrs)", min_value=0.0
)

favorite_genre = st.selectbox(
    "Favorite Genre", ["Drama", "Comedy", "Action", "Romance", "Thriller"]
)

input_df = pd.DataFrame({
    "customer_id": [0],
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


if st.button("Predict Churn"):
    processed_input = preprocessor.transform(input_df)
    prediction = model.predict(processed_input)

    if prediction[0] == 1:
        st.error(" Customer is likely to CHURN")
    else:
        st.success(" Customer is NOT likely to churn")
