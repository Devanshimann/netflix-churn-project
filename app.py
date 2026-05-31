import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import shap

# ── LOAD FILES ────────────────────────────────────────────────────────────────
model        = joblib.load("churn_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
explainer    = joblib.load("shap_explainer.pkl")
feature_names = joblib.load("feature_names.pkl")

df = pd.read_csv("netflix_customer_churn.csv")

# ── UI SETUP ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Netflix Churn Predictor", layout="centered")
st.title("📺 Netflix Customer Churn Prediction")
st.write("Enter customer details to predict churn probability")

# ── USER INPUTS ───────────────────────────────────────────────────────────────
age = st.number_input("Age", min_value=1, max_value=100, value=25)

gender = st.selectbox("Gender", sorted(df["gender"].dropna().unique()))

subscription_type = st.selectbox(
    "Subscription Type", sorted(df["subscription_type"].dropna().unique())
)

watch_hours = st.number_input("Total Watch Hours", min_value=0.0, value=100.0)

last_login_days = st.number_input("Days Since Last Login", min_value=0, value=5)

region = st.selectbox("Region", sorted(df["region"].dropna().unique()))

device = st.selectbox("Device", sorted(df["device"].dropna().unique()))

monthly_fee = st.number_input("Monthly Fee", min_value=0.0, value=500.0)

payment_method = st.selectbox(
    "Payment Method", sorted(df["payment_method"].dropna().unique())
)

number_of_profiles = st.number_input(
    "Number of Profiles", min_value=1, max_value=10, value=2
)

avg_watch_time_per_day = st.number_input(
    "Average Watch Time Per Day (hours)", min_value=0.0, value=2.5
)

favorite_genre = st.selectbox(
    "Favorite Genre", sorted(df["favorite_genre"].dropna().unique())
)

input_df = pd.DataFrame({
    "customer_id":            [0],
    "age":                    [age],
    "gender":                 [gender],
    "subscription_type":      [subscription_type],
    "watch_hours":            [watch_hours],
    "last_login_days":        [last_login_days],
    "region":                 [region],
    "device":                 [device],
    "monthly_fee":            [monthly_fee],
    "payment_method":         [payment_method],
    "number_of_profiles":     [number_of_profiles],
    "avg_watch_time_per_day": [avg_watch_time_per_day],
    "favorite_genre":         [favorite_genre]
})

X_train_schema = df.drop("churned", axis=1)
input_df = input_df.astype(X_train_schema.dtypes.to_dict())

if st.button("Predict Churn"):

  
    processed_input = preprocessor.transform(input_df)

    if hasattr(processed_input, "toarray"):
        processed_input_dense = processed_input.toarray()
    else:
        processed_input_dense = processed_input

    prediction  = model.predict(processed_input)[0]
    probability = model.predict_proba(processed_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to CHURN\n\nChurn Probability: {probability:.2%}")
    else:
        st.success(f"✅ Customer is NOT likely to churn\n\nChurn Probability: {probability:.2%}")

    st.markdown("---")
    st.subheader("🔍 Why this prediction? (SHAP Explanation)")
    st.write(
        "SHAP shows which features pushed the prediction toward churn (red) "
        "or away from churn (blue) for this specific customer."
    )

    shap_vals_single = explainer.shap_values(processed_input_dense)
    # shap_vals_single[1] = SHAP values for class 1 (churn)
    shap_for_churn = shap_vals_single[1][0]  # shape: (n_features,)

   
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_for_churn
    }).sort_values("SHAP Value", key=abs, ascending=False).head(10)

    # Bar chart 
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#e53935" if v > 0 else "#1e88e5" for v in shap_df["SHAP Value"]]
    ax.barh(shap_df["Feature"][::-1], shap_df["SHAP Value"][::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on churn prediction)")
    ax.set_title("Top 10 features driving this prediction")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


    top_push    = shap_df[shap_df["SHAP Value"] > 0].head(3)["Feature"].tolist()
    top_protect = shap_df[shap_df["SHAP Value"] < 0].head(3)["Feature"].tolist()

    if top_push:
        st.markdown(
            f"**Factors increasing churn risk:** {', '.join(top_push)}"
        )
    if top_protect:
        st.markdown(
            f"**Factors reducing churn risk:** {', '.join(top_protect)}"
        )
\
    with st.expander("Show detailed SHAP waterfall chart"):
        shap_explanation = shap.Explanation(
            values=shap_for_churn,
            base_values=explainer.expected_value[1],
            data=processed_input_dense[0],
            feature_names=feature_names
        )
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(shap_explanation, show=False)
        st.pyplot(plt.gcf())
        plt.close()