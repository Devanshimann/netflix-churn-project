import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import shap

model        = joblib.load("churn_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
# explainer    = joblib.load("shap_explainer.pkl")
explainer = shap.TreeExplainer(model)
feature_names = joblib.load("feature_names.pkl")

df = pd.read_csv("netflix_customer_churn.csv")
st.set_page_config(page_title="Netflix Churn Predictor", layout="wide")

page = st.sidebar.radio(
    "Navigation",
    ["📺 Churn Prediction", "📊 SQL Showcase"]
)

if page == "📺 Churn Prediction":

    st.title("📺 Netflix Customer Churn Prediction")
    st.write("Enter customer details to predict churn probability")

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
    
    X_train_schema = df.drop(["churned", "customer_id"], axis=1)
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

        shap_vals_single = explainer.shap_values(
            processed_input_dense,
            check_additivity=False
        )
        
        if isinstance(shap_vals_single, list):
            shap_for_churn = shap_vals_single[1][0]
        else:
            shap_for_churn = shap_vals_single[0, :, 1]
            
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

        with st.expander("Show detailed SHAP waterfall chart"):
            shap_explanation = shap.Explanation(
                values=shap_for_churn,
                base_values=(
                    explainer.expected_value[1]
                    if isinstance(explainer.expected_value, (list, np.ndarray))
                    else explainer.expected_value
                ),
                data=processed_input_dense[0],
                feature_names=feature_names
            )
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(shap_explanation, show=False)
            st.pyplot(plt.gcf())
            plt.close()

elif page == "📊 SQL Showcase":

    st.title("📊 SQL Analytics Showcase")

    st.markdown("""
    This section demonstrates SQL skills used for churn analysis,
    customer segmentation, revenue analysis and business intelligence.
    """)

    st.divider()

    st.subheader("1️⃣ Churn Rate by Subscription Type")

    st.code("""
SELECT
    subscription_type,
    ROUND(AVG(churned::numeric)*100,2) AS churn_rate
FROM netflix
GROUP BY subscription_type
ORDER BY churn_rate DESC;
    """, language="sql")

    st.info("Identify which subscription plans have the highest churn rate.")

    st.divider()

    st.subheader("2️⃣ Churn by Age Group")

    st.code("""
SELECT
    CASE
        WHEN age < 25 THEN '18-24'
        WHEN age < 35 THEN '25-34'
        WHEN age < 45 THEN '35-44'
        ELSE '45+'
    END AS age_group,
    ROUND(AVG(churned::numeric)*100,2) AS churn_rate
FROM netflix
GROUP BY age_group
ORDER BY churn_rate DESC;
    """, language="sql")

    st.info("Find which age groups are most likely to churn.")

    st.divider()

    st.subheader("3️⃣ Customer Segmentation (Window Functions)")

    st.code("""
SELECT
    customer_id,
    NTILE(4) OVER(ORDER BY watch_hours DESC) AS engagement_score,
    NTILE(4) OVER(ORDER BY monthly_fee DESC) AS revenue_score
FROM netflix;
    """, language="sql")

    st.success(
        "Uses PostgreSQL Window Functions (NTILE) for customer segmentation."
    )

    st.divider()

    st.subheader("4️⃣ High-Risk Customers")

    st.code("""
SELECT *
FROM netflix
WHERE last_login_days > 30
AND watch_hours < 50;
    """, language="sql")

    st.info(
        "Identify customers likely to churn and target retention campaigns."
    )

    st.divider()

    st.subheader("5️⃣ Revenue Analysis")

    st.code("""
SELECT
    subscription_type,
    AVG(monthly_fee) AS avg_revenue,
    SUM(monthly_fee) AS total_revenue
FROM netflix
GROUP BY subscription_type
ORDER BY total_revenue DESC;
    """, language="sql")

    st.info(
        "Measure revenue contribution of each subscription plan."
    )

    st.divider()

    st.subheader("6️⃣ Most Engaged Customers")

    st.code("""
SELECT
    customer_id,
    watch_hours
FROM netflix
ORDER BY watch_hours DESC
LIMIT 10;
    """, language="sql")

    st.info(
        "Identify highly engaged customers for loyalty programs."
    )

    st.divider()

    st.subheader("7️⃣ Payment Method Analysis")

    st.code("""
SELECT
    payment_method,
    COUNT(*) AS customers,
    ROUND(AVG(churned::numeric)*100,2) AS churn_rate
FROM netflix
GROUP BY payment_method
ORDER BY churn_rate DESC;
    """, language="sql")

    st.info(
        "Analyze whether payment methods influence churn."
    )

    st.divider()

    