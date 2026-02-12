import streamlit as st
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="CollectIQ - Payment Risk AI", layout="wide")

st.title("💳 CollectIQ - Real-Time Payment Risk Intelligence")
st.markdown("AI-powered invoice risk prediction and smart reminder system")

st.divider()
model_version = st.selectbox(
    "Select Model Version",
    ["v1"]
)

# -------------------------
# INPUT SECTION
# -------------------------

st.subheader("🧾 Enter Invoice Details")

col1, col2 = st.columns(2)

with col1:
    invoice_amount = st.number_input("Invoice Amount", min_value=0.0, value=5000.0)
    avg_delay_days = st.number_input("Average Delay Days", min_value=0.0, value=2.0)
    num_past_invoices = st.number_input("Number of Past Invoices", min_value=0.0, value=5.0)

with col2:
    invoice_gap_days = st.number_input("Invoice Gap Days", min_value=0.0, value=30.0)
    industry_category = st.number_input("Industry Category (Encoded)", min_value=0.0, value=1.0)
    reliability_score = st.number_input("Reliability Score", min_value=0.0, value=0.8)

predict_button = st.button("🔍 Predict Payment Risk")

# -------------------------
# PREDICTION SECTION
# -------------------------

if predict_button:

    payload = {
        "invoice_amount": invoice_amount,
        "avg_delay_days": avg_delay_days,
        "num_past_invoices": num_past_invoices,
        "invoice_gap_days": invoice_gap_days,
        "industry_category": industry_category,
        "reliability_score": reliability_score
    }

    try:
       response = requests.post(
            f"https://collectiq-api.onrender.com/predict?model_version={model_version}",
            json=payload,
            timeout=15
        )

        if response.status_code == 200:

            result = response.json()

            st.subheader("📊 Prediction Result")

            probability = result.get("late_payment_probability", None)

            if probability is not None:

                tone = result.get("tone", "N/A")
                action = result.get("recommended_action", "N/A")
                model_version = result.get("model_version", "N/A")

                # Risk Gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    title={"text": "Late Payment Risk (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "red"},
                        "steps": [
                            {"range": [0, 40], "color": "green"},
                            {"range": [40, 70], "color": "yellow"},
                            {"range": [70, 100], "color": "red"},
                        ],
                    },
                ))

                st.plotly_chart(fig, width="stretch")

                col1, col2, col3 = st.columns(3)

                col1.metric("Tone", tone)
                col2.metric("Recommended Action", action)
                col3.metric("Model Version", model_version)

            else:
                st.error("❌ Probability not found in API response")

        else:
            st.error(f"API Error: {response.status_code}")

    except Exception as e:
        st.error("⚠️ Connection Error")
        st.write(str(e))

# -------------------------
# ANALYTICS SECTION
# -------------------------

st.divider()
st.subheader("📈 Business Analytics")

try:
    stats_response = requests.get(
        "https://collectiq-api.onrender.com/stats",
        timeout=10
    )

    if stats_response.status_code == 200:

        stats = stats_response.json()

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Predictions", stats.get("total_predictions", 0))
        col2.metric("Average Risk", f"{stats.get('average_risk', 0):.3f}")
        col3.metric("High Risk Cases", stats.get("high_risk_predictions", 0))

    else:
        st.warning("Stats not available")

except:
    st.warning("Could not load stats")

st.divider()
st.subheader("📊 Feature Importance")

try:
    fi_response = requests.get(
        "https://collectiq-api.onrender.com/feature-importance"
    )

    if fi_response.status_code == 200:
        fi_data = fi_response.json()

        if "feature_importance" in fi_data:
            import pandas as pd

            df_fi = pd.DataFrame(
                fi_data["feature_importance"].items(),
                columns=["Feature", "Importance"]
            ).sort_values(by="Importance", ascending=False)

            st.bar_chart(df_fi.set_index("Feature"), width="stretch")

    else:
        st.info("Feature importance unavailable")

except:
    st.warning("Could not fetch feature importance")
