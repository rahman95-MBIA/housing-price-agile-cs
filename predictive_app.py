
# predictive_app.py
import time

import joblib
import pandas as pd
import streamlit as st

from log_utils import log_prediction

st.set_page_config(page_title="Housing Proce Prediction App with Monitoring",
                   layout="centered")

st.title("Housing Price Prediction App with Live Monitoring")

@st.cache_resource
def load_models():
    old_model = joblib.load("housing_price_model_v1.pkl")  # trained on ["area"]
    new_model = joblib.load("housing_price_model_v2.pkl")  # trained on ["area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "parking", "prefarea", "furnishingstatus"]
    return old_model, new_model

old_model, new_model = load_models()

# ---------- Initialise session state ----------
if "pred_ready" not in st.session_state:
    st.session_state["pred_ready"] = False
if "old_pred" not in st.session_state:
    st.session_state["old_pred"] = None
if "new_pred" not in st.session_state:
    st.session_state["new_pred"] = None
if "latency_ms" not in st.session_state:
    st.session_state["latency_ms"] = None
if "input_summary" not in st.session_state:
    st.session_state["input_summary"] = ""

# ---------- INPUT SECTION ----------
st.sidebar.header("Input Parameters")

area = st.sidebar.slider("area", min_value=1000, max_value=20000, value=200)
bedrooms = st.sidebar.selectbox("bedrooms", ["1", "2", "3", "4", "5", "6"])
bathrooms = st.sidebar.selectbox ("bathrooms", ["1", "2", "3", "4"])
stories = st.sidebar.selectbox("stories", ["1", "2", "3", "4"])
parking = st.sidebar.selectbox("parking", ["0", "1", "2", "3"])
mainroad = st.sidebar.selectbox("mainroad", ["yes", "no"])
guestroom = st.sidebar.selectbox("guestroom", ["yes", "no"])
basement = st.sidebar.selectbox("basement", ["yes", "no"])
hotwaterheating = st.sidebar.selectbox("hotwaterheating", ["yes", "no"])
airconditioning = st.sidebar.selectbox("airconditioning", ["yes", "no"])
prefarea = st.sidebar.selectbox("prefarea", ["yes", "no"])
furnishingstatus = st.sidebar.selectbox("furnishingstatus", ["furnished", "semi-furnished", "unfurnished"])

# Canonical input dataframe
input_df = pd.DataFrame({
    "price": [price],
    "furnishingstatus": [furnishingstatus],
})

st.subheader("Input Summary")
st.write(input_df)

# ---------- BUTTON 1: RUN PREDICTION ----------
if st.button("Run Prediction"):
    start_time = time.time()

    # v1: baseline – only uses area
    input_v1 = input_df[["area"]]
    old_pred = old_model.predict(input_v1)[0]

    # v2: improved – uses all twelve features
    input_v2 = input_df[["area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "parking", "prefarea", "furnishingstatus"]]
    new_pred = new_model.predict(input_v2)[0]

    latency_ms = (time.time() - start_time) * 1000.0

    # Store in session_state so they survive reruns
    st.session_state["old_pred"] = float(old_pred)
    st.session_state["new_pred"] = float(new_pred)
    st.session_state["latency_ms"] = float(latency_ms)
    st.session_state["input_summary"] = f"area={area}, bedrooms={bedrooms}, bathrooms={bathrooms}, stories={stories}, mainroad={mainroad}, guestroom={guestroom}, basement={basement}, hotwaterheating={hotwaterheating}, airconditioning={airconditioning}, parking={parking}, prefarea={prefarea}, furnishingstatus={furnishingstatus}"
    st.session_state["pred_ready"] = True

# ---------- SHOW PREDICTIONS IF READY ----------
if st.session_state["pred_ready"]:
    st.subheader("Predictions")
    st.write(f"Old Model (v1 - area only): **${st.session_state['old_pred']:,.2f}**")
    st.write(f"New Model (v2 - all features): **${st.session_state['new_pred']:,.2f}**")
    st.write(f"Latency: {st.session_state['latency_ms']:.1f} ms")
else:
    st.info("Click **Run Prediction** to see model outputs before giving feedback.")

# ---------- FEEDBACK SECTION ----------
st.subheader("Your Feedback on These Predictions")

feedback_score = st.slider(
    "How useful were these predictions? (1 = Poor, 5 = Excellent)",
    min_value=1,
    max_value=5,
    value=4,
    key="feedback_score",
)
feedback_text = st.text_area("Comments (optional)", key="feedback_text")

# ---------- BUTTON 2: SUBMIT FEEDBACK ----------
if st.button("Submit Feedback"):
    if not st.session_state["pred_ready"]:
        st.warning("Please run the prediction first, then submit your feedback.")
    else:
        # Log both models using saved predictions and input summary
        log_prediction(
            model_version="v1_old",
            model_type="baseline",
            input_summary=st.session_state["input_summary"],
            prediction=st.session_state["old_pred"],
            latency_ms=st.session_state["latency_ms"],
            feedback_score=feedback_score,
            feedback_text=feedback_text,
        )

        log_prediction(
            model_version="v2_new",
            model_type="improved",
            input_summary=st.session_state["input_summary"],
            prediction=st.session_state["new_pred"],
            latency_ms=st.session_state["latency_ms"],
            feedback_score=feedback_score,
            feedback_text=feedback_text,
        )

        st.success(
            "Feedback and predictions have been saved to monitoring_logs.csv. "
            "You can now view them in the monitoring dashboard."
        )
