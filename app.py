# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import google.generativeai as genai
import io
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ----------------------------------------------------
# STREAMLIT CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="🔋 Battery SOH Predictor + Gemini Chatbot",
    page_icon="🔋",
    layout="centered"
)

st.title("🔋 Battery Pack SOH Predictor & Gemini Chatbot")
st.write("Predict battery SOH, evaluate model performance, and ask Gemini questions about your uploaded dataset.")

EXPECTED_COLS = [f"U{i}" for i in range(1, 22)]
DEFAULT_THRESHOLD = 0.6


# ----------------------------------------------------
# UTILITY FUNCTIONS
# ----------------------------------------------------
def safe_load_joblib(path):
    try:
        return joblib.load(path)
    except Exception:
        return None


def load_feature_order(path="feature_order.pkl"):
    try:
        return joblib.load(path)
    except Exception:
        return EXPECTED_COLS


def reorder_columns(df, feature_order):
    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    return df[feature_order]


def predict_pack_soh(model, vec, scaler=None):
    X = np.array(vec).reshape(1, -1)
    if scaler:
        X = scaler.transform(X)
    return float(model.predict(X)[0])


def plot_soh_gauge(soh, threshold=DEFAULT_THRESHOLD):
    fig, ax = plt.subplots(figsize=(6, 1))
    percent = np.clip(soh, 0, 1) * 100
    ax.barh([0], [percent], color="green" if soh >= threshold else "red")
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("SOH (%)")
    ax.bar_label(ax.containers[0], fmt="%.1f%%")
    st.pyplot(fig)


def plot_pred_vs_actual(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true, y_pred)
    ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], linestyle="--")
    ax.set_xlabel("Actual SOH")
    ax.set_ylabel("Predicted SOH")
    ax.set_title("Actual vs Predicted SOH")
    st.pyplot(fig)


def evaluate_predictions(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


# ----------------------------------------------------
# DATASET SUMMARY (Used by Chatbot)
# ----------------------------------------------------
def summarize_dataframe(df, max_rows=5):
    summary = []
    summary.append(f"Columns: {', '.join(df.columns)}")
    summary.append(f"Number of rows: {len(df)}")

    sample = df.head(max_rows).to_string(index=False)
    summary.append("\nSample rows:\n" + sample)

    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        stats = df[numeric_cols].describe().to_string()
        summary.append("\nNumeric statistics:\n" + stats)

    return "\n".join(summary)


# ----------------------------------------------------
# GEMINI CHATBOT WITH DATASET ACCESS
# ----------------------------------------------------
def gemini_chat(
    user_prompt,
    api_key,
    model_name="gemini-2.0-flash",
    soh_info=None,
    chat_history=None,
    df_uploaded=None
):
    if not api_key:
        return "Gemini API key is required."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)

        context_parts = []

        if soh_info:
            context_parts.append(
                f"Latest predicted SOH: {soh_info['soh']:.3f} "
                f"(status: {soh_info['status'].upper()}, threshold={soh_info['threshold']})."
            )

        if df_uploaded is not None:
            dataset_summary = summarize_dataframe(df_uploaded)
            context_parts.append("User uploaded dataset summary:\n" + dataset_summary)

        if chat_history:
            hist = "\n".join([f"{role}: {msg}" for role, msg in chat_history])
            context_parts.append("Chat history:\n" + hist)

        context_parts.append(f"User: {user_prompt}\nAssistant:")

        full_prompt = "\n\n".join(context_parts)

        response = model.generate_content(full_prompt)
        return response.text.strip()

    except Exception as e:
        return f"⚠️ Gemini API Error: {e}"


# ----------------------------------------------------
# SESSION STATE
# ----------------------------------------------------
if "df_uploaded" not in st.session_state:
    st.session_state.df_uploaded = None
if "soh_info" not in st.session_state:
    st.session_state.soh_info = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ----------------------------------------------------
# SIDEBAR MODEL LOADING
# ----------------------------------------------------
st.sidebar.header("Model Files")

model_path = st.sidebar.text_input("Model (.pkl)", "model.pkl")
scaler_path = st.sidebar.text_input("Scaler (optional)", "")
feat_path = st.sidebar.text_input("Feature Order (.pkl)", "feature_order.pkl")

if st.sidebar.button("Load Model"):
    st.session_state.model = safe_load_joblib(model_path)
    st.session_state.scaler = safe_load_joblib(scaler_path) if scaler_path else None
    st.session_state.feature_order = load_feature_order(feat_path)
    st.sidebar.success("✅ Model loaded")


# Autoload fallback
if "model" not in st.session_state:
    st.session_state.model = safe_load_joblib(model_path)
if "scaler" not in st.session_state:
    st.session_state.scaler = safe_load_joblib(scaler_path) if scaler_path else None
if "feature_order" not in st.session_state:
    st.session_state.feature_order = load_feature_order(feat_path)


# ----------------------------------------------------
# TABS
# ----------------------------------------------------
tab1, tab2 = st.tabs(["📊 SOH Prediction", "💬 Gemini Chatbot"])


# ----------------------------------------------------
# TAB 1 — Prediction
# ----------------------------------------------------
with tab1:
    st.header("📊 Predict Battery Pack SOH")

    uploaded = st.file_uploader("Upload CSV/XLSX with U1–U21", type=["csv", "xlsx"])
    manual_mode = st.checkbox("Enter 21 values manually")

    threshold = st.slider("Healthy threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01)

    df_uploaded = None

    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df_uploaded = pd.read_csv(uploaded)
            else:
                df_uploaded = pd.read_excel(uploaded)

            st.session_state.df_uploaded = df_uploaded
            st.write(df_uploaded.head())

        except Exception as e:
            st.error(f"Upload error: {e}")

    input_vec = None

    if manual_mode:
        cols = st.columns(3)
        vals = []
        for i, col in enumerate(EXPECTED_COLS):
            v = cols[i % 3].number_input(col, 0.0, 1.0, 0.95)
            vals.append(v)
        input_vec = np.array(vals)

    elif st.session_state.df_uploaded is not None:
        try:
            df_reordered = reorder_columns(st.session_state.df_uploaded,
                                           st.session_state.feature_order)
            input_vec = df_reordered.iloc[0].astype(float).to_numpy()
            st.info("Using first row from uploaded file for prediction.")
        except Exception as e:
            st.error(f"Column error: {e}")

    if st.session_state.model and input_vec is not None:
        soh_pred = predict_pack_soh(
            st.session_state.model,
            input_vec,
            st.session_state.scaler
        )

        status = "healthy" if soh_pred >= threshold else "problem"

        st.session_state.soh_info = {
            "soh": soh_pred,
            "status": status,
            "threshold": threshold
        }

        st.metric("Predicted SOH", f"{soh_pred:.3f}")
        plot_soh_gauge(soh_pred, threshold)

        if status == "healthy":
            st.success("✅ Battery is HEALTHY")
        else:
            st.error("⚠️ Battery has a PROBLEM")


    # Full dataset evaluation
    if st.session_state.df_uploaded is not None:
        if st.checkbox("Evaluate entire dataset"):
            try:
                df_reordered = reorder_columns(
                    st.session_state.df_uploaded,
                    st.session_state.feature_order
                )

                X_raw = df_reordered.astype(float)
                if st.session_state.scaler:
                    X = st.session_state.scaler.transform(X_raw)
                else:
                    X = X_raw.values

                preds = st.session_state.model.predict(X)
                st.session_state.df_uploaded["Predicted_SOH"] = preds

                st.write(st.session_state.df_uploaded.head())

                if "Actual_SOH" in st.session_state.df_uploaded.columns:
                    y_true = st.session_state.df_uploaded["Actual_SOH"].astype(float).values
                    scores = evaluate_predictions(y_true, preds)
                    st.metric("R²", f"{scores['R2']:.4f}")
                    st.metric("MSE", f"{scores['MSE']:.4f}")
                    st.metric("MAE", f"{scores['MAE']:.4f}")
                    plot_pred_vs_actual(y_true, preds)

            except Exception as e:
                st.error(f"Evaluation failed: {e}")


# ----------------------------------------------------
# TAB 2 — Gemini Chatbot
# ----------------------------------------------------
with tab2:
    st.header("💬 Gemini Chatbot")

    gemini_api_key = st.text_input("Gemini API Key", type="password")
    gemini_model = st.selectbox("Gemini Model", ["gemini-2.0-flash", "gemini-1.5-pro"])

    df_for_chat = st.session_state.df_uploaded

    if st.session_state.soh_info:
        st.subheader("🔎 Latest Predicted SOH")
        sohi = st.session_state.soh_info
        st.write(f"SOH: **{sohi['soh']:.3f}** — {sohi['status'].upper()}")
        plot_soh_gauge(sohi["soh"], sohi["threshold"])

    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    for role, msg in st.session_state.chat_history:
        st.markdown(f"**{role}:** {msg}")

    user_msg = st.text_input("Ask something:")

    if st.button("Send") and user_msg:
        st.session_state.chat_history.append(("You", user_msg))

        reply = gemini_chat(
            user_msg,
            api_key=gemini_api_key,
            model_name=gemini_model,
            soh_info=st.session_state.soh_info,
            chat_history=st.session_state.chat_history,
            df_uploaded=df_for_chat
        )

        st.session_state.chat_history.append(("Bot", reply))
        st.rerun()

st.markdown("---")
st.caption("✅ Fully Integrated Battery SOH Predictor + Gemini Chatbot (Dataset-Aware)")
