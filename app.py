# ====================================================
#  app.py — Cleaned & Optimized Version (Full Features)
# ====================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import google.generativeai as genai
import io
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------------------------------------------
# STREAMLIT APP CONFIG
# ---------------------------------------------
st.set_page_config(
    page_title="🔋 Battery SOH Predictor + Gemini Chatbot",
    page_icon="🔋",
    layout="centered"
)

st.title("🔋 Battery Pack SOH Predictor & Gemini Chatbot")
st.write("Predict battery SOH, evaluate dataset performance, and chat with Gemini.")


# ---------------------------------------------
# GLOBAL CONSTANTS
# ---------------------------------------------
EXPECTED_COLS = [f"U{i}" for i in range(1, 22)]
DEFAULT_THRESHOLD = 0.6


# ---------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------
def safe_load_joblib(path):
    """Load joblib file safely."""
    try:
        return joblib.load(path)
    except:
        return None


def load_feature_order(path="feature_order.pkl"):
    """Load training-time feature order."""
    try:
        return joblib.load(path)
    except:
        return EXPECTED_COLS


def reorder_columns(df, feature_order):
    """Ensure uploaded dataset follows model's training column order."""
    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    return df[feature_order]


def predict_pack_soh(model, vec, scaler=None):
    """Make a single SOH prediction."""
    X = np.array(vec).reshape(1, -1)
    if scaler is not None:
        X = scaler.transform(X)
    return float(model.predict(X)[0])


def plot_soh_gauge(soh, threshold):
    """Horizontal SOH gauge bar."""
    fig, ax = plt.subplots(figsize=(6, 1))
    percent = np.clip(soh, 0, 1) * 100
    color = "green" if soh >= threshold else "red"

    ax.barh([0], [percent], color=color)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("SOH (%)")
    ax.bar_label(ax.containers[0], fmt="%.1f%%")

    st.pyplot(fig)


def evaluate_predictions(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def summarize_dataframe(df, max_rows=5):
    """Create a text summary for LLM context."""
    numeric_cols = df.select_dtypes(include='number').columns

    summary = [
        f"Columns: {', '.join(df.columns)}",
        f"Number of rows: {len(df)}",
        "\nSample rows:\n" + df.head(max_rows).to_string(index=False)
    ]

    if len(numeric_cols) > 0:
        summary.append("\nNumeric statistics:\n" + df[numeric_cols].describe().to_string())

    return "\n".join(summary)


# ---------------------------------------------
# GEMINI CHATBOT — Dataset + SOH Aware
# ---------------------------------------------
def gemini_chat(user_prompt, api_key, model_name, soh_info, chat_history, df_uploaded):
    if not api_key:
        return "❌ Gemini API key is missing."

    # --- Basic Intent Detection ---
    prompt_lower = user_prompt.lower()
    wants_dataset = any([
        "data" in prompt_lower,
        "dataset" in prompt_lower,
        "csv" in prompt_lower,
        "value" in prompt_lower,
        "row" in prompt_lower,
        "column" in prompt_lower,
        "u1" in prompt_lower or "u21" in prompt_lower,
        "soh" in prompt_lower and df_uploaded is not None,
        "analyze" in prompt_lower,
        "summarize" in prompt_lower
    ])

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)

        # -------------------------------
        # ✅ Build context dynamically
        # -------------------------------
        context = []

        # Only include SOH if prediction exists
        if soh_info:
            context.append(
                f"The battery pack's latest predicted SOH is {soh_info['soh']:.3f}, "
                f"which is considered '{soh_info['status']}' based on a threshold of "
                f"{soh_info['threshold']:.2f}."
            )

        # ✅ Include dataset summary ONLY WHEN RELEVANT
        if wants_dataset and df_uploaded is not None:
            context.append("Here is the dataset summary:\n" + summarize_dataframe(df_uploaded))
        elif wants_dataset and df_uploaded is None:
            context.append("⚠️ The user asked about the dataset, but no dataset is uploaded.")

        # ✅ Add conversation history (shortened for clarity)
        if chat_history:
            formatted_history = "\n".join([f"{r}: {m}" for r, m, _ in chat_history[-6:]])
            context.append("Recent conversation:\n" + formatted_history)

        # ✅ Final user prompt
        context.append(f"User: {user_prompt}\nAssistant:")

        full_prompt = "\n\n".join(context)

        response = model.generate_content(full_prompt)
        return response.text.strip()

    except Exception as e:
        return f"⚠️ Gemini Error: {e}"



# ---------------------------------------------
# SESSION STATE
# ---------------------------------------------
for key in ["df_uploaded", "soh_info", "chat_history"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat_history" else None


# ---------------------------------------------
# SIDEBAR — MODEL LOADING
# ---------------------------------------------
st.sidebar.header("Model Files")

model_path = st.sidebar.text_input("Model (.pkl)", "model.pkl")
scaler_path = st.sidebar.text_input("Scaler (optional)", "")
feat_path = st.sidebar.text_input("Feature order (.pkl)", "feature_order.pkl")

if st.sidebar.button("Load Model"):
    st.session_state.model = safe_load_joblib(model_path)
    st.session_state.scaler = safe_load_joblib(scaler_path) if scaler_path else None
    st.session_state.feature_order = load_feature_order(feat_path)

    if st.session_state.model:
        st.sidebar.success("✅ Model loaded successfully")
    else:
        st.sidebar.error("❌ Failed to load model.")


# ---------------------------------------------
# MAIN TABS
# ---------------------------------------------
tab1, tab2 = st.tabs(["📊 SOH Prediction", "💬 Gemini Chatbot"])


# =====================================================
# TAB 1 — PREDICTION
# =====================================================
with tab1:
    st.header("📊 Predict Battery Pack SOH")

    uploaded = st.file_uploader("Upload CSV/XLSX with U1–U21", ["csv", "xlsx"])
    manual_mode = st.checkbox("Enter values manually")
    threshold = st.slider("Healthy Threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01)

    # --- Load dataset ---
    if uploaded:
        try:
            df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            st.write(df.head())
            st.session_state.df_uploaded = df
        except Exception as e:
            st.error(f"Error reading file: {e}")

    # --- Manual input or row selection ---
    input_vec = None

    if manual_mode:
        cols = st.columns(3)
        vals = []
        for i, col in enumerate(EXPECTED_COLS):
            vals.append(cols[i % 3].number_input(col, 0.0, 1.0, 0.95))
        input_vec = np.array(vals)

    elif st.session_state.df_uploaded is not None:
        try:
            df_reordered = reorder_columns(st.session_state.df_uploaded, st.session_state.feature_order)
            input_vec = df_reordered.iloc[0].astype(float).to_numpy()
            st.info("Using first row for prediction.")
        except Exception as e:
            st.error(f"Column error: {e}")

    # --- Single Prediction ---
    if input_vec is not None and st.session_state.model:
        soh_pred = predict_pack_soh(
            st.session_state.model, input_vec, st.session_state.scaler
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


    # --- Evaluate entire dataset ---
    if st.session_state.df_uploaded is not None:
        if st.checkbox("Evaluate entire dataset"):
            try:
                df_reordered = reorder_columns(st.session_state.df_uploaded, st.session_state.feature_order)

                X = df_reordered.astype(float)
                if st.session_state.scaler:
                    X = st.session_state.scaler.transform(X)

                preds = st.session_state.model.predict(X)
                st.session_state.df_uploaded["Predicted_SOH"] = preds

                st.write(st.session_state.df_uploaded.head())

                if "Actual_SOH" in st.session_state.df_uploaded.columns:
                    y_true = st.session_state.df_uploaded["Actual_SOH"].astype(float)
                    metrics = evaluate_predictions(y_true, preds)

                    st.metric("R²", f"{metrics['R2']:.4f}")
                    st.metric("MSE", f"{metrics['MSE']:.4f}")
                    st.metric("MAE", f"{metrics['MAE']:.4f}")

                    plot_pred_vs_actual(y_true, preds)

            except Exception as e:
                st.error(f"Dataset evaluation error: {e}")


# =====================================================
# TAB 2 — Gemini Chatbot (Improved Chat UI)
# =====================================================
with tab2:
    st.header("💬 Gemini Chatbot")

    gemini_api_key = st.text_input("Gemini API Key", type="password")
    gemini_model = st.selectbox("Gemini Model", ["gemini-2.0-flash", "gemini-1.5-pro"])

    df_for_chat = st.session_state.df_uploaded

    # --- Latest SOH Display ---
    if st.session_state.soh_info:
        sohi = st.session_state.soh_info
        st.subheader("🔎 Latest Prediction")
        st.write(f"SOH: **{sohi['soh']:.3f}** — {sohi['status'].upper()}")
        plot_soh_gauge(sohi["soh"], sohi["threshold"])

    # --- Clear Chat ---
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    # =====================================================
    # ✅ NEW & IMPROVED CHAT BUBBLE SYSTEM
    # =====================================================
    for role, msg, ts in st.session_state.chat_history:
        is_user = (role == "User")

        # Colors optimized for dark/light themes
        bg_color = "#2F80ED" if is_user else "#333333"    # blue (user) / dark gray (bot)
        text_color = "white"
        align = "right" if is_user else "left"

        with st.chat_message(role.lower()):
            st.markdown(
                f"""
                <div style="
                    text-align: {align};
                    margin-bottom: 8px;
                ">
                    <div style="
                        background-color: {bg_color};
                        color: {text_color};
                        padding: 12px 16px;
                        border-radius: 12px;
                        display: inline-block;
                        max-width: 80%;
                        font-size: 16px;
                        line-height: 1.4;
                        box-shadow: 0px 2px 6px rgba(0,0,0,0.25);
                        word-wrap: break-word;
                    ">
                        {msg}
                        <div style="
                            font-size: 12px;
                            color: #BBBBBB;
                            text-align: right;
                            margin-top: 4px;
                        ">
                            {ts}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # =====================================================
    # ✅ Chat Input (unchanged, but works with new UI)
    # =====================================================
    user_msg = st.chat_input("Ask Gemini something:")

    if user_msg:
        # Store visible user message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.chat_history.append(("User", user_msg, timestamp))

        # Inject threshold into model prompt if needed
        if st.session_state.soh_info:
            thr = st.session_state.soh_info["threshold"]
            internal_prompt = f"(IMPORTANT: Use threshold {thr:.2f} for SOH checks.) {user_msg}"
        else:
            internal_prompt = user_msg

        # Call Gemini
        reply = gemini_chat(
            internal_prompt,
            gemini_api_key,
            gemini_model,
            st.session_state.soh_info,
            st.session_state.chat_history,
            df_for_chat
        )

        # Store bot reply
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.chat_history.append(("Bot", reply, ts))

        st.rerun()
