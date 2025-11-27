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


def plot_pred_vs_actual(y_true, y_pred):
    """Scatter plot comparing predicted vs actual SOH."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k")
    min_v = min(min(y_true), min(y_pred))
    max_v = max(max(y_true), max(y_pred))
    ax.plot([min_v, max_v], [min_v, max_v], 'r--', label='Ideal')
    ax.set_xlabel('Actual SOH')
    ax.set_ylabel('Predicted SOH')
    ax.set_title('Predicted vs Actual SOH')
    ax.legend()
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
# GEMINI MODEL UTILITIES
# ---------------------------------------------
def get_available_gemini_models(api_key):
    """Return a list of available Gemini model IDs that support generateContent.
    Falls back to empty list on any error. Strips the leading 'models/' prefix
    for easier selection in the UI."""
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        supported = []
        for m in models:
            # Some client versions expose supported_generation_methods
            if hasattr(m, "supported_generation_methods") and "generateContent" in getattr(m, "supported_generation_methods", []):
                name = getattr(m, "name", "")
                if name.startswith("models/"):
                    name = name[len("models/") :]
                supported.append(name)
        return sorted(set(supported))
    except Exception:
        return []


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
        # Enhanced error messaging: capture last error & suggest available models if 404/not found
        err_msg = str(e)
        try:
            st.session_state["gemini_last_error"] = err_msg
        except Exception:
            pass
        if any(code in err_msg.lower() for code in ["404", "not found"]):
            avail = get_available_gemini_models(api_key)
            suggestion = "\nAvailable models supporting generateContent: " + ", ".join(avail) if avail else "\n(No supported models returned – check API key / quota / client version.)"
            return f"⚠️ Gemini Error: {err_msg}{suggestion}"
        return f"⚠️ Gemini Error: {err_msg}"



# ---------------------------------------------
# SESSION STATE
# ---------------------------------------------
for key in ["df_uploaded", "soh_info", "chat_history"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat_history" else None
# Track last Gemini API error for diagnostics
if "gemini_last_error" not in st.session_state:
    st.session_state["gemini_last_error"] = None


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
tab1, tab2, tab3 = st.tabs(["📊 SOH Prediction", "💬 Gemini Chatbot", "ℹ️ About"])


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
    
    # Only show model selector after API key is entered
    gemini_model = None
    if gemini_api_key:
        # Dynamic model retrieval: list models only after API key entered
        default_models = ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
        dynamic_models = get_available_gemini_models(gemini_api_key)
        # Filter to gemini-* style to avoid unrelated endpoints
        dynamic_models = [m for m in dynamic_models if m.startswith("gemini-")]
        model_options = dynamic_models if dynamic_models else default_models
        gemini_model = st.selectbox("Gemini Model", model_options)

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

        # Battery health heuristic fallback if error returned
        lower_msg = user_msg.lower()
        if reply.startswith("⚠️ Gemini Error") and ("keep" in lower_msg and "battery" in lower_msg and "healthy" in lower_msg):
            reply += "\n\nStatic battery health tips (fallback):\n- Avoid deep discharges; keep SOC 20–80%.\n- Minimize time spent at 100% or 0%.\n- Charge/operate within recommended temperature window.\n- Ensure periodic cell balancing via BMS.\n- Store long-term at ~50–60% SOC in a cool place.\n- Monitor cell voltage spread; increasing delta signals aging."

        # Store bot reply
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.chat_history.append(("Bot", reply, ts))

        st.rerun()


    # =====================================================
    # TAB 3 — ABOUT / PROJECT INFO
    # =====================================================
    with tab3:
        st.header("ℹ️ About This Project")
    
        st.markdown("""
        ### 🎓 Academic Context
        **Course:** SOFE 3370 - Data Structures and Algorithms  
        **Institution:** Ontario Tech University  
        **Group:** Group 13  
        **Date:** Fall 2025
    
        ---
    
        ### 🔋 What is State of Health (SOH)?
    
        **State of Health (SOH)** measures a battery's current capacity relative to its original rated capacity.
    
        - **SOH = 1.0 (100%)**: Battery is in perfect condition
        - **SOH = 0.8 (80%)**: Battery shows signs of degradation
        - **SOH < 0.6 (60%)**: Battery requires attention or replacement
    
        **Why it matters:**
        - Critical for electric vehicle battery management
        - Enables predictive maintenance scheduling
        - Optimizes battery lifecycle and replacement planning
        - Ensures safety and reliability of energy storage systems
    
        ---
    
        ### 🧠 Model Architecture
    
        **Algorithm:** Linear Regression  
        **Input Features:** 21 voltage measurements (U1 through U21)  
        **Target Variable:** Battery SOH  
    
        **Training Configuration:**
        - Data split: 80% training, 20% testing
        - Data preprocessing: Merge sort algorithm (O(n log n))
        - Model serialization: Joblib (.pkl format)
    
        **Performance Metrics:**
        - **R² Score:** ~0.95+ (high correlation)
        - **Mean Squared Error (MSE):** < 0.01
        - **Mean Absolute Error (MAE):** < 0.05
    
        **Why Linear Regression?**
        - Fast training and prediction
        - Interpretable coefficients for each voltage sensor
        - Excellent performance on linearly correlated battery data
        - Low computational overhead for real-time deployment
    
        ---
    
        ### 📊 Dataset
    
        **Source:** PulseBat Dataset  
        **Format:** Excel (.xlsx) with 22 columns  
        **Features:** U1-U21 (individual cell voltages)  
        **Target:** SOH (State of Health)
    
        The dataset contains battery pack measurements across various charge/discharge cycles,
        enabling accurate SOH prediction based on voltage patterns.
    
        ---
    
        ### 🚀 Key Features
    
        ✅ **ML-Powered Prediction** - Linear regression trained on battery voltage data  
        ✅ **Interactive Interface** - User-friendly Streamlit web application  
        ✅ **AI Chatbot** - Google Gemini integration for intelligent analysis  
        ✅ **Real-Time Visualization** - Dynamic SOH gauge and performance metrics  
        ✅ **Batch Processing** - Evaluate entire datasets with R², MSE, MAE  
        ✅ **Flexible Input** - Support for CSV/XLSX uploads and manual entry
    
        ---
    
        ### 🛠️ Technologies Used
    
        - **Python 3.x** - Core programming language
        - **Streamlit** - Web application framework
        - **scikit-learn** - Machine learning (Linear Regression)
        - **pandas & numpy** - Data manipulation
        - **matplotlib** - Visualization
        - **google-generativeai** - Gemini AI chatbot
        - **joblib** - Model serialization
    
        ---
    
        ### 📖 How to Use
    
        **Tab 1 - SOH Prediction:**
        1. Upload a CSV/XLSX file with U1-U21 columns, or
        2. Enter voltage values manually
        3. Adjust the healthy threshold (default: 0.60)
        4. View predicted SOH, gauge, and status
        5. Enable "Evaluate entire dataset" for batch predictions
    
        **Tab 2 - Gemini Chatbot:**
        1. Enter your Google Gemini API key
        2. Select a model from the dropdown
        3. Ask questions about battery health, dataset, or SOH predictions
        4. Get AI-powered insights and recommendations
    
        ---
    
        ### 📞 Support
    
        For questions or issues:
        - Check the README.md in the project repository
        - Review the training code in LinearRegression.py
        - Ensure all dependencies are installed via requirements.txt
    
        ---
    
        *Last Updated: November 2025*
        """)
