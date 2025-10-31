import streamlit as st
import pandas as pd
import numpy as np
import joblib
import google.generativeai as genai
import matplotlib.pyplot as plt

# -----------------------------
# 🔧 CONFIGURATION
# -----------------------------
genai.configure(api_key="AIzaSyBzfFP1AzOb95GoZ8PfKyr6C3QvqZnSR5w")  # Replace with your Gemini API key
model = joblib.load("model.pkl")  # Trained Linear Regression model
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

st.set_page_config(page_title="🔋 Battery SOH Chatbot", page_icon="🔋", layout="centered")

# -----------------------------
# 🧠 HELPER FUNCTIONS
# -----------------------------
def predict_battery_soh(cell_data):
    """Predict overall SOH of battery pack (U1–U21)."""
    soh_pred = model.predict([cell_data])[0]
    status = "healthy" if soh_pred >= 0.6 else "problem"
    return soh_pred, status


def ask_gemini(prompt, chat_history=None, soh_info=None):
    """Send user message to Gemini with full context (SOH + history)."""
    context = ""

    if soh_info:
        context += f"The user's latest battery SOH prediction is {soh_info['soh']:.3f}, meaning the battery is {soh_info['status']}.\n"

    if chat_history:
        context += "Conversation history:\n"
        for role, msg in chat_history:
            context += f"{role}: {msg}\n"

    full_prompt = f"{context}\nUser: {prompt}\nAssistant:"
    try:
        response = gemini_model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error communicating with Gemini API: {e}"


def plot_soh_gauge(soh_value):
    """Visualize SOH as a horizontal bar gauge."""
    fig, ax = plt.subplots(figsize=(5, 0.6))
    ax.barh(["SOH"], [soh_value * 100], color="green" if soh_value >= 0.6 else "red")
    ax.set_xlim(0, 100)
    ax.set_xlabel("State of Health (%)")
    ax.set_title("Battery Pack SOH Visualization")
    ax.bar_label(ax.containers[0], fmt="%.1f%%")
    ax.get_yaxis().set_visible(False)
    st.pyplot(fig)


# -----------------------------
# 🧩 STREAMLIT APP LAYOUT
# -----------------------------
st.title("🔋 Battery Pack SOH Predictor & Gemini Chatbot")
st.write("Predict your battery’s state of health (SOH) and chat with Google Gemini for insights, explanations, and recommendations.")

# Initialize session state
if "soh_info" not in st.session_state:
    st.session_state.soh_info = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Tabs
tab1, tab2 = st.tabs(["📊 Battery SOH Prediction", "💬 Gemini Chatbot"])

# -----------------------------
# TAB 1 — Battery Prediction
# -----------------------------
with tab1:
    st.header("📊 Predict Battery SOH")
    st.write("Upload your cell SOH data (U1–U21) as a CSV file or enter manually below:")

    uploaded_file = st.file_uploader("Upload CSV (21 columns for U1–U21)", type=["xlsx"])
    manual_input = st.checkbox("Or enter values manually")

    cell_data = None

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    expected_cols = [f"U{i}" for i in range(1,22)]

    if all(col in df.columns for col in expected_cols):
        filtered_df = df[expected_cols].apply(pd.to_numeric, errors="coerce")

        st.success("✅ Data loaded successfully.")
        st.write(filtered_df)

        # ✅ Extract 1st row as model input
        cell_data = filtered_df.iloc[0].values.tolist()

        if st.button("🔮 Predict SOH"):
            soh_pred, status = predict_battery_soh(cell_data)
            st.session_state.soh_info = {"soh": soh_pred, "status": status}

            st.metric("Predicted SOH", f"{soh_pred:.3f}")
            if status == "healthy":
                st.success("✅ The battery is HEALTHY.")
            else:
                st.error("⚠️ The battery has a PROBLEM.")

            plot_soh_gauge(soh_pred)


# -----------------------------
# TAB 2 — Gemini Chatbot
# -----------------------------
with tab2:
    st.header("💬 Chat with Gemini")
    st.write("Ask questions like *‘What does my SOH mean?’* or *‘How can I extend battery life?’*")

    # 🔁 Reset chat option
    if st.button("🧹 Reset Chat"):
        st.session_state.chat_history = []
        st.success("Chat history cleared! Start a new conversation.")
        st.stop()  # prevent showing old messages on reset

    # Show latest SOH gauge
    if st.session_state.soh_info:
        soh = st.session_state.soh_info['soh']
        st.subheader("🔎 Latest Battery SOH")
        plot_soh_gauge(soh)
        st.caption(f"Current SOH: {soh:.3f} → {st.session_state.soh_info['status'].upper()}")

    # Show chat history
    for role, text in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"**🧍 {role}:** {text}")
        else:
            st.markdown(f"**🤖 {role}:** {text}")

    # Chat input
    user_input = st.text_input("You:", placeholder="Ask something about your battery...")

    if st.button("Send") and user_input:
        st.session_state.chat_history.append(("You", user_input))
        response = ask_gemini(
            user_input,
            chat_history=st.session_state.chat_history,
            soh_info=st.session_state.soh_info
        )
        st.session_state.chat_history.append(("Bot", response))
        st.rerun()

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Built using Streamlit, Scikit-learn, Matplotlib, and Google Gemini.")
