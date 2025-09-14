# medai_chatbot_app.py

import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np
import os

# ---------------------- Streamlit Page Setup ----------------------
st.set_page_config(page_title="MedAI Chatbot Expert System", layout="centered")

# ---------------------- Load Pre-trained Model & Scaler ----------------------
MODEL_PATH = 'logistic_model_tuned.pkl''
SCALER_PATH = 'medai_scaler.pkl'

medai_diabetes_model = None
medai_scaler = None

try:
    medai_diabetes_model = joblib.load(MODEL_PATH)
    medai_scaler = joblib.load(SCALER_PATH)
    st.sidebar.success("✅ Diabetes predictive model loaded")
except FileNotFoundError:
    st.sidebar.error("❌ Model or scaler file not found")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model/scaler: {e}")

# ---------------------- Knowledge Base ----------------------
chatbot_knowledge_base = {
    # Greetings
    r"hi|hello|hey": "Hello! Welcome to MedAI. How can I assist you today?",
    r"how are you": "I’m doing great and ready to help you.",

    # General Health
    r"(what to do|how to deal|remedy|treatment|tips).*(fever|cough|headache|ear pain|stomach ache|joint pain)": "For common health issues, rest, hydrate, and consult a doctor if symptoms persist.",
    r"fever": "Fever is a sign of infection. Rest, hydrate, and use fever reducers. See a doctor if it’s persistent.",
    r"cough": "For cough, drink warm fluids, rest, and use cough syrup if needed. See a doctor if it lasts more than 2 weeks.",
    r"headache": "Headaches may be from stress, dehydration, or illness. Rest, drink water, and use mild pain relief.",
    r"ear pain": "Ear pain may be caused by infections or wax buildup. Avoid inserting objects, and see a doctor if it persists.",
    r"stomach ache": "Stomach ache can be from indigestion, gas, or infection. Rest, hydrate, and avoid spicy foods. If severe, consult a doctor.",
    r"joint pain": "Joint pain can be due to strain, arthritis, or injury. Rest, apply ice packs, and consult a doctor if persistent.",
    r"healthy diet": "Eat fruits, vegetables, whole grains, lean proteins, and healthy fats.",
    r"exercise benefits": "Exercise improves heart health, mood, and overall fitness.",
    r"water intake": "Most adults need 2-3 liters of water daily.",
    r"mental health": "Take breaks, get enough sleep, talk to loved ones.",
    r"stress relief": "Practice mindfulness, exercise, and talk to someone you trust.",
    r"sleep tips": "Maintain a regular sleep schedule, create a relaxing bedtime routine, and limit screen time before bed.",
    r"quit smoking|stop smoking": "Seek support, set a quit date, and consider nicotine replacement therapy.",
    r"alcohol consumption": "Limit to moderate levels: up to 1 drink a day or less.",
    r"healthy weight|weight loss tips": "Eat a balanced diet, exercise regularly, and consult a healthcare professional for personalized advice.",
    r"healthy lifestyle": "Eat well, exercise regularly, sleep enough, and manage stress.",
    r"preventive care": "Regular check-ups, vaccinations, and screenings are key.",
    r"vaccinations|immunizations": "Stay updated with your immunization schedule.",
    r"diabetic wound care": "Keep the wound clean and dry. Consult a doctor for proper treatment.",
    r"heart health": "Eat healthy, exercise, and manage stress. Seek immediate medical attention for chest pain.",
    r"blood pressure|hypertension": "Eat healthy, exercise, and manage stress to maintain healthy blood pressure.",

    # Diabetes
    r"what is diabetes|diabetes explained": "Diabetes is when your body can’t properly use or make insulin.",
    r"symptoms of diabetes": "Thirst, frequent urination, fatigue, and blurred vision.",
    r"risk factors for diabetes": "Obesity, inactivity, family history, and high blood pressure.",
    r"manage diabetes": "Diet, exercise, medication, and regular checkups.",
    r"prevent diabetes": "Healthy weight, regular activity, and balanced diet.",
    r"diabetes prediction|risk assessment|check my risk": "Type 'predict diabetes' to start.",

    # Farewell
    r"bye|goodbye|see you": "Goodbye! Take care.",
    r"thank you|thanks": "You’re welcome!"
}

# ---------------------- States ----------------------
def init_states():
    st.session_state.prediction_state = {
        "active": False, "current_step": 0, "data": {},
        "fields": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                   "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
        "medians": {'Pregnancies': 0.0, 'Glucose': 117.0, 'BloodPressure': 72.0,
                    'SkinThickness': 29.0, 'Insulin': 125.0, 'BMI': 32.0,
                    'DiabetesPedigreeFunction': 0.3725, 'Age': 29.0}
    }
    st.session_state.appointment_state = {"active": False, "current_step": 0, "data": {}, "fields": ["Full Name", "Date", "Time", "Reason for Appointment","Contact Info (Mobile/Email)"]}
    st.session_state.consultant_state = {"active": False, "current_step": 0, "data": {}, "fields": ["Full Name", "Phone or Email", "Preferred Time to Contact"]}
    st.session_state.insurance_state = {"active": False, "current_step": 0, "data": {}, "fields": ["Full Name", "Phone or Email"]}

if 'prediction_state' not in st.session_state:
    init_states()

# ---------------------- Flow Handlers ----------------------
def cancel_check(user_input, flow_name):
    if user_input.lower().strip() in ["cancel", "exit"]:
        st.session_state[flow_name]["active"] = False
        return f"❌ {flow_name.replace('_', ' ').title()} cancelled."
    return None

def handle_prediction_input(user_input):
    cancel_msg = cancel_check(user_input, "prediction_state")
    if cancel_msg: return cancel_msg

    s = st.session_state.prediction_state
    current_field = s["fields"][s["current_step"]]
    if user_input.lower() in ['skip', '0']:
        s["data"][current_field] = np.nan
    else:
        try:
            val = float(re.sub(r'[^\d.]', '', user_input))
            s["data"][current_field] = val
        except ValueError:
            return f"Invalid input for {current_field}. Enter a number or 'skip'."
    s["current_step"] += 1
    if s["current_step"] >= len(s["fields"]):
        s["active"] = False
        return make_diabetes_prediction(s["data"])
    return f"Enter your **{s['fields'][s['current_step']]}** value:"

def make_diabetes_prediction(user_data):
    if not medai_diabetes_model or not medai_scaler:
        return "Prediction unavailable."
    df = pd.DataFrame([user_data])
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(st.session_state.prediction_state["medians"][col])
    scaled = medai_scaler.transform(df)
    proba = medai_diabetes_model.predict_proba(scaled)[0, 1]
    pred = medai_diabetes_model.predict(scaled)[0]
    risk = proba * 100
    return f"Your diabetes risk is **{risk:.1f}%**. {'High risk - consult a doctor.' if pred == 1 else 'Low risk - maintain healthy habits.'}"

def handle_generic_flow(user_input, flow_name, filename, success_msg):
    cancel_msg = cancel_check(user_input, flow_name)
    if cancel_msg: return cancel_msg

    s = st.session_state[flow_name]
    s["data"][s["fields"][s["current_step"]]] = user_input.strip()
    s["current_step"] += 1
    if s["current_step"] >= len(s["fields"]):
        s["active"] = False
        return save_to_csv(filename, s["data"], success_msg)
    return f"Please provide your **{s['fields'][s['current_step']]}**."

# ---------------------- CSV Save ----------------------
def save_to_csv(filename, data, success_msg):
    try:
        exists = os.path.exists(filename)
        pd.DataFrame([data]).to_csv(filename, mode='a', header=not exists, index=False)
        return success_msg
    except Exception as e:
        return f"⚠️ Could not save: {e}"

# ---------------------- Chatbot Logic ----------------------
def get_chatbot_response(user_input):
    if st.session_state.prediction_state["active"]:
        return handle_prediction_input(user_input)
    if st.session_state.appointment_state["active"]:
        return handle_generic_flow(user_input, "appointment_state", "appointments.csv", "✅ Appointment booked!")
    if st.session_state.consultant_state["active"]:
        return handle_generic_flow(user_input, "consultant_state", "consultant_requests.csv", "✅ Request received! Our team will contact you soon.")
    if st.session_state.insurance_state["active"]:
        return handle_generic_flow(user_input, "insurance_state", "insurance_requests.csv", "✅ Insurance info request received!")

    if re.search(r"predict diabetes", user_input):
        if medai_diabetes_model and medai_scaler:
            s = st.session_state.prediction_state
            s.update({"active": True, "current_step": 0, "data": {}})
            return "Let's start. Enter your **Pregnancies**:"
        return "Prediction unavailable."

    if re.search(r"book appointment|schedule appointment", user_input):
        available_dates = ["15 Aug 2025", "18 Aug 2025", "21 Aug 2025","Or a Enter a date of your choice after the latest date mentioned"]
        s = st.session_state.appointment_state
        s.update({"active": True, "current_step": 0, "data": {}})
        return f"Available dates: {', '.join(available_dates)}.\nPlease provide your **Full Name**."

    if re.search(r"talk to consultant", user_input):
        s = st.session_state.consultant_state
        s.update({"active": True, "current_step": 0, "data": {}})
        return "⚠️ Consultants busy. Please provide your **Full Name** so we can contact you."

    if re.search(r"insurance connect|insurance info", user_input):
        s = st.session_state.insurance_state
        s.update({"active": True, "current_step": 0, "data": {}})
        return "We accept Sukoon, Medent, Metlife, DubaiHealth, and more. Please provide your **Full Name** and your insurance name with its package details."

    for pattern, response in chatbot_knowledge_base.items():
        if re.search(pattern, user_input.lower()):
            return response

    return "I’m sorry, I didn’t understand. Can you rephrase?"

# ---------------------- Sidebar ----------------------
def trigger_sidebar_action(user_message):
    st.session_state.messages.append({"role": "user", "content": user_message})
    bot_reply = get_chatbot_response(user_message)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

st.sidebar.header("Quick Actions")
if st.sidebar.button("📅 Schedule Appointment"):
    trigger_sidebar_action("schedule appointment")
if st.sidebar.button("🩺 Talk to a Consultant"):
    trigger_sidebar_action("talk to consultant")
if st.sidebar.button("🛡 MEDAi Insurance Connect"):
    trigger_sidebar_action("insurance connect")

# ---------------------- UI ----------------------
st.title("👨🏻‍⚕️ MedAI Chatbot Expert System")
st.warning("⚠ Always consult a healthcare professional for medical advice. I am not a substitute for professional care.")

if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your message...")
if user_input:
    user_lower = user_input.lower().strip()

    # Global exit
    if user_lower in ['bye', 'goodbye', 'exit']:
        st.session_state.clear()
        init_states()
        st.session_state.messages = [{"role": "assistant", "content": "👋 Chat cleared. Goodbye!"}]
        st.rerun()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    bot_reply = get_chatbot_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)

