import streamlit as st
from tensorflow import keras
import numpy as np
import pandas as pd
import sqlite3
import json

from pathlib import Path
from PIL import Image, ImageOps
from datetime import date, datetime, timedelta
from io import BytesIO
from gtts import gTTS


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌿",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "plant_disease_model.keras"
CLASS_PATH = BASE_DIR / "model" / "class_names.json"
DATABASE_PATH = BASE_DIR / "plantcare.db"


# =========================================================
# 2. SESSION STATE
# =========================================================

if "language" not in st.session_state:
    st.session_state.language = "en"

if "theme" not in st.session_state:
    st.session_state.theme = "Light"

if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None


# =========================================================
# 3. LANGUAGE CONTENT
# =========================================================

TEXT = {
    "en": {
        "home": "Home",
        "detection": "Disease Detection",
        "history": "Prediction History",
        "reminders": "Follow-up Reminders",
        "about": "About Project",
        "chatbot": "PlantCare Chatbot",
        "navigation": "Navigation",
        "language": "Choose Language",
        "theme": "Appearance",
        "light": "Light",
        "dark": "Dark",
        "welcome": "Welcome to PlantCare AI",
        "welcome_text": "An intelligent plant disease detection system that helps identify possible diseases from plant leaf images.",
        "upload": "Upload Plant Leaf Image",
        "preview": "Image Preview",
        "detect": "Detect Disease",
        "results": "Detection Results",
        "plant": "Plant",
        "prediction": "Predicted Condition",
        "score": "Model Score",
        "reliability": "Prediction Reliability",
        "symptoms": "Common Symptoms",
        "causes": "Possible Causes",
        "remedies": "Management Suggestions",
        "prevention": "Prevention Methods",
        "severity": "Select Observed Severity",
        "early": "Early",
        "moderate": "Moderate",
        "advanced": "Advanced",
        "healthy": "The model classified this leaf as healthy.",
        "diseased": "The model detected a possible plant disease.",
        "low_warning": "The model score is low. The result may be unreliable. Upload a clearer image.",
        "medium_warning": "The model score is moderate. Please verify the result with an agricultural expert.",
        "high_warning": "The model score is relatively high. This is still not a confirmed diagnosis.",
        "disclaimer": "This application provides preliminary information only. It cannot replace professional agricultural diagnosis.",
        "voice": "Listen to Results",
        "save_reminder": "Save Follow-up Reminder",
        "reminder_date": "Choose Follow-up Date",
        "history_empty": "No prediction records are available.",
        "reminders_empty": "No reminders are available.",
        "date": "Date",
        "status": "Status",
        "pending": "Pending",
        "completed": "Completed",
        "mark_complete": "Mark as Completed",
        "reminder_due": "Your follow-up reminder is due.",
        "download": "Download Prediction History",
        "features": "Main Features",
        "objective": "Project Objective",
        "technology": "Technologies Used",
        "about_overview": "Project Overview",
        "problem": "Problem Statement",
        "solution": "Proposed Solution",
        "architecture": "System Architecture",
        "workflow": "Working Process",
        "dataset": "Dataset Description",
        "model": "CNN Model Architecture",
        "evaluation": "Model Evaluation",
        "limitations": "Project Limitations",
        "future": "Future Enhancements",
        "applications": "Applications",
        "conclusion": "Conclusion",
        "test_accuracy": "Test Accuracy",
        "categories": "Disease Categories",
        "model_name": "Model",
        "select_image": "Please upload a plant leaf image first.",
        "prediction_error": "An error occurred during prediction.",
        "reminder_saved": "Your follow-up reminder has been saved.",
        "voice_error": "Voice generation failed. Check your internet connection.",
        "voice_text": "Listen to the prediction and management suggestions.",
        "no_image": "No image selected.",
        "unknown": "Unknown",
        "disease_info_unavailable": "Disease information is unavailable.",
        "about_intro": "Learn about the purpose, design, technologies and working process of PlantCare AI."
    },

    "ta": {
        "home": "முகப்பு",
        "detection": "தாவர நோய் கண்டறிதல்",
        "history": "கணிப்பு வரலாறு",
        "reminders": "தொடர் பராமரிப்பு நினைவூட்டல்கள்",
        "about": "திட்டத்தைப் பற்றி",
        "chatbot": "PlantCare உரையாடல் உதவியாளர்",
        "navigation": "வழிசெலுத்தல்",
        "language": "மொழியைத் தேர்ந்தெடுக்கவும்",
        "theme": "தோற்ற அமைப்பு",
        "light": "பகல் தோற்றம்",
        "dark": "இரவு தோற்றம்",
        "welcome": "PlantCare AI-க்கு வரவேற்கிறோம்",
        "welcome_text": "தாவர இலைப் படங்களைப் பயன்படுத்தி ஏற்படக்கூடிய நோய்களைக் கண்டறிய உதவும் அறிவார்ந்த அமைப்பு.",
        "upload": "தாவர இலையின் படத்தைப் பதிவேற்றவும்",
        "preview": "படத்தின் முன்னோட்டம்",
        "detect": "நோயைக் கண்டறியவும்",
        "results": "கண்டறிதல் முடிவுகள்",
        "plant": "தாவரம்",
        "prediction": "கணிக்கப்பட்ட நிலை",
        "score": "மாதிரியின் மதிப்பெண்",
        "reliability": "கணிப்பின் நம்பகத்தன்மை",
        "symptoms": "பொதுவான அறிகுறிகள்",
        "causes": "சாத்தியமான காரணங்கள்",
        "remedies": "மேலாண்மை பரிந்துரைகள்",
        "prevention": "தடுப்பு முறைகள்",
        "severity": "நீங்கள் கவனித்துள்ள பாதிப்பு அளவைத் தேர்ந்தெடுக்கவும்",
        "early": "ஆரம்ப நிலை",
        "moderate": "மிதமான நிலை",
        "advanced": "தீவிர நிலை",
        "healthy": "இந்த இலை ஆரோக்கியமானது என மாதிரி வகைப்படுத்தியுள்ளது.",
        "diseased": "இந்த இலையில் நோய் இருக்கலாம் என மாதிரி கணித்துள்ளது.",
        "low_warning": "மாதிரியின் மதிப்பெண் குறைவாக உள்ளது. கணிப்பு நம்பகமற்றதாக இருக்கலாம். தெளிவான படத்தைப் பதிவேற்றவும்.",
        "medium_warning": "மாதிரியின் மதிப்பெண் மிதமாக உள்ளது. வேளாண் நிபுணரிடம் முடிவைச் சரிபார்க்கவும்.",
        "high_warning": "மாதிரியின் மதிப்பெண் ஒப்பீட்டளவில் அதிகமாக உள்ளது. இருப்பினும் இது உறுதிப்படுத்தப்பட்ட நோய் கண்டறிதல் அல்ல.",
        "disclaimer": "இந்தச் செயலி ஆரம்பகட்ட தகவல்களை மட்டுமே வழங்குகிறது. இது வேளாண் நிபுணரின் நோய் கண்டறிதலுக்கு மாற்றாகாது.",
        "voice": "முடிவுகளைக் குரலில் கேட்கவும்",
        "save_reminder": "தொடர் பராமரிப்பு நினைவூட்டலைச் சேமிக்கவும்",
        "reminder_date": "மீண்டும் சரிபார்க்க வேண்டிய தேதியைத் தேர்ந்தெடுக்கவும்",
        "history_empty": "கணிப்பு பதிவுகள் எதுவும் இல்லை.",
        "reminders_empty": "நினைவூட்டல்கள் எதுவும் இல்லை.",
        "date": "தேதி",
        "status": "நிலை",
        "pending": "நிலுவையில் உள்ளது",
        "completed": "நிறைவு செய்யப்பட்டது",
        "mark_complete": "நிறைவு செய்ததாகக் குறிக்கவும்",
        "reminder_due": "தொடர் பராமரிப்பு நினைவூட்டலுக்கான தேதி வந்துவிட்டது.",
        "download": "கணிப்பு வரலாற்றைப் பதிவிறக்கவும்",
        "features": "முக்கிய அம்சங்கள்",
        "objective": "திட்டத்தின் நோக்கம்",
        "technology": "பயன்படுத்தப்பட்ட தொழில்நுட்பங்கள்",
        "about_overview": "திட்டத்தின் கண்ணோட்டம்",
        "problem": "பிரச்சினை விளக்கம்",
        "solution": "முன்மொழியப்பட்ட தீர்வு",
        "architecture": "அமைப்பு கட்டமைப்பு",
        "workflow": "செயல்படும் முறை",
        "dataset": "தரவுத்தொகுப்பு விளக்கம்",
        "model": "CNN மாதிரி கட்டமைப்பு",
        "evaluation": "மாதிரியின் மதிப்பீடு",
        "limitations": "திட்டத்தின் வரம்புகள்",
        "future": "எதிர்கால மேம்பாடுகள்",
        "applications": "பயன்பாடுகள்",
        "conclusion": "முடிவுரை",
        "test_accuracy": "சோதனைத் துல்லியம்",
        "categories": "நோய் வகைகள்",
        "model_name": "மாதிரி",
        "select_image": "முதலில் தாவர இலையின் படத்தைப் பதிவேற்றவும்.",
        "prediction_error": "கணிப்பின்போது பிழை ஏற்பட்டது.",
        "reminder_saved": "தொடர் பராமரிப்பு நினைவூட்டல் சேமிக்கப்பட்டது.",
        "voice_error": "குரல் உருவாக்கம் தோல்வியடைந்தது. இணைய இணைப்பைச் சரிபார்க்கவும்.",
        "voice_text": "கணிப்பு மற்றும் பராமரிப்பு பரிந்துரைகளைக் கேட்கவும்.",
        "no_image": "படம் தேர்ந்தெடுக்கப்படவில்லை.",
        "unknown": "அறியப்படாதது",
        "disease_info_unavailable": "நோய் பற்றிய தகவல் கிடைக்கவில்லை.",
        "about_intro": "PlantCare AI திட்டத்தின் நோக்கம், வடிவமைப்பு, தொழில்நுட்பங்கள் மற்றும் செயல்படும் முறையை அறிந்துகொள்ளுங்கள்."
    }
}


def t(key):
    return TEXT[st.session_state.language][key]


# =========================================================
# 4. SIDEBAR SETTINGS
# =========================================================

st.sidebar.title("🌿 PlantCare AI")

st.sidebar.subheader("🌗 " + t("theme"))

theme_options = [t("light"), t("dark")]

current_theme_label = (
    t("light") if st.session_state.theme == "Light"
    else t("dark")
)

selected_theme = st.sidebar.radio(
    t("theme"),
    theme_options,
    index=theme_options.index(current_theme_label),
    horizontal=True,
    label_visibility="collapsed"
)

st.session_state.theme = (
    "Light" if selected_theme == t("light") else "Dark"
)

st.sidebar.markdown("---")

language_options = ["English", "தமிழ்"]

selected_language = st.sidebar.selectbox(
    "🌐 " + t("language"),
    language_options,
    index=0 if st.session_state.language == "en" else 1
)

st.session_state.language = (
    "en" if selected_language == "English" else "ta"
)


# =========================================================
# 5. LIGHT AND DARK MODE
# =========================================================

if st.session_state.theme == "Dark":
    BG = "#121A16"
    TEXT_COLOR = "#F1F5F2"
    SIDEBAR = "#1B2921"
    CARD = "#24352A"
    BORDER = "#405847"
    INPUT = "#26372C"
    BUTTON = "#2E7D32"
else:
    BG = "#F4F8F1"
    TEXT_COLOR = "#263328"
    SIDEBAR = "#E3F0DF"
    CARD = "#FFFFFF"
    BORDER = "#D5E5D0"
    INPUT = "#FFFFFF"
    BUTTON = "#2E7D32"

st.markdown(f"""
<style>
.stApp {{
    background-color: {BG};
    color: {TEXT_COLOR};
}}

[data-testid="stHeader"] {{
    background-color: {BG};
}}

h1, h2, h3, h4, p, label {{
    color: {TEXT_COLOR};
}}

section[data-testid="stSidebar"] {{
    background-color: {SIDEBAR};
}}

section[data-testid="stSidebar"] * {{
    color: {TEXT_COLOR};
}}

.stButton > button {{
    background-color: {BUTTON};
    color: white !important;
    border-radius: 9px;
    border: none;
    padding: 10px 18px;
}}

.stButton > button:hover {{
    background-color: #1B5E20;
    color: white !important;
}}

div[data-testid="stMetric"] {{
    background-color: {CARD};
    padding: 16px;
    border-radius: 12px;
    border: 1px solid {BORDER};
}}

div[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: {CARD};
    border-color: {BORDER};
}}

.stTextInput input,
.stDateInput input,
.stSelectbox div[data-baseweb="select"] {{
    background-color: {INPUT};
    color: {TEXT_COLOR};
}}

[data-testid="stFileUploader"] {{
    background-color: {CARD};
    border-radius: 12px;
    padding: 12px;
}}

hr {{
    border-color: {BORDER};
}}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 6. DISEASE INFORMATION
# =========================================================

DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "plant_en": "Pepper",
        "plant_ta": "குடைமிளகாய்",
        "name_en": "Bacterial spot",
        "name_ta": "பாக்டீரியா இலைப்புள்ளி நோய்",
        "symptoms_en": "Small dark, water-soaked spots may appear on leaves. In severe cases, affected areas may enlarge.",
        "symptoms_ta": "இலைகளில் சிறிய கருமையான, நீர் தேங்கியதுபோன்ற புள்ளிகள் தோன்றலாம். கடுமையான நிலையில் பாதிக்கப்பட்ட பகுதிகள் பெரிதாகலாம்.",
        "causes_en": "Bacteria may spread through splashing water, infected plants and contaminated tools.",
        "causes_ta": "நீர் தெறிப்பு, பாதிக்கப்பட்ட செடிகள் மற்றும் அசுத்தமான கருவிகள் மூலம் பாக்டீரியா பரவலாம்.",
        "remedies_en": "Remove severely affected leaves. Avoid overhead watering and seek local agricultural advice about suitable management.",
        "remedies_ta": "கடுமையாகப் பாதிக்கப்பட்ட இலைகளை அகற்றவும். மேலிருந்து நீர் தெளிப்பதைத் தவிர்த்து பொருத்தமான பராமரிப்பிற்கு வேளாண் நிபுணரை அணுகவும்.",
        "prevention_en": "Use healthy seeds, keep leaves dry, clean tools and avoid working with wet plants.",
        "prevention_ta": "ஆரோக்கியமான விதைகளைப் பயன்படுத்தவும். இலைகளை உலர்வாக வைத்துக் கருவிகளைச் சுத்தமாகப் பராமரிக்கவும்."
    },

    "Pepper__bell___healthy": {
        "plant_en": "Pepper",
        "plant_ta": "குடைமிளகாய்",
        "name_en": "Healthy",
        "name_ta": "ஆரோக்கியமானது",
        "symptoms_en": "The model did not identify obvious disease symptoms in the uploaded image.",
        "symptoms_ta": "பதிவேற்றப்பட்ட படத்தில் தெளிவான நோய் அறிகுறிகளை மாதிரி கண்டறியவில்லை.",
        "causes_en": "No disease was identified in this prediction.",
        "causes_ta": "இந்தக் கணிப்பில் நோய் எதுவும் கண்டறியப்படவில்லை.",
        "remedies_en": "Continue normal watering, sunlight and regular plant care.",
        "remedies_ta": "வழக்கமான நீர்ப்பாசனம், சூரிய ஒளி மற்றும் தாவரப் பராமரிப்பைத் தொடரவும்.",
        "prevention_en": "Monitor the plant regularly and maintain good garden hygiene.",
        "prevention_ta": "தாவரத்தைத் தொடர்ந்து கண்காணித்து தோட்டத்தைச் சுத்தமாக வைத்திருக்கவும்."
    },

    "Potato___Early_blight": {
        "plant_en": "Potato",
        "plant_ta": "உருளைக்கிழங்கு",
        "name_en": "Early blight",
        "name_ta": "ஆரம்பகட்ட இலைக்கருகல் நோய்",
        "symptoms_en": "Brown spots, often with circular ring patterns, may appear on older leaves.",
        "symptoms_ta": "பழைய இலைகளில் வட்ட வடிவ வளையங்களுடன் பழுப்பு நிறப் புள்ளிகள் தோன்றலாம்.",
        "causes_en": "A fungal disease that can develop under warm and humid conditions.",
        "causes_ta": "வெப்பமான மற்றும் ஈரப்பதமான சூழலில் உருவாகக்கூடிய பூஞ்சை நோயாகும்.",
        "remedies_en": "Remove affected leaves and improve airflow. Ask an agricultural expert about suitable control measures.",
        "remedies_ta": "பாதிக்கப்பட்ட இலைகளை அகற்றி காற்றோட்டத்தை மேம்படுத்தவும். கட்டுப்பாட்டு முறைகளுக்கு வேளாண் நிபுணரை அணுகவும்.",
        "prevention_en": "Rotate crops, avoid overcrowding and water near the soil.",
        "prevention_ta": "பயிர் சுழற்சி முறையைப் பின்பற்றி செடிகளை நெருக்கமாக நடுவதைத் தவிர்த்து மண்ணின் அருகில் நீர் பாய்ச்சவும்."
    },

    "Potato___Late_blight": {
        "plant_en": "Potato",
        "plant_ta": "உருளைக்கிழங்கு",
        "name_en": "Late blight",
        "name_ta": "தாமத இலைக்கருகல் நோய்",
        "symptoms_en": "Dark, spreading lesions may develop on leaves and stems. The disease can spread quickly in favourable weather.",
        "symptoms_ta": "இலைகள் மற்றும் தண்டுகளில் கருமையான பரவும் புள்ளிகள் தோன்றலாம். ஏற்ற காலநிலையில் நோய் விரைவாகப் பரவலாம்.",
        "causes_en": "A serious disease favoured by cool, wet and humid conditions.",
        "causes_ta": "குளிர்ச்சியான, ஈரமான மற்றும் அதிக ஈரப்பதமுள்ள சூழலில் அதிகரிக்கக்கூடிய நோயாகும்.",
        "remedies_en": "Remove affected material safely where appropriate and seek prompt agricultural advice.",
        "remedies_ta": "தேவையான இடங்களில் பாதிக்கப்பட்ட பகுதிகளைப் பாதுகாப்பாக அகற்றி உடனடியாக வேளாண் ஆலோசனை பெறவும்.",
        "prevention_en": "Use healthy seed potatoes, provide adequate spacing and avoid prolonged leaf wetness.",
        "prevention_ta": "ஆரோக்கியமான விதை உருளைக்கிழங்கைப் பயன்படுத்தி போதிய இடைவெளி அளித்து இலைகள் நீண்ட நேரம் ஈரமாக இருப்பதைத் தவிர்க்கவும்."
    },

    "Potato___healthy": {
        "plant_en": "Potato",
        "plant_ta": "உருளைக்கிழங்கு",
        "name_en": "Healthy",
        "name_ta": "ஆரோக்கியமானது",
        "symptoms_en": "The model did not identify obvious disease symptoms in the uploaded image.",
        "symptoms_ta": "பதிவேற்றப்பட்ட படத்தில் தெளிவான நோய் அறிகுறிகளை மாதிரி கண்டறியவில்லை.",
        "causes_en": "No disease was identified in this prediction.",
        "causes_ta": "இந்தக் கணிப்பில் நோய் எதுவும் கண்டறியப்படவில்லை.",
        "remedies_en": "Continue normal watering and crop care.",
        "remedies_ta": "வழக்கமான நீர்ப்பாசனம் மற்றும் பயிர் பராமரிப்பைத் தொடரவும்.",
        "prevention_en": "Monitor plants regularly and use healthy seed potatoes.",
        "prevention_ta": "செடிகளைக் கண்காணித்து ஆரோக்கியமான விதை உருளைக்கிழங்கைப் பயன்படுத்தவும்."
    },

    "Tomato_Bacterial_spot": {
        "plant_en": "Tomato",
        "plant_ta": "தக்காளி",
        "name_en": "Bacterial spot",
        "name_ta": "பாக்டீரியா இலைப்புள்ளி நோய்",
        "symptoms_en": "Small dark spots may appear on leaves, stems or fruit.",
        "symptoms_ta": "இலைகள், தண்டுகள் அல்லது பழங்களில் சிறிய கருமையான புள்ளிகள் தோன்றலாம்.",
        "causes_en": "Bacteria may spread through splashing water and infected plant material.",
        "causes_ta": "நீர் தெறிப்பு மற்றும் பாதிக்கப்பட்ட தாவரப் பகுதிகள் மூலம் பாக்டீரியா பரவலாம்.",
        "remedies_en": "Remove affected plant material and avoid handling wet plants. Seek local agricultural advice.",
        "remedies_ta": "பாதிக்கப்பட்ட தாவரப் பகுதிகளை அகற்றி ஈரமான செடிகளைக் கையாள்வதைத் தவிர்க்கவும். வேளாண் ஆலோசனை பெறவும்.",
        "prevention_en": "Use clean tools, healthy seedlings and avoid overhead irrigation.",
        "prevention_ta": "சுத்தமான கருவிகள், ஆரோக்கியமான நாற்றுகளைப் பயன்படுத்தி மேலிருந்து நீர் பாய்ச்சுவதைத் தவிர்க்கவும்."
    },

    "Tomato_Early_blight": {
        "plant_en": "Tomato",
        "plant_ta": "தக்காளி",
        "name_en": "Early blight",
        "name_ta": "ஆரம்பகட்ட இலைக்கருகல் நோய்",
        "symptoms_en": "Brown leaf spots with circular or target-like rings may appear, particularly on older leaves.",
        "symptoms_ta": "குறிப்பாகப் பழைய இலைகளில் வட்ட வடிவ வளையங்களுடன் பழுப்பு நிறப் புள்ளிகள் தோன்றலாம்.",
        "causes_en": "A fungal disease that can spread in warm and moist conditions.",
        "causes_ta": "வெப்பமான மற்றும் ஈரமான சூழலில் பரவக்கூடிய பூஞ்சை நோயாகும்.",
        "remedies_en": "Remove affected leaves and improve airflow around the plants.",
        "remedies_ta": "பாதிக்கப்பட்ட இலைகளை அகற்றி செடிகளுக்கு இடையே காற்றோட்டத்தை மேம்படுத்தவும்.",
        "prevention_en": "Rotate crops and water near the soil instead of wetting leaves.",
        "prevention_ta": "பயிர் சுழற்சி முறையைப் பின்பற்றி இலைகளின் மீது அல்லாமல் மண்ணின் அருகில் நீர் பாய்ச்சவும்."
    },

    "Tomato_Late_blight": {
        "plant_en": "Tomato",
        "plant_ta": "தக்காளி",
        "name_en": "Late blight",
        "name_ta": "தாமத இலைக்கருகல் நோய்",
        "symptoms_en": "Large dark or water-soaked patches may spread quickly across leaves and stems.",
        "symptoms_ta": "இலைகள் மற்றும் தண்டுகளில் பெரிய கருமையான அல்லது நீர் தேங்கியதுபோன்ற புள்ளிகள் விரைவாகப் பரவலாம்.",
        "causes_en": "A disease favoured by cool and wet weather.",
        "causes_ta": "குளிர்ச்சியான மற்றும் ஈரமான வானிலையில் அதிகரிக்கக்கூடிய நோயாகும்.",
        "remedies_en": "Remove affected material safely and seek immediate local agricultural advice.",
        "remedies_ta": "பாதிக்கப்பட்ட பகுதிகளைப் பாதுகாப்பாக அகற்றி உடனடியாக வேளாண் நிபுணரிடம் ஆலோசனை பெறவும்.",
        "prevention_en": "Provide good spacing and avoid prolonged leaf wetness.",
        "prevention_ta": "செடிகளுக்கு போதிய இடைவெளி அளித்து இலைகள் நீண்ட நேரம் ஈரமாக இருப்பதைத் தவிர்க்கவும்."
    },

    "Tomato_Leaf_Mold": {
        "plant_en": "Tomato",
        "plant_ta": "தக்காளி",
        "name_en": "Leaf mold",
        "name_ta": "இலைப் பூஞ்சை நோய்",
        "symptoms_en": "Yellow patches may appear on the upper leaf surface, with mold-like growth underneath.",
        "symptoms_ta": "இலையின் மேற்புறத்தில் மஞ்சள் புள்ளிகளும் கீழ்ப்புறத்தில் பூஞ்சை போன்ற வளர்ச்சியும் காணப்படலாம்.",
        "causes_en": "High humidity and poor air circulation can encourage this disease.",
        "causes_ta": "அதிக ஈரப்பதம் மற்றும் குறைந்த காற்றோட்டம் இந்த நோயை அதிகரிக்கலாம்.",
        "remedies_en": "Remove affected leaves and improve ventilation around plants.",
        "remedies_ta": "பாதிக்கப்பட்ட இலைகளை அகற்றி செடிகளைச் சுற்றிய காற்றோட்டத்தை மேம்படுத்தவும்.",
        "prevention_en": "Avoid overcrowding and water at the base of plants.",
        "prevention_ta": "செடிகளை நெருக்கமாக நடுவதைத் தவிர்த்து அவற்றின் அடிப்பகுதியில் நீர் பாய்ச்சவும்."
    },

    "Tomato_Septoria_leaf_spot": {
        "plant_en": "Tomato",
        "plant_ta": "தக்காளி",
        "name_en": "Septoria leaf spot",
        "name_ta": "செப்டோரியா இலைப்புள்ளி நோய்",
        "symptoms_en": "Many small circular spots with pale centres may develop on leaves.",
        "symptoms_ta": "இலைகளில் வெளிர் நிற மையத்துடன் பல சிறிய வட்டப் புள்ளிகள் தோன்றலாம்.",
        "causes_en": "A fungal disease that can spread through wet foliage and splashing water.",
        "causes_ta": "ஈரமான இலைகள் மற்றும் நீர் தெறிப்பு மூலம் பரவக்கூடிய பூஞ்சை நோயாகும்.",
        "remedies_en": "Remove affected lower leaves and avoid overhead watering.",
        "remedies_ta": "பாதிக்கப்பட்ட கீழ் இலைகளை அகற்றி மேலிருந்து நீர் தெளிப்பதைத் தவிர்க்கவும்.",
        "prevention_en": "Keep the growing area clean and rotate crops.",
        "prevention_ta": "பயிர் வளரும் பகுதியைச் சுத்தமாக வைத்துப் பயிர் சுழற்சி முறையைப் பின்பற்றவும்."
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "plant_en": "Tomato",
        "plant_ta": "தக்காளி",
        "name_en": "Two-spotted spider mites",
        "name_ta": "இரண்டு புள்ளி சிலந்திப்பேன் தாக்குதல்",
        "symptoms_en": "Fine pale speckles, leaf bronzing and fine webbing may be visible.",
        "symptoms_ta": "இலைகளில் சிறிய வெளிர் புள்ளிகள், வெண்கல நிற மாற்றம் மற்றும் மெல்லிய வலை போன்ற அமைப்பு காணப்படலாம்.",
        "causes_en": "Tiny mites can multiply rapidly in hot and dry conditions.",
        "causes_ta": "வெப்பமான மற்றும் வறண்ட சூழலில் சிறிய சிலந்திப்பேன்கள் வேகமாகப் பெருகலாம்.",
        "remedies_en": "Inspect leaf undersides and seek advice on appropriate mite management.",
        "remedies_ta": "இலைகளின் கீழ்ப்புறத்தைச் சரிபார்த்து சிலந்திப்பேன்களைக் கட்டுப்படுத்த நிபுணர் ஆலோசனை பெறவும்.",
        "prevention_en": "Monitor plants regularly and reduce dust around the crop.",
        "prevention_ta": "செடிகளைத் தொடர்ந்து கண்காணித்து பயிரைச் சுற்றியுள்ள தூசியைக் குறைக்கவும்."
    },

    "Tomato__Target_Spot": {
        "plant_en": "Tomato",
        "plant_ta": "தக்காளி",
        "name_en": "Target spot",
        "name_ta": "இலக்கு வடிவ இலைப்புள்ளி நோய்",
        "symptoms_en": "Brown circular spots with concentric rings may appear on leaves.",
        "symptoms_ta": "இலைகளில் ஒரே மையத்தைச் சுற்றிய வளையங்களுடன் பழுப்பு நிறப் புள்ளிகள் தோன்றலாம்.",
        "causes_en": "A fungal disease favoured by warm and humid conditions.",
        "causes_ta": "வெப்பமான மற்றும் ஈரப்பதமான சூழலில் அதிகரிக்கக்கூடிய பூஞ்சை நோயாகும்.",
        "remedies_en": "Remove affected leaves and improve airflow around the plant.",
        "remedies_ta": "பாதிக்கப்பட்ட இலைகளை அகற்றி செடியைச் சுற்றிய காற்றோட்டத்தை மேம்படுத்தவும்.",
        "prevention_en": "Avoid overcrowding and keep foliage as dry as possible.",
        "prevention_ta": "செடிகளை நெருக்கமாக நடுவதைத் தவிர்த்து இலைகளை உலர்வாக வைத்திருக்கவும்."
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "plant_en": "Tomato",
        "plant_ta": "தக்காளி",
        "name_en": "Tomato yellow leaf curl virus",
        "name_ta": "தக்காளி மஞ்சள் இலை சுருள் வைரஸ் நோய்",
        "symptoms_en": "Leaves may turn yellow, curl upward and show reduced plant growth.",
        "symptoms_ta": "இலைகள் மஞ்சளாகி மேல்நோக்கிச் சுருண்டு தாவரத்தின் வளர்ச்சி குறையலாம்.",
        "causes_en": "The virus is commonly transmitted by whiteflies.",
        "causes_ta": "இந்த வைரஸ் பொதுவாக வெள்ளை ஈக்கள் மூலம் பரவுகிறது.",
        "remedies_en": "Seek local advice on managing whiteflies and handling severely affected plants.",
        "remedies_ta": "வெள்ளை ஈக்களைக் கட்டுப்படுத்தவும் கடுமையாகப் பாதிக்கப்பட்ட செடிகளைப் பராமரிக்கவும் உள்ளூர் வேளாண் ஆலோசனை பெறவும்.",
        "prevention_en": "Monitor whiteflies and use healthy seedlings.",
        "prevention_ta": "வெள்ளை ஈக்களைக் கண்காணித்து ஆரோக்கியமான நாற்றுகளைப் பயன்படுத்தவும்."
    },

    "Tomato__Tomato_mosaic_virus": {
        "plant_en": "Tomato",
        "plant_ta": "தக்காளி",
        "name_en": "Tomato mosaic virus",
        "name_ta": "தக்காளி மொசைக் வைரஸ் நோய்",
        "symptoms_en": "Mottled light and dark green patterns may appear on leaves.",
        "symptoms_ta": "இலைகளில் வெளிர் மற்றும் அடர் பச்சை நிறக் கலவையுடன் கூடிய வடிவங்கள் தோன்றலாம்.",
        "causes_en": "The virus can spread through contaminated hands, tools and plant material.",
        "causes_ta": "அசுத்தமான கைகள், கருவிகள் மற்றும் தாவரப் பகுதிகள் மூலம் வைரஸ் பரவலாம்.",
        "remedies_en": "Remove affected plants where appropriate and disinfect tools.",
        "remedies_ta": "தேவையான இடங்களில் பாதிக்கப்பட்ட செடிகளை அகற்றி கருவிகளைத் தூய்மைப்படுத்தவும்.",
        "prevention_en": "Use clean tools and healthy seeds or seedlings.",
        "prevention_ta": "சுத்தமான கருவிகள் மற்றும் ஆரோக்கியமான விதைகள் அல்லது நாற்றுகளைப் பயன்படுத்தவும்."
    },

    "Tomato_healthy": {
        "plant_en": "Tomato",
        "plant_ta": "தக்காளி",
        "name_en": "Healthy",
        "name_ta": "ஆரோக்கியமானது",
        "symptoms_en": "The model did not identify obvious disease symptoms in the uploaded image.",
        "symptoms_ta": "பதிவேற்றப்பட்ட படத்தில் தெளிவான நோய் அறிகுறிகளை மாதிரி கண்டறியவில்லை.",
        "causes_en": "No disease was identified in this prediction.",
        "causes_ta": "இந்தக் கணிப்பில் நோய் எதுவும் கண்டறியப்படவில்லை.",
        "remedies_en": "Continue normal watering, sunlight and regular plant care.",
        "remedies_ta": "வழக்கமான நீர்ப்பாசனம், சூரிய ஒளி மற்றும் தாவரப் பராமரிப்பைத் தொடரவும்.",
        "prevention_en": "Monitor the plant regularly and maintain good garden hygiene.",
        "prevention_ta": "தாவரத்தைத் தொடர்ந்து கண்காணித்து தோட்டத்தைச் சுத்தமாக வைத்திருக்கவும்."
    }
}


# =========================================================
# 7. DATABASE
# =========================================================

def get_connection():
    # Ensure the folder containing the database exists.
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def setup_database():
    """Create the tables and safely add columns missing from older databases.

    Existing records are preserved. This function never deletes or recreates
    an existing table.
    """
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plant_key TEXT,
                disease_key TEXT,
                score REAL,
                severity TEXT,
                created_at TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plant_key TEXT,
                disease_key TEXT,
                reminder_date TEXT,
                status TEXT DEFAULT 'Pending'
            )
        """)

        # CREATE TABLE IF NOT EXISTS does not change an older table.
        # Inspect its columns and add only the fields that are missing.
        required_columns = {
            "history": {
                "plant_key": "TEXT",
                "disease_key": "TEXT",
                "score": "REAL",
                "severity": "TEXT",
                "created_at": "TEXT"
            },
            "reminders": {
                "plant_key": "TEXT",
                "disease_key": "TEXT",
                "reminder_date": "TEXT",
                "status": "TEXT DEFAULT 'Pending'"
            }
        }

        for table_name, columns in required_columns.items():
            cursor.execute(f"PRAGMA table_info({table_name})")
            existing_columns = {
                row["name"] for row in cursor.fetchall()
            }

            for column_name, column_type in columns.items():
                if column_name not in existing_columns:
                    cursor.execute(
                        f"ALTER TABLE {table_name} "
                        f"ADD COLUMN {column_name} {column_type}"
                    )

        # Give old reminder rows a usable status if the field was newly added
        # or contained NULL values.
        cursor.execute("""
            UPDATE reminders
            SET status = 'Pending'
            WHERE status IS NULL OR TRIM(status) = ''
        """)

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


setup_database()


# =========================================================
# 8. LOAD MODEL AND CLASS NAMES
# =========================================================

@st.cache_resource
def load_model() -> keras.Model:

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = keras.models.load_model(MODEL_PATH)

    if not isinstance(model, keras.Model):
        raise TypeError("Loaded file is not a valid Keras model.")

    return model


@st.cache_data
def load_class_names():

    if not CLASS_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_PATH}"
        )

    with open(CLASS_PATH, "r", encoding="utf-8") as file:
        names = json.load(file)

    if isinstance(names, dict):
        names = [
            names[key]
            for key in sorted(names, key=lambda x: int(x))
        ]

    return names


# =========================================================
# 9. HELPER FUNCTIONS
# =========================================================

def get_disease_text(disease_key, field):

    language = st.session_state.language

    suffix = "ta" if language == "ta" else "en"

    if not isinstance(disease_key, str) or disease_key not in DISEASE_INFO:
        return t("disease_info_unavailable")

    return DISEASE_INFO[disease_key][f"{field}_{suffix}"]


def get_plant_text(disease_key):

    language = st.session_state.language

    suffix = "ta" if language == "ta" else "en"

    if not isinstance(disease_key, str) or disease_key not in DISEASE_INFO:
        return t("unknown")

    return DISEASE_INFO[disease_key][f"plant_{suffix}"]


def get_name_text(disease_key):

    language = st.session_state.language

    suffix = "ta" if language == "ta" else "en"

    if not isinstance(disease_key, str) or disease_key not in DISEASE_INFO:
        return t("unknown")

    return DISEASE_INFO[disease_key][f"name_{suffix}"]




# =========================================================
# 10. SIMPLE OFFLINE CHATBOT RESPONSE RULES
# =========================================================

def get_chatbot_reply(question):
    """
    Create a reply using local keyword rules and the disease information
    already stored in this application. No API key or internet is required.
    """
    q = question.lower().strip()
    tamil = st.session_state.language == "ta"

    def reply(en, ta):
        return ta if tamil else en

    # Greetings and common conversation.
    if any(word in q for word in ["hello", "hi", "hey", "good morning", "good evening", "வணக்கம்", "ஹலோ"]):
        return reply(
            "Hello! 🌱 I am PlantCare AI. You can ask me about plant diseases, symptoms, watering, sunlight, prevention and plant care.",
            "வணக்கம்! 🌱 நான் PlantCare AI. தாவர நோய்கள், அறிகுறிகள், நீர்ப்பாசனம், சூரிய ஒளி மற்றும் தாவரப் பராமரிப்பு பற்றி கேட்கலாம்."
        )

    if any(word in q for word in ["thank", "thanks", "நன்றி"]):
        return reply(
            "You're welcome! Keep checking your plants regularly. 🌿",
            "மகிழ்ச்சி! உங்கள் தாவரங்களைத் தொடர்ந்து கவனியுங்கள். 🌿"
        )

    # Identify a plant/disease from the application's existing 15 categories.
    matched_key = None
    for key, info in DISEASE_INFO.items():
        # Match a disease/category name here, not just a plant name.
        # Plant names are handled separately below so a question such as
        # "How often should I water tomato plants?" gets a watering answer.
        candidates = [
            key.lower().replace("_", " "),
            info["name_en"].lower(),
            info["name_ta"]
        ]
        if any(term and term in q for term in candidates):
            matched_key = key
            break

    # If a specific disease or plant is mentioned, use the stored project data.
    if matched_key:
        info = DISEASE_INFO[matched_key]
        plant = info["plant_ta"] if tamil else info["plant_en"]
        disease = info["name_ta"] if tamil else info["name_en"]
        symptoms = info["symptoms_ta"] if tamil else info["symptoms_en"]
        causes = info["causes_ta"] if tamil else info["causes_en"]
        remedies = info["remedies_ta"] if tamil else info["remedies_en"]
        prevention = info["prevention_ta"] if tamil else info["prevention_en"]

        is_healthy = info["name_en"].lower() == "healthy"
        if is_healthy:
            return reply(
                f"{plant} – Healthy 🌿\n\nThe selected category is Healthy. {symptoms}\n\nContinue regular watering, suitable sunlight and routine monitoring.",
                f"{plant} – ஆரோக்கியமானது 🌿\n\nஇந்த வகை ஆரோக்கியமானது எனக் குறிப்பிடுகிறது. {symptoms}\n\nவழக்கமான நீர்ப்பாசனம், தேவையான சூரிய ஒளி மற்றும் தொடர்ந்த கண்காணிப்பை மேற்கொள்ளுங்கள்."
            )

        return (
            f"**{plant} – {disease}**\n\n"
            f"**{ 'அறிகுறிகள்' if tamil else 'Symptoms' }:** {symptoms}\n\n"
            f"**{ 'காரணங்கள்' if tamil else 'Possible causes' }:** {causes}\n\n"
            f"**{ 'பராமரிப்பு' if tamil else 'Management' }:** {remedies}\n\n"
            f"**{ 'தடுப்பு' if tamil else 'Prevention' }:** {prevention}\n\n"
            + reply(
                "This is general information, not a confirmed diagnosis. Please verify serious or spreading symptoms with a local agricultural expert.",
                "இது பொதுவான தகவல் மட்டுமே; உறுதிப்படுத்தப்பட்ட நோய் கண்டறிதல் அல்ல. கடுமையான அல்லது பரவும் அறிகுறிகளுக்கு உள்ளூர் வேளாண் நிபுணரை அணுகவும்."
            )
        )

    # Plant-category questions: list the supported categories from this model.
    plant_terms = ["tomato", "potato", "pepper", "தக்காளி", "உருளைக்கிழங்கு", "குடைமிளகாய்"]
    if any(term in q for term in plant_terms) and any(
        word in q for word in ["disease", "diseases", "available", "support", "நோய்", "நோய்கள்", "என்னென்ன"]
    ):
        chosen_plant = None
        for key, info in DISEASE_INFO.items():
            if any(term in q for term in [info["plant_en"].lower(), info["plant_ta"]]):
                chosen_plant = info["plant_en"]
                break
        if chosen_plant:
            entries = [
                (v["name_ta"] if tamil else v["name_en"])
                for v in DISEASE_INFO.values()
                if v["plant_en"] == chosen_plant and v["name_en"].lower() != "healthy"
            ]
            label = next(
                (v["plant_ta"] if tamil else v["plant_en"]
                 for v in DISEASE_INFO.values() if v["plant_en"] == chosen_plant),
                chosen_plant
            )
            return reply(
                f"This model includes these {label} categories: " + ", ".join(entries) + ".",
                f"இந்த மாதிரியில் {label} தொடர்பாக உள்ள வகைகள்: " + "、".join(entries) + "。"
            )

    # General care questions.
    if any(word in q for word in ["water", "watering", "irrigation", "நீர்", "தண்ணீர்", "நீர்ப்பாசனம்"]):
        return reply(
            "Water the soil when the top layer begins to feel dry. The amount depends on the plant, pot, soil and weather. Avoid keeping the soil constantly waterlogged, and make sure excess water can drain.",
            "மண்ணின் மேற்பரப்பு உலரத் தொடங்கும்போது நீர் ஊற்றுங்கள். தேவையான நீரின் அளவு தாவரம், தொட்டி, மண் மற்றும் வானிலையைப் பொறுத்தது. மண்ணில் எப்போதும் அதிக நீர் தேங்காமல் பார்த்துக் கொள்ளுங்கள்; அதிகப்படியான நீர் வெளியேற வழி இருக்க வேண்டும்."
        )

    if any(word in q for word in ["sunlight", "sun", "light", "சூரிய", "வெயில்", "ஒளி"]):
        return reply(
            "Most plants need suitable light, but the amount differs by crop. Check the growing requirements of your plant. If leaves scorch or wilt in strong midday sun, provide appropriate shade and monitor the plant.",
            "பெரும்பாலான தாவரங்களுக்கு ஏற்ற ஒளி தேவை; ஆனால் தேவையான அளவு தாவர வகையைப் பொறுத்து மாறும். உங்கள் தாவரத்திற்கான வளர்ப்பு வழிமுறைகளைப் பின்பற்றுங்கள். கடுமையான மதிய வெயிலில் இலைகள் கருகினால் பொருத்தமான நிழல் வழங்குங்கள்."
        )

    if any(word in q for word in ["yellow", "yellowing", "மஞ்சள்", "மஞ்சளாக"]):
        return reply(
            "Yellow leaves can be caused by overwatering, underwatering, nutrient problems, pests or disease. Check soil moisture, the pattern of yellowing and the underside of leaves. Upload a clear leaf image in Disease Detection for a model-based screening.",
            "இலைகள் மஞ்சளாக மாறுவதற்கு அதிக நீர், குறைந்த நீர், ஊட்டச்சத்து குறைபாடு, பூச்சிகள் அல்லது நோய்கள் காரணமாக இருக்கலாம். மண்ணின் ஈரப்பதம், மஞ்சள் நிறம் பரவும் விதம் மற்றும் இலைகளின் கீழ்ப்புறத்தைச் சரிபார்க்கவும். மாதிரியின் ஆரம்பகட்ட ஆய்விற்கு Disease Detection பகுதியில் தெளிவான படத்தைப் பதிவேற்றலாம்."
        )

    if any(word in q for word in ["spot", "spots", "patch", "புள்ளி", "புள்ளிகள்", "தழும்பு"]):
        return reply(
            "Leaf spots may have several causes, including fungal or bacterial diseases, insects or physical damage. Avoid wetting the leaves, remove badly affected fallen leaves and compare the symptoms with the disease information. Ask an agricultural expert if the problem spreads.",
            "இலைப்புள்ளிகள் பூஞ்சை அல்லது பாக்டீரியா நோய், பூச்சிகள் அல்லது உடல் சேதம் போன்ற பல காரணங்களால் ஏற்படலாம். இலைகளை அதிகமாக நனைப்பதைத் தவிர்த்து, கடுமையாகப் பாதிக்கப்பட்ட விழுந்த இலைகளை அகற்றுங்கள். பாதிப்பு பரவினால் வேளாண் நிபுணரை அணுகவும்."
        )

    if any(word in q for word in ["pest", "insect", "whitefly", "mite", "aphid", "பூச்சி", "வெள்ளை ஈ", "சிலந்திப்பேன்"]):
        return reply(
            "Inspect both sides of leaves and young shoots for insects, eggs, webbing or sticky residue. Keep the area clean and identify the pest before choosing a treatment. Use any plant-protection product only according to its label and local expert advice.",
            "இலைகளின் இருபுறங்களையும் இளம் தளிர்களையும் பூச்சிகள், முட்டைகள், வலை போன்ற அமைப்பு அல்லது ஒட்டும் திரவம் உள்ளதா எனப் பாருங்கள். சிகிச்சையைத் தேர்வு செய்வதற்கு முன் பூச்சியை அடையாளம் காணுங்கள். தாவரப் பாதுகாப்புப் பொருட்களை அதன் லேபிள் மற்றும் உள்ளூர் நிபுணர் ஆலோசனைப்படி மட்டுமே பயன்படுத்துங்கள்."
        )

    if any(word in q for word in ["fertilizer", "fertiliser", "nutrient", "உரம்", "ஊட்டச்சத்து"]):
        return reply(
            "Use a fertilizer suitable for your crop and follow the package directions. Avoid applying extra fertilizer without checking the plant's needs; too much can damage roots and leaves. A soil test or local agricultural advice can help.",
            "உங்கள் பயிருக்கு ஏற்ற உரத்தைத் தேர்ந்தெடுத்து அதன் பொட்டலத்தில் உள்ள வழிமுறைகளைப் பின்பற்றுங்கள். தேவையை அறியாமல் அதிக உரம் இடாதீர்கள்; அதிக உரம் வேர்களையும் இலைகளையும் பாதிக்கலாம். மண் பரிசோதனை அல்லது உள்ளூர் வேளாண் ஆலோசனை உதவும்."
        )

    if any(word in q for word in ["prevent", "prevention", "healthy", "care", "maintain", "தடுப்பு", "ஆரோக்கியம்", "பராமரிப்பு"]):
        return reply(
            "For general plant care: use healthy seedlings, provide suitable sunlight, water according to soil moisture, allow good airflow, keep tools clean and inspect leaves regularly. Remove fallen diseased plant material safely.",
            "பொதுவான தாவரப் பராமரிப்பிற்கு: ஆரோக்கியமான நாற்றுகளைப் பயன்படுத்துங்கள், ஏற்ற சூரிய ஒளி வழங்குங்கள், மண்ணின் ஈரப்பதத்திற்கேற்ப நீர் ஊற்றுங்கள், நல்ல காற்றோட்டம் ஏற்படுத்துங்கள், கருவிகளைச் சுத்தமாக வைத்திருங்கள், இலைகளைத் தொடர்ந்து பரிசோதியுங்கள். நோயுற்ற விழுந்த தாவரப் பகுதிகளைப் பாதுகாப்பாக அகற்றுங்கள்."
        )

    if any(word in q for word in ["help", "what can", "என்ன கேட்க", "உதவி"]):
        return reply(
            "You can ask me about:\n- Diseases included in the app (pepper, potato and tomato)\n- Symptoms and possible causes\n- Watering and sunlight\n- Yellow leaves and leaf spots\n- Pests, prevention and general plant care",
            "நீங்கள் என்னிடம் கேட்கக்கூடியவை:\n- செயலியில் உள்ள குடைமிளகாய், உருளைக்கிழங்கு மற்றும் தக்காளி நோய் வகைகள்\n- அறிகுறிகள் மற்றும் சாத்தியமான காரணங்கள்\n- நீர்ப்பாசனம் மற்றும் சூரிய ஒளி\n- மஞ்சள் இலைகள் மற்றும் இலைப்புள்ளிகள்\n- பூச்சிகள், தடுப்பு முறைகள் மற்றும் பொதுவான தாவரப் பராமரிப்பு"
        )

    return reply(
        "I can answer common plant-care questions using the information built into this app. Try asking about a supported plant (pepper, potato or tomato), a disease name, yellow leaves, leaf spots, watering, sunlight or pests. I do not generate open-ended answers like an online AI service.",
        "இந்தச் செயலியில் உள்ள தகவல்களின் அடிப்படையில் பொதுவான தாவரப் பராமரிப்பு கேள்விகளுக்கு பதிலளிக்க முடியும். குடைமிளகாய், உருளைக்கிழங்கு, தக்காளி, நோய்களின் பெயர்கள், மஞ்சள் இலைகள், இலைப்புள்ளிகள், நீர்ப்பாசனம், சூரிய ஒளி அல்லது பூச்சிகள் பற்றி கேளுங்கள். இணைய AI சேவையைப் போல எல்லா வகையான கேள்விகளுக்கும் புதிய பதில்களை உருவாக்காது."
    )


# =========================================================
# 10. NAVIGATION
# =========================================================

page_keys = [
    "home",
    "detection",
    "history",
    "reminders",
    "chatbot",
    "about"
]

page_labels = [t(key) for key in page_keys]

selected_page = st.sidebar.radio(
    t("navigation"),
    page_labels
)

page = page_keys[page_labels.index(selected_page)]


# =========================================================
# 11. HOME PAGE
# =========================================================

if page == "home":

    st.title("🌿 " + t("welcome"))

    st.write(t("welcome_text"))

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(t("test_accuracy"), "89.66%")

    with col2:
        st.metric(t("categories"), "15")

    with col3:
        st.metric(t("model_name"), "CNN")

    st.markdown("---")

    st.subheader("🌱 " + t("features"))

    feature1, feature2, feature3 = st.columns(3)

    if st.session_state.language == "ta":

        with feature1:
            st.markdown("### 📷 படப் பதிவேற்றம்")
            st.write("தாவர இலையின் படத்தைப் பதிவேற்றி ஆய்வு செய்யலாம்.")

        with feature2:
            st.markdown("### 🔍 நோய் கண்டறிதல்")
            st.write("பயிற்சி பெற்ற மாதிரி இலையின் நிலையை வகைப்படுத்தும்.")

        with feature3:
            st.markdown("### 📋 தாவரப் பராமரிப்பு")
            st.write("பரிந்துரைகள், பழைய கணிப்புகள் மற்றும் நினைவூட்டல்களைப் பார்க்கலாம்.")

    else:

        with feature1:
            st.markdown("### 📷 Image Upload")
            st.write("Upload a plant leaf image for analysis.")

        with feature2:
            st.markdown("### 🔍 Disease Detection")
            st.write("The trained model classifies the leaf condition.")

        with feature3:
            st.markdown("### 📋 Plant Care")
            st.write("View suggestions, previous predictions and reminders.")

    st.info(t("disclaimer"))


# =========================================================
# 12. DISEASE DETECTION
# =========================================================

elif page == "detection":

    st.title("🔍 " + t("detection"))

    uploaded_file = st.file_uploader(
        t("upload"),
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        st.subheader(t("preview"))
        st.image(image, width=400)

        severity_options = (
            ["ஆரம்ப நிலை", "மிதமான நிலை", "தீவிர நிலை"]
            if st.session_state.language == "ta"
            else ["Early", "Moderate", "Advanced"]
        )

        selected_severity = st.selectbox(
            t("severity"),
            severity_options
        )

        if st.button("🌿 " + t("detect")):

            try:

                model = load_model()
                class_names = load_class_names()

                if len(class_names) != len(DISEASE_INFO):
                    st.error("Class names do not match the disease information.")
                    st.stop()

                resized_image = image.resize((128, 128))

                image_array = np.array(
                    resized_image,
                    dtype=np.float32
                ) / 255.0

                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                predictions = model.predict(image_array)[0]

                predicted_index = int(np.argmax(predictions))
                disease_key = class_names[predicted_index]

                score = float(predictions[predicted_index])

                if disease_key not in DISEASE_INFO:
                    st.error("Disease information is not available.")
                    st.stop()

                connection = get_connection()
                cursor = connection.cursor()

                cursor.execute("""
                    INSERT INTO history
                    (plant_key, disease_key, score, severity, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    DISEASE_INFO[disease_key]["plant_en"],
                    disease_key,
                    score,
                    selected_severity,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ))

                connection.commit()
                connection.close()

                st.session_state.latest_prediction = {
                    "disease_key": disease_key,
                    "score": score,
                    "severity": selected_severity
                }

            except Exception as error:

                st.error(f"{t('prediction_error')} {error}")

    result = st.session_state.latest_prediction

    if result is not None:

        disease_key = result["disease_key"]
        score = result["score"]

        info = DISEASE_INFO[disease_key]

        is_healthy = info["name_en"] == "Healthy"

        st.markdown("---")

        st.header("📋 " + t("results"))

        st.subheader(
            f"{t('plant')}: {get_plant_text(disease_key)}"
        )

        st.subheader(
            f"{t('prediction')}: {get_name_text(disease_key)}"
        )

        st.metric(
            t("score"),
            f"{score * 100:.2f}%"
        )

        st.markdown("---")

        st.subheader("⚠️ " + t("reliability"))

        if score < 0.60:
            st.warning(t("low_warning"))
        elif score < 0.80:
            st.warning(t("medium_warning"))
        else:
            st.success(t("high_warning"))

        if is_healthy:
            st.success(t("healthy"))
        else:
            st.warning(t("diseased"))

        st.markdown("---")

        st.subheader("🩺 " + t("symptoms"))
        st.write(get_disease_text(disease_key, "symptoms"))

        st.subheader("🔎 " + t("causes"))
        st.write(get_disease_text(disease_key, "causes"))

        st.subheader("🌱 " + t("remedies"))
        st.write(get_disease_text(disease_key, "remedies"))

        st.subheader("🛡️ " + t("prevention"))
        st.write(get_disease_text(disease_key, "prevention"))

        st.info(t("disclaimer"))

        # Voice assistant

        st.markdown("---")

        st.subheader("🔊 " + t("voice"))

        if st.button("🔊 " + t("voice_text")):

            try:

                voice_text = (
                    get_name_text(disease_key) + ". "
                    + get_disease_text(disease_key, "remedies") + ". "
                    + get_disease_text(disease_key, "prevention")
                )

                language_code = (
                    "ta" if st.session_state.language == "ta"
                    else "en"
                )

                speech = gTTS(
                    text=voice_text,
                    lang=language_code
                )

                audio_buffer = BytesIO()
                speech.write_to_fp(audio_buffer)
                audio_buffer.seek(0)

                st.audio(
                    audio_buffer,
                    format="audio/mp3"
                )

            except Exception:
                st.error(t("voice_error"))

        # Follow-up reminder

        st.markdown("---")

        st.subheader("📅 " + t("reminders"))

        reminder_date = st.date_input(
            t("reminder_date"),
            value=date.today() + timedelta(days=7),
            min_value=date.today()
        )

        if st.button("📌 " + t("save_reminder")):

            connection = get_connection()
            cursor = connection.cursor()

            cursor.execute("""
                INSERT INTO reminders
                (plant_key, disease_key, reminder_date, status)
                VALUES (?, ?, ?, ?)
            """, (
                info["plant_en"],
                disease_key,
                reminder_date.isoformat(),
                "Pending"
            ))

            connection.commit()
            connection.close()

            st.success(t("reminder_saved"))


# =========================================================
# 13. PREDICTION HISTORY
# =========================================================

elif page == "history":

    st.title("📚 " + t("history"))

    connection = get_connection()

    history_df = pd.read_sql_query(
        """
        SELECT disease_key, score, severity, created_at
        FROM history
        ORDER BY id DESC
        """,
        connection
    )

    connection.close()

    if not history_df.empty:

        history_df["Plant"] = history_df["disease_key"].apply(
            lambda key: get_plant_text(key)
        )

        history_df["Prediction"] = history_df["disease_key"].apply(
            lambda key: get_name_text(key)
        )

        history_df["Model Score (%)"] = (
            history_df["score"] * 100
        ).round(2)

        if st.session_state.language == "ta":

            history_df["நிலை"] = history_df["severity"]
            history_df["தேதி"] = history_df["created_at"]

            display_df = history_df[
                ["Plant", "Prediction", "Model Score (%)", "நிலை", "தேதி"]
            ].copy()

            display_df.columns = [
                "தாவரம்",
                "கணிப்பு",
                "மாதிரி மதிப்பெண் (%)",
                "தீவிரம்",
                "தேதி"
            ]

        else:

            display_df = history_df[
                ["Plant", "Prediction", "Model Score (%)", "severity", "created_at"]
            ].copy()

            display_df.columns = [
                "Plant",
                "Prediction",
                "Model Score (%)",
                "Severity",
                "Date"
            ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            t("download"),
            data=display_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="plantcare_history.csv",
            mime="text/csv"
        )

    else:

        st.info(t("history_empty"))


# =========================================================
# 14. FOLLOW-UP REMINDERS
# =========================================================

elif page == "reminders":

    st.title("📅 " + t("reminders"))

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, disease_key, reminder_date, status
        FROM reminders
        ORDER BY reminder_date ASC
    """)

    reminders = cursor.fetchall()

    connection.close()

    if reminders:

        today = date.today().isoformat()

        for reminder in reminders:

            reminder_id, disease_key, reminder_date, status = reminder

            with st.container(border=True):

                st.write(
                    f"🌱 **{get_plant_text(disease_key)}**"
                )

                st.write(
                    f"**{t('prediction')}:** {get_name_text(disease_key)}"
                )

                st.write(
                    f"**{t('date')}:** {reminder_date}"
                )

                if status == "Completed":

                    st.success(t("completed"))

                elif reminder_date <= today:

                    st.warning(t("reminder_due"))

                else:

                    st.info(t("pending"))

                if status != "Completed":

                    if st.button(
                        t("mark_complete"),
                        key=f"complete_{reminder_id}"
                    ):

                        connection = get_connection()
                        cursor = connection.cursor()

                        cursor.execute("""
                            UPDATE reminders
                            SET status = 'Completed'
                            WHERE id = ?
                        """, (reminder_id,))

                        connection.commit()
                        connection.close()

                        st.rerun()

    else:

        st.info(t("reminders_empty"))


# =========================================================
# 15. CHATBOT
# =========================================================

elif page == "chatbot":

    st.title("🌱 PlantCare AI Chatbot")

    if st.session_state.language == "ta":
        st.write("தாவர நோய்கள் மற்றும் தாவரப் பராமரிப்பு பற்றி கேள்விகள் கேளுங்கள்.")
        input_placeholder = "உங்கள் கேள்வியை இங்கே எழுதுங்கள்..."
        clear_label = "உரையாடலை அழிக்கவும்"
        intro = "வணக்கம்! தாவர நோய்கள், அறிகுறிகள் மற்றும் பராமரிப்பு பற்றி கேளுங்கள்."
        disclaimer = "குறிப்பு: இந்த உதவி செயலியில் உள்ள விதிகள் மற்றும் தகவல்களைப் பயன்படுத்துகிறது. இது உறுதிப்படுத்தப்பட்ட நோய் கண்டறிதல் அல்ல."
    else:
        st.write("Ask me about plant diseases, symptoms and plant care.")
        input_placeholder = "Type your question here..."
        clear_label = "Clear conversation"
        intro = "Hello! Ask me about plant diseases, symptoms and plant care."
        disclaimer = "Note: This chatbot uses built-in rules and information. It does not provide a confirmed diagnosis."

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    if st.button("🗑️ " + clear_label):
        st.session_state.chat_messages = []
        st.rerun()

    if not st.session_state.chat_messages:
        with st.chat_message("assistant"):
            st.markdown(intro)

    # Display previous messages.
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input(input_placeholder)

    if user_question:
        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):
            st.markdown(user_question)

        answer = get_chatbot_reply(user_question)

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": answer
        })

    st.caption(disclaimer)


# =========================================================
# 16. ABOUT PROJECT
# =========================================================

elif page == "about":

    st.title("🌿 " + t("about"))

    st.write(t("about_intro"))

    st.markdown("---")

    if st.session_state.language == "en":

        st.header("1. Project Overview")

        st.write("""
        PlantCare AI is an AI-powered plant disease detection application
        designed to identify possible diseases from images of plant leaves.

        The application uses a Convolutional Neural Network (CNN) trained
        on the PlantVillage image dataset. A user uploads a leaf image,
        and the trained model processes the image and predicts one of
        fifteen supported plant-health categories.

        The application also provides general disease information,
        symptoms, possible causes, management suggestions, prevention
        methods, prediction history and follow-up reminders.
        """)

        st.header("2. Problem Statement")

        st.write("""
        Farmers and home gardeners may find it difficult to identify
        plant diseases at an early stage. Access to agricultural experts
        may be limited in some locations, and identifying diseases
        manually requires experience.

        Delayed identification can allow diseases to spread and may
        affect plant growth and crop production.

        Therefore, an accessible image-based preliminary screening
        system can help users obtain initial information about
        possible plant diseases.
        """)

        st.header("3. Proposed Solution")

        st.write("""
        PlantCare AI provides an image-based disease classification
        system using deep learning.

        The user uploads a photograph of a plant leaf. The image is
        resized and normalized before being passed to the trained CNN
        model. The model produces scores for the supported categories.

        The application displays the predicted category, model score,
        relevant disease information and general management suggestions.
        It also stores previous predictions and allows users to create
        follow-up reminders.
        """)

        st.header("4. Project Objectives")

        st.markdown("""
        - Develop an image-based plant disease classification application.
        - Apply a CNN model to recognize visual patterns in leaf images.
        - Support 15 plant-health categories covering pepper, potato and tomato.
        - Display possible symptoms and causes of identified conditions.
        - Provide general management and prevention information.
        - Maintain prediction history using a local database.
        - Provide follow-up reminders for plant monitoring.
        - Support English and Tamil interfaces.
        - Provide spoken output for prediction information.
        """)

        st.header("5. System Architecture")

        st.markdown("""
        **Step 1 – User Interface**

        The user interacts with the Streamlit application and uploads a
        plant leaf photograph.

        **Step 2 – Image Processing**

        The uploaded image is converted to RGB, resized to 128 × 128
        pixels and normalized to a numerical range suitable for the model.

        **Step 3 – CNN Model**

        The processed image is passed to the trained CNN model.
        Convolutional and pooling layers extract visual features, and
        the final classification layer produces scores for 15 categories.

        **Step 4 – Prediction Output**

        The application selects the category with the highest model score
        and displays the corresponding information.

        **Step 5 – Database**

        SQLite stores prediction records and follow-up reminders locally.

        **Step 6 – Additional Features**

        The application provides language selection, theme selection,
        prediction history and text-to-speech output.
        """)

        st.header("6. Working Process")

        st.markdown("""
        1. Open the PlantCare AI application.
        2. Select English or Tamil.
        3. Select Light Mode or Dark Mode.
        4. Open the Disease Detection page.
        5. Upload a clear photograph of a plant leaf.
        6. Select the observed severity level.
        7. Click Detect Disease.
        8. The application preprocesses the image.
        9. The CNN model predicts the leaf category.
        10. The result and relevant information are displayed.
        11. The prediction is stored in the local database.
        12. The user can listen to the advice or save a follow-up reminder.
        """)

        st.header("7. Dataset Description")

        st.write("""
        The project uses the PlantVillage dataset, which contains
        labelled images of healthy and diseased plant leaves.

        The selected dataset contains three plant groups:
        pepper, potato and tomato.

        After dataset preparation, the images were divided into
        training, validation and testing sets.
        """)

        st.markdown("""
        | Dataset Split | Images | Percentage |
        |---|---:|---:|
        | Training | 14,446 | 70% |
        | Validation | 3,096 | 15% |
        | Testing | 3,096 | 15% |
        | **Total** | **20,638** | **100%** |
        """)

        st.write("The model supports these 15 categories:")

        st.markdown("""
        **Pepper**
        - Bacterial spot
        - Healthy

        **Potato**
        - Early blight
        - Late blight
        - Healthy

        **Tomato**
        - Bacterial spot
        - Early blight
        - Late blight
        - Leaf mold
        - Septoria leaf spot
        - Spider mites
        - Target spot
        - Yellow leaf curl virus
        - Mosaic virus
        - Healthy
        """)

        st.header("8. CNN Model Architecture")

        st.write("""
        A Convolutional Neural Network is a deep learning model commonly
        used for image classification.

        The project CNN contains the following main layers:
        """)

        st.markdown("""
        1. **Input Layer:** Accepts a 128 × 128 RGB image.
        2. **First Convolution Layer:** Extracts basic visual features.
        3. **First Max Pooling Layer:** Reduces feature-map dimensions.
        4. **Second Convolution Layer:** Learns more detailed patterns.
        5. **Second Max Pooling Layer:** Reduces spatial dimensions.
        6. **Third Convolution Layer:** Extracts deeper image features.
        7. **Third Max Pooling Layer:** Further reduces dimensions.
        8. **Flatten Layer:** Converts feature maps into a one-dimensional vector.
        9. **Dense Layer:** Learns classification relationships.
        10. **Dropout Layer:** Helps reduce overfitting.
        11. **Output Layer:** Produces scores for 15 categories using Softmax.
        """)

        st.header("9. Model Evaluation")

        st.write("""
        The original CNN model was evaluated using a separate test dataset
        that was not used for training.

        The observed test accuracy was approximately 89.66%.

        Accuracy alone does not describe every class equally well.
        Precision, recall and F1-score are also useful for understanding
        performance across individual disease categories.
        """)

        st.metric("Original CNN Test Accuracy", "89.66%")

        st.header("10. Technologies Used")

        st.markdown("""
        | Technology | Purpose |
        |---|---|
        | Python | Main programming language |
        | TensorFlow | Deep learning framework |
        | Keras | CNN model development |
        | NumPy | Numerical image processing |
        | Pillow | Image loading and resizing |
        | Streamlit | Web application interface |
        | SQLite | Local database |
        | Pandas | Prediction history and data tables |
        | gTTS | Text-to-speech generation |
        | VS Code | Development environment |
        """)

        st.header("11. Main Application Features")

        st.markdown("""
        - Plant leaf image upload and preview.
        - CNN-based disease classification.
        - Model-score display and reliability warnings.
        - Disease symptoms, possible causes and management suggestions.
        - English and Tamil language selection.
        - Light Mode and Dark Mode.
        - Prediction history and CSV export.
        - Follow-up reminder creation and completion tracking.
        - Spoken prediction information.
        """)

        st.header("12. Project Limitations")

        st.markdown("""
        - The model supports only the 15 categories used during training.
        - Performance may differ on real-world photographs from the training dataset.
        - Poor lighting, blurry images and complex backgrounds may affect predictions.
        - The model score is not a calibrated probability of correctness.
        - The application does not independently confirm disease severity.
        - Follow-up reminders appear inside the application and are not push notifications.
        - Voice generation requires an internet connection.
        - The system provides preliminary information and cannot replace an agricultural expert.
        """)

        st.header("13. Future Enhancements")

        st.markdown("""
        - Expand the dataset to support more crops and disease categories.
        - Improve performance on real-world field photographs.
        - Develop an offline Android application.
        - Add more regional languages.
        - Integrate weather and environmental information.
        - Add expert consultation and location-based agricultural support.
        - Introduce notifications for scheduled follow-ups.
        - Explore lightweight models for mobile deployment.
        """)

        st.header("14. Applications")

        st.markdown("""
        - Preliminary plant-health screening for farmers.
        - Plant monitoring in home gardens.
        - Agricultural education and demonstrations.
        - College-level AI and machine learning projects.
        - Initial plant disease awareness in areas with limited expert access.
        """)

        st.header("15. Conclusion")

        st.write("""
        PlantCare AI demonstrates how deep learning can be applied to
        image-based plant disease classification.

        By combining a CNN model with an accessible application interface,
        disease information, prediction history and follow-up reminders,
        the system provides a useful preliminary plant-health screening tool.

        Further improvements in dataset diversity, field testing and
        model validation can help make the system more reliable for
        practical agricultural use.
        """)

    else:

        st.header("1. திட்டத்தின் கண்ணோட்டம்")

        st.write("""
        PlantCare AI என்பது தாவர இலைகளின் படங்களை ஆய்வு செய்து
        ஏற்படக்கூடிய நோய்களைக் கண்டறிய உதவும் செயற்கை நுண்ணறிவு
        அடிப்படையிலான செயலியாகும்.

        இந்தச் செயலி PlantVillage படத் தரவுத்தொகுப்பைப் பயன்படுத்திப்
        பயிற்சி அளிக்கப்பட்ட சுருள் நரம்பு வலையமைப்பு மாதிரியைப்
        பயன்படுத்துகிறது.

        பயனர் தாவர இலையின் படத்தைப் பதிவேற்றும்போது, பயிற்சி பெற்ற
        மாதிரி அந்தப் படத்தை ஆய்வு செய்து 15 வகைகளில் ஒன்றாக
        வகைப்படுத்துகிறது.

        மேலும் நோய் அறிகுறிகள், சாத்தியமான காரணங்கள், பராமரிப்பு
        பரிந்துரைகள், தடுப்பு முறைகள், கணிப்பு வரலாறு மற்றும்
        தொடர் பராமரிப்பு நினைவூட்டல்களையும் வழங்குகிறது.
        """)

        st.header("2. பிரச்சினை விளக்கம்")

        st.write("""
        விவசாயிகள் மற்றும் வீட்டுத் தோட்டம் அமைப்பவர்கள் தாவரங்களில்
        ஏற்படும் நோய்களை ஆரம்ப நிலையிலேயே கண்டறிவதில் சிரமங்களை
        எதிர்கொள்ளலாம்.

        சில பகுதிகளில் வேளாண் நிபுணர்களை உடனடியாக அணுக முடியாமல்
        இருக்கலாம். நோய்களை கைமுறையாக அடையாளம் காண்பதற்கும்
        அனுபவம் தேவைப்படுகிறது.

        நோய்களைத் தாமதமாகக் கண்டறிவதால் அவை மற்ற தாவரங்களுக்குப்
        பரவக்கூடும். இதனால் தாவர வளர்ச்சியும் பயிர் உற்பத்தியும்
        பாதிக்கப்படலாம்.

        எனவே, தாவர இலைப் படங்களைப் பயன்படுத்தி ஆரம்பகட்ட தகவலை
        வழங்கக்கூடிய எளிய அமைப்பு தேவைப்படுகிறது.
        """)

        st.header("3. முன்மொழியப்பட்ட தீர்வு")

        st.write("""
        PlantCare AI ஆழ்கற்றல் தொழில்நுட்பத்தைப் பயன்படுத்தி
        தாவர இலைப் படங்களை வகைப்படுத்துகிறது.

        பயனர் இலையின் படத்தைப் பதிவேற்றியதும், படம் தேவையான
        அளவிற்கு மாற்றப்பட்டு எண் மதிப்புகளாகச் சீரமைக்கப்படுகிறது.

        பின்னர் பயிற்சி பெற்ற சுருள் நரம்பு வலையமைப்பு மாதிரி
        படத்தை ஆய்வு செய்து 15 வகைகளுக்கான மதிப்பெண்களை உருவாக்குகிறது.

        அதிக மதிப்பெண் பெற்ற வகை கணிப்பாகக் காட்டப்படுகிறது.
        அதனுடன் தொடர்புடைய நோய் விவரங்கள் மற்றும் பொதுவான
        பராமரிப்பு பரிந்துரைகளும் வழங்கப்படுகின்றன.
        """)

        st.header("4. திட்டத்தின் நோக்கங்கள்")

        st.markdown("""
        - தாவர இலைப் படங்களைப் பயன்படுத்தி நோய்களை வகைப்படுத்துதல்.
        - படங்களின் காட்சிப் பண்புகளைக் கண்டறிய ஆழ்கற்றலைப் பயன்படுத்துதல்.
        - குடைமிளகாய், உருளைக்கிழங்கு மற்றும் தக்காளி ஆகிய தாவரங்களின் 15 வகைகளை ஆதரித்தல்.
        - நோய்களின் அறிகுறிகள் மற்றும் சாத்தியமான காரணங்களை வழங்குதல்.
        - பொதுவான பராமரிப்பு மற்றும் தடுப்பு முறைகளைப் பரிந்துரைத்தல்.
        - பழைய கணிப்புகளை உள்ளூர் தரவுத்தளத்தில் சேமித்தல்.
        - மீண்டும் பரிசோதிப்பதற்கான நினைவூட்டல்களை உருவாக்குதல்.
        - தமிழ் மற்றும் ஆங்கில மொழி வசதியை வழங்குதல்.
        - கணிப்பு விவரங்களை குரல் மூலம் கேட்கும் வசதியை வழங்குதல்.
        """)

        st.header("5. அமைப்பு கட்டமைப்பு")

        st.markdown("""
        **படி 1 – பயனர் இடைமுகம்**

        பயனர் செயலியைத் திறந்து தாவர இலையின் படத்தைப் பதிவேற்றுகிறார்.

        **படி 2 – படச் செயலாக்கம்**

        பதிவேற்றப்பட்ட படம் மூன்று வண்ண அலைவரிசைகளாக மாற்றப்பட்டு
        128 × 128 அளவிற்கு மாற்றப்படுகிறது. பின்னர் அதன் எண் மதிப்புகள்
        மாதிரிக்குத் தேவையான வகையில் சீரமைக்கப்படுகின்றன.

        **படி 3 – CNN மாதிரி**

        செயலாக்கப்பட்ட படம் பயிற்சி பெற்ற மாதிரிக்கு அனுப்பப்படுகிறது.
        சுருள் அடுக்குகள் படத்தின் பண்புகளைக் கண்டறிந்து இறுதி
        வகைப்படுத்தல் அடுக்கு 15 வகைகளுக்கான மதிப்பெண்களை உருவாக்குகிறது.

        **படி 4 – கணிப்பு முடிவு**

        அதிக மதிப்பெண் பெற்ற வகை தேர்ந்தெடுக்கப்பட்டு அதன் விவரங்கள்
        பயனருக்குக் காட்டப்படுகின்றன.

        **படி 5 – தரவுத்தளம்**

        கணிப்பு பதிவுகள் மற்றும் நினைவூட்டல்கள் உள்ளூர் தரவுத்தளத்தில்
        சேமிக்கப்படுகின்றன.

        **படி 6 – கூடுதல் வசதிகள்**

        மொழி மாற்றம், பகல் மற்றும் இரவு தோற்றம், கணிப்பு வரலாறு மற்றும்
        குரல் வெளியீடு ஆகியவை வழங்கப்படுகின்றன.
        """)

        st.header("6. செயல்படும் முறை")

        st.markdown("""
        1. PlantCare AI செயலியைத் திறக்கவும்.
        2. தமிழ் அல்லது ஆங்கிலத்தைத் தேர்ந்தெடுக்கவும்.
        3. பகல் தோற்றம் அல்லது இரவு தோற்றத்தைத் தேர்ந்தெடுக்கவும்.
        4. நோய் கண்டறிதல் பக்கத்தைத் திறக்கவும்.
        5. தாவர இலையின் தெளிவான படத்தைப் பதிவேற்றவும்.
        6. நீங்கள் கவனித்துள்ள பாதிப்பு அளவைத் தேர்ந்தெடுக்கவும்.
        7. நோயைக் கண்டறியும் பொத்தானை அழுத்தவும்.
        8. செயலி படத்தைச் செயலாக்கும்.
        9. பயிற்சி பெற்ற மாதிரி இலையின் வகையைக் கணிக்கும்.
        10. கணிப்பு மற்றும் தொடர்புடைய தகவல்கள் காட்டப்படும்.
        11. கணிப்பு உள்ளூர் தரவுத்தளத்தில் சேமிக்கப்படும்.
        12. பயனர் பரிந்துரைகளைக் குரலில் கேட்கலாம் அல்லது நினைவூட்டலைச் சேமிக்கலாம்.
        """)

        st.header("7. தரவுத்தொகுப்பு விளக்கம்")

        st.write("""
        இந்தத் திட்டத்தில் PlantVillage எனப்படும் பெயரிடப்பட்ட
        தாவர இலைப் படங்களின் தரவுத்தொகுப்பு பயன்படுத்தப்படுகிறது.

        இந்தத் திட்டத்திற்காக குடைமிளகாய், உருளைக்கிழங்கு மற்றும்
        தக்காளி ஆகிய மூன்று தாவரக் குழுக்களின் படங்கள் தேர்ந்தெடுக்கப்பட்டன.

        பயிற்சி, சரிபார்ப்பு மற்றும் சோதனை ஆகிய மூன்று பிரிவுகளாக
        படங்கள் பிரிக்கப்பட்டன.
        """)

        st.markdown("""
        | தரவுப் பிரிவு | படங்களின் எண்ணிக்கை | சதவீதம் |
        |---|---:|---:|
        | பயிற்சி | 14,446 | 70% |
        | சரிபார்ப்பு | 3,096 | 15% |
        | சோதனை | 3,096 | 15% |
        | **மொத்தம்** | **20,638** | **100%** |
        """)

        st.write("இந்த மாதிரி ஆதரிக்கும் 15 வகைகள்:")

        st.markdown("""
        **குடைமிளகாய்**
        - பாக்டீரியா இலைப்புள்ளி நோய்
        - ஆரோக்கியமான இலை

        **உருளைக்கிழங்கு**
        - ஆரம்பகட்ட இலைக்கருகல் நோய்
        - தாமத இலைக்கருகல் நோய்
        - ஆரோக்கியமான இலை

        **தக்காளி**
        - பாக்டீரியா இலைப்புள்ளி நோய்
        - ஆரம்பகட்ட இலைக்கருகல் நோய்
        - தாமத இலைக்கருகல் நோய்
        - இலைப் பூஞ்சை நோய்
        - செப்டோரியா இலைப்புள்ளி நோய்
        - இரண்டு புள்ளி சிலந்திப்பேன் தாக்குதல்
        - இலக்கு வடிவ இலைப்புள்ளி நோய்
        - மஞ்சள் இலை சுருள் வைரஸ் நோய்
        - மொசைக் வைரஸ் நோய்
        - ஆரோக்கியமான இலை
        """)

        st.header("8. CNN மாதிரி கட்டமைப்பு")

        st.write("""
        சுருள் நரம்பு வலையமைப்பு என்பது படங்களை வகைப்படுத்தப்
        பயன்படும் ஆழ்கற்றல் மாதிரியாகும்.

        இந்தத் திட்டத்தில் பயன்படுத்தப்படும் மாதிரியில் பின்வரும்
        முக்கிய அடுக்குகள் உள்ளன:
        """)

        st.markdown("""
        1. **உள்ளீட்டு அடுக்கு:** 128 × 128 அளவுள்ள வண்ணப் படத்தைப் பெறுகிறது.
        2. **முதல் சுருள் அடுக்கு:** அடிப்படை காட்சிப் பண்புகளைக் கண்டறிகிறது.
        3. **முதல் அதிகபட்சத் தொகுப்பாக்க அடுக்கு:** படத் தகவலின் அளவைக் குறைக்கிறது.
        4. **இரண்டாவது சுருள் அடுக்கு:** மேலும் விரிவான பண்புகளைக் கற்றுக்கொள்கிறது.
        5. **இரண்டாவது தொகுப்பாக்க அடுக்கு:** படத் தகவலின் பரிமாணத்தைக் குறைக்கிறது.
        6. **மூன்றாவது சுருள் அடுக்கு:** ஆழமான காட்சிப் பண்புகளைக் கண்டறிகிறது.
        7. **மூன்றாவது தொகுப்பாக்க அடுக்கு:** பரிமாணத்தைக் மேலும் குறைக்கிறது.
        8. **தட்டையாக்கும் அடுக்கு:** பண்புகளை ஒரு வரிசையாக மாற்றுகிறது.
        9. **அடர்த்தி அடுக்கு:** வகைப்படுத்தலுக்கான தொடர்புகளைக் கற்றுக்கொள்கிறது.
        10. **விடுபடுத்தல் அடுக்கு:** அளவுக்கு அதிகமாகப் பயிற்சி பெறுவதைத் தடுக்க உதவுகிறது.
        11. **வெளியீட்டு அடுக்கு:** 15 வகைகளுக்கான மதிப்பெண்களை உருவாக்குகிறது.
        """)

        st.header("9. மாதிரியின் மதிப்பீடு")

        st.write("""
        பயிற்சியில் பயன்படுத்தப்படாத தனியான சோதனைத் தரவுத்தொகுப்பைப்
        பயன்படுத்தி அசல் CNN மாதிரி மதிப்பீடு செய்யப்பட்டது.

        சோதனையில் கிடைத்த துல்லியம் சுமார் 89.66% ஆகும்.

        துல்லியம் மட்டும் எல்லா நோய் வகைகளின் செயல்திறனையும்
        முழுமையாக விளக்காது. ஒவ்வொரு வகைக்கும் சரியான கணிப்புகளின்
        விகிதம், கண்டறியும் திறன் மற்றும் F1 மதிப்பெண் போன்ற
        அளவுகளும் முக்கியமானவை.
        """)

        st.metric("அசல் CNN மாதிரியின் சோதனைத் துல்லியம்", "89.66%")

        st.header("10. பயன்படுத்தப்பட்ட தொழில்நுட்பங்கள்")

        st.markdown("""
        | தொழில்நுட்பம் | பயன்பாடு |
        |---|---|
        | Python | முதன்மை நிரலாக்க மொழி |
        | TensorFlow | ஆழ்கற்றல் கட்டமைப்பு |
        | Keras | CNN மாதிரி உருவாக்கம் |
        | NumPy | எண் கணக்கீடுகள் மற்றும் படச் செயலாக்கம் |
        | Pillow | படங்களைத் திறத்தல் மற்றும் அளவு மாற்றம் |
        | Streamlit | இணையச் செயலி இடைமுகம் |
        | SQLite | உள்ளூர் தரவுத்தளம் |
        | Pandas | கணிப்பு வரலாறு மற்றும் அட்டவணைகள் |
        | gTTS | உரையை குரலாக மாற்றுதல் |
        | VS Code | நிரல் உருவாக்கும் சூழல் |
        """)

        st.header("11. முக்கிய செயலி அம்சங்கள்")

        st.markdown("""
        - தாவர இலைப் படத்தைப் பதிவேற்றுதல் மற்றும் முன்னோட்டம்.
        - CNN மாதிரியைப் பயன்படுத்தி நோய் வகைப்படுத்துதல்.
        - மாதிரி மதிப்பெண் மற்றும் நம்பகத்தன்மை எச்சரிக்கைகள்.
        - நோய் அறிகுறிகள், காரணங்கள் மற்றும் பராமரிப்பு பரிந்துரைகள்.
        - தமிழ் மற்றும் ஆங்கில மொழித் தேர்வு.
        - பகல் மற்றும் இரவு தோற்ற அமைப்பு.
        - கணிப்பு வரலாறு மற்றும் கோப்பாகப் பதிவிறக்கும் வசதி.
        - தொடர் பராமரிப்பு நினைவூட்டல்கள்.
        - கணிப்பு விவரங்களை குரலில் கேட்கும் வசதி.
        """)

        st.header("12. திட்டத்தின் வரம்புகள்")

        st.markdown("""
        - பயிற்சியில் பயன்படுத்தப்பட்ட 15 வகைகளை மட்டுமே மாதிரி ஆதரிக்கிறது.
        - பயிற்சிப் படங்களிலிருந்து மாறுபடும் உண்மையான வயல்வெளிப் படங்களில் செயல்திறன் மாறலாம்.
        - மங்கலான படங்கள், குறைந்த வெளிச்சம் மற்றும் பின்னணி மாற்றங்கள் கணிப்பைப் பாதிக்கலாம்.
        - மாதிரி மதிப்பெண் என்பது சரியான கணிப்பிற்கான உறுதிப்படுத்தப்பட்ட நிகழ்தகவு அல்ல.
        - நோயின் தீவிரத்தை மாதிரி தானாகக் கண்டறிவதில்லை.
        - செயலி திறந்திருக்கும் போது மட்டுமே நினைவூட்டல்கள் காட்டப்படும்.
        - குரல் உருவாக்கத்திற்கு இணைய இணைப்பு தேவை.
        - இந்த அமைப்பு வேளாண் நிபுணரின் ஆலோசனைக்கு மாற்றாகாது.
        """)

        st.header("13. எதிர்கால மேம்பாடுகள்")

        st.markdown("""
        - மேலும் பல பயிர்கள் மற்றும் நோய் வகைகளைச் சேர்த்தல்.
        - உண்மையான வயல்வெளிப் படங்களில் மாதிரியின் செயல்திறனை மேம்படுத்துதல்.
        - இணைய இணைப்பு இல்லாமல் இயங்கும் கைப்பேசி செயலியை உருவாக்குதல்.
        - கூடுதல் பிராந்திய மொழிகளைச் சேர்த்தல்.
        - வானிலை மற்றும் சுற்றுச்சூழல் தகவல்களை ஒருங்கிணைத்தல்.
        - வேளாண் நிபுணர் ஆலோசனை வசதியைச் சேர்த்தல்.
        - திட்டமிட்ட பராமரிப்பிற்கான அறிவிப்புகளை வழங்குதல்.
        - கைப்பேசியில் இயங்கக்கூடிய குறைந்த எடை கொண்ட மாதிரிகளை உருவாக்குதல்.
        """)

        st.header("14. பயன்பாடுகள்")

        st.markdown("""
        - விவசாயிகளுக்கு ஆரம்பகட்ட தாவர நோய் பரிசோதனை.
        - வீட்டுத் தோட்டத் தாவரங்களைக் கண்காணித்தல்.
        - வேளாண் கல்வி மற்றும் செயல்விளக்கங்கள்.
        - கல்லூரி செயற்கை நுண்ணறிவு மற்றும் இயந்திரக் கற்றல் திட்டங்கள்.
        - நிபுணர்களை எளிதில் அணுக முடியாத பகுதிகளில் ஆரம்பகட்ட தகவல் வழங்குதல்.
        """)

        st.header("15. முடிவுரை")

        st.write("""
        PlantCare AI திட்டம் தாவர இலைப் படங்களை வகைப்படுத்துவதற்கு
        ஆழ்கற்றல் தொழில்நுட்பத்தை எவ்வாறு பயன்படுத்தலாம் என்பதை
        விளக்குகிறது.

        CNN மாதிரியையும் எளிய செயலி இடைமுகத்தையும் ஒருங்கிணைத்து,
        நோய் விவரங்கள், கணிப்பு வரலாறு மற்றும் தொடர் பராமரிப்பு
        நினைவூட்டல்களை வழங்குகிறது.

        தரவுத்தொகுப்பின் பல்வகைத்தன்மை, உண்மையான வயல்வெளிப்
        பரிசோதனைகள் மற்றும் மாதிரி மதிப்பீடு ஆகியவற்றை மேம்படுத்துவதன்
        மூலம் இந்த அமைப்பை எதிர்காலத்தில் மேலும் நம்பகமானதாக மாற்றலாம்.
        """)

    st.markdown("---")
    st.info(t("disclaimer"))


# =========================================================
# 17. FOOTER
# =========================================================

st.markdown("---")

if st.session_state.language == "ta":
    st.markdown(
        "<center>🌿 PlantCare AI | தாவர நோய் கண்டறிதல் அமைப்பு</center>",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        "<center>🌿 PlantCare AI | Plant Disease Detection System</center>",
        unsafe_allow_html=True
    )
    