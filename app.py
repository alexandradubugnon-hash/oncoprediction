"""
=============================================================================
OncoPrediction — Application Streamlit
Prédiction de survie — Cancers gastro-œsophagiens
Outil d'aide à la décision pour les Tumor Boards
=============================================================================
Données : TCGA via cBioPortal  |  n = 617 patients
Modèles : Cox sksurv (C-index = 0.665)  +  RSF (C-index = 0.635)
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import os
import json
import warnings
import requests
import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from streamlit_option_menu import option_menu
warnings.filterwarnings('ignore')

# =============================================================================
# CHEMINS — relatifs à ce fichier (fonctionnent en local ET sur Render)
# =============================================================================
ROOT_DIR   = os.path.dirname(os.path.abspath(__file__))
COX_PATH   = os.path.join(ROOT_DIR, 'models', 'cox_model.pkl')
RSF_PATH   = os.path.join(ROOT_DIR, 'models', 'rsf_model.pkl')

# 6 variables du modèle de Cox
COX_VARS = [
    'Age', 'Grade',
    'AJCC_Stage_Simplified_III', 'AJCC_Stage_Simplified_IV',
    'Radiation_Therapy', 'TMB',
]

# =============================================================================
# CONFIGURATION FIREBASE
# =============================================================================
FIREBASE_API_KEY = st.secrets.get("FIREBASE_API_KEY") or os.environ.get("FIREBASE_API_KEY")

FIREBASE_CONFIG = {
    "apiKey": FIREBASE_API_KEY,
    "authDomain": "oncoprediction.firebaseapp.com",
    "projectId": "oncoprediction",
    "storageBucket": "oncoprediction.firebasestorage.app",
    "messagingSenderId": "242172897083",
    "appId": "1:242172897083:web:2e2d211e86f97f79edf91e",
    "measurementId": "G-09766X2VL1",
    "databaseURL": "",  # Requis par pyrebase même si non utilisé
}

# URL de l'API Firebase Auth REST
FIREBASE_AUTH_URL = "https://identitytoolkit.googleapis.com/v1/accounts"
FIREBASE_API_KEY = FIREBASE_CONFIG["apiKey"]


# =============================================================================
# FONCTIONS D'AUTHENTIFICATION FIREBASE (API REST directe)
# =============================================================================
def firebase_sign_up(email, password):
    """Crée un nouveau compte Firebase avec email/mot de passe."""
    url = f"{FIREBASE_AUTH_URL}:signUp?key={FIREBASE_API_KEY}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        data = response.json()
        if "error" in data:
            error_msg = data["error"]["message"]
            if "EMAIL_EXISTS" in error_msg:
                return None, "Cette adresse email est déjà utilisée."
            elif "WEAK_PASSWORD" in error_msg:
                return None, "Le mot de passe doit contenir au moins 6 caractères."
            elif "INVALID_EMAIL" in error_msg:
                return None, "L'adresse email est invalide."
            else:
                return None, f"Erreur : {error_msg}"
        return data, None
    except requests.exceptions.RequestException:
        return None, "Erreur de connexion au serveur. Réessayez."


def firebase_sign_in(email, password):
    """Connecte un utilisateur Firebase avec email/mot de passe."""
    url = f"{FIREBASE_AUTH_URL}:signInWithPassword?key={FIREBASE_API_KEY}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        data = response.json()
        if "error" in data:
            error_msg = data["error"]["message"]
            if "EMAIL_NOT_FOUND" in error_msg or "INVALID_PASSWORD" in error_msg:
                return None, "Email ou mot de passe incorrect."
            elif "INVALID_LOGIN_CREDENTIALS" in error_msg:
                return None, "Email ou mot de passe incorrect."
            elif "TOO_MANY_ATTEMPTS_TRY_LATER" in error_msg:
                return None, "Trop de tentatives. Réessayez plus tard."
            else:
                return None, f"Erreur : {error_msg}"
        return data, None
    except requests.exceptions.RequestException:
        return None, "Erreur de connexion au serveur. Réessayez."


def firebase_reset_password(email):
    """Envoie un email de réinitialisation de mot de passe."""
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={FIREBASE_API_KEY}"
    payload = {
        "requestType": "PASSWORD_RESET",
        "email": email
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        data = response.json()
        if "error" in data:
            error_msg = data["error"]["message"]
            if "EMAIL_NOT_FOUND" in error_msg:
                return False, "Aucun compte trouvé avec cette adresse email."
            else:
                return False, f"Erreur : {error_msg}"
        return True, None
    except requests.exceptions.RequestException:
        return False, "Erreur de connexion au serveur. Réessayez."


def firebase_send_verification_email(id_token):
    """Envoie un email de vérification d'adresse à l'utilisateur connecté."""
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={FIREBASE_API_KEY}"
    payload = {"requestType": "VERIFY_EMAIL", "idToken": id_token}
    try:
        response = requests.post(url, json=payload, timeout=10)
        data = response.json()
        return "error" not in data
    except requests.exceptions.RequestException:
        return False


def firebase_get_account_info(id_token):
    """Récupère les informations du compte (dont emailVerified)."""
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key={FIREBASE_API_KEY}"
    payload = {"idToken": id_token}
    try:
        response = requests.post(url, json=payload, timeout=10)
        data = response.json()
        if "error" in data or "users" not in data:
            return None
        return data
    except requests.exceptions.RequestException:
        return None


# =============================================================================
# PAGE CONFIG ET CSS
# =============================================================================
st.set_page_config(
    page_title="OncoPrediction — Tumor Board",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# FIRESTORE — initialisation (une seule fois par processus)
# =============================================================================
_db = None
try:
    if not firebase_admin._apps:
        # Option 1 : variable d'environnement (production sur Render)
        firebase_key_json = os.environ.get("FIREBASE_KEY_JSON")
        if firebase_key_json:
            try:
                key_dict = json.loads(firebase_key_json)
                cred = credentials.Certificate(key_dict)
                firebase_admin.initialize_app(cred)
            except Exception as e:
                print(f"Erreur init Firebase depuis env: {e}")
        # Option 2 : fichier local (développement local)
        elif os.path.exists("firebase_key.json"):
            try:
                cred = credentials.Certificate("firebase_key.json")
                firebase_admin.initialize_app(cred)
            except Exception as e:
                print(f"Erreur init Firebase depuis fichier: {e}")
    _db = firestore.client()
except Exception:
    _db = None  # L'app fonctionne sans Firestore si la clé est absente

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Masquer le chrome Streamlit ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
.stDeployButton       { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stToolbar"]    { display: none !important; }


/* ── Fond général ── */
.stApp { background: #f4f6f9 !important; }
[data-testid="stAppViewContainer"] > .main { background: #f4f6f9 !important; }

/* ── Sidebar — verrouillée toujours ouverte ── */
/* 1. Masque le bouton ">" de réouverture (zone principale, sidebar réduite) */
[data-testid="collapsedControl"] { display: none !important; }

/* 2. Masque le bouton de fermeture INTERNE à la sidebar (icône Material Icons cassée "keyb...") */
/*    La sidebar est verrouillée ouverte via min-width + transform : ce bouton est inutile */
section[data-testid="stSidebar"] > div:first-child > button:first-of-type {
    display: none !important;
}

/* 3. Empêche physiquement la sidebar de se fermer */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
    min-width: 320px !important;
    max-width: 320px !important;
    transform: none !important;
}
[data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 320px !important;
    max-width: 320px !important;
    margin-left: 0 !important;
    transform: none !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 12px; }

/* ── Titres formulaire sidebar ── */
.form-title {
    font-size: 17px; font-weight: 800; color: #0b1628;
    letter-spacing: -0.02em; margin: 4px 0 2px;
}
.form-subtitle { font-size: 11px; color: #94a3b8; margin: 0 0 14px; }

/* ── Expanders → accordéons dans la sidebar ── */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    border: 1px solid #e8ecf0 !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
    background: #ffffff !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    font-size: 13px !important; font-weight: 600 !important;
    color: #0b1628 !important; padding: 10px 14px !important;
    background: #f8fafc !important; border-radius: 10px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
    background: #f1f5f9 !important;
}

/* ── Radio → boutons chip dans la sidebar ── */
[data-testid="stSidebar"] [data-testid="stRadioGroup"] {
    display: flex !important;
    flex-direction: row !important;
    gap: 4px !important;
    flex-wrap: nowrap !important;
}
[data-testid="stSidebar"] [data-testid="stRadioGroup"] label {
    flex: 1 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 6px 4px !important;
    border-radius: 6px !important;
    border: 1.5px solid #e2e8f0 !important;
    background: #f8fafc !important;
    cursor: pointer !important;
    margin: 0 !important;
    min-width: 0 !important;
    transition: all 0.12s ease !important;
}
[data-testid="stSidebar"] [data-testid="stRadioGroup"] label:has(input:checked) {
    background: #0b1628 !important;
    border-color: #0b1628 !important;
}
[data-testid="stSidebar"] [data-testid="stRadioGroup"] label p {
    font-size: 12px !important; font-weight: 500 !important;
    color: #64748b !important; margin: 0 !important; line-height: 1.2 !important;
}
[data-testid="stSidebar"] [data-testid="stRadioGroup"] label:has(input:checked) p {
    color: #ffffff !important; font-weight: 600 !important;
}
[data-testid="stSidebar"] [data-testid="stRadioGroup"] label input[type="radio"] {
    position: absolute !important; opacity: 0 !important;
    width: 0 !important; height: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadioGroup"] label > div {
    display: none !important;
}

/* ── Bouton principal Streamlit dans la sidebar ── */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
    color: #ffffff !important; border: none !important;
    border-radius: 8px !important; font-weight: 700 !important;
    font-size: 14px !important; padding: 11px 16px !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 2px 8px rgba(14,165,233,0.30) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0284c7 0%, #0b1628 100%) !important;
}

/* ── Header app ── */
.app-header {
    background: #0b1628;
    padding: 16px 24px; border-radius: 14px; margin-bottom: 20px;
    display: flex; align-items: center; justify-content: space-between;
}
.app-header-left  { display: flex; align-items: center; gap: 12px; }
.app-header-icon  {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #1e3a5f, #0ea5e9);
    border-radius: 9px; display: flex; align-items: center;
    justify-content: center; font-size: 13px; font-weight: 800; color: white;
}
.app-header-title { font-size: 17px; font-weight: 800; color: white;
                    letter-spacing: -0.02em; margin: 0; line-height: 1.2; }
.app-header-title em { color: #0ea5e9; font-style: normal; }
.app-header-sub   { font-size: 11px; color: rgba(255,255,255,0.42);
                    margin: 0; letter-spacing: 0.01em; }
.app-header-right { display: flex; align-items: center; gap: 14px; }
.proto-badge {
    background: rgba(14,165,233,0.12); border: 1px solid rgba(14,165,233,0.25);
    color: #38bdf8; font-size: 10px; font-weight: 700;
    padding: 4px 12px; border-radius: 100px;
    letter-spacing: 0.06em; text-transform: uppercase;
}
.user-badge {
    background: rgba(255,255,255,0.08); color: rgba(255,255,255,0.65);
    padding: 5px 14px; border-radius: 20px; font-size: 12px; font-weight: 500;
}
.back-link { color: rgba(255,255,255,0.38); font-size: 12px; text-decoration: none; }
.back-link:hover { color: rgba(255,255,255,0.75); }

/* ── Titres de section (contenu principal) ── */
.section-title {
    font-size: 11px; font-weight: 700; color: #0ea5e9;
    letter-spacing: 0.08em; text-transform: uppercase;
    padding-bottom: 8px; border-bottom: 1px solid #e8ecf0; margin: 18px 0 12px;
}

/* ── Carte AI Consensus ── */
.ai-card {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
    padding: 20px 24px; display: flex; align-items: stretch;
    gap: 0; margin-bottom: 14px;
}
.ai-card-left { flex: 1; padding-right: 24px; }
.ai-card-right {
    min-width: 210px; text-align: center; padding: 0 0 0 24px;
    border-left: 1px solid #f1f5f9;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.ai-card-header {
    font-size: 10px; font-weight: 700; color: #94a3b8;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 10px;
}
.ai-card-badges { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.confidence-txt { font-size: 12px; color: #94a3b8; font-weight: 500; }
.ai-insight-text { font-size: 13px; color: #475569; line-height: 1.65; }
.ai-median-label {
    font-size: 10px; font-weight: 700; color: #94a3b8;
    letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 6px;
}
.ai-median-value {
    font-size: 34px; font-weight: 800; color: #0b1628;
    letter-spacing: -0.03em; line-height: 1;
}
.ai-median-bar {
    height: 3px; background: linear-gradient(90deg, #0b1628, #0ea5e9);
    border-radius: 2px; margin-top: 10px; width: 70%;
}

/* ── Cartes métriques horizontales avec barres ── */
.metric-card-h {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 16px 14px;
}
.metric-h-label {
    font-size: 11px; color: #94a3b8; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;
}
.metric-h-value {
    font-size: 26px; font-weight: 800; color: #0b1628;
    letter-spacing: -0.03em; margin-bottom: 8px; line-height: 1;
}
.progress-bar-bg { height: 5px; background: #f1f5f9; border-radius: 3px; overflow: hidden; }
.progress-bar-fill { height: 100%; border-radius: 3px; }
.progress-green  { background: #22c55e; }
.progress-orange { background: #f59e0b; }
.progress-red    { background: #ef4444; }
.progress-navy   { background: linear-gradient(90deg, #0b1628, #0ea5e9); }
.metric-h-sub { font-size: 10px; color: #cbd5e1; margin-top: 5px; }

/* ── Cartes métriques ── */
.metrics-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 8px; margin-bottom: 14px;
}
.metric-card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 12px 10px; text-align: center;
}
.metric-label {
    font-size: 10px; font-weight: 700; color: #94a3b8;
    letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 4px;
}
.metric-value {
    font-size: 22px; font-weight: 800; color: #0b1628;
    letter-spacing: -0.03em; line-height: 1;
}
.metric-value em { color: #0ea5e9; font-style: normal; font-size: 16px; }
.metric-sub { font-size: 10px; color: #cbd5e1; margin-top: 3px; }

/* ── Risque ── */
.risk-card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 14px 12px;
    text-align: center; margin-bottom: 10px;
}
.risk-label-txt {
    font-size: 10px; font-weight: 700; color: #94a3b8;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 7px;
}
.risk-badge {
    display: inline-block; padding: 6px 20px; border-radius: 8px;
    font-weight: 700; font-size: 15px;
}
.risk-low    { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
.risk-medium { background: #fef9c3; color: #854d0e; border: 1px solid #fef08a; }
.risk-high   { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
.risk-median { font-size: 11px; color: #94a3b8; margin-top: 5px; }

/* ── Résumé patient ── */
.summary-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 0; border-bottom: 1px solid #f1f5f9; font-size: 13px;
}
.summary-row:last-child { border-bottom: none; }
.summary-key { color: #94a3b8; font-weight: 500; }
.summary-val { color: #0b1628; font-weight: 700; }

/* ── Titre graphe ── */
.graph-header { margin-bottom: 10px; }
.graph-title  { font-size: 14px; font-weight: 700; color: #0b1628;
                letter-spacing: -0.01em; margin: 0; }
.graph-sub    { font-size: 11px; color: #94a3b8; margin: 2px 0 0; font-style: italic; }

/* ── État vide / Welcome state ── */
.welcome-state {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 56px 24px 40px;
    text-align: center;
}
.welcome-icon-wrap {
    width: 72px; height: 72px;
    background: linear-gradient(135deg, #0b1628 0%, #1e3a5f 50%, #0ea5e9 100%);
    border-radius: 20px; display: flex; align-items: center; justify-content: center;
    margin-bottom: 20px; box-shadow: 0 8px 24px rgba(14,165,233,0.22);
}
.welcome-icon-txt { font-size: 26px; font-weight: 900; color: #ffffff; letter-spacing: -0.04em; }
.welcome-title {
    font-size: 22px; font-weight: 800; color: #0b1628;
    letter-spacing: -0.03em; margin-bottom: 8px;
}
.welcome-title em { color: #0ea5e9; font-style: normal; }
.welcome-sub {
    font-size: 14px; color: #94a3b8; line-height: 1.6;
    max-width: 480px; margin: 0 auto 32px;
}
.welcome-cards {
    display: flex; gap: 14px; justify-content: center; flex-wrap: wrap;
    max-width: 720px; margin: 0 auto;
}
.welcome-card {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 18px 20px; flex: 1; min-width: 160px; max-width: 220px;
    text-align: left;
}
.welcome-card-num {
    font-size: 20px; font-weight: 900; color: #0ea5e9;
    letter-spacing: -0.04em; margin-bottom: 6px;
}
.welcome-card-title {
    font-size: 13px; font-weight: 700; color: #0b1628; margin-bottom: 4px;
}
.welcome-card-desc { font-size: 12px; color: #94a3b8; line-height: 1.5; }
.welcome-hint {
    margin-top: 28px; padding: 12px 20px;
    background: rgba(14,165,233,0.06); border: 1px solid rgba(14,165,233,0.18);
    border-radius: 8px; font-size: 12px; color: #0ea5e9; font-weight: 500;
}

/* ── AI Insight comparaison ── */
.comp-insight-card {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
    padding: 20px 24px; margin-top: 18px;
}
.comp-insight-header {
    font-size: 10px; font-weight: 700; color: #94a3b8;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 12px;
    padding-bottom: 10px; border-bottom: 1px solid #f1f5f9;
}
.comp-insight-grid {
    display: flex; gap: 14px; flex-wrap: wrap;
}
.comp-insight-item {
    flex: 1; min-width: 120px; padding: 12px 14px;
    background: #f8fafc; border-radius: 8px; border: 1px solid #f1f5f9;
}
.comp-insight-item-label {
    font-size: 10px; color: #94a3b8; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;
}
.comp-insight-item-val {
    font-size: 16px; font-weight: 800; color: #0b1628; letter-spacing: -0.02em;
}
.comp-insight-text {
    font-size: 13px; color: #475569; line-height: 1.65; margin-top: 14px;
}

/* ── Footer ── */
.app-footer {
    text-align: center; padding: 12px; margin-top: 8px;
    font-size: 11px; color: #cbd5e1; border-top: 1px solid #f1f5f9;
}
.app-footer a { color: #0ea5e9; text-decoration: none; }

/* ── Panneau patient (Vue 2 — résultats) ── */
.patient-panel {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 14px; padding: 20px 16px;
}
.patient-avatar {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, #0b1628 0%, #1e3a5f 50%, #0ea5e9 100%);
    border-radius: 14px; display: flex; align-items: center; justify-content: center;
    margin: 0 auto 10px;
}
.patient-name {
    font-size: 15px; font-weight: 800; color: #0b1628;
    letter-spacing: -0.02em; text-align: center; margin-bottom: 2px;
}
.patient-date {
    font-size: 11px; color: #94a3b8; text-align: center; margin-bottom: 14px;
}
.patient-section-title {
    font-size: 9px; font-weight: 700; color: #94a3b8;
    letter-spacing: 0.1em; text-transform: uppercase;
    margin: 12px 0 8px; padding-bottom: 6px;
    border-bottom: 1px solid #f1f5f9;
}
.param-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 5px 0; border-bottom: 1px solid #f8fafc; font-size: 12px;
}
.param-row:last-child { border-bottom: none; }
.param-key { color: #94a3b8; font-weight: 500; }
.param-val { color: #0b1628; font-weight: 700; }

/* ── Footer disclaimer (Vue 2) ── */
.view2-footer {
    margin-top: 20px; padding: 14px 18px;
    background: rgba(14,165,233,0.04); border: 1px solid rgba(14,165,233,0.12);
    border-radius: 8px; font-size: 11px; color: #64748b; line-height: 1.7;
}

/* ── Auth page ── */
.auth-header {
    background: #0b1628; padding: 28px 24px; border-radius: 14px;
    text-align: center; margin-bottom: 28px;
}
.auth-header h1 { margin: 0; font-size: 22px; font-weight: 800;
                  color: white; letter-spacing: -0.02em; }
.auth-header h1 em { color: #0ea5e9; font-style: normal; }
.auth-header p  { margin: 6px 0 0; font-size: 13px; color: rgba(255,255,255,0.5); }
.auth-title {
    text-align: center; font-size: 16px; font-weight: 700;
    color: #0b1628; margin-bottom: 20px; letter-spacing: -0.01em;
}
.auth-footer {
    text-align: center; margin-top: 16px; font-size: 11px; color: #94a3b8; line-height: 1.7;
}

/* ── Sidebar : profil utilisateur ── */
.sidebar-user-card {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 4px 12px; border-bottom: 1px solid #f1f5f9; margin-bottom: 10px;
}
.sidebar-avatar {
    width: 38px; height: 38px; flex-shrink: 0;
    background: linear-gradient(135deg, #0b1628 0%, #1e3a5f 50%, #0ea5e9 100%);
    border-radius: 10px; display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 800; color: white; letter-spacing: 0.02em;
}
.sidebar-user-info { min-width: 0; }
.sidebar-user-label {
    font-size: 9px; font-weight: 700; color: #94a3b8;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 2px;
}
.sidebar-user-email {
    font-size: 12px; font-weight: 600; color: #0b1628;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    max-width: 180px;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# FIRESTORE — SAUVEGARDE DES ANALYSES
# =============================================================================
def save_analysis_to_firestore(user_id, user_email, patient_id, parameters, results):
    """Sauvegarde une analyse patient dans Firestore. Retourne le doc_id ou None."""
    if _db is None:
        return None
    try:
        doc_data = {
            "user_id":    user_id,
            "user_email": user_email,
            "patient_id": patient_id,
            "timestamp":  firestore.SERVER_TIMESTAMP,
            "parameters": parameters,
            "results":    results,
            "notes":      [],
        }
        _timestamp, doc_ref = _db.collection("analyses").add(doc_data)
        return doc_ref.id
    except Exception as e:
        print(f"Erreur sauvegarde Firestore: {e}")
        return None


def add_note_to_analysis(analysis_doc_id, note_text, author_name):
    """Ajoute une note clinique à une analyse."""
    if _db is None:
        return False
    try:
        note = {
            "text":      note_text,
            "timestamp": datetime.datetime.now().strftime("%d/%m/%Y à %H:%M"),
            "author":    author_name,
        }
        _db.collection("analyses").document(analysis_doc_id).update(
            {"notes": firestore.ArrayUnion([note])}
        )
        return True
    except Exception as e:
        st.warning(f"Erreur ajout note : {e}")
        return False


def get_notes_for_analysis(analysis_doc_id):
    """Récupère les notes d'une analyse."""
    if _db is None:
        return []
    try:
        doc = _db.collection("analyses").document(analysis_doc_id).get()
        if doc.exists:
            return doc.to_dict().get("notes", [])
        return []
    except Exception:
        return []


def get_user_analyses(user_id, limit=50):
    """Récupère les analyses de l'utilisateur depuis Firestore, triées par date.

    NOTE : la requête where + order_by nécessite un index composite Firestore.
    Si une erreur apparaît dans le terminal avec un lien Firebase, cliquer sur
    ce lien pour créer l'index automatiquement dans la console Firebase.
    """
    if _db is None:
        return []
    try:
        query = (
            _db.collection("analyses")
            .where("user_id", "==", user_id)
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )
        docs = query.stream()
        analyses = []
        for doc in docs:
            data = doc.to_dict()
            data["doc_id"] = doc.id
            analyses.append(data)
        return analyses
    except Exception as e:
        st.warning(f"Erreur chargement historique: {e}")
        return []


def get_user_profile(user_id):
    """Récupère le profil utilisateur depuis Firestore. Retourne None si absent."""
    if _db is None:
        return None
    try:
        doc = _db.collection("users").document(user_id).get()
        return doc.to_dict() if doc.exists else None
    except Exception as e:
        print(f"Erreur lecture profil: {e}")
        return None


def save_user_profile(user_id, profile_data):
    """Sauvegarde ou met à jour le profil utilisateur dans Firestore (merge=True)."""
    if _db is None:
        return False
    try:
        profile_data["updated_at"] = firestore.SERVER_TIMESTAMP
        _db.collection("users").document(user_id).set(profile_data, merge=True)
        return True
    except Exception as e:
        print(f"Erreur sauvegarde profil: {e}")
        return False


# =============================================================================
# FONCTIONS TUMOR BOARD — Firestore
# =============================================================================

def create_tb_session(user_id, user_email, title, date_str):
    """Crée une nouvelle session Tumor Board dans Firestore."""
    if _db is None:
        return None
    try:
        doc_data = {
            "user_id":     user_id,
            "user_email":  user_email,
            "title":       title,
            "date":        date_str,
            "patient_ids": [],
            "created_at":  firestore.SERVER_TIMESTAMP,
        }
        _, ref = _db.collection("tumor_board_sessions").add(doc_data)
        return ref.id
    except Exception as e:
        st.warning(f"Erreur création session Tumor Board : {e}")
        return None


def get_tb_sessions(user_id):
    """Récupère toutes les sessions Tumor Board de l'utilisateur."""
    if _db is None:
        return []
    try:
        query = (
            _db.collection("tumor_board_sessions")
            .where("user_id", "==", user_id)
            .order_by("created_at", direction=firestore.Query.DESCENDING)
        )
        sessions = []
        for doc in query.stream():
            data = doc.to_dict()
            data["doc_id"] = doc.id
            sessions.append(data)
        return sessions
    except Exception as e:
        st.warning(f"Erreur chargement sessions Tumor Board : {e}")
        return []


def get_tb_session(session_id):
    """Récupère une session Tumor Board par son ID."""
    if _db is None:
        return None
    try:
        doc = _db.collection("tumor_board_sessions").document(session_id).get()
        if doc.exists:
            data = doc.to_dict()
            data["doc_id"] = doc.id
            return data
        return None
    except Exception as e:
        st.warning(f"Erreur lecture session Tumor Board : {e}")
        return None


def add_patient_to_tb_session(session_id, analysis_id):
    """Ajoute un ID d'analyse à la liste des patients d'une session."""
    if _db is None:
        return False
    try:
        _db.collection("tumor_board_sessions").document(session_id).update(
            {"patient_ids": firestore.ArrayUnion([analysis_id])}
        )
        return True
    except Exception as e:
        st.warning(f"Erreur ajout patient : {e}")
        return False


def remove_patient_from_tb_session(session_id, analysis_id):
    """Retire un ID d'analyse de la liste des patients d'une session."""
    if _db is None:
        return False
    try:
        _db.collection("tumor_board_sessions").document(session_id).update(
            {"patient_ids": firestore.ArrayRemove([analysis_id])}
        )
        return True
    except Exception as e:
        st.warning(f"Erreur suppression patient : {e}")
        return False


def delete_tb_session(session_id):
    """Supprime une session Tumor Board."""
    if _db is None:
        return False
    try:
        _db.collection("tumor_board_sessions").document(session_id).delete()
        return True
    except Exception as e:
        st.warning(f"Erreur suppression session Tumor Board : {e}")
        return False


def get_analysis_by_id(doc_id):
    """Récupère une analyse par son doc_id Firestore."""
    if _db is None:
        return None
    try:
        doc = _db.collection("analyses").document(doc_id).get()
        if doc.exists:
            data = doc.to_dict()
            data["doc_id"] = doc.id
            return data
        return None
    except Exception:
        return None


def show_profile_page(user_id, user_email, profile):
    """Affiche la page Mon Profil dans la zone principale."""
    if profile is None:
        profile = {}

    TITLES = ['', 'Dr', 'Pr', 'M.', 'Mme']
    ROLES  = ['', 'Oncologue médical', 'Chirurgien', 'Radio-oncologue', 'Pathologiste',
              'Radiologue', 'Interne', 'Chef de clinique', 'Médecin cadre', 'Autre']

    first_val   = profile.get('first_name', '')
    last_val    = profile.get('last_name', '')
    title_val   = profile.get('title', '')
    role_val    = profile.get('role', '')
    inst_val    = profile.get('institution', '')
    service_val = profile.get('service', '')

    full_name = f"{title_val} {first_val} {last_val}".strip() if first_val else user_email
    sub_line  = " · ".join(filter(None, [role_val, inst_val]))
    av_init   = (first_val[0] + last_val[0]).upper() if (first_val and last_val) else user_email[:2].upper()

    # ── Carte d'identité ──
    st.markdown(f"""
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;
                padding:24px 28px;margin-bottom:28px;display:flex;align-items:center;gap:20px;">
        <div style="width:64px;height:64px;flex-shrink:0;
                    background:linear-gradient(135deg,#0b1628 0%,#1e3a5f 50%,#0ea5e9 100%);
                    border-radius:16px;display:flex;align-items:center;justify-content:center;
                    font-size:22px;font-weight:800;color:white;letter-spacing:0.02em;">{av_init}</div>
        <div>
            <div style="font-size:20px;font-weight:800;color:#0b1628;
                        letter-spacing:-0.02em;margin-bottom:3px;">{full_name}</div>
            <div style="font-size:13px;color:#64748b;">{user_email}</div>
            {f'<div style="font-size:12px;color:#94a3b8;margin-top:3px;">{sub_line}</div>' if sub_line else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Formulaire ──
    st.markdown('<div class="section-title">Informations personnelles</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 2])
    new_title = c1.selectbox(
        "Titre", TITLES,
        index=TITLES.index(title_val) if title_val in TITLES else 0,
        key="prof_title")
    new_first = c2.text_input("Prénom", value=first_val, key="prof_first")
    new_last  = c3.text_input("Nom",    value=last_val,  key="prof_last")

    st.markdown('<div class="section-title" style="margin-top:20px;">Informations professionnelles</div>',
                unsafe_allow_html=True)
    new_role = st.selectbox(
        "Fonction", ROLES,
        index=ROLES.index(role_val) if role_val in ROLES else 0,
        key="prof_role")
    c4, c5 = st.columns(2)
    new_inst    = c4.text_input("Institution", value=inst_val,
                                placeholder="Ex : HUG - Hôpitaux Universitaires de Genève",
                                key="prof_inst")
    new_service = c5.text_input("Service", value=service_val,
                                placeholder="Ex : Oncologie médicale",
                                key="prof_service")

    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    if st.button("Sauvegarder le profil", type="primary", key="prof_save"):
        new_profile = {
            "email":       user_email,
            "title":       new_title,
            "first_name":  new_first,
            "last_name":   new_last,
            "role":        new_role,
            "institution": new_inst,
            "service":     new_service,
        }
        if save_user_profile(user_id, new_profile):
            st.session_state["user_profile"] = new_profile
            st.toast("Profil sauvegardé ✓", icon="✅")
            st.rerun()
        else:
            st.error("Erreur lors de la sauvegarde. Vérifiez votre connexion.")


def show_welcome_profile_page(user_id, user_email, profile):
    """Page de bienvenue pour la première connexion — force le remplissage du profil."""
    if profile is None:
        profile = {}

    TITLES = ['', 'Dr', 'Pr', 'M.', 'Mme']
    ROLES  = ['', 'Oncologue médical', 'Chirurgien', 'Radio-oncologue', 'Pathologiste',
              'Radiologue', 'Interne', 'Chef de clinique', 'Médecin cadre', 'Autre']

    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("""
        <div style="text-align:center;padding:32px 0 24px;">
            <div style="font-size:36px;margin-bottom:12px;">🎉</div>
            <div style="font-size:22px;font-weight:800;color:#0b1628;
                        letter-spacing:-0.02em;margin-bottom:10px;">
                Bienvenue sur OncoPrediction !
            </div>
            <div style="font-size:14px;color:#64748b;line-height:1.6;max-width:420px;margin:0 auto;">
                Avant de commencer, veuillez compléter votre profil.
                Ces informations permettent de personnaliser votre expérience
                et d'identifier vos analyses.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Informations personnelles</div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 2, 2])
        title = c1.selectbox(
            "Titre", TITLES,
            index=TITLES.index(profile.get('title', '')) if profile.get('title', '') in TITLES else 0,
            key="welcome_title")
        first_name = c2.text_input("Prénom *", value=profile.get('first_name', ''),
                                   key="welcome_first")
        last_name  = c3.text_input("Nom *",    value=profile.get('last_name', ''),
                                   key="welcome_last")

        st.markdown('<div class="section-title" style="margin-top:20px;">Informations professionnelles</div>',
                    unsafe_allow_html=True)
        _role_val = profile.get('role', '')
        role = st.selectbox(
            "Fonction *", ROLES,
            index=ROLES.index(_role_val) if _role_val in ROLES else 0,
            key="welcome_role")
        c4, c5 = st.columns(2)
        institution = c4.text_input("Institution", value=profile.get('institution', ''),
                                    placeholder="Ex : HUG", key="welcome_inst")
        service     = c5.text_input("Service", value=profile.get('service', ''),
                                    placeholder="Ex : Oncologie médicale", key="welcome_service")

        st.markdown("<div style='color:#94a3b8;font-size:12px;margin-top:6px;'>* Champs obligatoires</div>",
                    unsafe_allow_html=True)
        st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

        if st.button("Sauvegarder et commencer →", type="primary",
                     use_container_width=True, key="welcome_save"):
            if not first_name.strip():
                st.error("Veuillez saisir votre prénom.")
            elif not last_name.strip():
                st.error("Veuillez saisir votre nom.")
            elif not role.strip():
                st.error("Veuillez sélectionner votre fonction.")
            else:
                _profile_data = {
                    "email":       user_email,
                    "title":       title,
                    "first_name":  first_name.strip(),
                    "last_name":   last_name.strip(),
                    "role":        role,
                    "institution": institution.strip(),
                    "service":     service.strip(),
                }
                if save_user_profile(user_id, _profile_data):
                    st.session_state["user_profile"] = _profile_data
                    st.toast("Profil sauvegardé ✓", icon="✅")
                    st.rerun()
                else:
                    st.error("Erreur lors de la sauvegarde. Vérifiez votre connexion.")


# =============================================================================
# CHARGEMENT DES MODÈLES (mis en cache — exécuté une seule fois au démarrage)
# =============================================================================
@st.cache_resource(show_spinner="⏳ Chargement des modèles IA…")
def load_models():
    cox_data = joblib.load(COX_PATH)
    rsf_data = joblib.load(RSF_PATH)
    return (
        cox_data['model'],
        cox_data['stats'],
        rsf_data['model'],
        rsf_data['feature_cols'],
    )

# =============================================================================
# CONSTRUCTION DES VECTEURS PATIENT
# =============================================================================
def build_rsf_vector(inp, feature_cols):
    """Vecteur 44 colonnes (one-hot encodé) pour le RSF."""
    row = {col: 0.0 for col in feature_cols}
    row.update({
        'Age': inp['age'],  'Sex': inp['sex'],
        'Anatomic_Site': inp['site'],  'Grade': inp['grade'],
        'Aneuploidy_Score': inp['aneuploidy'],
        'Fraction_Genome_Altered': inp['fga'],
        'MSIsensor_Score': inp['msi'],
        'Mutation_Count': inp['mutation_count'],
        'TMB': inp['tmb'],  'Tumor_Break_Load': inp['tbl'],
        'Radiation_Therapy': inp['radiation'],
    })
    row[f'AJCC_Stage_Simplified_{inp["ajcc_stage"]}'] = 1.0
    row[f'T_Stage_Clean_{inp["t_stage"]}']            = 1.0
    row[f'N_Stage_Clean_{inp["n_stage"]}']            = 1.0
    row[f'M_Stage_Clean_{inp["m_stage"]}']            = 1.0
    row[f'Histology_Detailed_{inp["histology"]}']     = 1.0
    row[f'Subtype_{inp["subtype"]}']                  = 1.0
    return pd.DataFrame([row])[feature_cols]


def build_cox_vector(inp):
    """Vecteur 6 colonnes pour le Cox."""
    return pd.DataFrame([{
        'Age':                       inp['age'],
        'Grade':                     inp['grade'],
        'AJCC_Stage_Simplified_III': 1.0 if inp['ajcc_stage'] == 'III' else 0.0,
        'AJCC_Stage_Simplified_IV':  1.0 if inp['ajcc_stage'] == 'IV'  else 0.0,
        'Radiation_Therapy':         float(inp['radiation']),
        'TMB':                       inp['tmb'],
    }])

# =============================================================================
# PRÉDICTIONS
# =============================================================================
def get_predictions(inp, cox_model, rsf_model, feature_cols):
    """Calcule les courbes de survie et métriques clés pour un patient."""
    times = np.linspace(0.5, 100, 300)
    key_t = np.array([12.0, 24.0, 60.0])

    # ── Cox (sksurv) ──
    cox_vec  = build_cox_vector(inp)
    cox_fns  = cox_model.predict_survival_function(cox_vec)
    cox_curve = np.clip([cox_fns[0](t) for t in times], 0, 1)
    cox_key   = np.clip([cox_fns[0](t) for t in key_t], 0, 1)

    cox_med = None
    below   = np.where(np.array(cox_curve) <= 0.5)[0]
    if len(below):
        cox_med = times[below[0]]

    # ── RSF ──
    rsf_vec  = build_rsf_vector(inp, feature_cols)
    rsf_fns  = rsf_model.predict_survival_function(rsf_vec, return_array=False)
    rsf_curve = np.clip([rsf_fns[0](t) for t in times], 0, 1)
    rsf_key   = np.clip([rsf_fns[0](t) for t in key_t], 0, 1)

    rsf_med = None
    below_r = np.where(np.array(rsf_curve) <= 0.5)[0]
    if len(below_r):
        rsf_med = times[below_r[0]]

    return {
        'times':     times,
        'cox_curve': np.array(cox_curve), 'cox_key': np.array(cox_key), 'cox_med': cox_med,
        'rsf_curve': np.array(rsf_curve), 'rsf_key': np.array(rsf_key), 'rsf_med': rsf_med,
    }

# =============================================================================
# FIGURE
# =============================================================================
PALETTE = ['#0ea5e9','#e74c3c','#10b981','#8b5cf6','#f59e0b','#06b6d4']

def plot_survival(patients, show_cox=True, show_rsf=True):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    for i, pat in enumerate(patients):
        c = PALETTE[i % len(PALETTE)]
        p = pat['preds']
        lb = pat['label']
        # Zone ombrée entre Cox et RSF
        if show_cox and show_rsf:
            ax.fill_between(p['times'], p['cox_curve'] * 100, p['rsf_curve'] * 100,
                            alpha=0.10, color=c)
        if show_cox:
            ax.plot(p['times'], p['cox_curve'] * 100,
                    color=c, lw=2.5, ls='-', label=f"{lb} — Modèle Cox")
            if p['cox_med']:
                ax.axvline(p['cox_med'], color=c, ls=':', lw=1.2, alpha=0.4)
        if show_rsf:
            ax.plot(p['times'], p['rsf_curve'] * 100,
                    color=c, lw=1.8, ls='--', label=f"{lb} — Modèle RSF", alpha=0.7)

    ax.axhline(50, color='#cbd5e1', ls='--', lw=1.0, alpha=0.8)
    ax.text(1, 51.5, 'Médiane (50%)', fontsize=8, color='#cbd5e1')
    ax.set_xlabel('Temps (mois)', fontsize=11, color='#64748b', labelpad=8)
    ax.set_ylabel('Probabilité de survie (%)', fontsize=11, color='#64748b', labelpad=8)
    ax.set_xlim(0, 72); ax.set_ylim(0, 105)
    ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
    ax.tick_params(colors='#94a3b8', labelsize=9)
    ax.legend(fontsize=9, loc='upper right', framealpha=1.0,
              edgecolor='#e2e8f0', facecolor='white')
    ax.grid(alpha=0.25, ls='-', color='#e8ecf0')
    for spine in ax.spines.values():
        spine.set_color('#e8ecf0')
    plt.tight_layout(pad=1.0)
    return fig


def plot_survival_v2(patient, show_cox=True, show_rsf=True):
    """Courbe de survie — Vue 2 (patient unique, style dashboard)."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    p = patient['preds']

    # Marqueurs temporels : 12, 24, 60 mois
    for t_mark, t_lbl in [(12, '1 an'), (24, '2 ans'), (60, '5 ans')]:
        ax.axvline(t_mark, color='#e2e8f0', ls='--', lw=1.0, alpha=0.9, zorder=1)
        ax.text(t_mark + 0.8, 103, t_lbl, fontsize=8, color='#94a3b8', ha='left')

    # Zone ombrée entre les deux modèles
    if show_cox and show_rsf:
        ax.fill_between(p['times'], p['cox_curve'] * 100, p['rsf_curve'] * 100,
                        alpha=0.08, color='#1B2A4A', zorder=2)

    # Courbe Cox — navy foncé, trait plein
    if show_cox:
        ax.plot(p['times'], p['cox_curve'] * 100,
                color='#1B2A4A', lw=2.5, ls='-', label='Cox (risques proportionnels)', zorder=4)

    # Courbe RSF — bleu moyen, tirets
    if show_rsf:
        ax.plot(p['times'], p['rsf_curve'] * 100,
                color='#2980B9', lw=2.0, ls='--', label='Forêt aléatoire de survie',
                alpha=0.9, zorder=3)

    # Ligne médiane 50%
    ax.axhline(50, color='#cbd5e1', ls='--', lw=0.9, alpha=0.7, zorder=1)
    ax.text(1, 51.5, 'Médiane 50%', fontsize=8, color='#cbd5e1')

    ax.set_xlabel('Temps (mois)', fontsize=11, color='#64748b', labelpad=8)
    ax.set_ylabel('Probabilité de survie (%)', fontsize=11, color='#64748b', labelpad=8)
    ax.set_xlim(0, 72)
    ax.set_ylim(0, 108)
    ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
    ax.tick_params(colors='#94a3b8', labelsize=9)
    ax.legend(fontsize=9, loc='upper right', framealpha=1.0,
              edgecolor='#e2e8f0', facecolor='white')
    ax.grid(alpha=0.20, ls='-', color='#e8ecf0', zorder=0)
    for spine in ax.spines.values():
        spine.set_color('#e8ecf0')
    plt.tight_layout(pad=1.0)
    return fig


# =============================================================================
# CATÉGORIE DE RISQUE
# =============================================================================
def risk_cat(med):
    if med is None or med >= 36: return "Faible",        "risk-low"
    if med >= 15:                 return "Intermédiaire", "risk-medium"
    return                              "Élevé",          "risk-high"

# =============================================================================
# TEXTE AI CONSENSUS (généré dynamiquement)
# =============================================================================
def generate_insight(inp, rl):
    stage = inp['ajcc_stage']
    rad   = inp['radiation'] == 1
    grade = inp['grade']
    age   = inp['age']

    risk_f, protect_f = [], []
    if stage == 'IV':     risk_f.append("stade AJCC IV (maladie métastatique)")
    elif stage == 'III':  risk_f.append("stade AJCC III")
    if not rad:           risk_f.append("absence de radiothérapie")
    else:                 protect_f.append("radiothérapie administrée")
    if grade == 3:        risk_f.append("grade histologique élevé (G3)")
    if age >= 70:         risk_f.append(f"âge avancé ({age} ans)")

    if rl == 'Élevé':
        factors = risk_f[:2] if risk_f else ["plusieurs facteurs défavorables"]
        return (f"L'analyse combinée Cox + RSF identifie un profil de risque élevé. "
                f"Les principaux facteurs pronostiques sont : {' et '.join(factors)}.")
    elif rl == 'Intermédiaire':
        if risk_f and protect_f:
            return (f"Les deux modèles s'accordent sur un profil de risque intermédiaire, "
                    f"avec un équilibre entre le {risk_f[0]} et la {protect_f[0]}.")
        elif risk_f:
            return (f"Les deux modèles identifient un profil de risque intermédiaire, "
                    f"principalement lié au {risk_f[0]}.")
        return "Les deux modèles s'accordent sur un profil de risque intermédiaire."
    else:
        if protect_f:
            return (f"L'analyse combinée identifie un profil de risque favorable. "
                    f"La {protect_f[0]} contribue positivement au pronostic.")
        return "Les deux modèles s'accordent sur un profil de risque favorable, avec une survie médiane prolongée."

# =============================================================================
# TEXTE AI INSIGHT — COMPARAISON MULTI-PATIENTS
# =============================================================================
def generate_group_insight(patients):
    """Génère un texte d'analyse pour un groupe de patients comparés."""
    if not patients:
        return ""

    risk_levels = []
    medians = []
    for p in patients:
        rl, _ = risk_cat(p['preds']['cox_med'])
        risk_levels.append(rl)
        if p['preds']['cox_med']:
            medians.append(p['preds']['cox_med'])

    n = len(patients)
    n_high    = risk_levels.count("Élevé")
    n_medium  = risk_levels.count("Intermédiaire")
    n_low     = risk_levels.count("Faible")

    med_min = min(medians) if medians else None
    med_max = max(medians) if medians else None

    # Trouver le patient avec la meilleure et la moins bonne survie
    best  = max(patients, key=lambda p: p['preds']['cox_med'] or 200)
    worst = min(patients, key=lambda p: p['preds']['cox_med'] or 200)

    txt = f"Cette comparaison porte sur {n} profil{'s' if n > 1 else ''} patient. "

    if n_high > 0 and n_low > 0:
        txt += (f"Le groupe présente une hétérogénéité pronostique marquée : "
                f"{n_high} profil{'s' if n_high > 1 else ''} de risque élevé "
                f"et {n_low} de risque favorable. ")
    elif n_high == n:
        txt += "L'ensemble des profils présente un risque élevé selon les deux modèles. "
    elif n_low == n:
        txt += "L'ensemble des profils présente un pronostic favorable selon les deux modèles. "
    else:
        txt += f"Le groupe présente une répartition mixte ({n_low} faible, {n_medium} intermédiaire, {n_high} élevé). "

    if med_min and med_max and med_max - med_min > 10:
        txt += (f"L'écart de survie médiane entre {worst['label']} "
                f"(~{med_min:.0f} mois) et {best['label']} "
                f"(~{med_max:.0f} mois) suggère des différences biologiques ou thérapeutiques significatives.")
    elif med_max:
        txt += f"Les survies médianes estimées sont relativement homogènes (~{med_max:.0f} mois)."

    return txt


# =============================================================================
# FOREST PLOT — HAZARD RATIOS (modèle Cox)
# =============================================================================
def plot_hr_forest(cox_model, inp):
    hrs = np.exp(cox_model.coef_)   # [Age, Grade, StageIII, StageIV, Radiation, TMB]

    labels = [
        f"Age ({inp['age']} ans)",
        f"Grade (G{inp['grade']})",
        "Stade AJCC III",
        "Stade AJCC IV",
        "Radiothérapie",
        f"TMB ({inp['tmb']:.1f} mut/Mb)",
    ]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    n = len(labels)
    y_pos = list(range(n - 1, -1, -1))
    x_min_plot, x_max_plot = 0.15, 5.2

    for y, label, hr in zip(y_pos, labels, hrs):
        color = '#0b1628' if hr >= 1.0 else '#22c55e'
        ax.plot([x_min_plot, x_max_plot], [y, y], color='#e8ecf0', lw=1.2, zorder=1)
        ax.scatter([hr], [y], color=color, s=80, zorder=3, linewidths=0)
        ax.text(-0.05, y, label, ha='right', va='center', fontsize=10, color='#374151')
        ax.text(x_max_plot + 0.08, y, f"{hr:.2f}", ha='left', va='center',
                fontsize=10, fontweight='700', color=color)

    ax.axvline(1.0, color='#94a3b8', ls='--', lw=1.2, alpha=0.8, zorder=2)
    ax.text(1.0, -0.8, "Base neutre (HR = 1,0)", ha='center', fontsize=8, color='#94a3b8')

    ax.scatter([], [], color='#0b1628', s=50, label="Facteur de risque (HR > 1,0)")
    ax.scatter([], [], color='#22c55e', s=50, label="Facteur protecteur (HR < 1,0)")
    ax.legend(loc='upper right', fontsize=9, framealpha=1.0, edgecolor='#e2e8f0',
              facecolor='white', handletextpad=0.4)

    ax.set_xlim(x_min_plot - 1.8, x_max_plot + 0.6)
    ax.set_ylim(-1.2, n)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout(pad=0.4)
    return fig


def plot_rsf_importance(rsf_model, feature_cols, top_n=10):
    """Barres horizontales — Top N importances MDI du RSF."""
    try:
        importances = rsf_model.feature_importances_
    except (AttributeError, NotImplementedError):
        # Fallback : importance uniforme si non disponible
        importances = np.ones(len(feature_cols)) / len(feature_cols)

    idx = np.argsort(importances)[::-1][:top_n]
    idx_rev = idx[::-1]  # du moins important au plus important (bas → haut)

    feat_names = []
    for i in idx_rev:
        name = feature_cols[i]
        name = name.replace('AJCC_Stage_Simplified_', 'Stade ')
        name = name.replace('T_Stage_Clean_', 'Stade T ')
        name = name.replace('N_Stage_Clean_', 'Stade N ')
        name = name.replace('M_Stage_Clean_', 'Stade M ')
        name = name.replace('Histology_Detailed_', 'Hist. ')
        name = name.replace('Subtype_', 'Sous-type ')
        name = name.replace('_', ' ')
        if len(name) > 30:
            name = name[:28] + '…'
        feat_names.append(name)

    values = [importances[i] for i in idx_rev]
    max_v  = max(values) if values else 1.0
    med_v  = float(np.median(values))

    bar_colors = ['#0b1628' if v == max_v else '#1e3a5f' if v >= med_v else '#64748b'
                  for v in values]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    bars = ax.barh(range(len(feat_names)), values, color=bar_colors,
                   height=0.65, alpha=0.85, edgecolor='none')

    for bar, val in zip(bars, values):
        ax.text(val + max_v * 0.012, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=8.5, color='#374151')

    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(feat_names, fontsize=9, color='#374151')
    ax.set_xlabel('Importance des variables (MDI)', fontsize=10, color='#64748b', labelpad=6)
    ax.set_xlim(0, max_v * 1.28)

    for spine in ax.spines.values():
        spine.set_color('#e8ecf0')
    ax.tick_params(colors='#94a3b8', labelsize=9)
    ax.xaxis.grid(alpha=0.25, color='#e8ecf0')
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.8)
    return fig


# =============================================================================
# FORMULAIRE PATIENT (SIDEBAR)
# =============================================================================
def patient_form(stats, pfx="p1"):

    st.sidebar.markdown("""
    <div class="form-title">Profil Patient</div>
    <div class="form-subtitle">Paramètres cliniques pour l'analyse de survie</div>
    """, unsafe_allow_html=True)

    label = st.sidebar.text_input("ID Patient", f"Patient {pfx.upper()}",
                                  key=f"{pfx}_lbl")

    # ── Section 1 : Clinical Data ──
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📋 Données cliniques**")
    with st.sidebar:
        age = st.slider("Âge au diagnostic (ans)",
                        stats['age_min'], stats['age_max'], stats['age_med'],
                        key=f"{pfx}_age")
        sex  = st.radio("Sexe biologique", ["Homme", "Femme"],
                        horizontal=True, key=f"{pfx}_sex")
        site = st.selectbox("Site tumoral", ["Estomac", "Œsophage"],
                            key=f"{pfx}_site")

    # ── Section 2 : Staging & Histology ──
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔬 Stadification et histologie**")
    with st.sidebar:
        col_t, col_n, col_m = st.columns(3)
        t_stage = col_t.selectbox("Stade T", ["T1","T2","T3","T4"], index=1,
                                  key=f"{pfx}_t")
        n_stage = col_n.selectbox("Stade N", ["N0","N1","N2","N3"],
                                  key=f"{pfx}_n")
        m_stage = col_m.selectbox("Stade M", ["M0","M1"], key=f"{pfx}_m")

        ajcc = st.selectbox("Stade AJCC", ["I","II","III","IV"], index=1,
                            key=f"{pfx}_ajcc")

        grade_sel = st.selectbox("Grade de la tumeur", ["G1","G2","G3"], index=1,
                                 key=f"{pfx}_grade")
        grade = int(grade_sel[1])  # "G1"→1, "G2"→2, "G3"→3

        HIST = {
            "Adénocarcinome intestinal (estomac)":      "Intestinal Type Stomach Adenocarcinoma",
            "Adénocarcinome diffus (estomac)":          "Diffuse Type Stomach Adenocarcinoma",
            "Adénocarcinome mucineux (estomac)":        "Mucinous Stomach Adenocarcinoma",
            "Adénocarcinome papillaire (estomac)":      "Papillary Stomach Adenocarcinoma",
            "Adénocarcinome tubulaire (estomac)":       "Tubular Stomach Adenocarcinoma",
            "Carcinome à cellules en bague (estomac)":  "Signet Ring Cell Carcinoma of the Stomach",
            "Adénocarcinome gastrique NOS":             "Stomach Adenocarcinoma",
            "Adénocarcinome oesophagien":               "Esophageal Adenocarcinoma",
            "Carcinome epidermoide oesophagien":        "Esophageal Squamous Cell Carcinoma",
        }
        hist_lbl  = st.selectbox("Histologie détaillée", list(HIST.keys()),
                                 key=f"{pfx}_hist")
        histology = HIST[hist_lbl]

        SUB = {
            "STAD CIN — Instabilité chromosomique (estomac)":   "STAD_CIN",
            "STAD EBV — Virus Epstein-Barr (estomac)":          "STAD_EBV",
            "STAD GS — Génomiquement stable (estomac)":         "STAD_GS",
            "STAD MSI — Instabilité microsatellites (estomac)": "STAD_MSI",
            "STAD POLE — Mutation POLE (estomac)":               "STAD_POLE",
            "ESCA CIN — Instabilité chromosomique (oesophage)": "ESCA_CIN",
            "ESCA ESCC — Carcinome epidermoide (oesophage)":    "ESCA_ESCC",
            "ESCA GS — Génomiquement stable (oesophage)":       "ESCA_GS",
            "ESCA MSI — Instabilité microsatellites (oesophage)":"ESCA_MSI",
            "ESCA POLE — Mutation POLE (oesophage)":             "ESCA_POLE",
        }
        sub_lbl = st.selectbox("Sous-type moléculaire (TCGA)", list(SUB.keys()),
                               key=f"{pfx}_sub")
        subtype = SUB[sub_lbl]

    # ── Section 3 : Genomic Markers ──
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🧬 Marqueurs génomiques**")
    with st.sidebar:
        tmb = st.slider(
            "TMB — Charge mutationnelle tumorale (mut/Mb)",
            0.0, float(stats['tmb_p95']), float(stats['tmb_med']), step=0.5,
            help=f"Médiane de la cohorte : {stats['tmb_med']} mut/Mb",
            key=f"{pfx}_tmb")
        msi = st.slider(
            "Score MSIsensor", 0.0, float(stats['msi_max']),
            float(stats['msi_med']), step=0.1, key=f"{pfx}_msi")
        mutation_count = st.slider(
            "Nombre total de mutations", 1, int(stats['mut_p95']),
            int(stats['mut_med']), step=10, key=f"{pfx}_mut")
        fga = st.slider(
            "Fraction du génome altérée (0–1)",
            0.0, 1.0, float(stats['fga_med']), step=0.01, key=f"{pfx}_fga")
        aneuploidy = st.slider(
            "Score d'aneuploïdie", 0, int(stats['aneu_max']),
            int(stats['aneu_med']), key=f"{pfx}_aneu")
        tbl = st.slider(
            "Charge de cassures tumorales", 0.0, float(stats['tbl_p95']),
            float(stats['tbl_med']), step=10.0, key=f"{pfx}_tbl")

    # ── Radiothérapie (standalone) ──
    st.sidebar.markdown("<div style='margin: 8px 0 2px;'></div>", unsafe_allow_html=True)
    radiation = st.sidebar.checkbox("Radiothérapie administrée", key=f"{pfx}_rad")

    return {
        'label':          label,
        'age':            age,
        'sex':            1 if sex == "Homme" else 0,
        'site':           1 if site == "Estomac" else 0,
        'grade':          grade,
        'ajcc_stage':     ajcc,
        't_stage':        t_stage,
        'n_stage':        n_stage,
        'm_stage':        m_stage,
        'histology':      histology,
        'subtype':        subtype,
        'radiation':      1 if radiation else 0,
        'tmb':            tmb,
        'msi':            msi,
        'mutation_count': mutation_count,
        'fga':            fga,
        'aneuploidy':     aneuploidy,
        'tbl':            tbl,
    }


# =============================================================================
# PAGE D'AUTHENTIFICATION
# =============================================================================
def show_auth_page():
    """Affiche la page de connexion / inscription / mot de passe oublié."""

    # En-tête
    st.markdown("""
    <div class="auth-header">
        <h1>Onco<em>Prediction</em></h1>
        <p>Prédiction de survie · Cancers gastro-œsophagiens · Tumor Boards</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialiser l'état de la page auth
    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "login"

    # Centrer le formulaire
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:

        # ─── MODE CONNEXION ───
        if st.session_state.auth_mode == "login":
            st.markdown('<div class="auth-title">Connexion</div>',
                        unsafe_allow_html=True)

            with st.form("login_form"):
                email = st.text_input("Adresse email", placeholder="exemple@email.com")
                password = st.text_input("Mot de passe", type="password",
                                         placeholder="Votre mot de passe")
                submit = st.form_submit_button("Se connecter", type="primary",
                                                use_container_width=True)

            if submit:
                if not email or not password:
                    st.error("Veuillez remplir tous les champs.")
                else:
                    with st.spinner("Connexion en cours…"):
                        user, error = firebase_sign_in(email, password)
                    if error:
                        st.error(error)
                        st.session_state.pop("_unverified_token", None)
                    else:
                        account_info = firebase_get_account_info(user["idToken"])
                        email_verified = (
                            account_info is not None
                            and account_info.get("users", [{}])[0].get("emailVerified", False)
                        )
                        if not email_verified:
                            st.session_state["_unverified_token"] = user["idToken"]
                            st.rerun()
                        else:
                            st.session_state.pop("_unverified_token", None)
                            st.session_state["authenticated"] = True
                            st.session_state["user_email"] = user["email"]
                            st.session_state["user_token"] = user["idToken"]
                            st.session_state["user_id"]    = user.get("localId", user["email"])
                            st.rerun()

            # Avertissement email non vérifié (persiste entre reruns via session_state)
            if st.session_state.get("_unverified_token"):
                st.warning(
                    "⚠️ Veuillez d'abord vérifier votre email. "
                    "Vérifiez votre boîte de réception (et les spams)."
                )
                if st.button("Renvoyer l'email de vérification", use_container_width=True):
                    firebase_send_verification_email(st.session_state["_unverified_token"])
                    st.success("Email de vérification renvoyé !")

            st.markdown("")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Créer un compte", use_container_width=True):
                    st.session_state.auth_mode = "signup"
                    st.rerun()
            with col_b:
                if st.button("Mot de passe oublié", use_container_width=True):
                    st.session_state.auth_mode = "reset"
                    st.rerun()

        # ─── MODE INSCRIPTION ───
        elif st.session_state.auth_mode == "signup":
            st.markdown('<div class="auth-title">Créer un compte</div>',
                        unsafe_allow_html=True)

            with st.form("signup_form"):
                email = st.text_input("Adresse email", placeholder="exemple@email.com")
                password = st.text_input("Mot de passe (min. 6 caractères)",
                                         type="password", placeholder="Choisissez un mot de passe")
                password2 = st.text_input("Confirmer le mot de passe",
                                          type="password", placeholder="Répétez le mot de passe")
                submit = st.form_submit_button("Créer mon compte", type="primary",
                                                use_container_width=True)

            if submit:
                if not email or not password or not password2:
                    st.error("Veuillez remplir tous les champs.")
                elif password != password2:
                    st.error("Les mots de passe ne correspondent pas.")
                elif len(password) < 6:
                    st.error("Le mot de passe doit contenir au moins 6 caractères.")
                else:
                    with st.spinner("Création du compte…"):
                        user, error = firebase_sign_up(email, password)
                    if error:
                        st.error(error)
                    else:
                        # Connecter temporairement pour envoyer l'email de vérification
                        temp_user, _ = firebase_sign_in(email, password)
                        if temp_user:
                            firebase_send_verification_email(temp_user["idToken"])
                        st.success(
                            "✅ Compte créé ! Un email de vérification a été envoyé à votre "
                            "adresse. Cliquez sur le lien dans l'email puis connectez-vous."
                        )

            st.markdown("")
            if st.button("← Retour à la connexion", use_container_width=True):
                st.session_state.auth_mode = "login"
                st.rerun()

        # ─── MODE MOT DE PASSE OUBLIÉ ───
        elif st.session_state.auth_mode == "reset":
            st.markdown('<div class="auth-title">Réinitialiser le mot de passe</div>',
                        unsafe_allow_html=True)

            with st.form("reset_form"):
                email = st.text_input("Adresse email de votre compte",
                                      placeholder="exemple@email.com")
                submit = st.form_submit_button("Envoyer le lien de réinitialisation",
                                                type="primary", use_container_width=True)

            if submit:
                if not email:
                    st.error("Veuillez entrer votre adresse email.")
                else:
                    with st.spinner("Envoi de l'email…"):
                        success, error = firebase_reset_password(email)
                    if error:
                        st.error(error)
                    else:
                        st.success("Un email de réinitialisation a été envoyé. Vérifiez votre boîte de réception.")

            st.markdown("")
            if st.button("← Retour à la connexion", use_container_width=True):
                st.session_state.auth_mode = "login"
                st.rerun()

        # Pied de page
        st.markdown("""
        <div class="auth-footer">
            OncoPrediction · Université de Genève<br>
            Données TCGA/cBioPortal · Usage académique uniquement<br>
            <a href="https://oncoprediction.ch" style="color:#0ea5e9;text-decoration:none;">← oncoprediction.ch</a>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# MODE PRÉSENTATION — Tumor Board (plein écran, épuré)
# =============================================================================
def show_presentation_mode():
    """Affiche le mode présentation épuré pour les réunions de tumor board."""

    # Cacher sidebar, header Streamlit et forcer plein écran
    st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        header[data-testid="stHeader"] { display: none !important; }
        .block-container { padding-top: 1rem !important; max-width: 95% !important; }
    </style>
    """, unsafe_allow_html=True)

    _p = st.session_state.get("last_prediction", {})

    # ── Helpers de formatage ──
    def _fmt_s(v):
        if v is None: return "—"
        try:
            v = float(v)
            return f"{v * 100:.0f}%" if v < 1 else f"{v:.0f}%"
        except Exception:
            return "—"

    def _fmt_m(v):
        if v is None: return "—"
        try:
            return f"~{round(float(v))} mois"
        except Exception:
            return "—"

    # ── Barre du haut : Quitter + branding ──
    _top1, _top2, _top3 = st.columns([1, 6, 1])
    with _top1:
        if st.button("← Quitter", type="secondary", key="pres_quit"):
            st.session_state["presentation_mode"] = False
            st.rerun()
    with _top3:
        st.markdown(
            "<div style='text-align:right;padding-top:6px;font-weight:700;"
            "color:#0b1628;font-size:15px;'>Onco<em>Prediction</em></div>",
            unsafe_allow_html=True)

    # ── Identité du patient ──
    _patient_id = _p.get("patient_id", "Patient")
    _age        = _p.get("age", "—")
    _sex_lbl    = _p.get("sex_label", "—")
    _site       = _p.get("tumor_site", "—")
    _stage      = _p.get("ajcc_stage", "—")

    st.markdown(f"""
    <div style="text-align:center;margin:24px 0 8px;">
        <h1 style="font-size:52px;font-weight:900;color:#0b1628;
                   letter-spacing:-0.03em;margin-bottom:6px;">{_patient_id}</h1>
        <p style="font-size:22px;color:#64748b;font-weight:400;">
            {_age} ans &nbsp;·&nbsp; {_sex_lbl} &nbsp;·&nbsp; {_site}
            &nbsp;·&nbsp; Stade {_stage}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Badge de risque ──
    _rl = _p.get("risk_level", "")
    if _rl == "Faible":
        _rc, _rbg, _rl_lbl = "#22c55e", "#dcfce7", "RISQUE FAIBLE"
    elif _rl == "Élevé":
        _rc, _rbg, _rl_lbl = "#ef4444", "#fee2e2", "RISQUE ÉLEVÉ"
    else:
        _rc, _rbg, _rl_lbl = "#f59e0b", "#fef3c7", "RISQUE MODÉRÉ"

    st.markdown(f"""
    <div style="text-align:center;margin:20px 0 28px;">
        <span style="display:inline-block;padding:14px 44px;border-radius:40px;
             background:{_rbg};border:3px solid {_rc};
             color:{_rc};font-weight:800;font-size:26px;letter-spacing:2px;">
            {_rl_lbl}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── 4 grandes cases de survie ──
    def _big_card(col, titre, valeur, couleur):
        with col:
            st.markdown(f"""
            <div style="border:3px solid {couleur};border-radius:16px;padding:28px 16px;
                 text-align:center;background:#ffffff;min-height:150px;
                 display:flex;flex-direction:column;justify-content:center;">
                <div style="color:#94a3b8;font-size:15px;text-transform:uppercase;
                     font-weight:600;letter-spacing:1px;margin-bottom:10px;">{titre}</div>
                <div style="color:{couleur};font-size:52px;font-weight:900;
                     line-height:1;">{valeur}</div>
            </div>
            """, unsafe_allow_html=True)

    _sc1, _sc2, _sc3, _sc4 = st.columns(4)
    _big_card(_sc1, "Survie à 1 an",   _fmt_s(_p.get("survival_1yr_cox")), "#22c55e")
    _big_card(_sc2, "Survie à 2 ans",  _fmt_s(_p.get("survival_2yr_cox")), "#f59e0b")
    _big_card(_sc3, "Survie à 5 ans",  _fmt_s(_p.get("survival_5yr_cox")), "#ef4444")
    _big_card(_sc4, "Médiane (Cox)",   _fmt_m(_p.get("median_survival_cox")), "#0b1628")

    # ── Courbe de survie ──
    _current_entry = st.session_state.get("current")
    if _current_entry is not None:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        try:
            _fig_pres = plot_survival_v2(_current_entry, show_cox=True, show_rsf=True)
            st.pyplot(_fig_pres, use_container_width=True)
            plt.close(_fig_pres)
        except Exception:
            pass

    # ── Paramètres cliniques résumés ──
    _t    = _p.get("t_stage",   "—")
    _n    = _p.get("n_stage",   "—")
    _m    = _p.get("m_stage",   "—")
    _g    = _p.get("grade",     "—")
    _tmb  = _p.get("tmb",       "—")
    _msi  = _p.get("msi_score", "—")
    _rad  = "Oui" if _p.get("radiation") else "Non"

    st.markdown(f"""
    <div style="display:flex;justify-content:center;gap:36px;flex-wrap:wrap;
         margin:24px 0 8px;font-size:17px;color:#374151;">
        <span><strong>T</strong> : {_t}</span>
        <span><strong>N</strong> : {_n}</span>
        <span><strong>M</strong> : {_m}</span>
        <span><strong>Grade</strong> : {_g}</span>
        <span><strong>TMB</strong> : {_tmb}</span>
        <span><strong>MSI</strong> : {_msi}</span>
        <span><strong>Radiothérapie</strong> : {_rad}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Pied de page ──
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#94a3b8;font-size:12px;margin-top:8px;">
        OncoPrediction · Outil d'aide à la décision — Usage académique uniquement
        · Ne remplace pas le jugement clinique
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TUMOR BOARD — Pages
# =============================================================================

def _tb_fmt_surv(v):
    """Formate une valeur de survie (proportion ou %) en '68.4%'."""
    if v is None:
        return "—"
    v = float(v)
    return f"{v * 100:.1f}%" if v < 1 else f"{v:.1f}%"


def _tb_fmt_med(v):
    """Formate une médiane de survie en '~58 mois'."""
    if v is None:
        return "—"
    return f"~{round(float(v))} mois"


def show_tb_session_list(user_id):
    """Affiche la liste des sessions Tumor Board."""
    if "tb_sessions" not in st.session_state:
        st.session_state["tb_sessions"] = get_tb_sessions(user_id)

    sessions = st.session_state["tb_sessions"]

    st.markdown('<div class="section-title">Mes sessions Tumor Board</div>',
                unsafe_allow_html=True)

    # ── Créer une nouvelle session ──
    _nc1, _nc2, _nc3 = st.columns([3, 1, 1])
    _tb_title = _nc1.text_input(
        "Titre de la session",
        placeholder="ex : RCP Gastrique — 04/03/2026",
        key="tb_new_title",
    )
    _tb_date = _nc2.date_input("Date", value=datetime.date.today(), key="tb_new_date")
    _nc3.markdown("<br>", unsafe_allow_html=True)
    if _nc3.button("Créer", type="primary", use_container_width=True, key="tb_create_btn"):
        if _tb_title.strip():
            _sid = create_tb_session(
                user_id,
                st.session_state.get("user_email", ""),
                _tb_title.strip(),
                str(_tb_date),
            )
            if _sid:
                st.success("Session créée.")
                st.session_state.pop("tb_sessions", None)
                st.rerun()
        else:
            st.error("Le titre de la session est obligatoire.")

    if not sessions:
        st.info("Aucune session Tumor Board. Créez-en une ci-dessus.")
        return

    # ── Confirmation de suppression ──
    if st.session_state.get("tb_confirm_delete_id"):
        _del_id    = st.session_state["tb_confirm_delete_id"]
        _del_title = st.session_state.get("tb_confirm_delete_title", "cette session")
        st.warning(f"⚠️ Supprimer définitivement **{_del_title}** ?")
        _dc1, _dc2, _dc3 = st.columns([1, 1, 4])
        if _dc1.button("Confirmer", type="primary", key="tb_del_confirm_yes"):
            delete_tb_session(_del_id)
            st.session_state.pop("tb_sessions", None)
            st.session_state.pop("tb_confirm_delete_id", None)
            st.session_state.pop("tb_confirm_delete_title", None)
            st.toast("Session supprimée.", icon="🗑️")
            st.rerun()
        if _dc2.button("Annuler", key="tb_del_confirm_no"):
            st.session_state.pop("tb_confirm_delete_id", None)
            st.session_state.pop("tb_confirm_delete_title", None)
            st.rerun()

    # ── Liste des sessions ──
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    for _sess in sessions:
        _sid   = _sess.get("doc_id", "")
        _title = _sess.get("title", "—")
        _date  = _sess.get("date", "—")
        _n_pts = len(_sess.get("patient_ids", []))

        _lc1, _lc2, _lc3 = st.columns([5, 1.5, 0.5])
        with _lc1:
            st.markdown(f"""
            <div style="padding:6px 0;">
                <div style="font-size:15px;font-weight:600;color:#0b1628;">{_title}</div>
                <div style="font-size:12px;color:#94a3b8;">
                    {_date} &nbsp;·&nbsp; {_n_pts} patient{'s' if _n_pts != 1 else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with _lc2:
            if st.button("Ouvrir →", key=f"tb_open_{_sid}", use_container_width=True):
                st.session_state["tb_open_session"] = _sid
                st.session_state.pop("tb_sessions", None)
                st.rerun()
        with _lc3:
            if st.button("🗑️", key=f"tb_del_{_sid}", use_container_width=True):
                st.session_state["tb_confirm_delete_id"]    = _sid
                st.session_state["tb_confirm_delete_title"] = _title
                st.rerun()

        st.markdown(
            "<hr style='margin:4px 0;border:none;border-top:1px solid #e2e8f0;'>",
            unsafe_allow_html=True,
        )


def show_tb_session_detail(user_id):
    """Affiche le détail d'une session Tumor Board."""
    session_id = st.session_state["tb_open_session"]
    session = get_tb_session(session_id)

    if session is None:
        st.error("Session introuvable.")
        st.session_state.pop("tb_open_session", None)
        return

    # Header avec bouton retour et bouton présenter
    col_back, col_title, col_present = st.columns([1, 4, 2])
    with col_back:
        if st.button("← Retour"):
            st.session_state.pop("tb_open_session", None)
            st.session_state.pop("tb_sessions_cache", None)
            st.rerun()
    with col_title:
        st.markdown(f"## {session.get('title', 'Session')}")
        st.caption(f"Réunion prévue le {session.get('session_date', session.get('date', '—'))}")

    patient_doc_ids = session.get("patient_analyses", session.get("patient_ids", []))

    with col_present:
        if patient_doc_ids:
            if st.button("▶ Présenter", type="primary", use_container_width=True):
                st.session_state["tb_presenting"]            = True
                st.session_state["tb_presenting_session_id"] = session_id
                st.session_state["tb_presenting_index"]      = 0
                st.rerun()

    st.markdown("---")

    # === Ajouter un patient depuis l'historique ===
    st.markdown("### Ajouter un patient")

    if "user_analyses" not in st.session_state:
        st.session_state["user_analyses"] = get_user_analyses(user_id)

    all_analyses = st.session_state.get("user_analyses", [])

    if all_analyses:
        options = {}
        for a in all_analyses:
            pid    = a.get("patient_id", "?")
            stage  = a.get("parameters", {}).get("ajcc_stage", "?")
            ts     = a.get("timestamp")
            date_str = ts.strftime("%d/%m/%Y") if ts else "?"
            label  = f"{pid} — Stade {stage} — {date_str}"
            doc_id = a.get("doc_id", "")
            if doc_id not in patient_doc_ids:
                options[label] = doc_id

        if options:
            add_col1, add_col2 = st.columns([4, 1])
            with add_col1:
                selected_label = st.selectbox(
                    "Sélectionner depuis l'historique",
                    options=[""] + list(options.keys()),
                    label_visibility="collapsed",
                )
            with add_col2:
                if (st.button("Ajouter", type="primary")
                        and selected_label and selected_label in options):
                    analysis_doc_id = options[selected_label]
                    if add_patient_to_tb_session(session_id, analysis_doc_id):
                        st.toast("Patient ajouté ✓")
                        st.session_state.pop("tb_sessions_cache", None)
                        st.rerun()
        else:
            st.info("Tous vos patients sont déjà dans cette session.")
    else:
        st.info("Aucune analyse disponible. Allez dans l'onglet Prédiction pour analyser un patient, puis revenez l'ajouter ici.")

    st.markdown("---")

    # === Liste des patients dans la session ===
    st.markdown(f"### Patients dans cette session ({len(patient_doc_ids)})")

    if not patient_doc_ids:
        st.info("Aucun patient dans cette session. Ajoutez des patients depuis votre historique.")
        return

    for i, doc_id in enumerate(patient_doc_ids):
        analysis = get_analysis_by_id(doc_id)
        if analysis is None:
            st.warning(f"Analyse introuvable (ID: {doc_id})")
            continue

        params     = analysis.get("parameters", {})
        results    = analysis.get("results", {})
        patient_id = analysis.get("patient_id", "?")

        risk = results.get("risk_level", "—")
        if "LOW" in str(risk).upper() or risk == "Faible":
            risk_label = "🟢 Risque faible"
        elif "HIGH" in str(risk).upper() or risk == "Élevé":
            risk_label = "🔴 Risque élevé"
        else:
            risk_label = "🟡 Risque modéré"

        median = results.get("median_survival_cox")
        median_str = (f"~{round(median)} mois"
                      if median and isinstance(median, (int, float)) else "—")

        notes    = analysis.get("notes", []) or []
        nb_notes = len(notes)

        col1, col2, col3, col4, col5, col6 = st.columns([0.5, 3, 2, 2, 0.5, 0.5])
        with col1:
            st.markdown(f"**{i + 1}.**")
        with col2:
            st.markdown(f"**{patient_id}**")
            stage = params.get("ajcc_stage", "?")
            site  = params.get("tumor_site", "?")
            st.caption(f"Stade {stage} · {site}")
            if nb_notes > 0:
                st.caption(f"📝 {nb_notes} note{'s' if nb_notes > 1 else ''}")
        with col3:
            st.markdown(risk_label)
        with col4:
            st.markdown(f"Médiane : {median_str}")
        with col5:
            if st.button("📝", key=f"tb_notes_{session_id}_{doc_id}",
                         help="Voir / ajouter des notes"):
                current_open = st.session_state.get("tb_show_notes_for")
                st.session_state["tb_show_notes_for"] = (
                    None if current_open == doc_id else doc_id
                )
                st.rerun()
        with col6:
            if st.button("✕", key=f"remove_{session_id}_{doc_id}"):
                remove_patient_from_tb_session(session_id, doc_id)
                st.toast("Patient retiré ✓")
                st.rerun()

        # Panneau de notes dépliable (sans expander)
        if st.session_state.get("tb_show_notes_for") == doc_id:
            _tb_profile    = st.session_state.get("user_profile", {}) or {}
            _tb_title      = _tb_profile.get("title", "")
            _tb_first      = _tb_profile.get("first_name", "")
            _tb_last       = _tb_profile.get("last_name", "")
            _tb_author     = (f"{_tb_title} {_tb_first} {_tb_last}".strip()
                              if (_tb_first or _tb_last)
                              else st.session_state.get("user_email", "Utilisateur"))
            show_notes_section(doc_id, _tb_author,
                               context_key=f"tb_session_{session_id}")

        st.markdown("---")


def show_tb_presentation(user_id):
    """Mode présentation plein écran pour une session Tumor Board."""
    session_id    = st.session_state.get("tb_presenting_session_id")
    current_index = st.session_state.get("tb_presenting_index", 0)

    session = get_tb_session(session_id)
    if session is None:
        st.session_state["tb_presenting"] = False
        st.rerun()
        return

    patient_doc_ids = session.get("patient_analyses", session.get("patient_ids", []))
    if not patient_doc_ids:
        st.session_state["tb_presenting"] = False
        st.rerun()
        return

    if current_index >= len(patient_doc_ids):
        current_index = len(patient_doc_ids) - 1
    if current_index < 0:
        current_index = 0

    analysis = get_analysis_by_id(patient_doc_ids[current_index])
    if analysis is None:
        st.error("Analyse introuvable.")
        return

    params  = analysis.get("parameters", {})
    results = analysis.get("results", {})

    # === CSS plein écran ===
    st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        header[data-testid="stHeader"] { display: none !important; }
        .block-container { padding-top: 1rem !important; max-width: 95% !important; }
    </style>
    """, unsafe_allow_html=True)

    # === Barre du haut ===
    top1, top2, top3 = st.columns([1, 4, 2])
    with top1:
        if st.button("← Quitter"):
            st.session_state["tb_presenting"] = False
            st.session_state.pop("tb_presenting_session_id", None)
            st.session_state.pop("tb_presenting_index", None)
            st.rerun()
    with top2:
        st.markdown(f"**{session.get('title', 'Tumor Board')}**")
    with top3:
        st.markdown(f"Patient **{current_index + 1}** / {len(patient_doc_ids)}")

    st.markdown("---")

    # === Identité patient ===
    patient_id = analysis.get("patient_id", "?")
    age  = params.get("age", "?")
    sex  = "Homme" if params.get("sex") == 1 else "Femme"
    site = params.get("tumor_site", "?")
    stage = params.get("ajcc_stage", "?")

    st.markdown(f"""
    <div style="text-align: center; margin: 20px 0;">
        <h1 style="font-size: 48px; margin-bottom: 5px;">{patient_id}</h1>
        <p style="font-size: 20px; color: #666;">{age} ans · {sex} · {site} · Stade {stage}</p>
    </div>
    """, unsafe_allow_html=True)

    # === Badge risque ===
    risk = str(results.get("risk_level", "—")).upper()
    if "LOW" in risk or risk == "FAIBLE":
        risk_color, risk_bg, risk_label = "#28a745", "#d4edda", "RISQUE FAIBLE"
    elif "HIGH" in risk or risk == "ÉLEVÉ":
        risk_color, risk_bg, risk_label = "#dc3545", "#f8d7da", "RISQUE ÉLEVÉ"
    else:
        risk_color, risk_bg, risk_label = "#ffc107", "#fff3cd", "RISQUE MODÉRÉ"

    st.markdown(f"""
    <div style="text-align: center; margin: 30px 0;">
        <span style="display: inline-block; padding: 15px 40px; border-radius: 30px;
             background-color: {risk_bg}; border: 3px solid {risk_color};
             color: {risk_color}; font-weight: bold; font-size: 28px; letter-spacing: 2px;">
             {risk_label}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # === 4 grandes cases de survie ===
    def format_surv(val):
        if val is None:
            return "—"
        if isinstance(val, (int, float)):
            return f"{val * 100:.0f}%" if val < 1 else f"{val:.0f}%"
        return "—"

    def big_card(col, title, value, color):
        with col:
            st.markdown(f"""
            <div style="border: 3px solid {color}; border-radius: 15px; padding: 25px;
                 text-align: center; background-color: white; min-height: 140px;
                 display: flex; flex-direction: column; justify-content: center;">
                <div style="color: #666; font-size: 16px; text-transform: uppercase;
                     font-weight: 600; letter-spacing: 1px;">{title}</div>
                <div style="color: {color}; font-size: 48px; font-weight: bold;
                     margin-top: 10px;">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    surv_cols = st.columns(4)
    big_card(surv_cols[0], "Survie à 1 an",  format_surv(results.get("survival_1yr_cox")), "#28a745")
    big_card(surv_cols[1], "Survie à 2 ans", format_surv(results.get("survival_2yr_cox")), "#ffc107")
    big_card(surv_cols[2], "Survie à 5 ans", format_surv(results.get("survival_5yr_cox")), "#dc3545")

    median = results.get("median_survival_cox")
    median_str = (f"~{round(median)} mois"
                  if median and isinstance(median, (int, float)) else "—")
    big_card(surv_cols[3], "Médiane (Cox)", median_str, "#1B4F72")

    # === Paramètres cliniques résumés ===
    t         = params.get("t_stage",  "?")
    n         = params.get("n_stage",  "?")
    m         = params.get("m_stage",  "?")
    grade     = params.get("grade",    "?")
    tmb       = params.get("tmb",      "?")
    radiation = "Oui" if params.get("radiation") else "Non"

    st.markdown(f"""
    <div style="display: flex; justify-content: center; gap: 40px; flex-wrap: wrap;
         margin: 30px 0; font-size: 16px;">
        <span><strong>T</strong> : {t}</span>
        <span><strong>N</strong> : {n}</span>
        <span><strong>M</strong> : {m}</span>
        <span><strong>Grade</strong> : {grade}</span>
        <span><strong>TMB</strong> : {tmb}</span>
        <span><strong>Radiothérapie</strong> : {radiation}</span>
    </div>
    """, unsafe_allow_html=True)

    # === Notes cliniques (présentation) ===
    current_doc_id = patient_doc_ids[current_index]
    pres_notes = get_notes_for_analysis(current_doc_id)

    st.markdown("---")
    notes_col1, notes_col2 = st.columns([3, 2])

    with notes_col1:
        st.markdown("**📝 Notes cliniques**")
        if pres_notes:
            for _pn in pres_notes:
                st.markdown(f"""
                <div style="background-color:#f8f9fa;border-left:3px solid #1B4F72;
                     padding:8px 12px;margin-bottom:6px;border-radius:0 6px 6px 0;
                     font-size:14px;">
                    <span style="color:#333;">{_pn.get('text', '')}</span>
                    <span style="color:#aaa;font-size:11px;margin-left:10px;">
                        — {_pn.get('author', '')} · {_pn.get('timestamp', '')}
                    </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("Aucune note.")

    with notes_col2:
        st.markdown("**Ajouter une note rapide**")
        _pres_profile = st.session_state.get("user_profile", {}) or {}
        _pres_title   = _pres_profile.get("title", "")
        _pres_first   = _pres_profile.get("first_name", "")
        _pres_last    = _pres_profile.get("last_name", "")
        _pres_author  = (f"{_pres_title} {_pres_first} {_pres_last}".strip()
                         if (_pres_first or _pres_last)
                         else st.session_state.get("user_email", "Utilisateur"))

        quick_note = st.text_area(
            "Note",
            placeholder="Décision TB, remarques…",
            key=f"tb_pres_note_{current_doc_id}",
            label_visibility="collapsed",
            height=60,
        )
        if st.button("Ajouter", key=f"tb_pres_add_{current_doc_id}", type="primary"):
            if quick_note.strip():
                if add_note_to_analysis(current_doc_id, quick_note.strip(), _pres_author):
                    st.toast("Note ajoutée ✓")
                    st.rerun()

    # === Navigation entre patients ===
    st.markdown("---")
    nav1, nav2, nav3 = st.columns([1, 4, 1])
    with nav1:
        if current_index > 0:
            if st.button("← Précédent", use_container_width=True):
                st.session_state["tb_presenting_index"] = current_index - 1
                st.rerun()
    with nav3:
        if current_index < len(patient_doc_ids) - 1:
            if st.button("Suivant →", use_container_width=True):
                st.session_state["tb_presenting_index"] = current_index + 1
                st.rerun()

    st.markdown("""
    <div style="text-align: center; color: #999; font-size: 12px; margin-top: 30px;">
        OncoPrediction · Outil d'aide à la décision — Usage académique uniquement
        · Ne remplace pas le jugement clinique
    </div>
    """, unsafe_allow_html=True)


def show_tumor_board_page(user_id):
    """Dispatcher pour la section Tumor Board."""
    if st.session_state.get("tb_open_session"):
        show_tb_session_detail(user_id)
        return

    show_tb_session_list(user_id)


# =============================================================================
# NOTES CLINIQUES — Composant réutilisable
# =============================================================================

def show_notes_section(analysis_doc_id, author_name, context_key=""):
    """Affiche les notes cliniques d'une analyse et le formulaire d'ajout."""
    st.markdown("---")
    st.markdown("### 📝 Notes cliniques")

    notes = get_notes_for_analysis(analysis_doc_id)

    if notes:
        for note in notes:
            st.markdown(f"""
            <div style="background-color:#f8f9fa;border-left:4px solid #1B4F72;
                 padding:12px 16px;margin-bottom:10px;border-radius:0 8px 8px 0;">
                <div style="font-size:14px;color:#333;">{note.get('text', '')}</div>
                <div style="font-size:11px;color:#999;margin-top:6px;">
                    {note.get('author', '')} · {note.get('timestamp', '')}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("Aucune note pour cette analyse.")

    st.markdown("")
    new_note = st.text_area(
        "Ajouter une note",
        placeholder="Ex : Chimiothérapie néoadjuvante proposée, réévaluation dans 3 mois…",
        key=f"new_note_{context_key}_{analysis_doc_id}",
        label_visibility="collapsed",
        height=80,
    )

    if st.button("Ajouter la note", key=f"add_note_{context_key}_{analysis_doc_id}"):
        if new_note.strip():
            if add_note_to_analysis(analysis_doc_id, new_note.strip(), author_name):
                st.toast("Note ajoutée ✓")
                st.rerun()
        else:
            st.warning("Veuillez écrire une note avant de l'ajouter.")


# =============================================================================
# APPLICATION PRINCIPALE (après authentification)
# =============================================================================
def main_app():
    cox_model, stats, rsf_model, feature_cols = load_models()

    if 'comparison_list' not in st.session_state:
        st.session_state.comparison_list = []

    # ── Mode Présentation (court-circuite tout le reste de l'app) ──
    if st.session_state.get("presentation_mode"):
        show_presentation_mode()
        return

    # ── Mode Présentation Tumor Board (court-circuite tout le reste de l'app) ──
    if st.session_state.get("tb_presenting"):
        show_tb_presentation(st.session_state.get("user_id"))
        return

    # ── En-tête principal ──
    user_email = st.session_state.get("user_email", "")
    st.markdown(f"""
    <div class="app-header">
        <div class="app-header-left">
            <div class="app-header-icon">OP</div>
            <div>
                <div class="app-header-title">Onco<em>Prediction</em></div>
                <div class="app-header-sub">Cancers gastro-œsophagiens · Tumor Boards · TCGA n=617</div>
            </div>
        </div>
        <div class="app-header-right">
            <span class="proto-badge">Prototype</span>
            <span class="user-badge">{user_email}</span>
            <a href="https://oncoprediction.ch" class="back-link">← oncoprediction.ch</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Chargement du profil (une seule fois par session) ──
    user_id = st.session_state.get("user_id", user_email)
    if "user_profile" not in st.session_state:
        st.session_state["user_profile"] = get_user_profile(user_id)
    profile = st.session_state["user_profile"] or {}

    # ── Nom de l'auteur pour les notes cliniques ──
    _pf_first = profile.get("first_name", "")
    _pf_last  = profile.get("last_name",  "")
    _pf_title = profile.get("title",      "")
    if _pf_first or _pf_last:
        author_name = f"{_pf_title} {_pf_first} {_pf_last}".strip()
    else:
        author_name = st.session_state.get("user_email", "Utilisateur")

    # Vérifier si le profil est complet (bypass si Firestore indisponible)
    if _db is None:
        profile_complete = True
    else:
        profile_complete = (
            bool(profile.get("first_name", "").strip())
            and bool(profile.get("last_name", "").strip())
            and bool(profile.get("role", "").strip())
        )

    # ── Sidebar — profil utilisateur (toujours visible) ──
    _local = user_email.split('@')[0]
    for _sep in ['_', '-', '.']:
        _local = _local.replace(_sep, '.')
    _parts = [p for p in _local.split('.') if p]
    initials = (_parts[0][0] + _parts[1][0]).upper() if len(_parts) >= 2 else _local[:2].upper()

    if profile.get("first_name") and profile.get("last_name"):
        _display_name = f"{profile['first_name']} {profile['last_name']}"
        initials = (profile["first_name"][0] + profile["last_name"][0]).upper()
    elif profile.get("first_name"):
        _display_name = profile["first_name"]
        initials = profile["first_name"][:2].upper()
    else:
        _display_name = user_email

    st.sidebar.markdown(f"""
    <div class="sidebar-user-card">
        <div class="sidebar-avatar">{initials}</div>
        <div class="sidebar-user-info">
            <div class="sidebar-user-label">Connecté</div>
            <div class="sidebar-user-email" title="{user_email}">{_display_name}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.sidebar.button("Se déconnecter", use_container_width=True):
        for key in ["authenticated", "user_email", "user_token", "user_id", "current",
                     "comparison_list", "auth_mode", "user_profile"]:
            st.session_state.pop(key, None)
        st.rerun()
    st.sidebar.markdown("<div style='margin-bottom:4px;'></div>", unsafe_allow_html=True)

    # ── Première connexion : profil incomplet → forcer le remplissage ──
    if not profile_complete:
        show_welcome_profile_page(user_id, user_email, profile)
        return

    # ── Barre de navigation horizontale ──
    _PAGES = ["Prédiction", "Historique", "Comparaison", "Tumor Board", "Mon Profil"]
    _ICONS = ["activity", "clock-history", "bar-chart", "people", "person"]

    selected = option_menu(
        menu_title=None,
        options=_PAGES,
        icons=_ICONS,
        default_index=st.session_state.get("_nav_idx", 0),
        orientation="horizontal",
        styles={
            "container": {
                "padding": "0", "margin": "0 0 20px 0",
                "background-color": "#ffffff",
                "border-bottom": "1px solid #e2e8f0",
            },
            "icon": {"color": "#94a3b8", "font-size": "14px"},
            "nav-link": {
                "font-size": "14px", "text-align": "center",
                "padding": "12px 16px", "color": "#64748b",
                "font-weight": "500", "background": "transparent",
                "border-bottom": "2px solid transparent",
            },
            "nav-link-selected": {
                "background-color": "transparent",
                "color": "#0b1628", "font-weight": "700",
                "border-bottom": "2px solid #0b1628",
            },
        },
        key="main_nav",
    )
    st.session_state["_nav_idx"] = _PAGES.index(selected)

    # ── Sidebar et boutons conditionnels (Prédiction + Comparaison uniquement) ──
    if selected in ["Prédiction", "Comparaison"]:
        inp = patient_form(stats)
        st.sidebar.markdown("<div style='margin: 10px 0 6px;'></div>", unsafe_allow_html=True)
        predict_btn = st.sidebar.button("Prédire la survie", type="primary", use_container_width=True)
        add_btn     = st.sidebar.button("+ Ajouter à la comparaison", use_container_width=True)
        clear_btn   = st.sidebar.button("Effacer la comparaison", use_container_width=True)
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Options d'affichage**")
        show_cox = st.sidebar.checkbox("Courbe Cox", value=True)
        show_rsf = st.sidebar.checkbox("Courbe RSF", value=True)
    else:
        inp = None
        predict_btn = add_btn = clear_btn = False
        show_cox = show_rsf = True

    if clear_btn:
        st.session_state.comparison_list = []

    if predict_btn or add_btn:
        with st.spinner("Calcul en cours…"):
            preds = get_predictions(inp, cox_model, rsf_model, feature_cols)
        entry = {'label': inp['label'], 'inp': inp, 'preds': preds}
        if predict_btn:
            st.session_state['current'] = entry
        if add_btn:
            if len(st.session_state.comparison_list) < 6:
                st.session_state.comparison_list.append(entry)
            else:
                st.sidebar.warning("Maximum 6 patients.")

        # ── Sauvegarde Firestore (uniquement sur "Prédire la survie") ──
        if predict_btn:
            _rl, _ = risk_cat(preds['cox_med'])
            _mean_diff = (
                abs(preds['cox_key'][0] - preds['rsf_key'][0]) +
                abs(preds['cox_key'][1] - preds['rsf_key'][1]) +
                abs(preds['cox_key'][2] - preds['rsf_key'][2])
            ) / 3
            _concordance = max(60, min(98, int(100 - _mean_diff * 100)))

            _parameters = {
                "age":                     inp['age'],
                "sex":                     inp['sex'],
                "tumor_site":              inp['site'],
                "ajcc_stage":              inp['ajcc_stage'],
                "t_stage":                 inp['t_stage'],
                "n_stage":                 inp['n_stage'],
                "m_stage":                 inp['m_stage'],
                "grade":                   inp['grade'],
                "radiation":               inp['radiation'],
                "tmb":                     inp['tmb'],
                "msi_score":               inp['msi'],
                "aneuploidy_score":        inp['aneuploidy'],
                "fraction_genome_altered": inp['fga'],
                "mutation_count":          inp['mutation_count'],
                "tumor_break_load":        inp['tbl'],
            }
            _results = {
                "risk_level":           _rl,
                "median_survival_cox":  preds['cox_med'],
                "median_survival_rsf":  preds['rsf_med'],
                "survival_1yr_cox":     float(preds['cox_key'][0]),
                "survival_2yr_cox":     float(preds['cox_key'][1]),
                "survival_5yr_cox":     float(preds['cox_key'][2]),
                "survival_1yr_rsf":     float(preds['rsf_key'][0]),
                "survival_2yr_rsf":     float(preds['rsf_key'][1]),
                "survival_5yr_rsf":     float(preds['rsf_key'][2]),
                "model_concordance":    _concordance,
            }
            _doc_id = save_analysis_to_firestore(
                user_id    = st.session_state.get("user_id", user_email),
                user_email = user_email,
                patient_id = inp['label'],
                parameters = _parameters,
                results    = _results,
            )
            if _doc_id:
                st.toast("Analyse sauvegardée ✓", icon="✅")
                # Invalider le cache historique pour que le prochain affichage soit à jour
                st.session_state.pop("user_analyses", None)
                st.session_state["last_analysis_doc_id"] = _doc_id

            # Stocker les données pour le mode présentation
            st.session_state["last_prediction"] = {
                "patient_id":         inp['label'],
                "age":                inp['age'],
                "sex_label":          "Homme" if inp['sex'] == 1 else "Femme",
                "tumor_site":         "Estomac" if inp['site'] == 1 else "Œsophage",
                "ajcc_stage":         inp['ajcc_stage'],
                "t_stage":            inp['t_stage'],
                "n_stage":            inp['n_stage'],
                "m_stage":            inp['m_stage'],
                "grade":              inp['grade'],
                "radiation":          inp['radiation'],
                "tmb":                inp['tmb'],
                "msi_score":          inp['msi'],
                "risk_level":         _rl,
                "median_survival_cox": preds['cox_med'],
                "survival_1yr_cox":   float(preds['cox_key'][0]),
                "survival_2yr_cox":   float(preds['cox_key'][1]),
                "survival_5yr_cox":   float(preds['cox_key'][2]),
            }

    # ── Préparer les données d'affichage ──
    current = st.session_state.get('current', None)
    display = []
    if current:
        display.append(current)
    seen = {current['label']} if current else set()
    for p in st.session_state.comparison_list:
        if p['label'] not in seen:
            display.append(p); seen.add(p['label'])

    fmt_med = lambda m: f"~{m:.0f} mois" if m else "> 100 mois"

    # ────────────────────────────────────────────────────────────────
    # PAGE PRÉDICTION
    # ────────────────────────────────────────────────────────────────
    if selected == "Prédiction":

        if not display:
            st.markdown("""
            <div class="welcome-state">
                <div class="welcome-icon-wrap">
                    <span class="welcome-icon-txt">OP</span>
                </div>
                <div class="welcome-title">Prêt à prédire<em>.</em></div>
                <div class="welcome-sub">
                    Renseignez le profil clinique et génomique du patient dans la barre latérale,
                    puis cliquez sur <strong>Calculer</strong> pour afficher les courbes de survie
                    et l'analyse IA combinée Cox + RSF.
                </div>
                <div class="welcome-cards">
                    <div class="welcome-card">
                        <div class="welcome-card-num">617</div>
                        <div class="welcome-card-title">Patients TCGA</div>
                        <div class="welcome-card-desc">Cancers gastriques et oesophagiens — données publiques cBioPortal</div>
                    </div>
                    <div class="welcome-card">
                        <div class="welcome-card-num">2</div>
                        <div class="welcome-card-title">Modèles IA</div>
                        <div class="welcome-card-desc">Cox (C-index 0.665) et Forêt Aléatoire de Survie (C-index 0.635)</div>
                    </div>
                    <div class="welcome-card">
                        <div class="welcome-card-num">50</div>
                        <div class="welcome-card-title">Variables cliniques</div>
                        <div class="welcome-card-desc">Stade, histologie, génomique, traitement — analyse personnalisée</div>
                    </div>
                    <div class="welcome-card">
                        <div class="welcome-card-num">6</div>
                        <div class="welcome-card-title">Patients max.</div>
                        <div class="welcome-card-desc">Comparaison multi-patients avec courbes superposées et tableau</div>
                    </div>
                </div>
                <div class="welcome-hint">
                    Outil d'aide à la décision — usage académique uniquement · Ne remplace pas le jugement clinique
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            main_p = display[0]
            pr     = main_p['preds']
            inp_d  = main_p['inp']

            s1_cox, s2_cox, s5_cox = pr['cox_key'] * 100
            s1_rsf, s2_rsf, s5_rsf = pr['rsf_key'] * 100
            rl, rc = risk_cat(pr['cox_med'])
            grade_lbl = {1:"G1",2:"G2",3:"G3"}[inp_d['grade']]

            # Labels de risque en français
            risk_fr = {"Faible": "RISQUE FAIBLE", "Intermédiaire": "RISQUE MODÉRÉ", "Élevé": "RISQUE ÉLEVÉ"}
            rl_en   = risk_fr.get(rl, rl)

            # Concordance entre modèles
            mean_diff   = (abs(s1_cox - s1_rsf) + abs(s2_cox - s2_rsf) + abs(s5_cox - s5_rsf)) / 3
            concordance = max(60, min(98, int(100 - mean_diff)))

            insight_text = generate_insight(inp_d, rl)
            med_display  = fmt_med(pr['cox_med'])

            sex_lbl  = "Homme"    if inp_d['sex'] == 1  else "Femme"
            site_lbl = "Estomac" if inp_d['site'] == 1 else "Œsophage"
            rad_lbl  = "Oui"     if inp_d['radiation'] else "Non"

            stage_style = {
                "I":   "background:#dcfce7;color:#166534;",
                "II":  "background:#fef9c3;color:#854d0e;",
                "III": "background:#fee2e2;color:#991b1b;",
                "IV":  "background:#fee2e2;color:#991b1b;",
            }
            ss        = stage_style.get(inp_d['ajcc_stage'], "background:#f1f5f9;color:#0b1628;")
            today_str = datetime.date.today().strftime("%d/%m/%Y")
            initials  = inp_d['label'][:2].upper()

            # ─────────────────────────────────────────────────
            # LAYOUT : panneau gauche (patient) + contenu principal
            # ─────────────────────────────────────────────────
            col_panel, col_main = st.columns([1, 3], gap="large")

            # ── PANNEAU GAUCHE : résumé patient ──
            with col_panel:
                st.markdown(f"""
                <div class="patient-panel">
                    <div style="text-align:center; margin-bottom:16px;">
                        <div class="patient-avatar">
                            <span style="font-size:18px;font-weight:900;color:#fff;">{initials}</span>
                        </div>
                        <div class="patient-name">{inp_d['label']}</div>
                        <div class="patient-date">{today_str}</div>
                    </div>
                    <div class="patient-section-title">Paramètres cliniques</div>
                    <div class="param-row">
                        <span class="param-key">Âge</span>
                        <span class="param-val">{inp_d['age']} ans</span>
                    </div>
                    <div class="param-row">
                        <span class="param-key">Sexe</span>
                        <span class="param-val">{sex_lbl}</span>
                    </div>
                    <div class="param-row">
                        <span class="param-key">Site primaire</span>
                        <span class="param-val">{site_lbl}</span>
                    </div>
                    <div class="param-row">
                        <span class="param-key">Stade global</span>
                        <span class="param-val">
                            <span style="{ss}padding:2px 8px;border-radius:5px;
                                         font-size:11px;font-weight:700;">
                                AJCC {inp_d['ajcc_stage']}
                            </span>
                        </span>
                    </div>
                    <div class="param-row">
                        <span class="param-key">T / N / M</span>
                        <span class="param-val">{inp_d['t_stage']} / {inp_d['n_stage']} / {inp_d['m_stage']}</span>
                    </div>
                    <div class="param-row">
                        <span class="param-key">Grade</span>
                        <span class="param-val">{grade_lbl}</span>
                    </div>
                    <div class="param-row">
                        <span class="param-key">Radiothérapie</span>
                        <span class="param-val">{rad_lbl}</span>
                    </div>
                    <div class="param-row">
                        <span class="param-key">TMB</span>
                        <span class="param-val">{inp_d['tmb']:.1f} mut/Mb</span>
                    </div>
                    <div class="param-row">
                        <span class="param-key">Score MSI</span>
                        <span class="param-val">{inp_d['msi']:.1f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── CONTENU PRINCIPAL ──
            with col_main:

                # ─────────────────────────────────────────────
                # 1. BANNIÈRE AI CONSENSUS
                # ─────────────────────────────────────────────
                st.markdown(f"""
                <div class="ai-card">
                    <div class="ai-card-left">
                        <div class="ai-card-header">CONSENSUS IA — COX + RSF</div>
                        <div class="ai-card-badges">
                            <span class="risk-badge {rc}">{rl_en}</span>
                            <span class="confidence-txt">Concordance des modèles : {concordance}%</span>
                        </div>
                        <div class="ai-insight-text">{insight_text}</div>
                    </div>
                    <div class="ai-card-right">
                        <div class="ai-median-label">Survie médiane estimée</div>
                        <div class="ai-median-value">{med_display}</div>
                        <div class="ai-median-bar"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ─────────────────────────────────────────────
                # 2. QUATRE MÉTRIQUES EN LIGNE
                # ─────────────────────────────────────────────
                def bar_color(pct):
                    if pct >= 70: return "progress-green"
                    if pct >= 50: return "progress-orange"
                    return "progress-red"

                def metric_html(label, value_str, bar_pct, bar_class, sub=""):
                    w = min(100, max(4, bar_pct))
                    return f"""
                    <div class="metric-card-h">
                        <div class="metric-h-label">{label}</div>
                        <div class="metric-h-value">{value_str}</div>
                        <div class="progress-bar-bg">
                            <div class="progress-bar-fill {bar_class}" style="width:{w:.0f}%"></div>
                        </div>
                        <div class="metric-h-sub">{sub}</div>
                    </div>"""

                mc1, mc2, mc3, mc4 = st.columns(4, gap="small")
                mc1.markdown(metric_html("SURVIE À 1 AN",  f"{s1_cox:.1f}%", s1_cox, bar_color(s1_cox), f"RSF : {s1_rsf:.1f}%"), unsafe_allow_html=True)
                mc2.markdown(metric_html("SURVIE À 2 ANS", f"{s2_cox:.1f}%", s2_cox, bar_color(s2_cox), f"RSF : {s2_rsf:.1f}%"), unsafe_allow_html=True)
                mc3.markdown(metric_html("SURVIE À 5 ANS", f"{s5_cox:.1f}%", s5_cox, bar_color(s5_cox), f"RSF : {s5_rsf:.1f}%"), unsafe_allow_html=True)
                med_val = pr['cox_med']
                med_bar = min(100, (med_val / 80 * 100)) if med_val else 100
                mc4.markdown(metric_html("MÉDIANE (COX)", med_display, med_bar, "progress-navy", f"RSF : {fmt_med(pr['rsf_med'])}"), unsafe_allow_html=True)

                st.markdown("<div style='margin-bottom:14px;'></div>", unsafe_allow_html=True)

                # ─────────────────────────────────────────────
                # 3. GRAPHE DE SURVIE
                # ─────────────────────────────────────────────
                st.markdown(f"""
                <div class="graph-header">
                    <div class="graph-title">Probabilité de survie — {main_p['label']}</div>
                    <div class="graph-sub">Trait plein = Modèle Cox (6 vars, C-index 0,665) · Pointillés = RSF (44 vars, C-index 0,635) · Zone ombrée = incertitude des modèles</div>
                </div>
                """, unsafe_allow_html=True)
                fig = plot_survival_v2(main_p, show_cox=show_cox, show_rsf=show_rsf)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                # ─────────────────────────────────────────────
                # 4. SOUS-ONGLETS
                # ─────────────────────────────────────────────
                sub1, sub2 = st.tabs(["Détails modèle Cox", "Importance des variables (RSF)"])

                with sub1:
                    st.markdown("""
                    <div style="font-size:13px; color:#64748b; margin: 8px 0 14px;">
                    Rapports de risque (HR) issus du modèle de Cox à risques proportionnels.
                    HR &gt; 1 indique un facteur de risque, HR &lt; 1 un facteur protecteur.
                    Les valeurs spécifiques au patient sont indiquées entre parenthèses.
                    </div>
                    """, unsafe_allow_html=True)
                    fig_hr = plot_hr_forest(cox_model, inp_d)
                    st.pyplot(fig_hr, use_container_width=True)
                    plt.close(fig_hr)

                with sub2:
                    st.markdown("""
                    <div style="font-size:13px; color:#64748b; margin: 8px 0 14px;">
                    Top 10 des variables par diminution moyenne de l'impureté (MDI) du modèle
                    de forêt aléatoire de survie. Les valeurs plus élevées indiquent une plus grande
                    contribution à la prédiction de survie.
                    </div>
                    """, unsafe_allow_html=True)
                    fig_imp = plot_rsf_importance(rsf_model, feature_cols)
                    st.pyplot(fig_imp, use_container_width=True)
                    plt.close(fig_imp)

            # ─────────────────────────────────────────────────
            # FOOTER DISCLAIMER
            # ─────────────────────────────────────────────────
            st.markdown("""
            <div class="view2-footer">
                <strong>Avertissement :</strong> Cet outil est réservé à la recherche académique et ne
                remplace pas le jugement clinique. Les prédictions sont basées sur les données TCGA (n=617)
                et ne doivent pas être utilisées pour la prise en charge individuelle sans interprétation
                clinique experte.
                <br><span style="color:#94a3b8;">OncoPrediction · Mémoire de Master · Université de Genève · TCGA/cBioPortal</span>
            </div>
            """, unsafe_allow_html=True)

            # ── Notes cliniques ──
            if st.session_state.get("last_analysis_doc_id"):
                show_notes_section(
                    st.session_state["last_analysis_doc_id"],
                    author_name,
                    context_key="prediction",
                )

            # ── Bouton Mode Présentation ──
            st.markdown("---")
            _pb1, _pb2, _pb3 = st.columns([1, 2, 1])
            with _pb2:
                if st.button("🖥️ Mode Présentation Tumor Board",
                             use_container_width=True, type="primary",
                             key="enter_presentation"):
                    st.session_state["presentation_mode"] = True
                    st.rerun()

        # ── Méthodologie (affichage direct, sans expander) ──
        st.markdown("---")
        _show_meth = st.checkbox("📖 Afficher la méthodologie et les sources",
                                 value=False, key="show_methodology")
        if _show_meth:
            st.markdown("""
#### Source des données
- **TCGA** (The Cancer Genome Atlas) via cBioPortal · données publiques
- **617 patients** : 440 cancers gastriques, 177 cancers œsophagiens
- **244 décès** (39.5 %) · Suivi médian : 28.7 mois

#### Modèles et performances (test set, n = 124)
| Modèle | Variables | C-index | IBS |
|---|---|---|---|
| **Cox (référence)** | 6 | **0.665** | **0.141** |
| RSF | 44 | 0.635 | 0.161 |

#### Catégories de risque (basées sur la survie médiane Cox)
- **Faible** : médiane ≥ 36 mois
- **Intermédiaire** : médiane 15–36 mois
- **Élevé** : médiane < 15 mois

#### Limites
- Outil de recherche — ne remplace pas le jugement clinique
- Entraîné sur données nord-américaines (TCGA) — extrapolation prudente
- Petit dataset (617 patients) — incertitude statistique importante
- Variables génomiques non disponibles dans tous les centres
            """)

    # ────────────────────────────────────────────────────────────────
    # PAGE HISTORIQUE
    # ────────────────────────────────────────────────────────────────
    elif selected == "Historique":

        # ── Helpers de formatage ──
        def _fmt_surv(value):
            if value is None:
                return "—"
            try:
                v = float(value)
                return f"{v * 100:.1f}%" if v < 1 else f"{v:.1f}%"
            except (TypeError, ValueError):
                return str(value)

        def _fmt_med(value):
            if value is None:
                return "—"
            try:
                return f"~{round(float(value))} mois"
            except (TypeError, ValueError):
                return str(value)

        # ── Cache des analyses (rechargé uniquement si invalidé) ──
        if "user_analyses" not in st.session_state:
            st.session_state["user_analyses"] = get_user_analyses(user_id)
        analyses = st.session_state["user_analyses"]

        # ── En-tête ──
        _h_col_title, _h_col_btn = st.columns([5, 1])
        with _h_col_title:
            st.markdown("## Historique des analyses")
            st.markdown("Retrouvez toutes vos analyses précédentes.")
        with _h_col_btn:
            st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
            if st.button("🔄 Rafraîchir", use_container_width=True):
                st.session_state["user_analyses"] = get_user_analyses(user_id)
                st.rerun()

        st.markdown("---")

        # ── Métriques ──
        _m1, _m2, _m3 = st.columns(3)
        with _m1:
            st.metric("Total analyses", len(analyses))
        with _m2:
            if analyses:
                _last_ts = analyses[0].get("timestamp")
                _last_str = _last_ts.strftime("%d/%m/%Y") if _last_ts else "—"
            else:
                _last_str = "—"
            st.metric("Dernière analyse", _last_str)
        with _m3:
            _unique_pts = len(set(a.get("patient_id", "") for a in analyses))
            st.metric("Patients uniques", _unique_pts)

        st.markdown("---")

        # ── Barre de recherche + filtre ──
        _s1, _s2 = st.columns([3, 1])
        with _s1:
            search_query = st.text_input(
                "Rechercher", placeholder="🔍 Rechercher un patient…",
                label_visibility="collapsed", key="hist_search")
        with _s2:
            risk_filter = st.selectbox(
                "Filtre risque",
                ["Tous", "Risque faible", "Risque modéré", "Risque élevé"],
                label_visibility="collapsed", key="hist_risk_filter")

        # ── Filtrage ──
        filtered = analyses
        if search_query:
            filtered = [a for a in filtered
                        if search_query.lower() in a.get("patient_id", "").lower()]
        if risk_filter != "Tous":
            _risk_map_filter = {
                "Risque faible":  "Faible",
                "Risque modéré":  "Intermédiaire",
                "Risque élevé":   "Élevé",
            }
            _target = _risk_map_filter.get(risk_filter, "")
            filtered = [a for a in filtered
                        if a.get("results", {}).get("risk_level", "") == _target]

        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)

        # ── Confirmation de suppression depuis la liste (en haut, bien visible) ──
        if st.session_state.get("confirm_delete_id"):
            _cdid   = st.session_state["confirm_delete_id"]
            _cdname = st.session_state.get("confirm_delete_patient", "")
            st.warning(f"Voulez-vous vraiment supprimer l'analyse de **{_cdname}** ? Cette action est irréversible.")
            _cc1, _cc2, _ = st.columns([1, 1, 4])
            with _cc1:
                if st.button("Oui, supprimer", type="primary", key="hist_list_del_confirm"):
                    try:
                        _db.collection("analyses").document(_cdid).delete()
                        st.session_state.pop("confirm_delete_id", None)
                        st.session_state.pop("confirm_delete_patient", None)
                        st.session_state.pop("user_analyses", None)
                        st.toast("Analyse supprimée ✓")
                        st.rerun()
                    except Exception as _e:
                        st.error(f"Erreur : {_e}")
            with _cc2:
                if st.button("Annuler", key="hist_list_del_cancel"):
                    st.session_state.pop("confirm_delete_id", None)
                    st.session_state.pop("confirm_delete_patient", None)
                    st.rerun()

        # ── Vue détaillée (si une analyse est sélectionnée) ──
        _sel = st.session_state.get("selected_analysis")
        if _sel is not None:
            _ap = _sel.get("parameters", {})
            _ar = _sel.get("results", {})

            st.markdown(f"### Détails — {_sel.get('patient_id', 'Patient')}")
            _ts = _sel.get("timestamp")
            st.caption(_ts.strftime("%d/%m/%Y à %H:%M") if _ts else "Date inconnue")

            if st.button("← Retour à la liste", key="hist_back"):
                st.session_state["selected_analysis"] = None
                st.session_state.pop("confirm_delete", None)
                st.rerun()

            # ── Badge de risque ──
            _risk_det = _ar.get("risk_level", "—")
            if "faible" in str(_risk_det).lower() or _risk_det == "Faible":
                _risk_color = "#22c55e"
                _risk_lbl   = "RISQUE FAIBLE"
            elif "élevé" in str(_risk_det).lower() or _risk_det == "Élevé":
                _risk_color = "#ef4444"
                _risk_lbl   = "RISQUE ÉLEVÉ"
            else:
                _risk_color = "#f59e0b"
                _risk_lbl   = "RISQUE MODÉRÉ"

            st.markdown(f"""
            <div style="display:inline-block;padding:8px 20px;border-radius:20px;
                 background:{_risk_color};color:white;font-weight:700;font-size:14px;
                 margin:8px 0 16px 0;">{_risk_lbl}</div>
            """, unsafe_allow_html=True)

            # ── 4 cases colorées de survie ──
            _sc1, _sc2, _sc3, _sc4 = st.columns(4)
            with _sc1:
                st.markdown(f"""
                <div style="border:2px solid #22c55e;border-radius:10px;padding:15px;
                     text-align:center;margin-bottom:16px;">
                    <div style="color:#64748b;font-size:11px;text-transform:uppercase;
                         margin-bottom:6px;">Survie à 1 an</div>
                    <div style="color:#22c55e;font-size:26px;font-weight:800;">
                         {_fmt_surv(_ar.get("survival_1yr_cox"))}</div>
                </div>""", unsafe_allow_html=True)
            with _sc2:
                st.markdown(f"""
                <div style="border:2px solid #f59e0b;border-radius:10px;padding:15px;
                     text-align:center;margin-bottom:16px;">
                    <div style="color:#64748b;font-size:11px;text-transform:uppercase;
                         margin-bottom:6px;">Survie à 2 ans</div>
                    <div style="color:#f59e0b;font-size:26px;font-weight:800;">
                         {_fmt_surv(_ar.get("survival_2yr_cox"))}</div>
                </div>""", unsafe_allow_html=True)
            with _sc3:
                st.markdown(f"""
                <div style="border:2px solid #ef4444;border-radius:10px;padding:15px;
                     text-align:center;margin-bottom:16px;">
                    <div style="color:#64748b;font-size:11px;text-transform:uppercase;
                         margin-bottom:6px;">Survie à 5 ans</div>
                    <div style="color:#ef4444;font-size:26px;font-weight:800;">
                         {_fmt_surv(_ar.get("survival_5yr_cox"))}</div>
                </div>""", unsafe_allow_html=True)
            with _sc4:
                st.markdown(f"""
                <div style="border:2px solid #0b1628;border-radius:10px;padding:15px;
                     text-align:center;margin-bottom:16px;">
                    <div style="color:#64748b;font-size:11px;text-transform:uppercase;
                         margin-bottom:6px;">Médiane (Cox)</div>
                    <div style="color:#0b1628;font-size:26px;font-weight:800;">
                         {_fmt_med(_ar.get("median_survival_cox"))}</div>
                </div>""", unsafe_allow_html=True)

            # ── Paramètres cliniques ──
            st.markdown("#### Paramètres cliniques")
            _dc1, _dc2, _dc3 = st.columns(3)
            with _dc1:
                st.markdown(f"**Âge** : {_ap.get('age', '—')} ans")
                st.markdown(f"**Sexe** : {'Homme' if _ap.get('sex') == 1 else 'Femme'}")
                st.markdown(f"**Site** : {'Estomac' if _ap.get('tumor_site') == 1 else 'Œsophage'}")
            with _dc2:
                st.markdown(f"**Stade AJCC** : {_ap.get('ajcc_stage', '—')}")
                st.markdown(f"**T** : {_ap.get('t_stage', '—')}")
                st.markdown(f"**N** : {_ap.get('n_stage', '—')}")
                st.markdown(f"**M** : {_ap.get('m_stage', '—')}")
            with _dc3:
                st.markdown(f"**Grade** : {_ap.get('grade', '—')}")
                st.markdown(f"**Radiothérapie** : {'Oui' if _ap.get('radiation') else 'Non'}")
                st.markdown(f"**TMB** : {_ap.get('tmb', '—')}")

            # ── Notes cliniques ──
            _detail_doc_id = _sel.get("doc_id", "")
            if _detail_doc_id:
                show_notes_section(_detail_doc_id, author_name, context_key="historique")

            # ── Suppression depuis la vue détaillée ──
            st.markdown("---")
            if st.button("🗑️ Supprimer cette analyse", type="secondary", key="hist_del"):
                st.session_state["confirm_delete"] = _sel.get("doc_id")

            if st.session_state.get("confirm_delete") == _sel.get("doc_id"):
                st.warning("Êtes-vous sûr de vouloir supprimer cette analyse ? Cette action est irréversible.")
                _conf1, _conf2, _ = st.columns([1, 1, 3])
                with _conf1:
                    if st.button("Oui, supprimer", type="primary", key="hist_del_confirm"):
                        try:
                            _db.collection("analyses").document(_sel["doc_id"]).delete()
                            st.session_state["selected_analysis"] = None
                            st.session_state.pop("confirm_delete", None)
                            st.session_state.pop("user_analyses", None)
                            st.toast("Analyse supprimée ✓")
                            st.rerun()
                        except Exception as _e:
                            st.error(f"Erreur lors de la suppression : {_e}")
                with _conf2:
                    if st.button("Annuler", key="hist_del_cancel"):
                        st.session_state.pop("confirm_delete", None)
                        st.rerun()

        else:
            # ── Liste des analyses ──
            if not filtered:
                if search_query or risk_filter != "Tous":
                    st.info("Aucune analyse ne correspond à votre recherche.")
                else:
                    st.info("Vous n'avez pas encore réalisé d'analyse. Rendez-vous dans l'onglet Prédiction pour commencer.")
            else:
                for _analysis in filtered:
                    _params  = _analysis.get("parameters", {})
                    _results = _analysis.get("results", {})
                    _pid     = _analysis.get("patient_id", "Inconnu")
                    _ts      = _analysis.get("timestamp")
                    _date_str = _ts.strftime("%d/%m/%Y à %H:%M") if _ts else "Date inconnue"

                    _risk = _results.get("risk_level", "")
                    if "faible" in _risk.lower() or _risk == "Faible":
                        _risk_label = "🟢 Risque faible"
                    elif "élevé" in _risk.lower() or _risk == "Élevé":
                        _risk_label = "🔴 Risque élevé"
                    else:
                        _risk_label = "🟡 Risque modéré"

                    _lc1, _lc2, _lc3, _lc4, _lc5, _lc6 = st.columns([2, 2, 2, 2, 1, 0.5])
                    with _lc1:
                        st.markdown(f"**{_pid}**")
                        st.caption(_date_str)
                        _nb_notes = len(_analysis.get("notes", []) or [])
                        if _nb_notes > 0:
                            st.caption(f"📝 {_nb_notes} note{'s' if _nb_notes > 1 else ''}")
                    with _lc2:
                        _stage = _params.get("ajcc_stage", "—")
                        _site  = "Estomac" if _params.get("tumor_site") == 1 else "Œsophage"
                        st.markdown(f"Stade {_stage}")
                        st.caption(_site)
                    with _lc3:
                        st.markdown(_risk_label)
                    with _lc4:
                        st.markdown(_fmt_med(_results.get("median_survival_cox")))
                        st.caption(f"Survie 1 an : {_fmt_surv(_results.get('survival_1yr_cox'))}")
                    with _lc5:
                        if st.button("Détails", key=f"detail_{_analysis.get('doc_id', '')}"):
                            st.session_state["selected_analysis"] = _analysis
                            st.rerun()
                    with _lc6:
                        if st.button("🗑️", key=f"delete_{_analysis.get('doc_id', '')}"):
                            st.session_state["confirm_delete_id"] = _analysis.get("doc_id")
                            st.session_state["confirm_delete_patient"] = _analysis.get("patient_id", "")
                            st.rerun()

                    st.markdown("---")

    # ────────────────────────────────────────────────────────────────
    # PAGE COMPARAISON
    # Lit directement comparison_list (source de vérité indépendante)
    # ────────────────────────────────────────────────────────────────
    elif selected == "Comparaison":
        comp_list = st.session_state.comparison_list

        if len(comp_list) >= 2:
            st.markdown('<div class="section-title">Comparaison multi-patients</div>',
                        unsafe_allow_html=True)
            rows = []
            for p in comp_list:
                rl2, _ = risk_cat(p['preds']['cox_med'])
                rows.append({
                    "Patient":       p['label'],
                    "Stade":         p['inp']['ajcc_stage'],
                    "Âge":           p['inp']['age'],
                    "Grade":         p['inp']['grade'],
                    "Radio.":        "Oui" if p['inp']['radiation'] else "Non",
                    "TMB":           f"{p['inp']['tmb']:.1f}",
                    "S 1an Cox":     f"{p['preds']['cox_key'][0]*100:.1f}%",
                    "S 2ans Cox":    f"{p['preds']['cox_key'][1]*100:.1f}%",
                    "S 5ans Cox":    f"{p['preds']['cox_key'][2]*100:.1f}%",
                    "Médiane Cox":   fmt_med(p['preds']['cox_med']),
                    "Risque":        rl2,
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            # Graphe comparatif
            st.markdown('<div class="section-title" style="margin-top:20px;">Courbes superposées</div>',
                        unsafe_allow_html=True)
            fig2 = plot_survival(comp_list, show_cox=show_cox, show_rsf=show_rsf)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

            # ── AI Insight Groupe ──
            n_low2  = sum(1 for p in comp_list if risk_cat(p['preds']['cox_med'])[0] == "Faible")
            n_med2  = sum(1 for p in comp_list if risk_cat(p['preds']['cox_med'])[0] == "Intermédiaire")
            n_high2 = sum(1 for p in comp_list if risk_cat(p['preds']['cox_med'])[0] == "Élevé")
            group_insight = generate_group_insight(comp_list)
            best_p  = max(comp_list, key=lambda p: p['preds']['cox_med'] or 200)
            worst_p = min(comp_list, key=lambda p: p['preds']['cox_med'] or 200)

            st.markdown(f"""
            <div class="comp-insight-card">
                <div class="comp-insight-header">ANALYSE IA — PROFIL DU GROUPE</div>
                <div class="comp-insight-grid">
                    <div class="comp-insight-item">
                        <div class="comp-insight-item-label">Risque faible</div>
                        <div class="comp-insight-item-val" style="color:#22c55e;">{n_low2}</div>
                    </div>
                    <div class="comp-insight-item">
                        <div class="comp-insight-item-label">Risque intermédiaire</div>
                        <div class="comp-insight-item-val" style="color:#f59e0b;">{n_med2}</div>
                    </div>
                    <div class="comp-insight-item">
                        <div class="comp-insight-item-label">Risque élevé</div>
                        <div class="comp-insight-item-val" style="color:#ef4444;">{n_high2}</div>
                    </div>
                    <div class="comp-insight-item">
                        <div class="comp-insight-item-label">Meilleur pronostic</div>
                        <div class="comp-insight-item-val">{best_p['label']}</div>
                    </div>
                    <div class="comp-insight-item">
                        <div class="comp-insight-item-label">Pronostic le plus sombre</div>
                        <div class="comp-insight-item-val">{worst_p['label']}</div>
                    </div>
                </div>
                <div class="comp-insight-text">{group_insight}</div>
            </div>
            """, unsafe_allow_html=True)

        elif len(comp_list) == 1:
            st.markdown(f"""
            <div class="welcome-state" style="padding: 40px 24px;">
                <div class="welcome-icon-wrap" style="width:56px;height:56px;border-radius:14px;
                     background:linear-gradient(135deg,#0b1628,#0ea5e9);">
                    <span class="welcome-icon-txt" style="font-size:18px;">1</span>
                </div>
                <div class="welcome-title" style="font-size:18px;">
                    {comp_list[0]['label']} ajouté
                </div>
                <div class="welcome-sub">
                    Modifiez les paramètres dans la barre latérale pour configurer
                    un deuxième patient, puis cliquez à nouveau sur
                    <strong>Comparer</strong> pour lancer la comparaison.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="welcome-state" style="padding: 40px 24px;">
                <div class="welcome-icon-wrap" style="width:56px;height:56px;border-radius:14px;">
                    <span class="welcome-icon-txt" style="font-size:18px;">+</span>
                </div>
                <div class="welcome-title" style="font-size:18px;">Comparaison multi-patients</div>
                <div class="welcome-sub">
                    Utilisez le bouton <strong>Comparer</strong> dans la barre latérale
                    pour ajouter des profils patients et les comparer côte à côte.
                    Jusqu'à 6 patients peuvent être comparés simultanément.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────
    # PAGE TUMOR BOARD
    # ────────────────────────────────────────────────────────────────
    elif selected == "Tumor Board":
        show_tumor_board_page(user_id)

    # ────────────────────────────────────────────────────────────────
    # PAGE MON PROFIL
    # ────────────────────────────────────────────────────────────────
    elif selected == "Mon Profil":
        show_profile_page(user_id, user_email, profile)

    st.markdown("""
    <div class="app-footer">
        <a href="https://oncoprediction.ch">← oncoprediction.ch</a>
        &nbsp;·&nbsp; Université de Genève · Données TCGA/cBioPortal
        &nbsp;·&nbsp; Usage académique uniquement
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# POINT D'ENTRÉE — Routeur principal
# =============================================================================
def main():
    """Vérifie l'authentification et affiche la page appropriée."""
    if st.session_state.get("authenticated", False):
        main_app()
    else:
        show_auth_page()


if __name__ == "__main__":
    main()
