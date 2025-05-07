import streamlit as st
import pandas as pd
import pickle
import os

@st.cache_data
def load_data():
    """Load and cache the processed fatality data"""
    df = pd.read_csv('fatality_data_processed.csv')
    return df

@st.cache_resource
def load_model():
    """Load and cache the trained model"""
    try:
        with open('fatality_risk_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_shap_values():
    """Load and cache SHAP values for model explanation"""
    try:
        with open('shap_values.pkl', 'rb') as f:
            shap_values, X_test_processed = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        return shap_values, X_test_processed, feature_names
    except Exception as e:
        print(f"Error loading SHAP values: {e}")
        return None, None, None

def load_glassmorphism_css():
    """Load custom CSS for glassmorphism effect"""
    st.markdown("""
    <style>
    /* Main color palette */
    :root {
        --dark-blue: #252D6E;
        --purple: #6F5B96;
        --mauve: #B3698A;
        --salmon: #F08A7B;
        --peach: #FFA477;
        --background: rgba(255, 255, 255, 0.85);
        --text: #252D6E;
        --glass-border: rgba(255, 255, 255, 0.18);
        --glass-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    /* Set page background with a subtle gradient */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf5 100%);
    }
    
    /* Typography */
    html, body, [class*="css"] {
        font-family: 'Optima', 'Libre Baskerville', serif !important;
        color: var(--text);
    }
    
    h1, h2, h3 {
        font-family: 'Optima', 'Libre Baskerville', serif !important;
        color: var(--dark-blue);
        font-weight: 600;
    }
    
    h1 {
        font-size: 28px;
    }
    
    h2 {
        font-size: 20px;
    }
    
    p, div, span, li {
        font-size: 16px;
    }
    
    .caption {
        font-size: 14px;
        color: var(--purple);
    }
    
    /* Navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: rgba(255, 255, 255, 0.4);
        padding: 5px;
        border-radius: 8px;
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        color: var(--dark-blue);
        padding: 10px 20px;
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--dark-blue);
        color: white !important;
        box-shadow: 0 4px 20px rgba(37, 45, 110, 0.25);
    }
    
    /* Glassmorphism cards for visualizations */
    .glass-card {
        background: rgba(255, 255, 255, 0.25);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 25px;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.5);
        transform: translateY(-2px);
    }
    
    .glass-card h3 {
        color: var(--dark-blue);
        margin-bottom: 15px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.3);
        padding-bottom: 8px;
    }
    
    /* Custom metric styles with glassmorphism */
    .metric-container {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0px;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 4px 20px rgba(37, 45, 110, 0.15);
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.3);
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: var(--dark-blue);
    }
    
    .metric-label {
        font-size: 14px;
        color: var(--purple);
    }
    
    /* Map container with glassmorphism */
    .map-container {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        overflow: hidden;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        padding: 10px;
        margin-bottom: 25px;
    }
    
    /* Filter section with glassmorphism */
    .filter-section {
        background: rgba(255, 255, 255, 0.15);
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 25px;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 4px 20px rgba(37, 45, 110, 0.15);
    }
    
    /* Key insights with glassmorphism */
    .insights-box {
        background: rgba(255, 255, 255, 0.25);
        border-radius: 12px;
        padding: 20px;
        margin-top: 25px;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-left: 4px solid var(--mauve);
        box-shadow: 0 4px 20px rgba(179, 105, 138, 0.2);
    }
    
    /* Apply custom glassmorphism to Streamlit elements */
    .stButton>button {
        background-color: rgba(37, 45, 110, 0.8);
        color: white;
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 6px;
        padding: 8px 16px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: rgba(37, 45, 110, 1);
        box-shadow: 0 4px 20px rgba(37, 45, 110, 0.3);
    }
    
    /* Customize select boxes and sliders */
    .stSelectbox>div>div, .stMultiSelect>div>div {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
    }
    
    .stSlider>div>div {
        background-color: rgba(255, 255, 255, 0.2);
    }
    
    .stSlider>div>div>div>div {
        background-color: var(--salmon);
    }
    
    /* Dataframe styling */
    .dataframe-container {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 10px;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 4px 20px rgba(37, 45, 110, 0.15);
        overflow: hidden;
    }
    
    /* Add subtle animated gradient background for added depth */
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    
    .stApp {
        background: linear-gradient(-45deg, #f5f7fa, #e8ecf5, #edf1f7, #e2e8f0);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    /* Title section with fancy glassmorphism */
    .title-section {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 25px;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .title-section h1 {
        margin-bottom: 5px;
        background: linear-gradient(135deg, var(--dark-blue), var(--purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    .title-section p {
        color: var(--purple);
        opacity: 0.8;
    }
    </style>
    """, unsafe_allow_html=True)