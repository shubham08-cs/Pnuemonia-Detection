from pathlib import Path
import sys
import numpy as np

import streamlit as st
from PIL import Image
import plotly.graph_objects as go

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from helpers import (
    load_demo_image,
    load_pneumonia_model,
    preprocess_image,
    create_saliency_map,
    create_3d_heatmap,
    create_attention_overlay,
    analyze_affected_regions,
    generate_pdf_report,
)

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== PREMIUM 3D ANIMATED DESIGN SYSTEM ====================
st.markdown("""
<style>
    /* ===== ADVANCED DESIGN SYSTEM ===== */
    :root {
        --primary: #0ea5e9;
        --primary-dark: #0284c7;
        --primary-light: #e0f2fe;
        --accent: #10b981;
        --accent-dark: #047857;
        --bg-light: #eff6ff;
        --bg-white: #FFFFFF;
        --border: rgba(14, 165, 233, 0.16);
        --text-primary: #111827;
        --text-secondary: #475569;
        --text-tertiary: #64748b;
        --success: #16a34a;
        --success-light: #dcfce7;
        --danger: #dc2626;
        --danger-light: #fee2e2;
        --shadow-sm: 0 12px 40px rgba(14, 165, 233, 0.08);
        --shadow-md: 0 24px 60px rgba(14, 165, 233, 0.12);
        --shadow-lg: 0 40px 100px rgba(14, 165, 233, 0.15);
        --radius: 24px;
    }

    /* ===== PREMIUM ANIMATIONS ===== */
    @keyframes floatUp {
        0% { transform: translateY(0px); opacity: 0.88; }
        50% { transform: translateY(-14px); opacity: 1; }
        100% { transform: translateY(0px); opacity: 0.88; }
    }

    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(24px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInScale {
        0% { opacity: 0; transform: scale(0.95); }
        100% { opacity: 1; transform: scale(1); }
    }

    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 0 0 rgba(14, 165, 233, 0.16); }
        50% { box-shadow: 0 0 50px 18px rgba(14, 165, 233, 0.08); }
    }

    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }

    @keyframes rotate3d {
        0% { transform: rotateX(0deg) rotateY(0deg) rotateZ(0deg); }
        100% { transform: rotateX(360deg) rotateY(360deg) rotateZ(0deg); }
    }

    @keyframes slideInRight {
        0% { opacity: 0; transform: translateX(40px); }
        100% { opacity: 1; transform: translateX(0); }
    }

    @keyframes slideInLeft {
        0% { opacity: 0; transform: translateX(-40px); }
        100% { opacity: 1; transform: translateX(0); }
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes particleFloat {
        0%, 100% { transform: translateY(0px) translateX(0px); opacity: 0.6; }
        50% { transform: translateY(-20px) translateX(10px); opacity: 1; }
    }

    /* ===== 3D PARTICLE BACKGROUND ===== */
    .particle-bg {
        position: fixed;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
        z-index: -1;
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 50%, #FFFFFF 100%);
        overflow: hidden;
    }

    .particle {
        position: absolute;
        border-radius: 50%;
        animation: particleFloat 8s ease-in-out infinite;
        opacity: 0.4;
        mix-blend-mode: screen;
    }

    /* ===== RESET & BASE ===== */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    .block-container {
        padding-top: 16px !important;
        margin-top: 0 !important;
        padding-left: 16px !important;
        padding-right: 16px !important;
        padding-bottom: 16px !important;
    }
    
    .stApp {
        background: linear-gradient(180deg, #ecfeff 0%, #ffffff 50%, #eff6ff 100%);
        color: var(--text-primary);
        min-height: 100vh;
        position: relative;
        padding-top: 16px !important;
        padding-bottom: 16px !important;
    }

    .stApp::before {
        content: '';
        position: fixed;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
        background: radial-gradient(circle at 15% 20%, rgba(14, 165, 233, 0.12) 0%, transparent 45%),
                    radial-gradient(circle at 85% 85%, rgba(16, 185, 129, 0.12) 0%, transparent 45%);
        pointer-events: none;
        z-index: 0;
    }

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: rgba(240, 249, 255, 0.98);
        border-right: 1px solid rgba(14, 165, 233, 0.14);
        box-shadow: 0 20px 60px rgba(14, 165, 233, 0.08);
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }
    
    /* ===== TYPOGRAPHY ===== */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        margin-top: 0 !important;
        animation: fadeInUp 0.8s ease both;
    }
    
    p, span, li {
        color: var(--text-secondary) !important;
    }
    
    /* ===== PREMIUM CARDS ===== */
    .card {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid var(--border);
        border-radius: 28px;
        padding: 16px;
        box-shadow: var(--shadow-md);
        backdrop-filter: blur(16px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInScale 0.6s ease both;
        margin-bottom: 16px;
    }
    
    .card:hover {
        border-color: rgba(14, 165, 233, 0.35);
        box-shadow: 0 30px 80px rgba(14, 165, 233, 0.15);
        transform: translateY(-6px);
    }
    
    /* ===== METRIC CARDS WITH 3D EFFECT ===== */
    .metric-card {
        background: linear-gradient(180deg, rgba(14, 165, 233, 0.12), rgba(249, 207, 232, 0.14));
        border: 1px solid rgba(14, 165, 233, 0.16);
        border-radius: 24px;
        padding: 14px 12px;
        text-align: center;
        box-shadow: var(--shadow-sm);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.45) 0%, transparent 70%);
        animation: rotate3d 20s linear infinite;
        pointer-events: none;
    }
    
    .metric-card:hover {
        border-color: var(--accent);
        box-shadow: 0 30px 60px rgba(14, 165, 233, 0.22);
        transform: translateY(-4px) scale(1.02);
    }
    
    .metric-value {
        font-size: 2.5em;
        font-weight: 800;
        color: var(--primary);
        margin: 6px 0;
        text-align: center;
        display: block;
        background: linear-gradient(135deg, var(--primary), var(--accent));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        color: var(--text-secondary);
        font-size: 0.85em;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        text-align: center;
    }
    
    /* ===== PREMIUM BUTTON ===== */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 999px !important;
        font-weight: 700 !important;
        padding: 10px 20px !important;
        font-size: 1.05em !important;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 16px 32px rgba(14, 165, 233, 0.22) !important;
        text-transform: none;
        letter-spacing: 0.45px;
        min-height: 46px !important;
        position: relative;
        overflow: hidden;
        margin-top: 8px;
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    /* ===== COLUMN SPACING ===== */
    .stColumn {
        padding: 0 6px;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 24px 48px rgba(14, 165, 233, 0.3) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    .stButton>button>span, .stButton>button>div {
        color: white !important;
    }
    
    /* ===== UPLOAD AREA ===== */
    [data-testid="stFileUploader"] {
        border: none !important;
        border-radius: var(--radius) !important;
        padding: 12px !important;
        background: transparent !important;
    }
    
    [data-testid="stFileUploader"] * {
        color: var(--text-primary) !important;
    }
    
    /* ===== SELECTBOX ===== */
    [data-testid="stSelectbox"] > div > div > div {
        background: #FFFFFF !important;
        border: 1px solid rgba(14, 165, 233, 0.16) !important;
        border-radius: var(--radius) !important;
        color: #111827 !important;
        padding: 10px 12px !important;
        box-shadow: 0 12px 24px rgba(14, 165, 233, 0.08) !important;
        font-weight: 600 !important;
        min-height: 48px !important;
        transition: all 0.3s ease !important;
        margin-bottom: 16px !important;
    }
    
    [data-testid="stSelectbox"] > div > div > div:hover,
    [data-testid="stSelectbox"] > div > div > div:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 14px 28px rgba(14, 165, 233, 0.15) !important;
    }
    
    [data-testid="stSelectbox"] select {
        color: #111827 !important;
    }
    
    /* ===== RADIO BUTTONS ===== */
    [data-testid="stRadio"] {
        background: var(--primary-light);
        padding: 8px 12px;
        border-radius: var(--radius);
        border: 1px solid #D1D5FE;
        margin-bottom: 12px;
    }
    
    /* ===== INPUT FIELDS ===== */
    input, textarea, select {
        background: var(--bg-white) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 8px 12px !important;
        transition: all 0.3s ease !important;
        line-height: 1.5 !important;
        margin-bottom: 12px !important;
    }
    
    input:focus, textarea:focus, select:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.1) !important;
        outline: none !important;
    }
    
    /* ===== SUCCESS/ERROR BOXES ===== */
    .success-box {
        background: linear-gradient(135deg, var(--success-light) 0%, rgba(34, 197, 94, 0.05) 100%);
        border: 2px solid #A7F3D0;
        border-left: 4px solid var(--success);
        padding: 16px 20px;
        border-radius: var(--radius);
        color: #065F46;
        animation: slideInRight 0.6s ease both;
        margin-bottom: 16px;
    }
    
    .success-box h3, .success-box h4 {
        color: var(--success) !important;
        margin-top: 0;
    }
    
    .success-box ul {
        margin: 12px 0 0 0;
        padding-left: 20px;
    }
    
    .success-box li {
        margin-bottom: 6px;
    }
    
    .error-box {
        background: linear-gradient(135deg, var(--danger-light) 0%, rgba(239, 68, 68, 0.05) 100%);
        border: 2px solid #FECACA;
        border-left: 4px solid var(--danger);
        padding: 16px 20px;
        border-radius: var(--radius);
        color: #7F1D1D;
        animation: slideInRight 0.6s ease both;
        margin-bottom: 16px;
    }
    
    .error-box h3, .error-box h4 {
        color: var(--danger) !important;
        margin-top: 0;
    }
    
    .error-box ul {
        margin: 12px 0 0 0;
        padding-left: 20px;
    }
    
    .error-box li {
        margin-bottom: 6px;
    }
    
    .info-box {
        background: linear-gradient(135deg, var(--primary-light) 0%, rgba(14, 165, 233, 0.05) 100%);
        border: 1px solid #D1D5FE;
        border-left: 4px solid var(--primary);
        padding: 16px 20px;
        border-radius: var(--radius);
        color: #1E3A8A;
        animation: slideInLeft 0.6s ease both;
        margin-bottom: 16px;
    }
    
    .info-box ul {
        margin: 12px 0 0 0;
        padding-left: 20px;
    }
    
    .info-box li {
        margin-bottom: 6px;
    }
    
    /* ===== DIVIDER ===== */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border) 20%, var(--border) 80%, transparent);
        margin: 20px 0;
    }
    
    /* ===== SECTION HEADER ===== */
    .section-header {
        font-size: 2em;
        font-weight: 800;
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 32px 0 20px 0;
        padding-bottom: 12px;
        border-bottom: 2px solid var(--primary);
        animation: fadeInUp 0.8s ease both;
    }
    
    h1, h2, h3 {
        margin-bottom: 12px !important;
    }
    
    p {
        margin-bottom: 12px !important;
    }
    
    /* ===== HERO SECTION WITH 3D EFFECT ===== */
    .hero {
        position: relative;
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.12), rgba(16, 185, 129, 0.08));
        border: 1px solid rgba(14, 165, 233, 0.16);
        border-radius: 34px;
        padding: 24px 28px;
        overflow: hidden;
        margin-bottom: 32px;
        box-shadow: var(--shadow-lg);
        animation: fadeInUp 0.9s ease both;
    }

    .hero::after {
        content: '';
        position: absolute;
        width: 420px;
        height: 420px;
        top: -120px;
        right: -120px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(14, 165, 233, 0.24), transparent 62%);
        pointer-events: none;
        animation: rotate3d 40s linear infinite;
    }

    .hero::before {
        content: '';
        position: absolute;
        width: 260px;
        height: 260px;
        bottom: -90px;
        left: -90px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(16, 185, 129, 0.24), transparent 70%);
        pointer-events: none;
        animation: rotate3d 60s linear infinite reverse;
    }

    .hero h1 {
        font-size: 3.2em;
        margin-bottom: 12px;
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: fadeInUp 1s ease both;
    }
    
    .hero p {
        font-size: 1.05em;
        color: var(--text-secondary);
        margin: 8px 0;
        line-height: 1.75;
        animation: fadeInUp 1.1s ease both;
    }

    .feature-chip {
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid rgba(14, 165, 233, 0.16);
        border-radius: 999px;
        color: var(--text-primary);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95em;
        font-weight: 600;
        padding: 8px 14px;
        margin-right: 12px;
        margin-bottom: 12px;
        transition: all 0.3s ease;
        animation: fadeInScale 0.8s ease both;
    }

    .feature-chip:hover {
        background: rgba(14, 165, 233, 0.1);
        border-color: var(--primary);
        transform: scale(1.05);
    }

    /* ===== INFO GRID ===== */
    .info-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 20px;
        margin-top: 24px;
        padding: 0 12px;
        animation: fadeInUp 1s ease both;
    }

    .info-card {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid rgba(14, 165, 233, 0.14);
        border-radius: 28px;
        padding: 16px;
        box-shadow: var(--shadow-sm);
        min-height: 200px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInScale 0.6s ease both;
        position: relative;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }
    
    .info-card h4 {
        margin-bottom: 16px;
        width: 100%;
    }

    .info-card::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(14, 165, 233, 0.1) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.3s ease;
        pointer-events: none;
    }

    .info-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 24px 60px rgba(14, 165, 233, 0.15);
        border-color: rgba(14, 165, 233, 0.25);
    }

    .info-card:hover::after {
        opacity: 1;
    }

    .info-card h4 {
        margin-bottom: 14px;
        color: var(--primary-dark);
        position: relative;
        z-index: 1;
    }

    /* ===== BADGE ===== */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(72, 181, 255, 0.14);
        border: 1px solid rgba(72, 181, 255, 0.25);
        border-radius: 999px;
        padding: 10px 16px;
        color: var(--text-primary);
        font-weight: 700;
        margin-right: 8px;
        margin-bottom: 12px;
        animation: fadeInScale 0.8s ease both;
    }

    /* ===== STEP CARDS ===== */
    .step-card {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid rgba(14, 165, 233, 0.12);
        border-radius: 28px;
        padding: 20px;
        box-shadow: var(--shadow-sm);
        min-height: 220px;
        position: relative;
        overflow: hidden;
        transition: all 0.4s ease;
        margin-bottom: 16px;
    }

    .step-card::after {
        content: '';
        position: absolute;
        width: 140px;
        height: 140px;
        background: rgba(14, 165, 233, 0.08);
        border-radius: 50%;
        right: -40px;
        bottom: -40px;
        z-index: 0;
        animation: rotate3d 30s linear infinite;
    }

    .step-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 50px rgba(14, 165, 233, 0.12);
    }

    .step-card h4 {
        position: relative;
        z-index: 1;
    }

    .step-card p, .step-card ul {
        position: relative;
        z-index: 1;
    }

    /* ===== TIMELINE CARD ===== */
    .timeline-card {
        display: flex;
        gap: 16px;
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.14), rgba(72, 181, 255, 0.08));
        border: 1px solid rgba(14, 165, 233, 0.25);
        border-radius: 28px;
        padding: 24px;
        box-shadow: var(--shadow-sm);
        align-items: flex-start;
        transition: all 0.3s ease;
        animation: slideInLeft 0.6s ease both;
        margin-bottom: 16px;
    }

    .timeline-card:hover {
        border-color: var(--primary);
        box-shadow: var(--shadow-md);
        transform: translateX(4px);
    }

    .timeline-card div:first-child {
        min-width: 52px;
        min-height: 52px;
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
        border-radius: 50%;
        color: white;
        display: grid;
        place-items: center;
        font-size: 1.2em;
        box-shadow: 0 14px 28px rgba(14, 165, 233, 0.2);
    }

    .timeline-card div:last-child {
        flex: 1;
    }

    .timeline-card h4 {
        margin: 0 0 10px 0;
        color: #111827;
        font-size: 1.07em;
    }

    .timeline-card p {
        margin: 0;
        color: #475569;
        line-height: 1.7;
    }

    /* ===== UTILITY ===== */
    .grid-col {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 20px;
    }
    
    /* ===== BADGE ===== */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(72, 181, 255, 0.14);
        border: 1px solid rgba(72, 181, 255, 0.25);
        border-radius: 999px;
        padding: 10px 16px;
        color: var(--text-primary);
        font-weight: 700;
        margin-right: 12px;
        margin-bottom: 12px;
        animation: fadeInScale 0.8s ease both;
    }

    .animated-panel {
        animation: floatUp 6s ease-in-out infinite;
    }

    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {
        .info-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 16px;
        }
        
        .grid-col {
            grid-template-columns: 1fr;
            gap: 16px;
        }

        .hero {
            padding: 16px 20px;
            margin-bottom: 24px;
        }

        .hero h1 {
            font-size: 2.8em;
            margin-bottom: 8px;
        }
        
        .hero p {
            margin: 6px 0;
        }

        .section-header {
            font-size: 1.8em;
            margin: 24px 0 16px 0;
        }
        
        .block-container {
            padding-left: 12px !important;
            padding-right: 12px !important;
        }
    }
    
    @media (max-width: 480px) {
        .info-grid {
            grid-template-columns: 1fr;
            gap: 12px;
            padding: 0 8px;
        }
        
        .grid-col {
            grid-template-columns: 1fr;
            gap: 12px;
        }
        
        .hero {
            padding: 12px 16px;
            margin-bottom: 16px;
        }
        
        .hero h1 {
            font-size: 2em;
            margin-bottom: 6px;
        }
        
        .hero p {
            margin: 4px 0;
            font-size: 0.95em;
        }
        
        .feature-chip {
            padding: 6px 10px;
            font-size: 0.85em;
            margin-bottom: 8px;
        }
        
        .section-header {
            font-size: 1.4em;
            margin: 16px 0 12px 0;
        }
        
        .step-card {
            padding: 16px;
            min-height: auto;
        }
        
        .info-card {
            padding: 12px;
            min-height: auto;
        }
        
        .card {
            padding: 12px;
        }
        
        .timeline-card div:first-child {
            min-width: 44px;
            min-height: 44px;
        }
        
        .block-container {
            padding-left: 8px !important;
            padding-right: 8px !important;
            padding-top: 12px !important;
            padding-bottom: 12px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ===== 3D VISUALIZATION HELPER FUNCTIONS =====
def create_3d_lung_model():
    """Create an interactive 3D rotating lung model visualization"""
    x = np.linspace(-1, 1, 20)
    y = np.linspace(-1, 1, 20)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2) * 3) * 0.5
    
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis',
        showscale=False,
        contours={"z": {"show": True, "usecolorscale": True, "highlightcolor": "limegreen", "project": {"z": True}}}
    )])
    
    fig.update_layout(
        title="🫁 3D Lung Analysis Model",
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            zaxis=dict(showgrid=False, zeroline=False),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        height=500,
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor='rgba(247, 248, 255, 0.5)',
        plot_bgcolor='rgba(247, 248, 255, 0.5)',
        showlegend=False,
        font=dict(size=12, color='#0ea5e9')
    )
    
    return fig

def create_healthy_lung_3d():
    """Create a 3D visualization of a healthy lung"""
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = 0.8 * np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    fig = go.Figure(data=[go.Surface(
        x=x, y=y, z=z,
        colorscale='Blues',
        showscale=False,
        hoverinfo='skip',
        contours={"z": {"show": True, "usecolorscale": False, "color": "rgba(150, 200, 255, 0.3)"}}
    )])
    
    fig.update_layout(
        title="✅ Healthy Lung (Normal)",
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showgrid=False, zeroline=False, visible=False),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        height=450,
        margin=dict(l=0, r=0, b=0, t=50),
        paper_bgcolor='rgba(226, 255, 243, 0.6)',
        plot_bgcolor='rgba(226, 255, 243, 0.6)',
        showlegend=False,
        font=dict(size=12, color='#22C55E', family='Arial Black')
    )
    
    return fig

def create_pneumonia_lung_3d():
    """Create a 3D visualization of a pneumonia-infected lung with inflamed areas"""
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = 0.8 * np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Add inflammation bump in certain areas
    infection_mask = np.exp(-((x-0.3)**2 + (y-0.2)**2) * 5) * 0.3
    z_infected = z + infection_mask
    
    # Create color array - red for infected areas
    colors = np.ones_like(z_infected)
    colors[infection_mask > 0.1] = 0.2
    
    fig = go.Figure(data=[go.Surface(
        x=x, y=y, z=z_infected,
        colorscale='Reds',
        showscale=False,
        hoverinfo='skip',
        contours={"z": {"show": True, "usecolorscale": False, "color": "rgba(255, 100, 100, 0.3)"}}
    )])
    
    # Add infection hotspot as sphere
    u_hot = np.linspace(0, 2 * np.pi, 30)
    v_hot = np.linspace(0, np.pi, 30)
    x_hot = 0.15 * np.outer(np.cos(u_hot), np.sin(v_hot)) + 0.3
    y_hot = 0.15 * np.outer(np.sin(u_hot), np.sin(v_hot)) + 0.2
    z_hot = 0.15 * np.outer(np.ones(np.size(u_hot)), np.cos(v_hot))
    
    fig.add_trace(go.Surface(
        x=x_hot, y=y_hot, z=z_hot,
        colorscale='Reds',
        showscale=False,
        hoverinfo='skip',
        opacity=0.9
    ))
    
    fig.update_layout(
        title="⚠️ Pneumonia-Infected Lung (Abnormal)",
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showgrid=False, zeroline=False, visible=False),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        height=450,
        margin=dict(l=0, r=0, b=0, t=50),
        paper_bgcolor='rgba(255, 243, 243, 0.6)',
        plot_bgcolor='rgba(255, 243, 243, 0.6)',
        showlegend=False,
        font=dict(size=12, color='#EF4444', family='Arial Black')
    )
    
    return fig

def create_3d_confidence_visualization(confidence: float, is_pneumonia: bool):
    """Create a 3D cone/sphere visualization showing confidence level"""
    theta = np.linspace(0, 2*np.pi, 30)
    z = np.linspace(0, 1, 30)
    Theta, Z = np.meshgrid(theta, z)
    
    # Create a cone shape that scales with confidence
    R = (1 - Z) * (confidence + 0.3)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    
    color_scale = 'Reds' if is_pneumonia else 'Greens'
    
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z,
        colorscale=color_scale,
        showscale=True,
        colorbar=dict(title=f"{'Risk' if is_pneumonia else 'Health'}<br>Level", tickformat=".0%")
    )])
    
    fig.update_layout(
        title=f"📊 Confidence: {confidence*100:.1f}%",
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            zaxis=dict(showgrid=False, zeroline=False),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        height=450,
        margin=dict(l=0, r=0, b=0, t=50),
        paper_bgcolor='rgba(247, 248, 255, 0.5)',
        plot_bgcolor='rgba(247, 248, 255, 0.5)',
        showlegend=False,
        hovermode=False,
        font=dict(size=12, color='#0ea5e9')
    )
    
    return fig

def create_floating_medical_icons():
    """Create animated floating medical icons in 3D space"""
    # Create multiple floating medical symbols
    symbols = ['🫁', '💊', '🩺', '🧬', '🔬', '⚕️', '🩸', '🦠']
    positions = [
        (0, 0, 0), (1, 1, 0.5), (-1, 1, 0.3), (1, -1, 0.7),
        (-1, -1, 0.4), (0, 1.5, 0.6), (0, -1.5, 0.2), (1.5, 0, 0.8)
    ]
    
    fig = go.Figure()
    
    for i, (symbol, (x, y, z)) in enumerate(zip(symbols, positions)):
        # Create a small sphere for each icon
        u = np.linspace(0, 2*np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_sphere = 0.1 * np.outer(np.cos(u), np.sin(v)) + x
        y_sphere = 0.1 * np.outer(np.sin(u), np.sin(v)) + y
        z_sphere = 0.1 * np.outer(np.ones(np.size(u)), np.cos(v)) + z
        
        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            colorscale=['#0ea5e9', '#22d3ee'],
            showscale=False,
            hoverinfo='text',
            hovertext=f"{symbol} Medical Technology",
            opacity=0.8
        ))
    
    fig.update_layout(
        title="✨ Floating Medical Ecosystem",
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showgrid=False, zeroline=False, visible=False),
            camera=dict(eye=dict(x=2, y=2, z=2))
        ),
        height=400,
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor='rgba(247, 248, 255, 0.8)',
        plot_bgcolor='rgba(247, 248, 255, 0.8)',
        showlegend=False,
        font=dict(size=14, color='#0ea5e9', family='Arial Black')
    )
    
    return fig

def create_dna_helix_3d():
    """Create a 3D DNA helix visualization representing genetic research"""
    t = np.linspace(0, 4*np.pi, 100)
    
    # DNA double helix
    x1 = np.cos(t)
    y1 = np.sin(t)
    z1 = t * 0.3
    
    x2 = np.cos(t + np.pi)
    y2 = np.sin(t + np.pi)
    z2 = t * 0.3
    
    # Connecting rungs
    x_rungs = []
    y_rungs = []
    z_rungs = []
    
    for i in range(0, len(t), 5):
        x_rungs.extend([x1[i], x2[i], None])
        y_rungs.extend([y1[i], y2[i], None])
        z_rungs.extend([z1[i], z2[i], None])
    
    fig = go.Figure()
    
    # Add the two DNA strands
    fig.add_trace(go.Scatter3d(
        x=x1, y=y1, z=z1,
        mode='lines',
        line=dict(color='#0ea5e9', width=8),
        name='DNA Strand 1'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=x2, y=y2, z=z2,
        mode='lines',
        line=dict(color='#22d3ee', width=8),
        name='DNA Strand 2'
    ))
    
    # Add connecting rungs
    fig.add_trace(go.Scatter3d(
        x=x_rungs, y=y_rungs, z=z_rungs,
        mode='lines',
        line=dict(color='#22C55E', width=4),
        name='Base Pairs'
    ))
    
    fig.update_layout(
        title="🧬 DNA Research & Genetic Analysis",
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showgrid=False, zeroline=False, visible=False),
            camera=dict(eye=dict(x=2, y=2, z=1))
        ),
        height=400,
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor='rgba(247, 248, 255, 0.8)',
        plot_bgcolor='rgba(247, 248, 255, 0.8)',
        showlegend=False,
        font=dict(size=14, color='#0ea5e9', family='Arial Black')
    )
    
    return fig

def create_lung_cross_section_3d():
    """Create educational 3D lung cross-section showing alveoli structure"""
    # Create alveoli-like structures
    fig = go.Figure()
    
    # Main lung outline
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_main = 1.2 * np.outer(np.cos(u), np.sin(v))
    y_main = 1.2 * np.outer(np.sin(u), np.sin(v))
    z_main = 0.8 * np.outer(np.ones(np.size(u)), np.cos(v))
    
    fig.add_trace(go.Surface(
        x=x_main, y=y_main, z=z_main,
        colorscale='Blues',
        showscale=False,
        opacity=0.3,
        hoverinfo='text',
        hovertext='Lung Tissue Structure'
    ))
    
    # Add alveoli (small air sacs)
    for i in range(8):
        angle = i * np.pi / 4
        x_center = 0.6 * np.cos(angle)
        y_center = 0.6 * np.sin(angle)
        z_center = np.random.uniform(-0.3, 0.3)
        
        u_alv = np.linspace(0, 2*np.pi, 20)
        v_alv = np.linspace(0, np.pi, 20)
        x_alv = 0.15 * np.outer(np.cos(u_alv), np.sin(v_alv)) + x_center
        y_alv = 0.15 * np.outer(np.sin(u_alv), np.sin(v_alv)) + y_center
        z_alv = 0.15 * np.outer(np.ones(np.size(u_alv)), np.cos(v_alv)) + z_center
        
        fig.add_trace(go.Surface(
            x=x_alv, y=y_alv, z=z_alv,
            colorscale='Greens',
            showscale=False,
            opacity=0.7,
            hoverinfo='text',
            hovertext='Alveoli - Air Sacs for Gas Exchange'
        ))
    
    # Add bronchioles
    t = np.linspace(0, 2, 50)
    for i in range(4):
        angle = i * np.pi / 2
        x_br = t * np.cos(angle) * 0.8
        y_br = t * np.sin(angle) * 0.8
        z_br = np.zeros_like(t)
        
        fig.add_trace(go.Scatter3d(
            x=x_br, y=y_br, z=z_br,
            mode='lines',
            line=dict(color='#F97316', width=6),
            hoverinfo='text',
            hovertext='Bronchiole - Airway Passage'
        ))
    
    fig.update_layout(
        title="🔬 Lung Cross-Section: Alveoli & Airways",
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showgrid=False, zeroline=False, visible=False),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=450,
        margin=dict(l=0, r=0, b=0, t=50),
        paper_bgcolor='rgba(226, 255, 243, 0.6)',
        plot_bgcolor='rgba(226, 255, 243, 0.6)',
        showlegend=False,
        font=dict(size=12, color='#22C55E', family='Arial Black')
    )
    
    return fig

def create_infection_progression_3d():
    """Create 3D visualization showing pneumonia infection progression"""
    fig = go.Figure()
    
    # Healthy lung base
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    x_base = 0.8 * np.outer(np.cos(u), np.sin(v))
    y_base = 0.8 * np.outer(np.sin(u), np.sin(v))
    z_base = 0.8 * np.outer(np.ones(np.size(u)), np.cos(v))
    
    fig.add_trace(go.Surface(
        x=x_base, y=y_base, z=z_base,
        colorscale='Blues',
        showscale=False,
        opacity=0.6,
        hoverinfo='text',
        hovertext='Healthy Lung Tissue'
    ))
    
    # Early infection (small red areas)
    early_infection = [
        (0.3, 0.2, 0.1), (-0.2, 0.4, -0.1), (0.1, -0.3, 0.2)
    ]
    
    for x_c, y_c, z_c in early_infection:
        u_inf = np.linspace(0, 2*np.pi, 20)
        v_inf = np.linspace(0, np.pi, 20)
        x_inf = 0.08 * np.outer(np.cos(u_inf), np.sin(v_inf)) + x_c
        y_inf = 0.08 * np.outer(np.sin(u_inf), np.sin(v_inf)) + y_c
        z_inf = 0.08 * np.outer(np.ones(np.size(u_inf)), np.cos(v_inf)) + z_c
        
        fig.add_trace(go.Surface(
            x=x_inf, y=y_inf, z=z_inf,
            colorscale='Reds',
            showscale=False,
            opacity=0.9,
            hoverinfo='text',
            hovertext='Early Infection - Bacterial/Viral Entry'
        ))
    
    # Advanced infection (larger inflamed areas)
    advanced_infection = [
        (0.5, -0.1, 0.0), (-0.4, -0.2, 0.3)
    ]
    
    for x_c, y_c, z_c in advanced_infection:
        u_adv = np.linspace(0, 2*np.pi, 25)
        v_adv = np.linspace(0, np.pi, 25)
        x_adv = 0.12 * np.outer(np.cos(u_adv), np.sin(v_adv)) + x_c
        y_adv = 0.12 * np.outer(np.sin(u_adv), np.sin(v_adv)) + y_c
        z_adv = 0.12 * np.outer(np.ones(np.size(u_adv)), np.cos(v_adv)) + z_c
        
        fig.add_trace(go.Surface(
            x=x_adv, y=y_adv, z=z_adv,
            colorscale='Reds',
            showscale=False,
            opacity=0.95,
            hoverinfo='text',
            hovertext='Advanced Infection - Fluid Accumulation'
        ))
    
    fig.update_layout(
        title="🔥 Pneumonia Progression: Infection Stages",
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showgrid=False, zeroline=False, visible=False),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2))
        ),
        height=450,
        margin=dict(l=0, r=0, b=0, t=50),
        paper_bgcolor='rgba(255, 243, 243, 0.6)',
        plot_bgcolor='rgba(255, 243, 243, 0.6)',
        showlegend=False,
        font=dict(size=12, color='#EF4444', family='Arial Black')
    )
    
    return fig

def create_neural_network_3d():
    """Create 3D neural network visualization for AI/tech theme"""
    fig = go.Figure()
    
    # Input layer
    input_nodes = []
    for i in range(8):
        z = (i - 3.5) * 0.3
        input_nodes.append((0, 0, z))
        # Add node spheres
        u = np.linspace(0, 2*np.pi, 15)
        v = np.linspace(0, np.pi, 15)
        x_node = 0.05 * np.outer(np.cos(u), np.sin(v))
        y_node = 0.05 * np.outer(np.sin(u), np.sin(v)) - 1.5
        z_node = 0.05 * np.outer(np.ones(np.size(u)), np.cos(v)) + z
        
        fig.add_trace(go.Surface(
            x=x_node, y=y_node, z=z_node,
            colorscale='Purples',
            showscale=False,
            opacity=0.8,
            hoverinfo='text',
            hovertext='Input Neuron - Data Reception'
        ))
    
    # Hidden layers
    for layer in range(1, 4):
        x_pos = layer * 0.8
        num_nodes = 6 if layer < 3 else 4
        
        for i in range(num_nodes):
            z = (i - (num_nodes-1)/2) * 0.4
            # Add node spheres
            u = np.linspace(0, 2*np.pi, 15)
            v = np.linspace(0, np.pi, 15)
            x_node = 0.05 * np.outer(np.cos(u), np.sin(v)) + x_pos
            y_node = 0.05 * np.outer(np.sin(u), np.sin(v)) - 1.5 + layer * 0.1
            z_node = 0.05 * np.outer(np.ones(np.size(u)), np.cos(v)) + z
            
            fig.add_trace(go.Surface(
                x=x_node, y=y_node, z=z_node,
                colorscale='Blues',
                showscale=False,
                opacity=0.8,
                hoverinfo='text',
                hovertext=f'Hidden Layer {layer} - Feature Processing'
            ))
    
    # Output layer
    for i in range(2):
        z = (i - 0.5) * 0.6
        u = np.linspace(0, 2*np.pi, 15)
        v = np.linspace(0, np.pi, 15)
        x_node = 0.05 * np.outer(np.cos(u), np.sin(v)) + 3.2
        y_node = 0.05 * np.outer(np.sin(u), np.sin(v)) - 1.5 + 0.3
        z_node = 0.05 * np.outer(np.ones(np.size(u)), np.cos(v)) + z
        
        color = 'Reds' if i == 0 else 'Greens'
        label = 'Pneumonia Detection' if i == 0 else 'Normal Result'
        
        fig.add_trace(go.Surface(
            x=x_node, y=y_node, z=z_node,
            colorscale=color,
            showscale=False,
            opacity=0.9,
            hoverinfo='text',
            hovertext=f'Output: {label}'
        ))
    
    # Add connection lines (simplified)
    connections = [
        (0, -1.5, 0, 0.8, -1.4, 0),  # Input to hidden
        (0.8, -1.4, 0, 1.6, -1.3, 0),  # Hidden to hidden
        (1.6, -1.3, 0, 2.4, -1.2, 0),  # Hidden to hidden
        (2.4, -1.2, 0, 3.2, -1.2, 0),  # Hidden to output
    ]
    
    for x1, y1, z1, x2, y2, z2 in connections:
        fig.add_trace(go.Scatter3d(
            x=[x1, x2], y=[y1, y2], z=[z1, z2],
            mode='lines',
            line=dict(color='#0ea5e9', width=2),
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title="🧠 AI Neural Network Architecture",
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showgrid=False, zeroline=False, visible=False),
            camera=dict(eye=dict(x=0, y=-3, z=1))
        ),
        height=450,
        margin=dict(l=0, r=0, b=0, t=50),
        paper_bgcolor='rgba(247, 248, 255, 0.8)',
        plot_bgcolor='rgba(247, 248, 255, 0.8)',
        showlegend=False,
        font=dict(size=14, color='#0ea5e9', family='Arial Black')
    )
    
    return fig

def create_tech_portfolio_3d():
    """Create 3D visualization of tech stack and portfolio elements"""
    fig = go.Figure()
    
    # Tech stack icons/symbols
    tech_items = [
        ('🐍', 'Python', 0, 0, 0),
        ('🤖', 'AI/ML', 1, 1, 0.5),
        ('🌐', 'Web Dev', -1, 1, 0.3),
        ('📊', 'Data Science', 1, -1, 0.7),
        ('☁️', 'Cloud', -1, -1, 0.4),
        ('⚛️', 'React', 0, 1.5, 0.6),
        ('🗄️', 'Database', 0, -1.5, 0.2),
        ('🔧', 'DevOps', 1.5, 0, 0.8)
    ]
    
    for emoji, name, x, y, z in tech_items:
        # Create floating cubes for each tech item
        size = 0.15
        x_cube = [x-size, x+size, x+size, x-size, x-size, x+size, x+size, x-size]
        y_cube = [y-size, y-size, y+size, y+size, y-size, y-size, y+size, y+size]
        z_cube = [z-size, z-size, z-size, z-size, z+size, z+size, z+size, z+size]
        
        fig.add_trace(go.Mesh3d(
            x=x_cube, y=y_cube, z=z_cube,
            color='lightblue',
            opacity=0.8,
            hovertext=f"{emoji} {name}",
            hoverinfo='text'
        ))
    
    # Add connecting lines to show relationships
    connections = [
        (0, 0, 0, 1, 1, 0.5),  # Python to AI/ML
        (0, 0, 0, -1, 1, 0.3),  # Python to Web Dev
        (1, 1, 0.5, 1, -1, 0.7),  # AI/ML to Data Science
        (-1, 1, 0.3, -1, -1, 0.4),  # Web Dev to Cloud
    ]
    
    for x1, y1, z1, x2, y2, z2 in connections:
        fig.add_trace(go.Scatter3d(
            x=[x1, x2], y=[y1, y2], z=[z1, z2],
            mode='lines',
            line=dict(color='#22d3ee', width=3),
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title="🚀 Tech Stack & Portfolio Universe",
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showgrid=False, zeroline=False, visible=False),
            camera=dict(eye=dict(x=2, y=2, z=2))
        ),
        height=450,
        margin=dict(l=0, r=0, b=0, t=50),
        paper_bgcolor='rgba(247, 248, 255, 0.8)',
        plot_bgcolor='rgba(247, 248, 255, 0.8)',
        showlegend=False,
        font=dict(size=14, color='#0ea5e9', family='Arial Black')
    )
    
    return fig

ROOT_DIR = SCRIPT_DIR
DATA_PATH = ROOT_DIR / "sample"
MODEL_PATH = ROOT_DIR / "model" / "resnet_model.h5"
LOGO_URL = "https://img.icons8.com/color/96/lungs.png"

@st.cache_data(show_spinner=False)
def load_sample_data(filename: str):
    return load_demo_image(DATA_PATH / filename)

model = None
if MODEL_PATH.exists():
    model = load_pneumonia_model(MODEL_PATH)

# ==================== UNIFIED SIDEBAR ====================

with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; margin: 12px auto 16px auto; max-width: 180px;'>
        <img src='{LOGO_URL}' style='width: 110px; height: 110px; border-radius: 50%; object-fit: cover; border: 3px solid #0ea5e9; box-shadow: 0 14px 30px rgba(14, 165, 233, 0.18); display: block; margin: 0 auto;'>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin: 0 0 16px 0; padding: 0 12px;'>
        <h2 style='margin: 0 0 8px 0; font-size: 1.6em;'>Pneumonia Detection</h2>
        <p style='font-size: 0.9em; margin: 0;'>Modern health diagnostics powered by AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='divider' style='margin: 12px 0;'></div>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align: center; font-size: 1em; margin: 20px 0 16px 0;'>📊 Quick Stats</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.markdown("""<div class='metric-card'>
            <div class='metric-label'>Accuracy</div>
            <div class='metric-value' style='font-size: 2em;'>95%</div>
        </div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""<div class='metric-card'>
            <div class='metric-label'>Dataset</div>
            <div class='metric-value' style='font-size: 2em;'>5.8K</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("<div class='divider' style='margin: 16px 0;'></div>", unsafe_allow_html=True)

    if model is None:
        st.error("Model file not found. Please add `app/model/resnet_model.h5` to enable prediction.")
    
    st.markdown("<h3 style='text-align: center; font-size: 1em; margin: 20px 0 16px 0;'>Navigation</h3>", unsafe_allow_html=True)
    page = st.radio(
        "Select:",
        ["🏥 Analyze", "📚 Learn", "👨‍💼 About"],
        label_visibility="collapsed"
    )
    
    st.markdown("<div class='divider' style='margin: 16px 0;'></div>", unsafe_allow_html=True)
    
    st.markdown("""<div class='info-box'>
        <strong>⚠️ Disclaimer</strong><br>
        <span style='font-size: 0.9em;'>For educational purposes only. Always consult healthcare professionals.</span>
    </div>""", unsafe_allow_html=True)
    
    st.markdown("""<div style='text-align: center; margin-top: 16px; padding-top: 12px; border-top: 1px solid var(--border);'>
        <p style='font-size: 0.9em; margin: 0 0 6px 0;'>Developed by</p>
        <p style='color: var(--primary); font-weight: 700; margin: 0 0 8px 0;'>Shubham Singh</p>
        <p style='font-size: 0.85em; margin: 0;'>© 2026</p>
    </div>""", unsafe_allow_html=True)

# ==================== MAIN CONTENT ====================

# ==================== HOME PAGE ====================
if page not in ["🏥 Analyze", "📚 Learn", "👨‍💼 About"]:
    # Hero Section with Premium Design
    st.markdown("""<div class='hero'>
        <div style='display:flex; align-items:center; justify-content:space-between; gap:24px; flex-wrap:wrap;'>
            <div style='max-width:620px; flex: 1; min-width: 250px;'>
                <h1>🫁 Pneumonia Detection AI</h1>
                <p style='font-size:1.05em; margin: 12px 0; color:var(--text-secondary); line-height:1.75;'>
                    Advanced artificial intelligence for early pneumonia detection through chest X-ray analysis. Empowering healthcare with cutting-edge technology.
                </p>
                <div style='display:grid; grid-template-columns:repeat(2,minmax(160px,1fr)); gap:12px; margin-top:20px;'>
                    <div class='feature-chip'>🧠 Deep Learning AI</div>
                    <div class='feature-chip'>📈 95% Accuracy</div>
                    <div class='feature-chip'>⚡ Real-time Analysis</div>
                    <div class='feature-chip'>🔬 Medical Research</div>
                </div>
            </div>
            <div class='info-card animated-panel' style='min-width:240px; max-width:320px;'>
                <h3 style='margin:0 0 12px 0; color:var(--primary-dark);'>Welcome to Advanced Diagnostics</h3>
                <p style='margin:0; color:var(--text-secondary); font-size:0.95em;'>Explore our interactive 3D visualizations and learn how AI revolutionizes pneumonia detection in modern healthcare.</p>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # 3D Model Display Section
    st.markdown("<div style='margin-top: 0px;'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; margin-bottom: 20px; background: linear-gradient(135deg, #0ea5e9 0%, #22d3ee 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;'>🫁 Explore 3D Lung Model</h3>", unsafe_allow_html=True)

    col_3d_intro1, col_3d_intro2 = st.columns([1.2, 1])

    with col_3d_intro1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(109,76,255,0.08), rgba(72,181,255,0.04)); padding: 24px; border-radius: 20px; border: 1px solid rgba(109,76,255,0.15);'>
            <h4 style='margin-top: 0; color: #0ea5e9;'>Advanced 3D Visualization</h4>
            <p>Explore interactive 3D representations of chest analysis. Click and drag to rotate, scroll to zoom. Our system converts X-ray findings into intuitive spatial models for better understanding.</p>
            <ul style='margin: 12px 0 0 0; padding-left: 20px;'>
                <li><strong>Rotate:</strong> Click and drag the visualization</li>
                <li><strong>Zoom:</strong> Scroll your mouse wheel</li>
                <li><strong>Interactive:</strong> Explore from all angles</li>
                <li><strong>Real-time:</strong> Instant 3D rendering</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_3d_intro2:
        try:
            fig_lung_intro = create_3d_lung_model()
            st.plotly_chart(fig_lung_intro, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
        except Exception as e:
            st.warning("3D Model loading... please ensure Plotly is properly installed")

    # ===== HEALTHY VS INFECTED LUNGS COMPARISON =====
    st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; margin-bottom: 24px; background: linear-gradient(135deg, #22C55E 0%, #EF4444 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-size: 1.8em;'>Compare: Healthy vs Infected Lungs</h3>", unsafe_allow_html=True)

    col_healthy, col_infected = st.columns(2, gap="large")

    with col_healthy:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(34,197,94,0.1), rgba(34,197,94,0.05)); padding: 20px; border-radius: 16px; border: 2px solid rgba(34,197,94,0.3); margin-bottom: 16px;'>
            <h4 style='margin-top: 0; color: #22C55E; text-align: center;'>✅ HEALTHY LUNGS</h4>
            <p style='margin: 8px 0; font-size: 0.95em;'>🫁 Clear respiratory passages<br/>💚 Optimal oxygen flow<br/>⭐ No inflammation detected<br/>✓ Normal chest imaging</p>
        </div>
        """, unsafe_allow_html=True)
        try:
            fig_healthy = create_healthy_lung_3d()
            st.plotly_chart(fig_healthy, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
        except Exception as e:
            st.error("Unable to load healthy lung model")

    with col_infected:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.05)); padding: 20px; border-radius: 16px; border: 2px solid rgba(239,68,68,0.3); margin-bottom: 16px;'>
            <h4 style='margin-top: 0; color: #EF4444; text-align: center;'>⚠️ PNEUMONIA-INFECTED LUNGS</h4>
            <p style='margin: 8px 0; font-size: 0.95em;'>🔴 Fluid/pus in air sacs<br/>❌ Restricted oxygen transfer<br/>🔥 Inflammation present<br/>⚠️ Abnormal patterns detected</p>
        </div>
        """, unsafe_allow_html=True)
        try:
            fig_infected = create_pneumonia_lung_3d()
            st.plotly_chart(fig_infected, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
        except Exception as e:
            st.error("Unable to load pneumonia lung model")

    # ===== NEW: ADDITIONAL AESTHETIC 3D VISUALS =====
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; margin-bottom: 24px; background: linear-gradient(135deg, #0ea5e9 0%, #22d3ee 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-size: 1.8em;'>✨ Advanced Medical Visualizations</h3>", unsafe_allow_html=True)

    col_aesthetic1, col_aesthetic2 = st.columns(2, gap="large")

    with col_aesthetic1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(109,76,255,0.1), rgba(72,181,255,0.05)); padding: 20px; border-radius: 16px; border: 2px solid rgba(109,76,255,0.3); margin-bottom: 16px;'>
            <h4 style='margin-top: 0; color: #0ea5e9; text-align: center;'>🧬 Genetic Research</h4>
            <p style='margin: 8px 0; font-size: 0.95em;'>DNA double helix structure representing the genetic foundation of medical research and personalized healthcare.</p>
        </div>
        """, unsafe_allow_html=True)
        try:
            fig_dna = create_dna_helix_3d()
            st.plotly_chart(fig_dna, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
        except Exception as e:
            st.error("Unable to load DNA visualization")

    with col_aesthetic2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(72,181,255,0.1), rgba(109,76,255,0.05)); padding: 20px; border-radius: 16px; border: 2px solid rgba(72,181,255,0.3); margin-bottom: 16px;'>
            <h4 style='margin-top: 0; color: #22d3ee; text-align: center;'>✨ Medical Technology</h4>
            <p style='margin: 8px 0; font-size: 0.95em;'>Interactive ecosystem of medical symbols and technologies working together for advanced healthcare solutions.</p>
        </div>
        """, unsafe_allow_html=True)
        try:
            fig_medical = create_floating_medical_icons()
            st.plotly_chart(fig_medical, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
        except Exception as e:
            st.error("Unable to load medical icons visualization")

    # ===== PNEUMONIA FACTS SECTION =====
    st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)
    st.markdown("""<div style='background: linear-gradient(135deg, rgba(239,68,68,0.05), rgba(249,115,22,0.05)); padding: 32px; border-radius: 24px; border: 1px solid rgba(239,68,68,0.2); margin: 24px 0;'>
        <h3 style='margin-top: 0; color: #EF4444;'>⚡ Critical Pneumonia Indicators</h3>
        <div style='display: grid; grid-template-columns: repeat(3, minmax(200px, 1fr)); gap: 20px; margin-top: 20px;'>
            <div style='background: white; padding: 16px; border-radius: 12px; border-left: 4px solid #EF4444;'>
                <h5 style='margin: 0 0 8px 0; color: #EF4444;'>🫁 Air Sac Inflammation</h5>
                <p style='margin: 0; font-size: 0.9em; color: #666;'>Alveoli fill with fluid or pus, preventing oxygen absorption and causing respiratory distress.</p>
            </div>
            <div style='background: white; padding: 16px; border-radius: 12px; border-left: 4px solid #F97316;'>
                <h5 style='margin: 0 0 8px 0; color: #F97316;'>🦠 Bacterial/Viral Infection</h5>
                <p style='margin: 0; font-size: 0.9em; color: #666;'>Pathogenic organisms trigger immune response, leading to swelling and fluid accumulation in lungs.</p>
            </div>
            <div style='background: white; padding: 16px; border-radius: 12px; border-left: 4px solid #DC2626;'>
                <h5 style='margin: 0 0 8px 0; color: #DC2626;'>🔥 Rapid Progression Risk</h5>
                <p style='margin: 0; font-size: 0.9em; color: #666;'>Without treatment, pneumonia can worsen rapidly, leading to respiratory failure and sepsis.</p>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='info-grid'>
            <div class='info-card'>
                <div style='font-size: 2em; margin-bottom: 12px;'>🎯</div>
                <h4>Precision scanning</h4>
                <p>Designed to identify subtle pneumonia indicators on chest X-rays with greater confidence.</p>
            </div>
            <div class='info-card'>
                <div style='font-size: 2em; margin-bottom: 12px;'>👥</div>
                <h4>Patient-first design</h4>
                <p>Provides simple, easy-to-understand insights so non-specialists can act sooner.</p>
            </div>
            <div class='info-card'>
                <div style='font-size: 2em; margin-bottom: 12px;'>📖</div>
                <h4>Clinical storytelling</h4>
                <p>Combines imaging results with health context like symptoms, causes, and prevention.</p>
            </div>
            <div class='info-card'>
                <div style='font-size: 2em; margin-bottom: 12px;'>🧪</div>
                <h4>Interactive demo</h4>
                <p>Try sample cases and compare healthy versus pneumonia images in seconds.</p>
            </div>
    </div>""", unsafe_allow_html=True)

    # ==================== PROFESSIONAL FOOTER ====================
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    
    # Professional footer with contact info
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(109,76,255,0.05), rgba(72,181,255,0.03)); 
         padding: 30px 20px; border-radius: 16px; border: 1px solid rgba(109,76,255,0.1); margin: 20px 0; 
         text-align: center;'>
        <h3 style='margin: 0 0 12px 0; background: linear-gradient(135deg, #0ea5e9 0%, #22d3ee 100%); 
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; 
             font-size: 1.3em;'>👨‍💼 Shubham Singh</h3>
        <p style='margin: 0 0 20px 0; color: #6B7280; font-size: 0.95em; line-height: 1.5; max-width: 600px; margin: 0 auto 20px auto;'>
            Full Stack Developer & AI Engineer specializing in healthcare technology, machine learning systems, and innovative medical solutions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Contact links using Streamlit columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <a href='https://github.com/shubham08-cs' target='_blank' 
           style='text-decoration: none; color: #0ea5e9; font-weight: 600; display: block; 
                  padding: 12px; border-radius: 12px; background: rgba(109,76,255,0.1); 
                  border: 1px solid rgba(109,76,255,0.2); text-align: center; margin: 5px;'>
            🐙 GitHub
        </a>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <a href='https://www.linkedin.com/in/shubham-ich/' target='_blank' 
           style='text-decoration: none; color: #0077B5; font-weight: 600; display: block; 
                  padding: 12px; border-radius: 12px; background: rgba(0,119,181,0.1); 
                  border: 1px solid rgba(0,119,181,0.2); text-align: center; margin: 5px;'>
            💼 LinkedIn
        </a>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <a href='mailto:me.shubhamsingh8998@gmail.com' 
           style='text-decoration: none; color: #EA4335; font-weight: 600; display: block; 
                  padding: 12px; border-radius: 12px; background: rgba(234,67,53,0.1); 
                  border: 1px solid rgba(234,67,53,0.2); text-align: center; margin: 5px;'>
            ✉️ Email
        </a>
        """, unsafe_allow_html=True)
    
    # Copyright
    st.markdown("""
    <div style='text-align: center; margin-top: 20px; padding-top: 16px; 
         border-top: 1px solid rgba(109,76,255,0.2); color: #9CA3AF; font-size: 0.85em;'>
        © 2026 Shubham Singh. All rights reserved. | Built with ❤️ using Python, Streamlit & AI
    </div>
    """, unsafe_allow_html=True)

# ==================== ANALYZE PAGE ====================
if page == "🏥 Analyze":
    # Hero Section with Premium Design
    st.markdown("""<div class='hero'>
        <div style='display:flex; align-items:center; justify-content:space-between; gap:24px; flex-wrap:wrap;'>
            <div style='max-width:620px;'>
                <h1>🎯 Intelligent Pneumonia Detection</h1>
                <p style='font-size:1.05em; margin-top:14px; color:var(--text-secondary); line-height:1.75;'>
                    Fast, clinical-grade chest X-ray analysis for informed health decisions and stronger patient guidance.
                </p>
                <div style='display:grid; grid-template-columns:repeat(2,minmax(160px,1fr)); gap:14px; margin-top:28px;'>
                    <div class='feature-chip'>✨ AI-powered diagnosis</div>
                    <div class='feature-chip'>📊 Medical-grade clarity</div>
                    <div class='feature-chip'>🔒 Safe insights</div>
                    <div class='feature-chip'>⚡ Rapid feedback</div>
                </div>
            </div>
            <div class='info-card animated-panel' style='min-width:240px; max-width:320px;'>
                <h3 style='margin:0 0 12px 0; color:var(--primary-dark);'>Ready to analyze?</h3>
                <p style='margin:0; color:var(--text-secondary); font-size:0.95em;'>Upload a chest X-ray image and our AI will classify the result with accuracy and practical next steps.</p>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)
    
    # 3D Model Display Section
    st.markdown("<div style='margin-top: 0px;'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; margin-bottom: 20px; background: linear-gradient(135deg, #0ea5e9 0%, #22d3ee 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;'>🫁 Explore 3D Lung Model</h3>", unsafe_allow_html=True)
    
    col_3d_intro1, col_3d_intro2 = st.columns([1.2, 1])
    
    with col_3d_intro1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(109,76,255,0.08), rgba(72,181,255,0.04)); padding: 24px; border-radius: 20px; border: 1px solid rgba(109,76,255,0.15);'>
            <h4 style='margin-top: 0; color: #0ea5e9;'>Advanced 3D Visualization</h4>
            <p>Explore interactive 3D representations of chest analysis. Click and drag to rotate, scroll to zoom. Our system converts X-ray findings into intuitive spatial models for better understanding.</p>
            <ul style='margin: 12px 0 0 0; padding-left: 20px;'>
                <li><strong>Rotate:</strong> Click and drag the visualization</li>
                <li><strong>Zoom:</strong> Scroll your mouse wheel</li>
                <li><strong>Interactive:</strong> Explore from all angles</li>
                <li><strong>Real-time:</strong> Instant 3D rendering</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_3d_intro2:
        try:
            fig_lung_intro = create_3d_lung_model()
            st.plotly_chart(fig_lung_intro, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
        except Exception as e:
            st.warning("3D Model loading... please ensure Plotly is properly installed")
    
    # ===== HEALTHY VS INFECTED LUNGS COMPARISON =====
    st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; margin-bottom: 24px; background: linear-gradient(135deg, #22C55E 0%, #EF4444 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-size: 1.8em;'>Compare: Healthy vs Infected Lungs</h3>", unsafe_allow_html=True)
    
    col_healthy, col_infected = st.columns(2, gap="large")
    
    with col_healthy:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(34,197,94,0.1), rgba(34,197,94,0.05)); padding: 20px; border-radius: 16px; border: 2px solid rgba(34,197,94,0.3); margin-bottom: 16px;'>
            <h4 style='margin-top: 0; color: #22C55E; text-align: center;'>✅ HEALTHY LUNGS</h4>
            <p style='margin: 8px 0; font-size: 0.95em;'>🫁 Clear respiratory passages<br/>💚 Optimal oxygen flow<br/>⭐ No inflammation detected<br/>✓ Normal chest imaging</p>
        </div>
        """, unsafe_allow_html=True)
        try:
            fig_healthy = create_healthy_lung_3d()
            st.plotly_chart(fig_healthy, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
        except Exception as e:
            st.error("Unable to load healthy lung model")
    
    with col_infected:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.05)); padding: 20px; border-radius: 16px; border: 2px solid rgba(239,68,68,0.3); margin-bottom: 16px;'>
            <h4 style='margin-top: 0; color: #EF4444; text-align: center;'>⚠️ PNEUMONIA-INFECTED LUNGS</h4>
            <p style='margin: 8px 0; font-size: 0.95em;'>🔴 Fluid/pus in air sacs<br/>❌ Restricted oxygen transfer<br/>🔥 Inflammation present<br/>⚠️ Abnormal patterns detected</p>
        </div>
        """, unsafe_allow_html=True)
        try:
            fig_infected = create_pneumonia_lung_3d()
            st.plotly_chart(fig_infected, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
        except Exception as e:
            st.error("Unable to load pneumonia lung model")
    
    # ===== PNEUMONIA FACTS SECTION =====
    st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)
    st.markdown("""<div style='background: linear-gradient(135deg, rgba(239,68,68,0.05), rgba(249,115,22,0.05)); padding: 32px; border-radius: 24px; border: 1px solid rgba(239,68,68,0.2); margin: 24px 0;'>
        <h3 style='margin-top: 0; color: #EF4444;'>⚡ Critical Pneumonia Indicators</h3>
        <div style='display: grid; grid-template-columns: repeat(3, minmax(200px, 1fr)); gap: 20px; margin-top: 20px;'>
            <div style='background: white; padding: 16px; border-radius: 12px; border-left: 4px solid #EF4444;'>
                <h5 style='margin: 0 0 8px 0; color: #EF4444;'>🫁 Air Sac Inflammation</h5>
                <p style='margin: 0; font-size: 0.9em; color: #666;'>Alveoli fill with fluid or pus, preventing oxygen absorption and causing respiratory distress.</p>
            </div>
            <div style='background: white; padding: 16px; border-radius: 12px; border-left: 4px solid #F97316;'>
                <h5 style='margin: 0 0 8px 0; color: #F97316;'>🦠 Bacterial/Viral Infection</h5>
                <p style='margin: 0; font-size: 0.9em; color: #666;'>Pathogenic organisms trigger immune response, leading to swelling and fluid accumulation in lungs.</p>
            </div>
            <div style='background: white; padding: 16px; border-radius: 12px; border-left: 4px solid #DC2626;'>
                <h5 style='margin: 0 0 8px 0; color: #DC2626;'>🔥 Rapid Progression Risk</h5>
                <p style='margin: 0; font-size: 0.9em; color: #666;'>Without treatment, pneumonia can worsen rapidly, leading to respiratory failure and sepsis.</p>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)
    
    st.markdown("""<div class='info-grid'>
            <div class='info-card'>
                <div style='font-size: 2em; margin-bottom: 12px;'>🎯</div>
                <h4>Precision scanning</h4>
                <p>Designed to identify subtle pneumonia indicators on chest X-rays with greater confidence.</p>
            </div>
            <div class='info-card'>
                <div style='font-size: 2em; margin-bottom: 12px;'>👥</div>
                <h4>Patient-first design</h4>
                <p>Provides simple, easy-to-understand insights so non-specialists can act sooner.</p>
            </div>
            <div class='info-card'>
                <div style='font-size: 2em; margin-bottom: 12px;'>📖</div>
                <h4>Clinical storytelling</h4>
                <p>Combines imaging results with health context like symptoms, causes, and prevention.</p>
            </div>
            <div class='info-card'>
                <div style='font-size: 2em; margin-bottom: 12px;'>🧪</div>
                <h4>Interactive demo</h4>
                <p>Try sample cases and compare healthy versus pneumonia images in seconds.</p>
            </div>
    </div>""", unsafe_allow_html=True)
    
    # Main Content Section
    col_upload, col_result = st.columns([1.3, 1.1])
    
    with col_upload:
        st.markdown("<h3 style='margin: 0 0 16px 0;'>📤 Upload X-Ray Image</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose JPG/JPEG", type=["jpg", "jpeg"], label_visibility="collapsed")
        
        if uploaded_file:
            st.markdown("<h4 style='color: #6B7280; margin: 20px 0 12px 0;'>Preview</h4>", unsafe_allow_html=True)
            st.image(uploaded_file, use_container_width=True)
    
    with col_result:
        st.markdown("<h3 style='margin: 0 0 16px 0;'>🔍 Results</h3>", unsafe_allow_html=True)
        
        if uploaded_file:
            if model is None:
                st.error("Prediction unavailable: model file is not loaded.")
            else:
                col_btn = st.columns([1])[0]
                with col_btn:
                    analyze_click = st.button("🚀 ANALYZE", type="primary", use_container_width=True)
                
                if analyze_click:
                    with st.spinner('Processing and generating 3D analysis...'):
                        img = Image.open(uploaded_file)
                        processed = preprocess_image(img)
                        prediction = model.predict(processed)
                        confidence = float(prediction[0][0])
                        is_pneumonia = confidence > 0.5

                        # Generate saliency map for the exact uploaded image
                        saliency = create_saliency_map(model, processed)
                        region_analysis = analyze_affected_regions(saliency, threshold=0.4)

                        if is_pneumonia:
                            st.markdown(f"""<div class='error-box'>
                                <h4 style='margin-top: 0;'>⚠️ Pneumonia Detected</h4>
                                <div style='font-size: 2.2em; font-weight: 800; color: var(--danger); margin: 16px 0;'>{confidence*100:.1f}%</div>
                                <ul style='margin: 16px 0 0 0; padding-left: 20px;'>
                                    <li>Seek medical attention</li>
                                    <li>Consult a healthcare provider</li>
                                    <li>Follow professional diagnosis</li>
                                </ul>
                            </div>""", unsafe_allow_html=True)
                        else:
                            st.markdown(f"""<div class='success-box'>
                                <h4 style='margin-top: 0;'>✅ Normal Result</h4>
                                <div style='font-size: 2.2em; font-weight: 800; color: var(--success); margin: 16px 0;'>{(1-confidence)*100:.1f}%</div>
                                <ul style='margin: 16px 0 0 0; padding-left: 20px;'>
                                    <li>Regular checkups recommended</li>
                                    <li>Consult medical professionals</li>
                                    <li>Maintain health practices</li>
                                </ul>
                            </div>""", unsafe_allow_html=True)

                        # Initialize session state for visualization options
                        if 'viz_view_mode' not in st.session_state:
                            st.session_state.viz_view_mode = '3D Surface Heatmap'
                        if 'viz_height_scale' not in st.session_state:
                            st.session_state.viz_height_scale = 35
                        if 'viz_downsample' not in st.session_state:
                            st.session_state.viz_downsample = 2

                        # Advanced view settings for the uploaded image only
                        with st.expander('🎨 Advanced 3D Visualization Options', expanded=False):
                            col_viz1, col_viz2 = st.columns(2)
                            with col_viz1:
                                st.session_state.viz_height_scale = st.slider('Surface height scale', 10, 80, st.session_state.viz_height_scale, step=5)
                                st.session_state.viz_view_mode = st.selectbox('Visualization Mode', 
                                    ['3D Surface Heatmap', '2D Attention Overlay', '3D Confidence Cone', '3D Lung Model'], 
                                    index=['3D Surface Heatmap', '2D Attention Overlay', '3D Confidence Cone', '3D Lung Model'].index(st.session_state.viz_view_mode),
                                    help='Choose your preferred 3D analysis view')
                            with col_viz2:
                                st.session_state.viz_downsample = st.selectbox('Performance Quality', [1, 2, 3], 
                                    index=[1, 2, 3].index(st.session_state.viz_downsample),
                                    help='1=High Quality (slower), 3=Fast (lower res)')
                                st.info("💡 Rotate & Zoom: Click and drag the 3D visualizations to explore all angles")

                        # Use the session state values for visualization
                        height_scale = st.session_state.viz_height_scale
                        view_mode = st.session_state.viz_view_mode
                        downsample = st.session_state.viz_downsample

                        # Display selected visualization
                        st.markdown("<h3 style='margin-top: 24px; margin-bottom: 16px; background: linear-gradient(135deg, #0ea5e9 0%, #22d3ee 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;'>📊 Advanced 3D Analysis</h3>", unsafe_allow_html=True)
                        
                        if view_mode == '3D Surface Heatmap':
                            st.markdown("<div style='background: rgba(109,76,255,0.08); padding: 12px; border-radius: 12px; margin-bottom: 16px;'><strong>🔍 3D Image-Based Heatmap Analysis</strong> - Surface height represents attention intensity</div>", unsafe_allow_html=True)
                            fig_3d = create_3d_heatmap(processed, saliency, is_pneumonia, height_scale=height_scale, downsample_factor=downsample)
                            st.plotly_chart(fig_3d, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
                            
                        elif view_mode == '2D Attention Overlay':
                            st.markdown("<div style='background: rgba(109,76,255,0.08); padding: 12px; border-radius: 12px; margin-bottom: 16px;'><strong>🖼️ 2D Attention Heatmap</strong> - Highlights regions the AI focused on</div>", unsafe_allow_html=True)
                            overlay = create_attention_overlay(img, saliency)
                            st.image(overlay, caption='AI Attention Overlay on Original X-ray', use_container_width=True)
                            
                        elif view_mode == '3D Confidence Cone':
                            st.markdown("<div style='background: rgba(109,76,255,0.08); padding: 12px; border-radius: 12px; margin-bottom: 16px;'><strong>📈 3D Confidence Visualization</strong> - Risk cone shows diagnosis confidence level</div>", unsafe_allow_html=True)
                            fig_confidence = create_3d_confidence_visualization(confidence, is_pneumonia)
                            st.plotly_chart(fig_confidence, use_container_width=True, config={'scrollZoom': True})
                            
                        elif view_mode == '3D Lung Model':
                            st.markdown("<div style='background: rgba(109,76,255,0.08); padding: 12px; border-radius: 12px; margin-bottom: 16px;'><strong>🫁 3D Interactive Lung Model</strong> - Rotating 3D visualization of analyzed lung structure</div>", unsafe_allow_html=True)
                            fig_lung = create_3d_lung_model()
                            st.plotly_chart(fig_lung, use_container_width=True, config={'scrollZoom': True})

                        if region_analysis:
                            st.markdown("<h3 style='margin-top: 8px; margin-bottom: 8px;'>📍 Detailed Findings & Analytics</h3>", unsafe_allow_html=True)
                            
                            # Premium metrics display with columns
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown(f"""
                                <div style='background: linear-gradient(135deg, rgba(109,76,255,0.1), rgba(72,181,255,0.05)); padding: 20px; border-radius: 16px; border: 1px solid rgba(109,76,255,0.2);'>
                                    <div style='font-size: 0.85em; text-transform: uppercase; color: #6B7280; font-weight: 600; margin-bottom: 8px;'>📊 Affected Area</div>
                                    <div style='font-size: 2.2em; font-weight: 800; background: linear-gradient(135deg, #0ea5e9, #22d3ee); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;'>{region_analysis['total_affected_percent']:.1f}%</div>
                                    <div style='font-size: 0.85em; color: #6B7280; margin-top: 8px;'>Area showing abnormalities</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                <div style='background: linear-gradient(135deg, rgba(72,181,255,0.1), rgba(109,76,255,0.05)); padding: 20px; border-radius: 16px; border: 1px solid rgba(72,181,255,0.2);'>
                                    <div style='font-size: 0.85em; text-transform: uppercase; color: #6B7280; font-weight: 600; margin-bottom: 8px;'>⚡ Intensity</div>
                                    <div style='font-size: 2.2em; font-weight: 800; background: linear-gradient(135deg, #22d3ee, #0ea5e9); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;'>{region_analysis['max_intensity']:.0%}</div>
                                    <div style='font-size: 0.85em; color: #6B7280; margin-top: 8px;'>Peak anomaly strength</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown(f"""
                                <div style='background: linear-gradient(135deg, rgba(109,76,255,0.1), rgba(72,181,255,0.05)); padding: 20px; border-radius: 16px; border: 1px solid rgba(109,76,255,0.2);'>
                                    <div style='font-size: 0.85em; text-transform: uppercase; color: #6B7280; font-weight: 600; margin-bottom: 8px;'>📍 Primary Location</div>
                                    <div style='font-size: 1.6em; font-weight: 800; color: #0ea5e9;'>{region_analysis['primary_location']}</div>
                                    <div style='font-size: 0.85em; color: #6B7280; margin-top: 8px;'>Main area of concern</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Additional details in premium card
                            st.markdown("""<div style='background: linear-gradient(135deg, rgba(109,76,255,0.08), rgba(72,181,255,0.04)); padding: 24px; border-radius: 20px; border: 1px solid rgba(109,76,255,0.15); margin-top: 20px;'>
                                <h4 style='margin-top: 0; color: #0ea5e9;'>📋 Clinical Analysis Summary</h4>""", unsafe_allow_html=True)
                            
                            if is_pneumonia:
                                severity = 'High' if region_analysis['total_affected_percent'] > 30 else 'Moderate' if region_analysis['total_affected_percent'] > 15 else 'Mild'
                                severity_color = '#EF4444' if severity == 'High' else '#F97316' if severity == 'Moderate' else '#EAB308'
                                st.markdown(f"""
<table style='width: 100%; border-collapse: collapse;'>
    <tr style='border-bottom: 1px solid rgba(109,76,255,0.1);'>
        <td style='padding: 12px 0; font-weight: 600;'>Status</td>
        <td style='padding: 12px 0; color: #EF4444;'>⚠️ Pneumonia Indicators Detected</td>
    </tr>
    <tr style='border-bottom: 1px solid rgba(109,76,255,0.1);'>
        <td style='padding: 12px 0; font-weight: 600;'>Location</td>
        <td style='padding: 12px 0;'>{region_analysis['primary_location']}</td>
    </tr>
    <tr style='border-bottom: 1px solid rgba(109,76,255,0.1);'>
        <td style='padding: 12px 0; font-weight: 600;'>Extent</td>
        <td style='padding: 12px 0;'>{region_analysis['total_affected_percent']:.1f}% with {region_analysis['mean_intensity']:.0%} avg intensity</td>
    </tr>
    <tr style='border-bottom: 1px solid rgba(109,76,255,0.1);'>
        <td style='padding: 12px 0; font-weight: 600;'>Severity</td>
        <td style='padding: 12px 0; color: {severity_color}; font-weight: 700;'>⚡ {severity}</td>
    </tr>
    <tr>
        <td style='padding: 12px 0; font-weight: 600;'>Recommendation</td>
        <td style='padding: 12px 0; color: #EF4444; font-weight: 600;'>🔴 Immediate Medical Consultation Required</td>
    </tr>
</table>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
<table style='width: 100%; border-collapse: collapse;'>
    <tr style='border-bottom: 1px solid rgba(109,76,255,0.1);'>
        <td style='padding: 12px 0; font-weight: 600;'>Status</td>
        <td style='padding: 12px 0; color: #22C55E;'>✅ No Pneumonia Indicators Detected</td>
    </tr>
    <tr style='border-bottom: 1px solid rgba(109,76,255,0.1);'>
        <td style='padding: 12px 0; font-weight: 600;'>Affected Area</td>
        <td style='padding: 12px 0;'>{region_analysis['total_affected_percent']:.1f}% (Normal baseline)</td>
    </tr>
    <tr style='border-bottom: 1px solid rgba(109,76,255,0.1);'>
        <td style='padding: 12px 0; font-weight: 600;'>Severity Assessment</td>
        <td style='padding: 12px 0; color: #22C55E;'>None - Lungs appear clear</td>
    </tr>
    <tr>
        <td style='padding: 12px 0; font-weight: 600;'>Recommendation</td>
        <td style='padding: 12px 0; color: #22C55E; font-weight: 600;'>🟢 Continue Regular Health Maintenance</td>
    </tr>
</table>
                                """, unsafe_allow_html=True)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # PDF Download Button
                            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                            col_pdf = st.columns([1, 1, 1])[0]
                            with col_pdf:
                                pdf_data = generate_pdf_report(img, confidence, region_analysis, is_pneumonia)
                                
                                timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"pneumonia_report_{timestamp}.pdf"
                                
                                st.download_button(
                                    label="📄 Download PDF Report",
                                    data=pdf_data,
                                    file_name=filename,
                                    mime="application/pdf",
                                    use_container_width=True
                                )
        else:
            st.markdown(f"""<div class='info-box' style='text-align: center; padding: 40px 20px; border: 2px dashed var(--border);'>
                <div style='font-size: 3em; margin-bottom: 12px;'>📁</div>
                <p style='font-size: 1.1em; margin: 0;'><strong>Upload to analyze</strong></p>
            </div>""", unsafe_allow_html=True)
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-header'>Clinical Recommendations & Workflow</h2>", unsafe_allow_html=True)
    st.markdown("""<div class='grid-col'>
        <div class='step-card'>
            <span class='badge'>💡 Step 1</span>
            <h4>Prepare the image</h4>
            <p>Crop to the chest region, ensure good contrast, and upload a clear frontal X-ray for consistent AI interpretation.</p>
            <ul style='margin: 16px 0 0 0; padding-left: 20px;'>
                <li>Recommended format: JPG or JPEG</li>
                <li>Resolution: 1024×1024 or higher</li>
                <li>Keep file size under 10 MB</li>
            </ul>
        </div>
        <div class='step-card'>
            <span class='badge'>🧠 Step 2</span>
            <h4>AI analysis</h4>
            <p>Our ResNet-based model scans the image for pneumonia patterns, then presents a confidence score and care advice.</p>
            <ul style='margin: 16px 0 0 0; padding-left: 20px;'>
                <li>Binary classification: Normal vs Pneumonia</li>
                <li>Fast inference in seconds</li>
                <li>Clear actionable output</li>
            </ul>
        </div>
    </div>""", unsafe_allow_html=True)
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-header'>Why This Matters</h2>", unsafe_allow_html=True)
    st.markdown("""<div class='timeline-card'>
        <div>⚡</div>
        <div>
            <h4>Early detection saves lives</h4>
            <p>Pneumonia can worsen rapidly. Identifying it early with imaging helps clinicians intervene before complications develop.</p>
        </div>
    </div>
    <div class='timeline-card' style='margin-top: 16px;'>
        <div>🌉</div>
        <div>
            <h4>Bridge between radiology and care</h4>
            <p>This dashboard provides a simple way for caregivers to review X-ray intelligence and understand next steps quickly.</p>
        </div>
    </div>
    <div class='timeline-card' style='margin-top: 16px;'>
        <div>✓</div>
        <div>
            <h4>Recommend follow-up actions</h4>
            <p>Use the results as a screening aid, then pursue clinical confirmation via doctors, lab tests, or CT imaging when needed.</p>
        </div>
    </div>""", unsafe_allow_html=True)
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-header'>How our project works</h2>", unsafe_allow_html=True)
    st.markdown("""<div class='grid-col'>
        <div class='info-card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>📋</div>
            <h4>1. Input preparation</h4>
            <p>Upload a clean frontal chest X-ray. The system standardizes size, color, and texture before AI analysis.</p>
        </div>
        <div class='info-card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>⚙️</div>
            <h4>2. Deep learning analysis</h4>
            <p>An optimized ResNet model reviews the X-ray and produces a pneumonia probability score.</p>
        </div>
        <div class='info-card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>📊</div>
            <h4>3. Interpret results</h4>
            <p>The dashboard shows clear detection status, confidence, and recommendations for follow-up action.</p>
        </div>
        <div class='info-card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>💊</div>
            <h4>4. Clinical advice</h4>
            <p>Use the output as a screening reference and support patient care conversations with professionals.</p>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-header'>🧪 Try Demo Samples</h2>", unsafe_allow_html=True)
    
    demo_col1, demo_col2 = st.columns([1, 1.2])
    
    with demo_col1:
        st.markdown("""<div class='card' style='text-align: center; height: 100%;'>
            <div style='font-size: 2.2em; margin-bottom: 12px;'>📋</div>
            <h4 style='margin: 0 0 8px 0;'>Pre-loaded Samples</h4>
            <p style='margin: 0; font-size: 0.9em;'>Try with example images</p>
        </div>""", unsafe_allow_html=True)
    
    with demo_col2:
        col_select, col_button = st.columns([2, 1])
        with col_select:
            sample_choice = st.selectbox("Select:", ["Normal X-Ray", "Pneumonia Case"], label_visibility="collapsed")
        with col_button:
            load_sample = st.button("Load", type="primary", use_container_width=True)
        
        if load_sample:
            st.session_state.demo_image = load_sample_data('data1n.jpeg') if sample_choice == "Normal X-Ray" else load_sample_data('data1p.jpeg')
    
    if 'demo_image' in st.session_state:
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        
        demo_col1, demo_col2 = st.columns([1.3, 1.1])
        
        with demo_col1:
            st.markdown("<h4 style='color: #6B7280; margin: 0 0 12px 0;'>Image</h4>", unsafe_allow_html=True)
            st.image(st.session_state.demo_image, use_container_width=True)
        
        with demo_col2:
            st.markdown("<h4 style='color: #6B7280; margin: 0 0 12px 0;'>Analysis</h4>", unsafe_allow_html=True)
            if model is None:
                st.error("Prediction unavailable: model file is not loaded.")
            elif st.button("Analyze Sample", type="primary", use_container_width=True):
                with st.spinner('Processing...'):
                    demo_processed = preprocess_image(st.session_state.demo_image)
                    demo_pred = model.predict(demo_processed)
                    demo_conf = demo_pred[0][0]
                    
                    if demo_conf > 0.5:
                        st.markdown(f"""<div class='error-box' style='padding: 16px;'>
                            <h5 style='margin-top: 0; color: var(--danger);'>⚠️ Detected</h5>
                            <div style='font-size: 2em; font-weight: 700; color: var(--danger);'>{demo_conf*100:.1f}%</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div class='success-box' style='padding: 16px;'>
                            <h5 style='margin-top: 0; color: var(--success);'>✅ Normal</h5>
                            <div style='font-size: 2em; font-weight: 700; color: var(--success);'>{(1-demo_conf)*100:.1f}%</div>
                        </div>""", unsafe_allow_html=True)

    # ==================== PROFESSIONAL FOOTER ====================
    st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
    # ==================== PROFESSIONAL FOOTER ====================
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    
    # Professional footer with contact info
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(109,76,255,0.05), rgba(72,181,255,0.03)); 
         padding: 30px 20px; border-radius: 16px; border: 1px solid rgba(109,76,255,0.1); margin: 20px 0; 
         text-align: center;'>
        <h3 style='margin: 0 0 12px 0; background: linear-gradient(135deg, #0ea5e9 0%, #22d3ee 100%); 
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; 
             font-size: 1.3em;'>👨‍💼 Shubham Singh</h3>
        <p style='margin: 0 0 20px 0; color: #6B7280; font-size: 0.95em; line-height: 1.5; max-width: 600px; margin: 0 auto 20px auto;'>
            Full Stack Developer & AI Engineer specializing in healthcare technology, machine learning systems, and innovative medical solutions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Contact links using Streamlit columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <a href='https://github.com/shubham08-cs' target='_blank' 
           style='text-decoration: none; color: #0ea5e9; font-weight: 600; display: block; 
                  padding: 12px; border-radius: 12px; background: rgba(109,76,255,0.1); 
                  border: 1px solid rgba(109,76,255,0.2); text-align: center; margin: 5px;'>
            🐙 GitHub
        </a>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <a href='https://www.linkedin.com/in/shubham-ich/' target='_blank' 
           style='text-decoration: none; color: #0077B5; font-weight: 600; display: block; 
                  padding: 12px; border-radius: 12px; background: rgba(0,119,181,0.1); 
                  border: 1px solid rgba(0,119,181,0.2); text-align: center; margin: 5px;'>
            💼 LinkedIn
        </a>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <a href='mailto:me.shubhamsingh8998@gmail.com' 
           style='text-decoration: none; color: #EA4335; font-weight: 600; display: block; 
                  padding: 12px; border-radius: 12px; background: rgba(234,67,53,0.1); 
                  border: 1px solid rgba(234,67,53,0.2); text-align: center; margin: 5px;'>
            ✉️ Email
        </a>
        """, unsafe_allow_html=True)
    
    # Copyright
    st.markdown("""
    <div style='text-align: center; margin-top: 20px; padding-top: 16px; 
         border-top: 1px solid rgba(109,76,255,0.2); color: #9CA3AF; font-size: 0.85em;'>
        © 2026 Shubham Singh. All rights reserved. | Built with ❤️ using Python, Streamlit & AI
    </div>
    """, unsafe_allow_html=True)

# ==================== LEARN PAGE ====================
elif page == "📚 Learn":
    st.markdown("""<div class='hero'>
        <h1>📚 Pneumonia Intelligence</h1>
        <p>Learn what pneumonia is, how it shows up in medical imaging, and how to prevent it with clear healthcare guidance.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<h2 class='section-header'>What is Pneumonia?</h2>", unsafe_allow_html=True)
    st.markdown("""<div class='grid-col'>
        <div class='info-card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>🫁</div>
            <h4>Medical overview</h4>
            <p>Pneumonia is an infection that inflames the air sacs in one or both lungs. It can cause fluid or pus, making it harder to breathe and reducing oxygen flow.</p>
        </div>
        <div class='info-card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>🦠</div>
            <h4>Causes</h4>
            <ul style='margin: 0; padding-left: 20px;'>
                <li>Bacteria</li>
                <li>Viruses</li>
                <li>Fungi</li>
                <li>Inhaled irritants</li>
            </ul>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<h2 class='section-header'>How Pneumonia Develops</h2>", unsafe_allow_html=True)
    st.markdown("""<div class='grid-col'>
        <div class='info-card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>🦠</div>
            <h4>Respiratory infection</h4>
            <p>An infection can spread to the lungs from the nose, throat, or sinuses. The immune system responds, swelling tissues and filling air sacs with fluid.</p>
        </div>
        <div class='info-card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>👥</div>
            <h4>High-risk groups</h4>
            <ul style='margin: 0; padding-left: 20px;'>
                <li>Young children</li>
                <li>Older adults</li>
                <li>Chronic illness</li>
                <li>Immune compromised</li>
            </ul>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<h2 class='section-header'>Symptoms to Watch</h2>", unsafe_allow_html=True)
    st.markdown("""<div class='grid-col'>
        <div class='info-card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>🤒</div>
            <h4>Common symptoms</h4>
            <ul style='margin: 0; padding-left: 20px;'>
                <li>Persistent cough</li>
                <li>Chest pain</li>
                <li>Shortness of breath</li>
                <li>Fever and chills</li>
            </ul>
        </div>
        <div class='info-card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>⚠️</div>
            <h4>Critical warning signs</h4>
            <ul style='margin: 0; padding-left: 20px;'>
                <li>❌ Severe fatigue</li>
                <li>❌ Confusion or disorientation</li>
                <li>❌ Rapid or labored breathing</li>
                <li>❌ Extremely low oxygen levels</li>
            </ul>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<h2 class='section-header'>Prevention & Care</h2>", unsafe_allow_html=True)
    st.markdown("""<div class='grid-col'>
        <div class='info-card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>💉</div>
            <h4>Prevention tips</h4>
            <ul style='margin: 0; padding-left: 20px;'>
                <li>Stay up to date with vaccines</li>
                <li>Practice good hygiene</li>
                <li>Avoid smoking and pollutants</li>
                <li>Stay hydrated and rest</li>
            </ul>
        </div>
        <div class='info-card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>🩺</div>
            <h4>When to seek help</h4>
            <p>If symptoms persist, breathing worsens, or fever stays high, contact a medical professional immediately.</p>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<h2 class='section-header'>How AI Supports Diagnosis</h2>", unsafe_allow_html=True)
    st.markdown("""<div class='info-grid'>
        <div class='info-card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>⚡</div>
            <h4>Faster screening</h4>
            <p>AI augments radiologist workflow by flagging suspect cases early, enabling quicker review and referral.</p>
        </div>
        <div class='info-card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>🧠</div>
            <h4>Pattern recognition</h4>
            <p>Machine learning learns from large X-ray datasets to detect patterns that may not be obvious at first glance.</p>
        </div>
        <div class='info-card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>✅</div>
            <h4>Decision support</h4>
            <p>Our system provides confidence scores and consistency checks to help clinicians make informed decisions.</p>
        </div>
        <div class='info-card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>📢</div>
            <h4>Patient education</h4>
            <p>Clear recommendations and visuals help patients understand their condition and follow treatment guidance.</p>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<h2 class='section-header'>Pneumonia Imaging</h2>", unsafe_allow_html=True)
    st.markdown("""<div class='grid-col'>
        <div class='image-card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>📸</div>
            <h4>X-ray appearance</h4>
            <p>Pneumonia often appears as cloudy, white opacities on chest X-rays due to fluid and inflammation in the lung tissue.</p>
        </div>
        <div class='image-card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>🔬</div>
            <h4>Clinical insight</h4>
            <p>Our AI evaluates these patterns and distinguishes pneumonia from normal lung structure using advanced convolutional networks.</p>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-header'>How pneumonia happens</h2>", unsafe_allow_html=True)
    st.markdown("""<div class='grid-col'>
        <div class='info-card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>💨</div>
            <h4>Exposure</h4>
            <p>Airborne pathogens enter the lungs through the nose or mouth and infect the alveoli.</p>
        </div>
        <div class='info-card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>⚔️</div>
            <h4>Immune response</h4>
            <p>The body sends white blood cells, causing swelling and fluid buildup in the air sacs.</p>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<h2 class='section-header'>How pneumonia is detected</h2>", unsafe_allow_html=True)
    st.markdown("""<div class='grid-col'>
        <div class='info-card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>🩺</div>
            <h4>Clinical observation</h4>
            <p>Doctors first look at symptoms like cough, fever, and breathing difficulty.</p>
        </div>
        <div class='info-card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>🖼️</div>
            <h4>Medical imaging</h4>
            <p>Chest X-rays reveal fluid-filled lungs and consolidation patterns associated with pneumonia.</p>
        </div>
    </div>""", unsafe_allow_html=True)

    # ===== EDUCATIONAL 3D VISUALIZATIONS =====
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-header' style='background: linear-gradient(135deg, #22C55E 0%, #EF4444 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;'>🔬 Interactive Lung Anatomy & Disease Progression</h2>", unsafe_allow_html=True)

    col_edu1, col_edu2 = st.columns(2, gap="large")

    with col_edu1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(34,197,94,0.1), rgba(34,197,94,0.05)); padding: 20px; border-radius: 16px; border: 2px solid rgba(34,197,94,0.3); margin-bottom: 16px;'>
            <h4 style='margin-top: 0; color: #22C55E; text-align: center;'>🫁 Lung Cross-Section</h4>
            <p style='margin: 8px 0; font-size: 0.95em;'>Explore the intricate structure of lung tissue, alveoli (air sacs), and bronchial passages that enable breathing and gas exchange.</p>
            <ul style='margin: 12px 0 0 0; padding-left: 20px; font-size: 0.9em;'>
                <li>🔵 Blue: Lung tissue structure</li>
                <li>🟢 Green: Alveoli air sacs</li>
                <li>🟠 Orange: Bronchial airways</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        try:
            fig_cross_section = create_lung_cross_section_3d()
            st.plotly_chart(fig_cross_section, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
        except Exception as e:
            st.error("Unable to load lung cross-section visualization")

    with col_edu2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.05)); padding: 20px; border-radius: 16px; border: 2px solid rgba(239,68,68,0.3); margin-bottom: 16px;'>
            <h4 style='margin-top: 0; color: #EF4444; text-align: center;'>🔥 Infection Progression</h4>
            <p style='margin: 8px 0; font-size: 0.95em;'>See how pneumonia develops from initial infection to advanced inflammation, showing the spread of pathogens through lung tissue.</p>
            <ul style='margin: 12px 0 0 0; padding-left: 20px; font-size: 0.9em;'>
                <li>🔵 Blue: Healthy lung tissue</li>
                <li>🔴 Red: Infection hotspots</li>
                <li>Progress: Early → Advanced stages</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        try:
            fig_progression = create_infection_progression_3d()
            st.plotly_chart(fig_progression, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
        except Exception as e:
            st.error("Unable to load infection progression visualization")

    st.markdown("<h2 class='section-header'>Protect yourself</h2>", unsafe_allow_html=True)
    st.markdown("""<div class='grid-col'>
        <div class='info-card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>💪</div>
            <h4>Vaccination</h4>
            <p>Vaccines against flu and pneumococcus reduce the risk of pneumonia and serious lung infection.</p>
        </div>
        <div class='info-card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>❤️</div>
            <h4>Healthy living</h4>
            <p>Good sleep, nutrition, and hand hygiene make your lungs stronger against infection.</p>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-header'>Pneumonia Types & Risk</h2>", unsafe_allow_html=True)
    st.markdown("""<div class='grid-col'>
        <div class='info-card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>🏘️</div>
            <h4>Community-acquired</h4>
            <p>Most common form; occurs outside hospitals and is often caused by bacteria or viruses.</p>
        </div>
        <div class='info-card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>🏥</div>
            <h4>Hospital-acquired</h4>
            <p>Develops during a hospital stay; can be more serious due to resistant organisms and weakened immunity.</p>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<h2 class='section-header'>Patient Guidance & Recommendations</h2>", unsafe_allow_html=True)
    st.markdown("""<div class='grid-col'>
        <div class='info-card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>⏰</div>
            <h4>Immediate advice</h4>
            <ul style='margin: 0; padding-left: 20px;'>
                <li>Keep hydrated</li>
                <li>Take rest and avoid strain</li>
                <li>Monitor breathing closely</li>
                <li>Follow prescribed medication</li>
            </ul>
        </div>
        <div class='info-card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>👨‍⚕️</div>
            <h4>Doctor consultation</h4>
            <ul style='margin: 0; padding-left: 20px;'>
                <li>Persistent fever over 38°C</li>
                <li>Difficult breathing</li>
                <li>Chest pain or worsening cough</li>
                <li>Confusion or dizziness</li>
            </ul>
        </div>
    </div>""", unsafe_allow_html=True)

    try:
        img_normal = Image.open(DATA_PATH / 'data1n.jpeg')
        img_pneumonia = Image.open(DATA_PATH / 'data1p.jpeg')
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""<div class='card' style='text-align:center;'>
                <h4>Normal Chest X-Ray</h4>
            </div>""", unsafe_allow_html=True)
            st.image(img_normal, use_container_width=True)
        with col2:
            st.markdown("""<div class='card' style='text-align:center;'>
                <h4>Pneumonia Chest X-Ray</h4>
            </div>""", unsafe_allow_html=True)
            st.image(img_pneumonia, use_container_width=True)
    except Exception:
        st.info("Sample X-ray images unavailable")

    st.markdown("<h2 class='section-header'>Model & Dataset Highlights</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    specs = [
        ("ResNet-50", "Robust backbone"),
        ("25.5M", "Model parameters"),
        ("95%", "Detection accuracy"),
        ("Realtime", "Fast evaluation")
    ]
    for col, (title, desc) in zip([col1, col2, col3, col4], specs):
        with col:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>{desc}</div>
                <div class='metric-value' style='font-size: 1.8em;'>{title}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<h2 class='section-header'>Technical Description</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class='card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>📊</div>
            <h4 style='margin-top: 0;'>Data & pre-processing</h4>
            <ul style='margin: 0; padding-left: 20px;'>
                <li>Input: 256×256 RGB X-rays</li>
                <li>Normalized intensities</li>
                <li>Batch-friendly tensor input</li>
                <li>Clean performance pipeline</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>🤖</div>
            <h4 style='margin-top: 0;'>Model design</h4>
            <ul style='margin: 0; padding-left: 20px;'>
                <li>ResNet-50 backbone</li>
                <li>Global average pooling</li>
                <li>Sigmoid output layer</li>
                <li>Optimized for binary detection</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""<div class='card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>✅</div>
            <h4 style='margin-top: 0;'>Performance metrics</h4>
            <ul style='margin: 0; padding-left: 20px;'>
                <li>Accuracy: 95%</li>
                <li>Sensitivity: 95%</li>
                <li>Specificity: 90%</li>
                <li>F1-score: 93%</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class='card'>
            <div style='font-size: 2em; margin-bottom: 12px;'>📄</div>
            <h4 style='margin-top: 0;'>Training overview</h4>
            <ul style='margin: 0; padding-left: 20px;'>
                <li>Loss: Binary Cross-Entropy</li>
                <li>Optimizer: Adam</li>
                <li>Batch size: 32</li>
                <li>Early stopping enabled</li>
            </ul>
        </div>""", unsafe_allow_html=True)

# ==================== ABOUT PAGE ====================
elif page == "👨‍💼 About":
    st.markdown("""<div class='hero'>
        <h1>👨‍💼 About Developer</h1>
        <p>Meet the Creator</p>
    </div>""", unsafe_allow_html=True)
    
    col1, col2 = st.columns([0.8, 1.5])
    
    with col1:
        st.markdown("""<div class='card' style='text-align: center; height: 100%;'>
            <div style='font-size: 4em; margin-bottom: 16px;'>👨‍💼</div>
            <h3 style='margin: 0 0 4px 0;'>Shubham Singh</h3>
            <p style='color: var(--text-secondary); font-weight: 600; margin: 0;'>Full Stack Developer</p>
            <p style='color: var(--text-secondary); margin: 4px 0 0 0;'>& AI Engineer</p>
        </div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""<div class='card' style='height: 100%;'>
            <h4 style='margin-top: 0;'>Professional Profile</h4>
            <p>Passionate Full Stack Developer & AI Engineer specializing in machine learning systems and healthcare technology. Expert in deep learning, computer vision, and cloud deployment.</p>
            <h4 style='margin-top: 20px;'>🎯 Core Competencies</h4>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px;'>
                <div style='background: var(--primary-light); padding: 10px; border-radius: 8px;'>
                    <strong style='color: var(--primary);'>🤖 Deep Learning</strong><br>
                    <span style='font-size: 0.85em; color: var(--text-secondary);'>Neural Networks</span>
                </div>
                <div style='background: var(--primary-light); padding: 10px; border-radius: 8px;'>
                    <strong style='color: var(--primary);'>🖼️ Computer Vision</strong><br>
                    <span style='font-size: 0.85em; color: var(--text-secondary);'>Image Processing</span>
                </div>
                <div style='background: var(--primary-light); padding: 10px; border-radius: 8px;'>
                    <strong style='color: var(--primary);'>⚕️ Medical AI</strong><br>
                    <span style='font-size: 0.85em; color: var(--text-secondary);'>Healthcare Tech</span>
                </div>
                <div style='background: var(--primary-light); padding: 10px; border-radius: 8px;'>
                    <strong style='color: var(--primary);'>🌐 Full Stack</strong><br>
                    <span style='font-size: 0.85em; color: var(--text-secondary);'>Web Development</span>
                </div>
                <div style='background: var(--primary-light); padding: 10px; border-radius: 8px;'>
                    <strong style='color: var(--primary);'>📊 Data Science</strong><br>
                    <span style='font-size: 0.85em; color: var(--text-secondary);'>Analytics & ML</span>
                </div>
                <div style='background: var(--primary-light); padding: 10px; border-radius: 8px;'>
                    <strong style='color: var(--primary);'>☁️ Cloud & DevOps</strong><br>
                    <span style='font-size: 0.85em; color: var(--text-secondary);'>Deployment</span>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-header'>🔗 Connect & Follow</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.link_button("🐙 GitHub", "https://github.com/shubham08-cs", use_container_width=True)
    with col2:
        st.link_button("💼 LinkedIn", "https://www.linkedin.com/in/shubham-ich/", use_container_width=True)
    with col3:
        st.link_button("✉️ Email", "mailto:me.shubhamsingh8998@gmail.com", use_container_width=True)
    
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-header'>🚀 Project Vision</h2>", unsafe_allow_html=True)
    
    st.markdown("""<div class='card'>
        <h4 style='margin-top: 0;'>💡 Our Mission</h4>
        <p>This Pneumonia Detection System leverages AI for early disease detection and improved healthcare outcomes. We aim to democratize access to medical diagnostic technology and support healthcare professionals with intelligent tools.</p>
    </div>""", unsafe_allow_html=True)

    # ===== TECH 3D VISUALIZATIONS =====
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-header' style='background: linear-gradient(135deg, #0ea5e9 0%, #22d3ee 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;'>🤖 AI Technology & Innovation</h2>", unsafe_allow_html=True)

    col_tech1, col_tech2 = st.columns(2, gap="large")

    with col_tech1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(109,76,255,0.1), rgba(72,181,255,0.05)); padding: 20px; border-radius: 16px; border: 2px solid rgba(109,76,255,0.3); margin-bottom: 16px;'>
            <h4 style='margin-top: 0; color: #0ea5e9; text-align: center;'>🧠 Neural Network Architecture</h4>
            <p style='margin: 8px 0; font-size: 0.95em;'>Explore the deep learning model that powers pneumonia detection, featuring convolutional layers, feature extraction, and binary classification.</p>
            <ul style='margin: 12px 0 0 0; padding-left: 20px; font-size: 0.9em;'>
                <li>🟣 Purple: Input neurons (X-ray data)</li>
                <li>🔵 Blue: Hidden layers (feature processing)</li>
                <li>🟢 Green: Output (pneumonia detection)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        try:
            fig_neural = create_neural_network_3d()
            st.plotly_chart(fig_neural, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
        except Exception as e:
            st.error("Unable to load neural network visualization")

    with col_tech2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(72,181,255,0.1), rgba(109,76,255,0.05)); padding: 20px; border-radius: 16px; border: 2px solid rgba(72,181,255,0.3); margin-bottom: 16px;'>
            <h4 style='margin-top: 0; color: #22d3ee; text-align: center;'>🚀 Tech Stack Universe</h4>
            <p style='margin: 8px 0; font-size: 0.95em;'>Interactive 3D representation of the technologies and frameworks that make this AI healthcare solution possible.</p>
            <ul style='margin: 12px 0 0 0; padding-left: 20px; font-size: 0.9em;'>
                <li>🐍 Python & AI frameworks</li>
                <li>🌐 Web technologies</li>
                <li>☁️ Cloud & DevOps</li>
                <li>📊 Data science tools</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        try:
            fig_tech = create_tech_portfolio_3d()
            st.plotly_chart(fig_tech, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
        except Exception as e:
            st.error("Unable to load tech portfolio visualization")

    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-header'>📈 Roadmap & Next Steps</h2>", unsafe_allow_html=True)
    st.markdown("""<div class='grid-col'>
        <div class='step-card'>
            <span class='badge'>🔮 Phase 1</span>
            <h4>Refine AI accuracy</h4>
            <p>Improve model robustness using more labeled X-ray data and diverse patient cohorts.</p>
        </div>
        <div class='step-card'>
            <span class='badge'>⚙️ Phase 2</span>
            <h4>Deploy clinical workflows</h4>
            <p>Create an integrated physician dashboard, real-time analysis API, and hospital-ready interface.</p>
        </div>
    </div>""", unsafe_allow_html=True)
    
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""<div class='card'>
            <h4 style='margin-top: 0;'>🌍 Key Objectives</h4>
            <ul style='margin: 0; padding-left: 20px;'>
                <li>🎯 Democratize Healthcare Tech</li>
                <li>🔍 Enable Early Detection</li>
                <li>💡 Support Clinical Decisions</li>
                <li>❤️ Improve Patient Outcomes</li>
                <li>🚀 Advance Medical AI</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""<div class='card'>
            <h4 style='margin-top: 0;'>🎓 Educational Value</h4>
            <ul style='margin: 0; padding-left: 20px;'>
                <li>📖 Open-Source Learning</li>
                <li>🔬 Real-World AI App</li>
                <li>⚕️ Healthcare Insights</li>
                <li>🔓 Model Transparency</li>
                <li>📚 ML Best Practices</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    
    # ==================== PROFESSIONAL FOOTER ====================
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    
    # Professional footer with contact info
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(109,76,255,0.05), rgba(72,181,255,0.03)); 
         padding: 30px 20px; border-radius: 16px; border: 1px solid rgba(109,76,255,0.1); margin: 20px 0; 
         text-align: center;'>
        <h3 style='margin: 0 0 12px 0; background: linear-gradient(135deg, #0ea5e9 0%, #22d3ee 100%); 
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; 
             font-size: 1.3em;'>👨‍💼 Shubham Singh</h3>
        <p style='margin: 0 0 20px 0; color: #6B7280; font-size: 0.95em; line-height: 1.5; max-width: 600px; margin: 0 auto 20px auto;'>
            Full Stack Developer & AI Engineer specializing in healthcare technology, machine learning systems, and innovative medical solutions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Contact links using Streamlit columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <a href='https://github.com/shubham08-cs' target='_blank' 
           style='text-decoration: none; color: #0ea5e9; font-weight: 600; display: block; 
                  padding: 12px; border-radius: 12px; background: rgba(109,76,255,0.1); 
                  border: 1px solid rgba(109,76,255,0.2); text-align: center; margin: 5px;'>
            🐙 GitHub
        </a>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <a href='https://www.linkedin.com/in/shubham-ich/' target='_blank' 
           style='text-decoration: none; color: #0077B5; font-weight: 600; display: block; 
                  padding: 12px; border-radius: 12px; background: rgba(0,119,181,0.1); 
                  border: 1px solid rgba(0,119,181,0.2); text-align: center; margin: 5px;'>
            💼 LinkedIn
        </a>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <a href='mailto:me.shubhamsingh8998@gmail.com' 
           style='text-decoration: none; color: #EA4335; font-weight: 600; display: block; 
                  padding: 12px; border-radius: 12px; background: rgba(234,67,53,0.1); 
                  border: 1px solid rgba(234,67,53,0.2); text-align: center; margin: 5px;'>
            ✉️ Email
        </a>
        """, unsafe_allow_html=True)
    
    # Copyright
    st.markdown("""
    <div style='text-align: center; margin-top: 20px; padding-top: 16px; 
         border-top: 1px solid rgba(109,76,255,0.2); color: #9CA3AF; font-size: 0.85em;'>
        © 2026 Shubham Singh. All rights reserved. | Built with ❤️ using Python, Streamlit & AI
    </div>
    """, unsafe_allow_html=True)

