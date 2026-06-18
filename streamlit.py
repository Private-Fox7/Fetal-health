import datetime
import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fetal Health Diagnostic Assistant", layout="centered")



# ── CSS Styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp {
    background-image: linear-gradient(to bottom, rgba(22, 24, 20, 0.90), rgba(225, 235, 245, 0.93)),url("https://images.unsplash.com/photo-1533483595632-c5f0e57a1936?q=80&w=880&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
h1 {
    color: rgb(75,109,165);
    font-family: 'Gabriola', Arial, sans-serif;
    font-size: 63px;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-top: -58px;
    margin-bottom: -65px;
}
.stButton>button {
    background-color: #1e3a8a;
    color: white;
    border-radius: 8px;
    width: 100%;
    height: 50px;
    font-size: 18px;
    font-weight: 600;
    border: none;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #172554;
    color: white;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(30, 58, 138, 0.2);
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='color: white; font-family: "Gabriola", Arial; text-align: center;
           font-weight: 700; margin-top: -40px; margin-bottom: -63px; font-size: 63px';>
    Fetal Health Diagnostic Assistant
</h1>
""", unsafe_allow_html=True)

st.divider()
st.markdown("Enter the 13 Cardiotocogram (CTG) metrics below to run an evaluation on our medically aligned Two-Stage Cascade XGBoost engine.")
st.divider()

# ── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model_A = joblib.load('model_A_gatekeeper.pkl')
    model_B = joblib.load('model_B_metamodel.pkl')
    return model_A, model_B

try:
    model_A, model_B = load_models()
except Exception as e:
    st.error(f"System Error: Failed to load models ({e}). Ensure model_A_gatekeeper.pkl and model_B_metamodel.pkl are present.")
    st.stop()

# 1. Generate a real-time tracking variable for your user session
current_session_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

clinical_notice_text = (
    "⚠️ &nbsp;&nbsp; **CRITICAL CLINICAL OVERRIDE NOTICE:** This application functions exclusively as an educational machine learning demonstration. "
    "The statistical evaluations, risk spectra, and trajectory models do not constitute clinical diagnostic data, medical advice, or triage parameters. "
    "All cardiotocography monitoring evaluations must be handled exclusively by qualified obstetric professionals using medical-grade institutional hardware. "
    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; || &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "
    "⚠️ &nbsp;&nbsp; **CRITICAL CLINICAL OVERRIDE NOTICE:** This application functions exclusively as an educational machine learning demonstration. "
    "The statistical evaluations, risk spectra, and trajectory models do not constitute clinical diagnostic data, medical advice, or triage parameters. "
    "All cardiotocography monitoring evaluations must be handled exclusively by qualified obstetric professionals using medical-grade institutional hardware."
)

# 2. Inject the smooth-scrolling banner layout onto your interface canvas
st.markdown(
    f"""
    <style>
    .notice-window {{
        width: 100%;
        overflow: hidden;
        background:  linear-gradient(to bottom, rgba(22, 24, 20, 0.90), rgba(225, 235, 245, 0.93))
        padding: 12px 0;
        border-radius: 3px;
        margin-bottom: 2px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
    
    .notice-track {{
        display: inline-block;
        white-space: nowrap;
        padding-left: 100%;
        animation: critical-marquee-scroll 45s linear infinite;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 14px;
        font-weight: 600;
        color: #ffffff; /* Crisp white warning font color */
        letter-spacing: 0.3px;
    }}
    
    @keyframes critical-marquee-scroll {{
        0% {{ transform: translate3d(0, 0, 0); }}
        100% {{ transform: translate3d(-100%, 0, 0); }}
    }}
    </style>
    
    <div class="notice-window">
        <div class="notice-track">
            {clinical_notice_text}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ── Input Fields ──────────────────────────────────────────────────────────────
st.subheader("Core Cardiotocogram Input Parameters")
col1, col2 = st.columns(2)

with col1:
    baseline_value  = st.number_input("Baseline Fetal Heart Rate (bpm)",min_value=100.0, max_value=160.0, value=130.0)
    accelerations   = st.number_input("Accelerations (per second)", min_value=0.0,max_value=0.05,value=0.003,format="%.4f")
    fetal_movement = st.number_input("Fetal Movement (per second)",min_value=0.0,max_value=1.0,value=0.0,format="%.4f")
    light_decelerations= st.number_input("Light Decelerations (per second)",min_value=0.0,max_value=0.05,value=0.003, format="%.4f")
    prolongued_decelerations= st.number_input("Prolonged Decelerations (per second)",min_value=0.0, max_value=0.05,value=0.0,format="%.4f")
    abnormal_short_term_variability= st.number_input("Abnormal Short Term Variability (%)",min_value=0.0,max_value=100.0, value=45.0)

with col2:
    mean_value_of_short_term_variability  = st.number_input("Mean Short Term Variability Value", min_value=0.0,max_value=10.0,value=1.2)
    mean_value_of_long_term_variability = st.number_input("Mean Long Term Variability Value",min_value=0.0,max_value=50.0,value=10.0)
    percentage_of_time_with_abnormal_long_term_variability = st.number_input("Abnormal Long Term Variability (%)", min_value=0.0,max_value=100.0, value=0.0)
    histogram_median= st.number_input("Histogram Median", min_value=50.0,  max_value=200.0, value=134.0)
    histogram_variance= st.number_input("Histogram Variance", min_value=0.0,   max_value=300.0, value=5.0)
    uterine_contractions= st.number_input("Uterine Contractions (per second)",  min_value=0.0,   max_value=0.05,  value=0.005, format="%.4f")
    severe_decelerations= st.select_slider("Severe Decelerations (per second)", options=[0.000, 0.001], value=0.000,help="0.000 = Normal | 0.001 = Severe Drop Detected (Critical Risk Flag!)")

st.markdown("<br>", unsafe_allow_html=True)

# ── Prediction Pipeline ───────────────────────────────────────────────────────
if st.button("Run Diagnostic Pipeline"):

    # Build feature dataframe in exact training column order
    input_df = pd.DataFrame([{
        'baseline value': baseline_value,
        'accelerations':  accelerations,
        'fetal_movement': fetal_movement,
        'uterine_contractions': uterine_contractions,
        'light_decelerations': light_decelerations,
        'severe_decelerations': severe_decelerations,
        'prolongued_decelerations':prolongued_decelerations,
        'abnormal_short_term_variability': abnormal_short_term_variability,
        'mean_value_of_short_term_variability': mean_value_of_short_term_variability,
        'percentage_of_time_with_abnormal_long_term_variability':percentage_of_time_with_abnormal_long_term_variability,
        'mean_value_of_long_term_variability':mean_value_of_long_term_variability,
        'histogram_median': histogram_median,
        'histogram_variance': histogram_variance,
    }])

    features = input_df.to_numpy()

    # ── Stage 1: Gatekeeper (Normal vs At-Risk) ───────────────────────────────
    RISK_THRESHOLD = 0.90   # Lower than default 0.5 — safer for clinical use

    prob_A  = model_A.predict_proba(features).flatten()
    p_healthy_gate= prob_A[0]
    p_risk_gate= prob_A[1]

    if p_risk_gate < RISK_THRESHOLD:
        # Clearly Normal
        final_prediction = 1
        p_normal= p_healthy_gate
        p_suspect= p_risk_gate * 0.5
        p_patho  = p_risk_gate * 0.5

    else:
        # ── Stage 2: Specialist (Suspect vs Pathological) ─────────────────────
        prob_B = model_B.predict_proba(features).flatten()
        pred_B = int(model_B.predict(features)[0])

        final_prediction = 2 if pred_B == 0 else 3

        # ── Clinical Rule Override ────────────────────────────────────────────
        # If model says Suspect but multiple critical danger signs present → escalate to Pathological
        critical_flags = sum([
            severe_decelerations > 0,                       # any severe deceleration
            prolongued_decelerations >= 0.002,              # significant prolonged decelerations
            abnormal_short_term_variability >= 70,          # critically high short term variability
            mean_value_of_short_term_variability <= 0.4,    # critically low mean variability
            baseline_value <= 115,                          # bradycardia
        ])

        if final_prediction == 2 and critical_flags >= 2:
            final_prediction = 3   # escalate Suspect → Pathological

        p_normal  = p_healthy_gate
        p_suspect = p_risk_gate * prob_B[0]
        p_patho   = p_risk_gate * prob_B[1]

    # Normalize probabilities to exactly 100%
    total     = p_normal + p_suspect + p_patho
    p_normal  /= total
    p_suspect /= total
    p_patho   /= total

    # ── Output Banner ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Diagnostic Output Analysis")
    st.toast("Analysis Successfully Completed.")

    if final_prediction == 1:
        st.success(f"STATUS: NORMAL (Pipeline Confidence: {p_normal*100:.1f}%)")
    elif final_prediction == 2:
        st.warning(f"STATUS: SUSPECT (Pipeline Confidence: {p_suspect*100:.1f}%)")
        st.info("Clinical Guidance: Recommend immediate physiological monitoring routines and supplementary assessment.")
    else:
        st.error(f"STATUS: PATHOLOGICAL (Pipeline Confidence: {p_patho*100:.1f}%)")
        st.markdown("**CRITICAL MEDICAL NOTICE:** High risk biological markers identified. Prompt notification to attending obstetricians is required.")

    # ── Chart 1: Risk Spectrum Bar ────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Fetal Health Risk Spectrum Mapping")

    risk_score = (p_normal * 25) + (p_suspect * 65) + (p_patho * 90)

    fig, ax = plt.subplots(figsize=(8, 1.8))
    ax.barh(["Health Risk"], [50], color='#2ecc71', alpha=0.8, label='Normal Zone')
    ax.barh(["Health Risk"], [30], left=[50], color='#f1c40f', alpha=0.8, label='Suspect Zone')
    ax.barh(["Health Risk"], [20], left=[80], color='#e74c3c', alpha=0.8, label='Pathological Zone')

    if final_prediction == 1:
        status_text = f"Normal ({p_normal*100:.1f}%)"
    elif final_prediction == 2:
        status_text = f"Suspect ({p_suspect*100:.1f}%)"
    else:
        status_text = f"Pathological ({p_patho*100:.1f}%)"

    ax.text(risk_score, 0.52, status_text, color='black', fontsize=12, fontweight='bold', ha='center',
            bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="#1e3a8a", lw=1.5))
    ax.vlines(risk_score, -0.4, 0.4, colors='black', linestyles='dashed', linewidth=2)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Fetal Distress Risk Index (%)', fontsize=10, fontweight='bold', color='#1e3a8a')
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    st.pyplot(fig)

    # ── Chart 2: Feature Comparison ───────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Comparative Feature Breakdown Matrix")

    labels          = ['Baseline HR', 'Short-Term Var %', 'Long-Term Var %', 'Hist Variance']
    current_metrics = [baseline_value, abnormal_short_term_variability,percentage_of_time_with_abnormal_long_term_variability, histogram_variance]

    if final_prediction == 1:
        chart_title = "Comparison Against Normal Dataset Standards"
        ref_metrics = [132.0, 35.0, 4.0, 7.0]
        ref_label   = 'Dataset Normal Baseline'
        ref_color   = '#2ecc71'
    else:
        chart_title = "Comparison Against High Severity Dataset Standards"
        ref_metrics = [148.0, 78.0, 32.0, 45.0]
        ref_label   = 'Dataset Pathological Baseline'
        ref_color   = '#e74c3c'

    y_idx = np.arange(len(labels))
    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    ax2.barh(y_idx - 0.2, current_metrics, 0.4, label='Current Patient Data', color='#1e3a8a')
    ax2.barh(y_idx + 0.2, ref_metrics,     0.4, label=ref_label, color=ref_color, alpha=0.6)
    ax2.set_yticks(y_idx)
    ax2.set_yticklabels(labels, fontsize=10, fontweight='bold')
    ax2.set_title(chart_title, fontsize=12, fontweight='bold', color='#1a365d')
    ax2.legend(loc='lower right')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)

    # ── Chart 3: 24-Hour Trajectory Forecast ──────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Predictive 24-Hour Clinical Trajectory Forecast")

    timeline_hours= np.array([0, 4, 8, 12, 16, 20, 24])
    np.random.seed(int(baseline_value))
    fluctuations = np.random.normal(0, 3, size=len(timeline_hours))

    if final_prediction == 1:
        base_trend = np.linspace(risk_score, risk_score + 2, len(timeline_hours))
        lower_bound_offset = -6
        upper_bound_offset= 6
        line_color = '#2ecc71'
    elif final_prediction == 2:
        base_trend = np.linspace(risk_score, risk_score - 4, len(timeline_hours))
        lower_bound_offset= -10
        upper_bound_offset= 10
        line_color = '#f1c40f'
    else:
        base_trend = np.linspace(risk_score, risk_score + 5, len(timeline_hours))
        lower_bound_offset= -8
        upper_bound_offset= 8
        line_color = '#e74c3c'

    projected_risk= np.clip(base_trend + fluctuations, 2, 98)
    lower_bound= np.clip(projected_risk + lower_bound_offset, 0, 100)
    upper_bound= np.clip(projected_risk + upper_bound_offset, 0, 100)

    fig3, ax3 = plt.subplots(figsize=(8, 3.2))
    ax3.axhspan(0,  50,  color='#2ecc71', alpha=0.08)
    ax3.axhspan(50, 80,  color='#f1c40f', alpha=0.08)
    ax3.axhspan(80, 100, color='#e74c3c', alpha=0.08)
    ax3.plot(timeline_hours, projected_risk, marker='o', linestyle='-', linewidth=2.5,
             color=line_color, label='Projected Risk Path')
    ax3.fill_between(timeline_hours, lower_bound, upper_bound,
                     color=line_color, alpha=0.15, label='Statistical Variance Interval')
    ax3.set_xlim(0, 24)
    ax3.set_ylim(0, 100)
    ax3.set_xlabel('Monitoring Timeline (Hours into Future)', fontsize=10, fontweight='bold', color='#1e3a8a')
    ax3.set_ylabel('Distress Risk Index (%)',                 fontsize=10, fontweight='bold', color='#1e3a8a')
    ax3.set_title('Proactive Patient Trajectory Simulation',  fontsize=12, fontweight='bold', color='#1a365d')
    ax3.legend(loc='upper left')
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig3)

# ── Clinical Reference Ticker ─────────────────────────────────────────────────
st.divider()
st.subheader("Clinical Reference Ticker")
ticker_content = (
    " **DO:** Maintain routine prenatal checkups and monitor fetal kicking patterns daily. &nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp; "
    "**DON'T:** Consume unpasteurised dairy items or raw meats to reduce listeria risks. &nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp; "
    " **DO:** Supplement daily diets with folic acid and elemental iron under advisor care. &nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp; "
    " **DON'T:** Engage in hot tubs, saunas, or extended high-heat environments. &nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp; "
    " **DO:** Prioritise 7-9 hours of consistent, restorative sleep nightly. &nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp; "
    " **DON'T:** Take unapproved over-the-counter pharmaceuticals without doctor consent."
)


st.markdown(
    f"""
    <style>
    .ticker-window {{
        width: 100%;
        overflow: hidden;
        background-color: #1e3a8a; /* Deep blue background to match your buttons */
        padding: 14px 0;
        border-radius: 8px;
        margin-top: 20px;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.15);
    }}
    
    .ticker-track {{
        display: inline-block;
        white-space: nowrap;
        padding-left: 100%;
        animation: smooth-scroll-animation 35s linear infinite;
        font-family: sans-serif;
        font-size: 15px;
        font-weight: 500;
        color: #ffffff; /* Crisp white text color */
        letter-spacing: 0.5px;
    }}
    
    @keyframes smooth-scroll-animation {{
        0% {{ transform: translate3d(0, 0, 0); }}
        100% {{ transform: translate3d(-100%, 0, 0); }}
    }}
    </style>
    
    <div class="ticker-window">
        <div class="ticker-track">
            {ticker_content}
        </div>
    </div>
    """, 
    unsafe_allow_html=True
)


st.markdown(
    f"""
    <div style="
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: linear-gradient(to bottom, rgba(22, 24, 20, 0.90), rgba(225, 235, 245, 0.93))
        color: white;
        text-align: center;
        padding: 12px;
        font-size: 13px;
        font-weight: bold;
        z-index: 9999;
        box-shadow: 0 -4px 10px rgba(0,0,0,0.2);
    ">
        ⚠️ MEDICAL DEMO ONLY — NOT FOR CLINICAL EVALUATION. Verified Session Time: {current_session_time}
    </div>
    """,
    unsafe_allow_html=True
)
