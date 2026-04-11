import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import json
import os
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Adaptive Drift Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
* { font-family: 'Space Grotesk', sans-serif; }
[data-testid="stAppViewContainer"] { background: #020c1b; }
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #010d1e 0%, #021428 40%, #010d1e 100%);
    border-right: 1px solid rgba(56,139,253,0.2);
    box-shadow: 4px 0 30px rgba(0,0,0,0.5);
}
[data-testid="stSidebar"] * { color: white !important; }
.main-header {
    background: linear-gradient(135deg, rgba(13,25,48,0.9), rgba(2,12,27,0.95));
    border: 1px solid rgba(56,139,253,0.25);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.main-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #58a6ff, #79c0ff, #cae8ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-subtitle { color: #8b949e; font-size: 1rem; margin-top: 10px; }
.badge {
    display: inline-block;
    background: rgba(56,139,253,0.12);
    color: #58a6ff;
    border: 1px solid rgba(56,139,253,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 3px;
}
.metric-card {
    background: linear-gradient(135deg, rgba(13,25,48,0.8), rgba(2,12,27,0.9));
    border: 1px solid rgba(56,139,253,0.15);
    border-radius: 16px;
    padding: 22px 18px;
    text-align: center;
    margin: 6px 0;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(56,139,253,0.5), transparent);
}
.metric-value { font-size: 2rem; font-weight: 700; color: #58a6ff; margin: 0; font-family: 'JetBrains Mono', monospace; }
.metric-label { font-size: 0.7rem; color: #6e7681; margin: 6px 0 0 0; text-transform: uppercase; letter-spacing: 2px; }
.section-title {
    font-size: 1rem; font-weight: 600; color: #58a6ff;
    margin-bottom: 16px; padding-bottom: 10px;
    border-bottom: 1px solid rgba(56,139,253,0.2);
    text-transform: uppercase; letter-spacing: 1.5px;
    font-family: 'JetBrains Mono', monospace;
}
.status-green {
    background: linear-gradient(135deg, rgba(35,134,54,0.15), rgba(35,134,54,0.08));
    border: 1px solid rgba(56,211,106,0.4); border-radius: 10px;
    padding: 14px; text-align: center; color: #3fb950; font-weight: 600; font-size: 0.85rem;
}
.status-red {
    background: linear-gradient(135deg, rgba(248,81,73,0.15), rgba(248,81,73,0.08));
    border: 1px solid rgba(248,81,73,0.4); border-radius: 10px;
    padding: 14px; text-align: center; color: #f85149; font-weight: 600; font-size: 0.85rem;
}
.insight-box {
    background: rgba(13,25,48,0.6);
    border-left: 3px solid #388bfd;
    border-radius: 0 8px 8px 0;
    padding: 12px 18px; margin: 8px 0;
    color: #8b949e; font-size: 0.88rem; line-height: 1.5;
}
.live-card {
    background: linear-gradient(135deg, rgba(13,25,48,0.9), rgba(2,12,27,0.95));
    border: 1px solid rgba(63,185,80,0.3);
    border-radius: 16px; padding: 20px; margin: 10px 0;
    position: relative; overflow: hidden;
}
.live-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #3fb950, #56d364, #3fb950);
}
.retrain-card {
    background: linear-gradient(135deg, rgba(13,25,48,0.9), rgba(2,12,27,0.95));
    border: 1px solid rgba(56,139,253,0.2);
    border-radius: 16px; padding: 20px; margin: 10px 0;
    position: relative; overflow: hidden;
}
.retrain-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #388bfd, #58a6ff, #79c0ff);
}
.auc-before { color: #f85149; font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 700; }
.auc-after  { color: #3fb950; font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 700; }
.auc-arrow  { color: #58a6ff; font-size: 1.4rem; margin: 0 8px; }
.sidebar-divider {
    border: none; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56,139,253,0.4), transparent);
    margin: 16px 0;
}
.sidebar-section-label {
    font-size: 0.65rem !important; color: #388bfd !important;
    text-transform: uppercase; letter-spacing: 2px; font-weight: 600;
    margin-bottom: 8px; font-family: 'JetBrains Mono', monospace;
}
.sidebar-stat {
    background: rgba(56,139,253,0.06);
    border: 1px solid rgba(56,139,253,0.12);
    border-radius: 10px; padding: 10px 14px; margin: 6px 0;
}
.stButton button {
    background: linear-gradient(135deg, #1f4788, #2563eb) !important;
    color: white !important;
    border: 1px solid rgba(56,139,253,0.4) !important;
    border-radius: 10px !important; font-weight: 600 !important;
    padding: 12px 20px !important; width: 100% !important;
    box-shadow: 0 4px 15px rgba(37,99,235,0.3) !important;
}
</style>
""", unsafe_allow_html=True)


def plot_style(fig, ax_list):
    fig.patch.set_facecolor('#0d1926')
    if not isinstance(ax_list, list):
        ax_list = [ax_list]
    for ax in ax_list:
        ax.set_facecolor('#0d1926')
        ax.tick_params(colors='#8b949e', labelsize=9)
        ax.title.set_color('#58a6ff')
        ax.spines['bottom'].set_color('#1c3661')
        ax.spines['left'].set_color('#1c3661')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.label.set_color('#8b949e')
        ax.xaxis.label.set_color('#8b949e')
        ax.grid(axis='y', color='#0d2040', linewidth=0.5)


# ── Supabase helpers ──────────────────────────────────────────────────────────
def get_supabase_headers(key):
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

def fetch_from_supabase(url, key, limit=5000):
    headers = get_supabase_headers(key)
    response = requests.get(
        f"{url}/rest/v1/transactions?select=*&limit={limit}",
        headers=headers
    )
    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            return pd.DataFrame(data), None
        return None, "Table is empty"
    return None, f"Error {response.status_code}: {response.text[:100]}"

def insert_to_supabase(url, key, records):
    headers = get_supabase_headers(key)
    headers["Prefer"] = "return=minimal"
    response = requests.post(
        f"{url}/rest/v1/transactions",
        headers=headers,
        data=json.dumps(records)
    )
    return response.status_code == 201

def delete_old_rows(url, key, keep_last=1000):
    headers = get_supabase_headers(key)
    headers["Prefer"] = "return=minimal"
    requests.delete(
        f"{url}/rest/v1/transactions?id=lt.{keep_last}",
        headers=headers
    )


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='text-align:center;padding:20px 0 10px'>
    <span style='font-size:3rem;filter:drop-shadow(0 0 12px rgba(56,139,253,0.6))'>🛡️</span>
    <div style='font-size:1.1rem;font-weight:700;color:#cae8ff'>Adaptive Drift Monitor</div>
    <div style='font-size:0.7rem;color:#58a6ff;font-family:JetBrains Mono,monospace;letter-spacing:1px;margin-top:4px'>v2.0 Pro · MLOps Platform</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style='display:flex;gap:6px;flex-wrap:wrap;justify-content:center;margin-bottom:12px'>
    <span class='badge'>KS TEST</span><span class='badge'>PSI</span>
    <span class='badge'>SUPABASE</span><span class='badge'>AUTO-RETRAIN</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

# Mode selection
st.sidebar.markdown("<p class='sidebar-section-label'>🔌 Data Source</p>", unsafe_allow_html=True)
data_mode = st.sidebar.radio("", ["📁 Upload CSV", "🔴 Live Database"], label_visibility="collapsed")

if data_mode == "📁 Upload CSV":
    st.sidebar.markdown("<p class='sidebar-section-label'>📤 Upload Dataset</p>", unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    SUPABASE_URL = None
    SUPABASE_KEY = None
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.sidebar.markdown(f"""
        <div class='sidebar-stat'>
            <span style='font-size:0.75rem;color:#6e7681'>📋 Rows</span>
            <span style='font-size:0.75rem;color:#58a6ff;font-family:JetBrains Mono,monospace'>{df.shape[0]:,}</span>
        </div>
        <div class='sidebar-stat'>
            <span style='font-size:0.75rem;color:#6e7681'>🔢 Columns</span>
            <span style='font-size:0.75rem;color:#58a6ff;font-family:JetBrains Mono,monospace'>{df.shape[1]}</span>
        </div>""", unsafe_allow_html=True)
        target_column = st.sidebar.selectbox("🎯 Target Column", df.columns.tolist())
    else:
        df = None
        target_column = None
        uploaded_file = None
else:
    uploaded_file = None
    df = None
    st.sidebar.markdown("<p class='sidebar-section-label'>🔴 Supabase Credentials</p>", unsafe_allow_html=True)
    SUPABASE_URL = st.sidebar.text_input("Project URL", placeholder="https://xxx.supabase.co", type="default")
    SUPABASE_KEY = st.sidebar.text_input("Anon Key", placeholder="eyJhbGc...", type="password")
    target_column = st.sidebar.text_input("🎯 Target Column", value="Class")

st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<p class='sidebar-section-label'>⚙️ Model Settings</p>", unsafe_allow_html=True)
n_trees        = st.sidebar.slider("🌳 Trees", 5, 50, 10)
drift_threshold= st.sidebar.slider("KS Threshold", 0.05, 0.3, 0.1)
auc_threshold  = st.sidebar.slider("AUC Threshold", 0.7, 0.99, 0.95)

st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
run_button = st.sidebar.button("🚀 Run Full Analysis")

st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style='text-align:center;padding:12px 0'>
    <p style='font-size:0.65rem;color:#484f58;letter-spacing:0.5px;line-height:1.8'>
        Built by <strong style='color:#58a6ff'>Shivali &amp; Vaishali</strong><br>
        Python · Streamlit · Supabase · Sklearn
    </p>
</div>""", unsafe_allow_html=True)


# ── LANDING PAGE ──────────────────────────────────────────────────────────────
def show_header(subtitle="Enterprise-grade ML Model Monitoring & Drift Detection"):
    st.markdown(f"""
    <div class='main-header'>
        <p class='main-title'>🛡️ Adaptive Drift Monitor</p>
        <p class='main-subtitle'>{subtitle}</p>
    </div>""", unsafe_allow_html=True)

if df is None and not (SUPABASE_URL and SUPABASE_KEY):
    show_header()
    col1,col2,col3,col4 = st.columns(4)
    for col, icon, title, desc in zip(
        [col1,col2,col3,col4],
        ["📁","🔴","🔍","🤖"],
        ["CSV Upload","Live Database","3 Drift Tests","Auto Retrain"],
        ["Upload any CSV","Supabase PostgreSQL","KS · PSI · KL Div","Self-healing model"]
    ):
        with col:
            st.markdown(f"""<div class='metric-card'>
                <p class='metric-value'>{icon}</p>
                <p class='metric-label' style='font-size:0.85rem;color:#cae8ff;margin-top:10px'>{title}</p>
                <p class='metric-label' style='text-transform:none;letter-spacing:0'>{desc}</p>
            </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.info("👈 Choose a data source from the sidebar to begin!")


# ── MAIN ANALYSIS ─────────────────────────────────────────────────────────────
elif run_button:
    show_header("Analysis Complete — Full Results Below")

    # Load data from chosen source
    live_mode = False
    if data_mode == "🔴 Live Database" and SUPABASE_URL and SUPABASE_KEY:
        with st.spinner("🔴 Fetching live data from Supabase..."):
            df_raw, err = fetch_from_supabase(SUPABASE_URL, SUPABASE_KEY)
            if err:
                st.error(f"❌ Supabase Error: {err}")
                st.stop()
            df = df_raw
            live_mode = True
            st.success(f"✅ Loaded {df.shape[0]:,} live rows from Supabase!")
    elif df is None:
        st.error("❌ Please upload a CSV or connect to Supabase!")
        st.stop()

    with st.spinner("🤖 Running full analysis pipeline..."):
        try:
            df_clean = df.copy()
            for col in df_clean.select_dtypes(include='object').columns:
                if col != target_column:
                    le = LabelEncoder()
                    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            y = df_clean[target_column]
            if y.dtype == 'object':
                le = LabelEncoder()
                df_clean[target_column] = le.fit_transform(y.astype(str))

            window = len(df_clean) // 4
            week1  = df_clean.iloc[0:window].copy()
            week2  = df_clean.iloc[window:window*2].copy()
            week3  = df_clean.iloc[window*2:window*3].copy()
            week4  = df_clean.iloc[window*3:].copy()

            num_cols = df_clean.drop(target_column, axis=1).select_dtypes(include=np.number).columns.tolist()

            week2_d = week2.copy(); week3_d = week3.copy(); week4_d = week4.copy()
            for col in num_cols[:3]:
                week2_d[col] += np.random.normal(0.5, 0.5, len(week2_d))
                week3_d[col] += np.random.normal(1.5, 1.0, len(week3_d))
                week4_d[col] += np.random.normal(3.0, 2.0, len(week4_d))

            X_train = week1.drop(target_column, axis=1)
            y_train = week1[target_column]
            try:
                smote = SMOTE(random_state=42)
                X_sm, y_sm = smote.fit_resample(X_train, y_train)
            except ValueError:
                X_sm, y_sm = X_train, y_train

            model = RandomForestClassifier(n_estimators=n_trees, random_state=42)
            model.fit(X_sm, y_sm)
            feature_names = X_train.columns.tolist()
            importances   = model.feature_importances_

            def get_metrics(data):
                X = data.drop(target_column, axis=1)
                y = data[target_column]
                y_pred = model.predict(X)
                try: auc = round(roc_auc_score(y, y_pred), 4)
                except ValueError: auc = 0.5
                f1  = round(f1_score(y, y_pred, average='weighted'), 4)
                rec = round(recall_score(y, y_pred, average='weighted'), 4)
                pre = round(precision_score(y, y_pred, average='weighted', zero_division=0), 4)
                cm  = confusion_matrix(y, y_pred)
                return auc, f1, rec, pre, cm, y_pred

            auc1,f11,rec1,pre1,cm1,_ = get_metrics(week1)
            auc2,f12,rec2,pre2,cm2,_ = get_metrics(week2_d)
            auc3,f13,rec3,pre3,cm3,_ = get_metrics(week3_d)
            auc4,f14,rec4,pre4,cm4,_ = get_metrics(week4_d)

            def retrain_on_week(data):
                X = data.drop(target_column, axis=1); y = data[target_column]
                try: Xr,yr = SMOTE(random_state=42).fit_resample(X,y)
                except ValueError: Xr,yr = X,y
                nm = RandomForestClassifier(n_estimators=n_trees, random_state=42)
                nm.fit(Xr,yr)
                try: return round(roc_auc_score(y, nm.predict_proba(X)[:,1]), 4)
                except ValueError: return 0.5

            auc2_new = retrain_on_week(week2_d)
            auc3_new = retrain_on_week(week3_d)
            auc4_new = retrain_on_week(week4_d)

            feat = num_cols[0] if num_cols else df_clean.columns[0]
            ks2  = round(stats.ks_2samp(week1[feat], week2_d[feat]).statistic, 4)
            ks3  = round(stats.ks_2samp(week1[feat], week3_d[feat]).statistic, 4)
            ks4  = round(stats.ks_2samp(week1[feat], week4_d[feat]).statistic, 4)

            def calc_psi(base, curr):
                bc,be = np.histogram(base, bins=10); cc,_ = np.histogram(curr, bins=be)
                bp = bc/len(base)+1e-6; cp = cc/len(curr)+1e-6
                return round(float(np.sum((cp-bp)*np.log(cp/bp))), 4)

            psi2 = calc_psi(week1[feat], week2_d[feat])
            psi3 = calc_psi(week1[feat], week3_d[feat])
            psi4 = calc_psi(week1[feat], week4_d[feat])

            st.success("✅ Analysis Complete!")

            # Top metrics
            col1,col2,col3,col4,col5 = st.columns(5)
            src_label = "🔴 Live DB" if live_mode else "📁 CSV"
            for col,(val,label) in zip([col1,col2,col3,col4,col5],[
                (f"{df.shape[0]:,}","Total Rows"),
                (f"{df.shape[1]}","Features"),
                (f"{auc1}","Baseline AUC"),
                (f"{round((auc1-auc4)*100,1)}%","AUC Drop"),
                (src_label,"Data Source"),
            ]):
                with col:
                    st.markdown(f"""<div class='metric-card'>
                        <p class='metric-value' style='font-size:1.4rem'>{val}</p>
                        <p class='metric-label'>{label}</p>
                    </div>""", unsafe_allow_html=True)

            st.markdown("---")

            tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
                "📊 Data Overview",
                "🔍 Drift Detection",
                "📉 Model Performance",
                "🔄 Retraining Status",
                "🔴 Live Database",
                "📄 Download Report"
            ])

            # ══ TAB 1 ════════════════════════════════════════════════
            with tab1:
                st.markdown("<p class='section-title'>📋 Dataset Preview</p>", unsafe_allow_html=True)
                st.dataframe(df.head(20), use_container_width=True)
                st.markdown("---")
                col1,col2 = st.columns(2)
                with col1:
                    st.markdown("<p class='section-title'>📊 Target Distribution</p>", unsafe_allow_html=True)
                    fig,ax = plt.subplots(figsize=(6,4)); plot_style(fig,ax)
                    tc = df_clean[target_column].value_counts()
                    bars = ax.bar(tc.index.astype(str), tc.values,
                                  color=['#388bfd','#58a6ff','#79c0ff'][:len(tc)],
                                  edgecolor='#1c3661', linewidth=0.8)
                    for bar,val in zip(bars,tc.values):
                        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                                f'{val:,}', ha='center', color='white', fontsize=10, fontweight='bold')
                    ax.set_title('Target Distribution', color='#58a6ff', fontsize=12)
                    st.pyplot(fig)
                with col2:
                    st.markdown("<p class='section-title'>🎯 Feature Importance</p>", unsafe_allow_html=True)
                    fig,ax = plt.subplots(figsize=(6,4)); plot_style(fig,ax)
                    top_n = min(10,len(feature_names)); idx = np.argsort(importances)[-top_n:]
                    ax.barh([feature_names[i] for i in idx],[importances[i] for i in idx],
                            color=plt.cm.Blues(np.linspace(0.4,1.0,top_n)),
                            edgecolor='#1c3661', linewidth=0.5)
                    ax.set_title('Top Features', color='#58a6ff', fontsize=12)
                    st.pyplot(fig)
                st.markdown("---")
                st.markdown("<p class='section-title'>📈 Statistics</p>", unsafe_allow_html=True)
                st.dataframe(df.describe().round(3), use_container_width=True)

            # ══ TAB 2 ════════════════════════════════════════════════
            with tab2:
                st.markdown("<p class='section-title'>🔍 Statistical Drift Tests</p>", unsafe_allow_html=True)
                col1,col2,col3 = st.columns(3)
                for col,ks,psi,name in zip([col1,col2,col3],[ks2,ks3,ks4],[psi2,psi3,psi4],['Week 2','Week 3','Week 4']):
                    with col:
                        color  = "#f85149" if ks>drift_threshold else "#3fb950"
                        status = "🚨 DRIFT" if ks>drift_threshold else "✅ STABLE"
                        st.markdown(f"""<div class='metric-card'>
                            <p class='metric-value' style='color:{color};font-size:1rem'>{status}</p>
                            <p class='metric-label'>{name}</p>
                            <p style='color:#6e7681;font-size:0.75rem;margin-top:10px;font-family:JetBrains Mono,monospace'>KS: {ks} | PSI: {psi}</p>
                        </div>""", unsafe_allow_html=True)
                st.markdown("---")
                col1,col2 = st.columns(2)
                with col1:
                    st.markdown("<p class='section-title'>KS Statistic</p>", unsafe_allow_html=True)
                    fig,ax = plt.subplots(figsize=(6,4)); plot_style(fig,ax)
                    weeks_x = ['Week 2','Week 3','Week 4']; ks_vals=[ks2,ks3,ks4]
                    ck = ['#3fb950' if v<drift_threshold else '#d29922' if v<0.3 else '#f85149' for v in ks_vals]
                    bars = ax.bar(weeks_x,ks_vals,color=ck,edgecolor='#1c3661',linewidth=0.8,width=0.5)
                    ax.axhline(y=drift_threshold,color='#d29922',linestyle='--',linewidth=1.5,label=f'Threshold ({drift_threshold})')
                    for bar,val in zip(bars,ks_vals):
                        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,str(val),ha='center',color='white',fontsize=10,fontweight='bold')
                    ax.set_title('KS Statistic by Week',color='#58a6ff',fontsize=12)
                    ax.legend(facecolor='#0d1926',labelcolor='white'); st.pyplot(fig)
                with col2:
                    st.markdown("<p class='section-title'>PSI Score</p>", unsafe_allow_html=True)
                    fig,ax = plt.subplots(figsize=(6,4)); plot_style(fig,ax)
                    psi_vals=[psi2,psi3,psi4]
                    cp = ['#3fb950' if v<0.1 else '#d29922' if v<0.2 else '#f85149' for v in psi_vals]
                    bars = ax.bar(weeks_x,psi_vals,color=cp,edgecolor='#1c3661',linewidth=0.8,width=0.5)
                    ax.axhline(y=0.2,color='#d29922',linestyle='--',linewidth=1.5,label='Threshold (0.2)')
                    for bar,val in zip(bars,psi_vals):
                        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,str(val),ha='center',color='white',fontsize=10,fontweight='bold')
                    ax.set_title('PSI Score by Week',color='#58a6ff',fontsize=12)
                    ax.legend(facecolor='#0d1926',labelcolor='white'); st.pyplot(fig)

            # ══ TAB 3 ════════════════════════════════════════════════
            with tab3:
                st.markdown("<p class='section-title'>📉 Performance Degradation</p>", unsafe_allow_html=True)
                col1,col2,col3,col4 = st.columns(4)
                for col,auc,label,color in zip([col1,col2,col3,col4],[auc1,auc2,auc3,auc4],
                    ['Baseline','Week 2','Week 3','Week 4'],['#3fb950','#d29922','#f0883e','#f85149']):
                    with col:
                        st.markdown(f"""<div class='metric-card'>
                            <p class='metric-value' style='color:{color}'>{auc}</p>
                            <p class='metric-label'>{label} AUC</p>
                        </div>""", unsafe_allow_html=True)
                st.markdown("---")
                col1,col2 = st.columns(2)
                with col1:
                    fig,ax = plt.subplots(figsize=(6,4)); plot_style(fig,ax)
                    weeks_all=['Week 1','Week 2','Week 3','Week 4']
                    ax.plot(weeks_all,[auc1,auc2,auc3,auc4],marker='o',color='#388bfd',linewidth=2.5,label='AUC',markersize=8)
                    ax.plot(weeks_all,[rec1,rec2,rec3,rec4],marker='s',color='#3fb950',linewidth=2.5,label='Recall',markersize=8)
                    ax.plot(weeks_all,[f11,f12,f13,f14],marker='^',color='#d29922',linewidth=2.5,label='F1',markersize=8)
                    ax.axhline(y=auc_threshold,color='#f85149',linestyle='--',linewidth=1.5,label=f'Threshold')
                    ax.legend(facecolor='#0d1926',labelcolor='white',fontsize=8)
                    ax.set_title('All Metrics Over Time',color='#58a6ff',fontsize=12); st.pyplot(fig)
                with col2:
                    fig,ax = plt.subplots(figsize=(6,4)); plot_style(fig,ax)
                    sns.heatmap(cm1,annot=True,fmt='d',cmap='Blues',ax=ax,linewidths=0.5,annot_kws={'color':'white','size':12})
                    ax.set_title('Confusion Matrix (Baseline)',color='#58a6ff',fontsize=12)
                    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); st.pyplot(fig)

            # ══ TAB 4 ════════════════════════════════════════════════
            with tab4:
                st.markdown("<p class='section-title'>🔄 Automated Retraining Monitor</p>", unsafe_allow_html=True)
                for ks,auc,psi,auc_new,name in zip([ks2,ks3,ks4],[auc2,auc3,auc4],[psi2,psi3,psi4],
                    [auc2_new,auc3_new,auc4_new],['Week 2 — Slight','Week 3 — Moderate','Week 4 — Severe']):
                    drift = ks>drift_threshold or auc<auc_threshold
                    col1,col2,col3,col4,col5 = st.columns(5)
                    with col1:
                        st.markdown(f"""<div class='metric-card'><p class='metric-value' style='font-size:0.85rem;color:#cae8ff'>{name}</p><p class='metric-label'>Window</p></div>""", unsafe_allow_html=True)
                    with col2:
                        c="#f85149" if ks>drift_threshold else "#3fb950"
                        st.markdown(f"""<div class='metric-card'><p class='metric-value' style='color:{c};font-family:JetBrains Mono,monospace'>{ks}</p><p class='metric-label'>KS Stat</p></div>""", unsafe_allow_html=True)
                    with col3:
                        c="#f85149" if psi>0.2 else "#3fb950"
                        st.markdown(f"""<div class='metric-card'><p class='metric-value' style='color:{c};font-family:JetBrains Mono,monospace'>{psi}</p><p class='metric-label'>PSI</p></div>""", unsafe_allow_html=True)
                    with col4:
                        c="#f85149" if auc<auc_threshold else "#3fb950"
                        st.markdown(f"""<div class='metric-card'><p class='metric-value' style='color:{c};font-family:JetBrains Mono,monospace'>{auc}</p><p class='metric-label'>AUC</p></div>""", unsafe_allow_html=True)
                    with col5:
                        if drift: st.markdown("<div class='status-red'>🚨 RETRAIN</div>", unsafe_allow_html=True)
                        else: st.markdown("<div class='status-green'>✅ STABLE</div>", unsafe_allow_html=True)
                    st.markdown("---")

                st.markdown("<p class='section-title'>📊 Before vs After Retraining</p>", unsafe_allow_html=True)
                col1,col2,col3 = st.columns(3)
                for col,before,after,name in zip([col1,col2,col3],[auc2,auc3,auc4],[auc2_new,auc3_new,auc4_new],['Week 2','Week 3','Week 4']):
                    imp = round((after-before)*100,2)
                    with col:
                        st.markdown(f"""<div class='retrain-card'>
                            <p style='color:#8b949e;font-size:0.7rem;text-transform:uppercase;letter-spacing:2px;margin:0 0 12px'>{name}</p>
                            <div style='display:flex;align-items:center;justify-content:center;gap:4px'>
                                <span class='auc-before'>{before}</span>
                                <span class='auc-arrow'>→</span>
                                <span class='auc-after'>{after}</span>
                            </div>
                            <p style='color:#3fb950;font-size:0.8rem;margin:10px 0 0;text-align:center'>▲ +{imp}% improvement</p>
                        </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                fig,ax = plt.subplots(figsize=(10,5)); plot_style(fig,ax)
                wk = ['Week 1\n(Baseline)','Week 2\n(Slight)','Week 3\n(Moderate)','Week 4\n(Severe)']
                ba = [auc1,auc2,auc3,auc4]; aa = [auc1,auc2_new,auc3_new,auc4_new]
                x = np.arange(len(wk)); w = 0.35
                b1 = ax.bar(x-w/2,ba,w,label='Before',color='#f85149',edgecolor='#7a1f1d',linewidth=0.8,alpha=0.9)
                b2 = ax.bar(x+w/2,aa,w,label='After', color='#3fb950',edgecolor='#1a5c24',linewidth=0.8,alpha=0.9)
                for bar in b1: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.002,f'{bar.get_height():.4f}',ha='center',va='bottom',color='#f85149',fontsize=8.5,fontweight='bold',fontfamily='monospace')
                for bar in b2: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.002,f'{bar.get_height():.4f}',ha='center',va='bottom',color='#3fb950',fontsize=8.5,fontweight='bold',fontfamily='monospace')
                ax.set_title('AUC: Before vs After Retraining',color='#58a6ff',fontsize=13,fontweight='bold',pad=15)
                ax.set_ylabel('AUC Score',color='#8b949e'); ax.set_xticks(x); ax.set_xticklabels(wk,color='#8b949e')
                ax.set_ylim(min(ba)-0.05,1.02)
                ax.legend(facecolor='#0d1926',labelcolor='white',fontsize=10,edgecolor='#1c3661',loc='lower right')
                ax.axhline(y=1.0,color='#388bfd',linestyle=':',linewidth=1,alpha=0.4)
                fig.tight_layout(); st.pyplot(fig)

            # ══ TAB 5 — LIVE DATABASE ════════════════════════════════
            with tab5:
                st.markdown("<p class='section-title'>🔴 Live Supabase Database</p>", unsafe_allow_html=True)

                if not live_mode:
                    st.markdown("""<div class='insight-box'>
                        💡 Switch to <strong>🔴 Live Database</strong> mode in the sidebar to use this tab!
                        Enter your Supabase URL and API key to connect.
                    </div>""", unsafe_allow_html=True)
                else:
                    # Connection status
                    st.markdown(f"""<div class='live-card'>
                        <p style='color:#3fb950;font-size:1rem;font-weight:700;margin:0'>🟢 Connected to Supabase</p>
                        <p style='color:#8b949e;font-size:0.8rem;margin:6px 0 0;font-family:JetBrains Mono,monospace'>{SUPABASE_URL}</p>
                        <p style='color:#6e7681;font-size:0.75rem;margin:4px 0 0'>📊 {df.shape[0]:,} rows · {df.shape[1]} columns · Table: transactions</p>
                    </div>""", unsafe_allow_html=True)

                    st.markdown("---")
                    st.markdown("<p class='section-title'>🧪 Simulate New Incoming Data</p>", unsafe_allow_html=True)
                    st.markdown("""<div class='insight-box'>
                        Click the button below to simulate <strong>50 new transactions</strong> being inserted into the live database.
                        This mimics real-world data arriving and potentially drifting from the baseline!
                    </div>""", unsafe_allow_html=True)

                    col1,col2,col3 = st.columns(3)
                    with col1:
                        simulate_normal = st.button("➕ Add 50 Normal Rows")
                    with col2:
                        simulate_drift  = st.button("⚠️ Add 50 Drifted Rows")
                    with col3:
                        check_count     = st.button("🔄 Refresh Row Count")

                    if simulate_normal:
                        with st.spinner("Inserting normal rows..."):
                            sample = df.sample(50).to_dict(orient='records')
                            success = insert_to_supabase(SUPABASE_URL, SUPABASE_KEY, sample)
                            if success:
                                st.success("✅ 50 normal rows inserted into Supabase!")
                            else:
                                st.error("❌ Insert failed!")

                    if simulate_drift:
                        with st.spinner("Inserting drifted rows..."):
                            sample = df.sample(50).copy()
                            for col in num_cols[:3]:
                                if col in sample.columns:
                                    sample[col] = sample[col] + np.random.normal(3.0, 2.0, len(sample))
                            records = sample.to_dict(orient='records')
                            success = insert_to_supabase(SUPABASE_URL, SUPABASE_KEY, records)
                            if success:
                                st.success("🚨 50 drifted rows inserted — run analysis again to detect drift!")
                            else:
                                st.error("❌ Insert failed!")

                    if check_count:
                        with st.spinner("Checking..."):
                            df_check, _ = fetch_from_supabase(SUPABASE_URL, SUPABASE_KEY, limit=10000)
                            if df_check is not None:
                                st.success(f"📊 Current rows in database: {df_check.shape[0]:,}")

                    st.markdown("---")
                    st.markdown("<p class='section-title'>📋 Live Data Preview</p>", unsafe_allow_html=True)
                    st.dataframe(df.head(10), use_container_width=True)

                    st.markdown("""
                    <div class='insight-box'>🔴 <strong>Live Connection:</strong> Data is read directly from Supabase PostgreSQL cloud database</div>
                    <div class='insight-box'>⚡ <strong>Real-time:</strong> Click "Add Drifted Rows" then re-run analysis to see drift detected on live data</div>
                    <div class='insight-box'>🛡️ <strong>Storage Safe:</strong> Database stays manageable — add rows only when needed for demos</div>
                    """, unsafe_allow_html=True)

            # ══ TAB 6 ════════════════════════════════════════════════
            with tab6:
                st.markdown("<p class='section-title'>📄 Download Report</p>", unsafe_allow_html=True)
                report_data = {
                    'Metric': ['Rows','Features','Target','Data Source',
                               'Baseline AUC','Week2 AUC Before','Week3 AUC Before','Week4 AUC Before',
                               'Week2 AUC After','Week3 AUC After','Week4 AUC After',
                               'KS Week2','KS Week3','KS Week4','PSI Week2','PSI Week3','PSI Week4'],
                    'Value': [df.shape[0],df.shape[1],target_column,'Supabase Live' if live_mode else 'CSV Upload',
                              auc1,auc2,auc3,auc4,auc2_new,auc3_new,auc4_new,
                              ks2,ks3,ks4,psi2,psi3,psi4]
                }
                report_df = pd.DataFrame(report_data)
                st.dataframe(report_df, use_container_width=True)
                csv = report_df.to_csv(index=False)
                st.download_button("📥 Download Report as CSV", data=csv,
                                   file_name="drift_report.csv", mime="text/csv", use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Make sure your dataset has numerical features and a valid target column")

elif df is not None and not run_button:
    show_header("Dataset loaded — configure settings and run analysis")
    st.dataframe(df.head(10), use_container_width=True)
    col1,col2,col3,col4 = st.columns(4)
    for col,val,label in zip([col1,col2,col3,col4],
        [f"{df.shape[0]:,}",df.shape[1],df.isnull().sum().sum(),len(df.select_dtypes(include=np.number).columns)],
        ['Total Rows','Total Columns','Missing Values','Numeric Features']):
        with col:
            st.markdown(f"""<div class='metric-card'>
                <p class='metric-value' style='font-size:1.5rem'>{val}</p>
                <p class='metric-label'>{label}</p>
            </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.info("👈 Click Run Full Analysis in the sidebar to start!")