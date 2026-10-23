import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
)
from imblearn.over_sampling import SMOTE

# ==============================================================================
# PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="VIT ML Mini-Project Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1f77b4 0%, #0d3b66 100%);
        padding: 22px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f8f9fa;
        border-left: 5px solid #1f77b4;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #0d3b66;
    }
    .metric-label {
        font-size: 13px;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: #f1f3f5;
        border-radius: 6px 6px 0 0;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to find data files
def find_data_file(filename):
    candidates = [
        filename,
        os.path.join("26MML0020_Rohan", filename),
        os.path.join("26MML0045_Vedant", filename),
        os.path.join("..", filename)
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return filename

# ==============================================================================
# CACHED DATA & MODEL PIPELINES
# ==============================================================================
@st.cache_data
def load_telco_data():
    csv_path = find_data_file("telco_preprocessed.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        raw_path = find_data_file("WA_Fn-UseC_-Telco-Customer-Churn.csv")
        raw_df = pd.read_csv(raw_path)
        df = raw_df.copy()
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
        df = df.dropna(subset=['TotalCharges']).reset_index(drop=True)
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'])
        binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
        binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map(binary_map)
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    return df

@st.cache_resource
def train_telco_models():
    df = load_telco_data()
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)
    
    # SMOTE for Logistic Regression
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)
    
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42).fit(X_train_sm, y_train_sm)
    rf = RandomForestClassifier(n_estimators=100, max_depth=9, class_weight='balanced', random_state=42).fit(X_train, y_train)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42).fit(X_train, y_train)
    
    return {
        'df': df,
        'feature_names': feature_names,
        'scaler': scaler,
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'models': {
            'Random Forest (Class-Weighted)': (rf, False),
            'Logistic Regression (SMOTE)': (lr, True),
            'Gradient Boosting (Ensemble)': (gb, False)
        }
    }

@st.cache_data
def load_car_data():
    csv_path = find_data_file("Car details v3.csv")
    raw_df = pd.read_csv(csv_path)
    df = raw_df.copy()
    
    CURRENT_YEAR = 2020
    df['vehicle_age'] = CURRENT_YEAR - df['year']
    
    df['mileage_kmpl'] = df['mileage'].str.extract(r'([0-9.]+)').astype(float)
    df.loc[df['mileage_kmpl'] == 0, 'mileage_kmpl'] = np.nan
    df['engine_cc'] = df['engine'].str.extract(r'([0-9.]+)').astype(float)
    df['max_power_bhp'] = df['max_power'].str.extract(r'([0-9.]+)').astype(float)
    df.loc[df['max_power_bhp'] == 0, 'max_power_bhp'] = np.nan
    df['seats'] = pd.to_numeric(df['seats'], errors='coerce')
    
    df['brand'] = df['name'].str.split().str[0]
    for col in ['mileage_kmpl', 'engine_cc', 'max_power_bhp', 'seats']:
        df[col] = df[col].fillna(df.groupby('brand')[col].transform('median')).fillna(df[col].median())
        
    df['log_selling_price'] = np.log(df['selling_price'])
    drop_cols = ['name', 'year', 'selling_price', 'mileage', 'engine', 'max_power', 'torque', 'brand']
    clean_df = df.drop(columns=[c for c in drop_cols if c in df.columns]).dropna().reset_index(drop=True)
    
    clean_df['km_per_year'] = clean_df['km_driven'] / (clean_df['vehicle_age'] + 0.5)
    clean_df['power_displacement_density'] = clean_df['max_power_bhp'] / (clean_df['engine_cc'] / 1000.0)
    clean_df['engine_age_interaction'] = (clean_df['engine_cc'] * clean_df['vehicle_age']) / 1000.0
    clean_df['efficiency_power_ratio'] = clean_df['mileage_kmpl'] / (clean_df['max_power_bhp'] + 1e-3)
    clean_df['log_odometer'] = np.log1p(clean_df['km_driven'])
    
    cat_cols = ['fuel', 'seller_type', 'transmission', 'owner']
    encoded_df = pd.get_dummies(clean_df, columns=cat_cols, drop_first=True, dtype=int)
    
    return clean_df, encoded_df

@st.cache_resource
def train_car_models():
    clean_df, encoded_df = load_car_data()
    X = encoded_df.drop(columns=['log_selling_price'])
    y = encoded_df['log_selling_price']
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)
    
    ridge = Ridge(alpha=6.38, random_state=42).fit(X_train_scaled, y_train)
    lasso = Lasso(alpha=0.0017, random_state=42).fit(X_train_scaled, y_train)
    ols = LinearRegression().fit(X_train_scaled, y_train)
    
    return {
        'clean_df': clean_df,
        'encoded_df': encoded_df,
        'feature_names': feature_names,
        'scaler': scaler,
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'models': {
            'Ridge Regression (L2, alpha=6.38)': ridge,
            'Lasso Regression (L1, alpha=0.0017)': lasso,
            'Multiple Linear Regression (OLS)': ols
        }
    }

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/c/c5/Vellore_Institute_of_Technology_seal_2017.svg/300px-Vellore_Institute_of_Technology_seal_2017.svg.png", width=120)
st.sidebar.markdown("### **VIT Machine Learning Lab**")
st.sidebar.markdown("**M.Tech Course: MACSE504**")
st.sidebar.markdown("Case Study Based Mini-Project (30 Marks)")
st.sidebar.markdown("---")

nav_choice = st.sidebar.radio(
    "Select Navigation Section:",
    [
        "🏠 Home & Project Overview",
        "📡 Case Study 1: Telecom Churn (Rohan)",
        "🚗 Case Study 2: Car Valuation (Vedant)",
        "📖 Project Documentation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Use the tabs in each case study to explore model diagnostics, live predictions, and business simulators.")

# ==============================================================================
# 1. HOME & OVERVIEW
# ==============================================================================
if nav_choice == "🏠 Home & Project Overview":
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size:28px;">🎓 VIT M.Tech Machine Learning Lab — Mini-Project Dashboard</h1>
        <p style="margin:6px 0 0 0; opacity:0.9;">School of Computer Science and Engineering | MACSE504 Case Study Mini-Project Evaluation</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Phase 1: Part A (10 Marks)</div>
            <div class="metric-value">EDA & Preprocessing</div>
            <p style="margin:4px 0 0 0; font-size:12px; color:#555;">Data hygiene, missing values, skewness, distribution analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Phase 2: Part B (10 Marks)</div>
            <div class="metric-value">Feature Eng. & Fitting</div>
            <p style="margin:4px 0 0 0; font-size:12px; color:#555;">SMOTE, Regularization, OHE, 5-Fold Stratified CV, Hyperparameter tuning.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Phase 3: Part C (10 Marks)</div>
            <div class="metric-value">Evaluation & Inferences</div>
            <p style="margin:4px 0 0 0; font-size:12px; color:#555;">ROC-AUC, 4-quadrant residuals, cost-sensitive thresholding, ROI simulators.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### 📋 Student Allocation & Research Roster")
    roster_data = pd.DataFrame([
        {
            "Student Name": "Rohan",
            "Register No.": "26MML0020",
            "Assigned Case Study": "Question 1: Predictive Churn Risk Modeling for Telecom Subscription Services",
            "Target Dataset": "Telco Customer Churn (7,032 accounts)",
            "Primary Models": "Random Forest, Logistic Regression (SMOTE), Gradient Boosting",
            "Key Metric": "ROC-AUC = 0.837 | PR-AUC = 0.628"
        },
        {
            "Student Name": "Vedant",
            "Register No.": "26MML0045",
            "Assigned Case Study": "Question 3: Used Car Valuation & Elasticity Analysis using Regularized Regression",
            "Target Dataset": "Car details v3 (8,128 vehicle listings)",
            "Primary Models": "Ridge Regression (L2), Lasso (L1), ElasticNet, OLS",
            "Key Metric": "R² = 0.875 | Test MAE = ₹1.40 Lakhs"
        },
        {
            "Student Name": "Vaishnavi Jadhav",
            "Register No.": "26MML0034",
            "Assigned Case Study": "Question 3: Used Car Valuation & Elasticity Analysis using Regularized Regression",
            "Target Dataset": "Car details v3 (8,128 vehicle listings)",
            "Primary Models": "Ridge Regression (L2), Lasso (L1), OLS Baseline",
            "Key Metric": "R² = 0.875 | Median Error = ₹0.77 Lakhs"
        }
    ])
    st.dataframe(roster_data, use_container_width=True, hide_index=True)
    
    st.markdown("### 🚀 Interactive Capabilities of this Dashboard")
    st.markdown("""
    - **Live Predictive Inference**: Adjust real-time customer and vehicle profiles to inspect model predictions instantaneously.
    - **Threshold Sensitivity Analysis**: Dynamically move the classification decision boundary $p \in [0.10, 0.90]$ and observe the resulting confusion matrix and trade-off curves.
    - **Economic ROI & Retention Simulator**: Model the dollar impact of proactive retention outreach packages vs. customer lifetime value preserved.
    - **Dynamic Depreciation Curves**: Simulate vehicle price decay trajectories across age ($1-15$ years) and compare Diesel vs. Petrol resale premiums.
    """)

# ==============================================================================
# 2. CASE STUDY 1: TELECOM CHURN (ROHAN - 26MML0020)
# ==============================================================================
elif nav_choice == "📡 Case Study 1: Telecom Churn (Rohan)":
    st.markdown("""
    <div class="main-header">
        <h2 style="margin:0; font-size:24px;">📡 Question 1: Predictive Churn Risk Modeling & Retention Policy</h2>
        <p style="margin:4px 0 0 0; opacity:0.9;">Student: Rohan (26MML0020) | Dataset: Telco Customer Churn (7,032 subscribers)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data & models
    telco_data = train_telco_models()
    models_dict = telco_data['models']
    X_test = telco_data['X_test']
    X_test_scaled = telco_data['X_test_scaled']
    y_test = telco_data['y_test']
    feature_names = telco_data['feature_names']
    
    tab1, tab2, tab3 = st.tabs([
        "📊 Model Evaluation & Threshold Diagnostics",
        "🔮 Live Customer Churn Risk Scorer",
        "💰 Retention Policy & Financial ROI Simulator"
    ])
    
    # --- TAB 1: Evaluation ---
    with tab1:
        st.subheader("Holdout Test Partition Performance Benchmark (N=1,407)")
        
        c_sel1, c_sel2 = st.columns([2, 2])
        with c_sel1:
            sel_model_name = st.selectbox("Select Classifier Architecture:", list(models_dict.keys()), index=0)
        with c_sel2:
            p_threshold = st.slider("Classification Decision Cutoff (p):", min_value=0.10, max_value=0.90, value=0.35, step=0.05)
            
        clf, is_scaled = models_dict[sel_model_name]
        eval_X = X_test_scaled if is_scaled else X_test
        y_prob = clf.predict_proba(eval_X)[:, 1]
        y_pred = (y_prob >= p_threshold).astype(int)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
        m_col1.metric("ROC-AUC", f"{roc_auc:.3f}")
        m_col2.metric("PR-AUC", f"{pr_auc:.3f}")
        m_col3.metric("Recall (Sensitivity)", f"{rec*100:.1f}%")
        m_col4.metric("Precision", f"{prec*100:.1f}%")
        m_col5.metric("F1-Score", f"{f1:.3f}")
        m_col6.metric("Accuracy", f"{acc*100:.1f}%")
        
        p_col1, p_col2 = st.columns(2)
        
        with p_col1:
            # Confusion Matrix Plotly
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            z_vals = [[tn, fp], [fn, tp]]
            x_labels = ['Retained (0)', 'Churned (1)']
            y_labels = ['Retained (0)', 'Churned (1)']
            
            fig_cm = px.imshow(
                z_vals, x=x_labels, y=y_labels, text_auto=True,
                color_continuous_scale='Blues',
                title=f"Confusion Matrix ({sel_model_name} @ p={p_threshold:.2f})"
            )
            fig_cm.update_layout(xaxis_title="Predicted Label", yaxis_title="True Label")
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with p_col2:
            # Multi-model ROC Curves
            fig_roc = go.Figure()
            colors = ['#1f77b4', '#2ca02c', '#d62728']
            for idx, (m_name, (m_obj, s_flag)) in enumerate(models_dict.items()):
                m_X = X_test_scaled if s_flag else X_test
                probs = m_obj.predict_proba(m_X)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, probs)
                auc_v = roc_auc_score(y_test, probs)
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{m_name} ({auc_v:.3f})", line=dict(color=colors[idx], width=2)))
                
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name="Chance Baseline", line=dict(dash='dash', color='gray')))
            fig_roc.update_layout(
                title="Receiver Operating Characteristic (ROC) Benchmark",
                xaxis_title="False Positive Rate", yaxis_title="True Positive Rate (Recall)",
                legend=dict(x=0.5, y=0.1)
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            
    # --- TAB 2: Live Predictor ---
    with tab2:
        st.subheader("Interactive Customer Churn Risk Scorer")
        st.write("Adjust subscriber account and demographic settings to estimate their real-time attrition probability.")
        
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            inp_tenure = st.slider("Customer Tenure (Months):", 1, 72, 6)
            inp_monthly = st.slider("Monthly Charges ($):", 18.0, 120.0, 85.0, 1.0)
            inp_contract = st.selectbox("Contract Commitment:", ["Month-to-month", "One year", "Two year"])
        with f_col2:
            inp_internet = st.selectbox("Internet Service Type:", ["Fiber optic", "DSL", "No"])
            inp_payment = st.selectbox("Payment Method:", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            inp_paperless = st.selectbox("Paperless Billing:", ["Yes", "No"])
        with f_col3:
            inp_tech_support = st.selectbox("Tech Support Subscribed:", ["No", "Yes"])
            inp_online_sec = st.selectbox("Online Security Subscribed:", ["No", "Yes"])
            inp_device_prot = st.selectbox("Device Protection Subscribed:", ["No", "Yes"])
            
        # Build synthetic row
        sample_row = {col: 0.0 for col in feature_names}
        sample_row['tenure'] = float(inp_tenure)
        sample_row['MonthlyCharges'] = float(inp_monthly)
        sample_row['TotalCharges'] = float(inp_tenure * inp_monthly)
        sample_row['PaperlessBilling'] = 1.0 if inp_paperless == "Yes" else 0.0
        
        # Mappings
        if inp_contract == "One year" and "Contract_One year" in sample_row:
            sample_row["Contract_One year"] = 1.0
        elif inp_contract == "Two year" and "Contract_Two year" in sample_row:
            sample_row["Contract_Two year"] = 1.0
            
        if inp_internet == "Fiber optic" and "InternetService_Fiber optic" in sample_row:
            sample_row["InternetService_Fiber optic"] = 1.0
        elif inp_internet == "No" and "InternetService_No" in sample_row:
            sample_row["InternetService_No"] = 1.0
            
        if inp_payment == "Electronic check" and "PaymentMethod_Electronic check" in sample_row:
            sample_row["PaymentMethod_Electronic check"] = 1.0
            
        if inp_tech_support == "Yes" and "TechSupport_Yes" in sample_row:
            sample_row["TechSupport_Yes"] = 1.0
        if inp_online_sec == "Yes" and "OnlineSecurity_Yes" in sample_row:
            sample_row["OnlineSecurity_Yes"] = 1.0
            
        input_df = pd.DataFrame([sample_row])[feature_names]
        
        # Run prediction on RF model
        rf_model, _ = models_dict['Random Forest (Class-Weighted)']
        pred_prob = rf_model.predict_proba(input_df)[0, 1]
        
        res_col1, res_col2 = st.columns([2, 3])
        with res_col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_prob * 100,
                number={'suffix': "%"},
                title={'text': "Predicted Churn Probability", 'font': {'size': 18}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#d62728" if pred_prob >= 0.50 else ("#ff7f0e" if pred_prob >= 0.35 else "#2ca02c")},
                    'steps': [
                        {'range': [0, 35], 'color': "#e8f5e9"},
                        {'range': [35, 60], 'color': "#fff3e0"},
                        {'range': [60, 100], 'color': "#ffebee"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.75,
                        'value': 35
                    }
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with res_col2:
            st.markdown("#### **Subscriber Risk Classification & Diagnosis:**")
            if pred_prob >= 0.60:
                st.error(f"🚨 **CRITICAL ATTRITION RISK** ({pred_prob*100:.1f}% probability)")
                st.markdown("""
                - **Primary Risk Catalysts**: Short tenure combined with high monthly spend on a Month-to-Month agreement.
                - **Actionable Playbook**: Immediate trigger of **Pillar 1 Onboarding VIP Care** + free 6-month TechSupport upgrade.
                """)
            elif pred_prob >= 0.35:
                st.warning(f"⚠️ **ELEVATED ATTRITION RISK** ({pred_prob*100:.1f}% probability)")
                st.markdown("""
                - **Primary Risk Catalysts**: Above the Bayesian cost-optimal cutoff ($p^* = 0.35$).
                - **Actionable Playbook**: Dispatch **Pillar 2 Annual Contract Transition Rebate** ($12/month loyalty discount for 1-year lock-in).
                """)
            else:
                st.success(f"✅ **STABLE / LOW CHURN RISK** ({pred_prob*100:.1f}% probability)")
                st.markdown("- Subscriber exhibits strong retention anchors (long tenure, security add-ons, or multi-year contract).")
                
    # --- TAB 3: Retention Simulator ---
    with tab3:
        st.subheader("Financial ROI & Customer Lifetime Value (CLV) Simulator")
        st.write("Simulate the dollar return of targeting high-risk subscribers with model-guided proactive retention offers.")
        
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        with sim_col1:
            sim_offer_cost = st.slider("Per-Subscriber Offer Cost ($):", 20, 150, 60, 5)
        with sim_col2:
            sim_success_rate = st.slider("Retention Acceptance Rate (%):", 20, 90, 55, 5) / 100.0
        with sim_col3:
            sim_target_pct = st.slider("Top Risk Cohort Targeted (%):", 5, 50, 20, 5) / 100.0
            
        rf_model, _ = models_dict['Random Forest (Class-Weighted)']
        all_probs = rf_model.predict_proba(X_test)[:, 1]
        
        sim_df = pd.DataFrame({
            'Actual_Churn': y_test.values,
            'Pred_Prob': all_probs,
            'MonthlyCharges': X_test['MonthlyCharges'].values,
            'CLV': X_test['MonthlyCharges'].values * 24 # 24-month horizon
        }).sort_values(by='Pred_Prob', ascending=False).reset_index(drop=True)
        
        n_target = int(len(sim_df) * sim_target_pct)
        targeted = sim_df.iloc[:n_target]
        churners_in_target = targeted['Actual_Churn'].sum()
        churners_saved = int(churners_in_target * sim_success_rate)
        
        total_campaign_cost = n_target * sim_offer_cost
        gross_clv_saved = targeted[targeted['Actual_Churn'] == 1]['CLV'].sum() * sim_success_rate
        net_profit_saved = gross_clv_saved - total_campaign_cost
        campaign_roi = (net_profit_saved / total_campaign_cost) * 100 if total_campaign_cost > 0 else 0
        
        r_col1, r_col2, r_col3, r_col4 = st.columns(4)
        r_col1.metric("Subscribers Targeted", f"{n_target:,d}")
        r_col2.metric("Churners Intercepted", f"{churners_saved:,d}")
        r_col3.metric("Net Preserved Value", f"${net_profit_saved:,.0f}")
        r_col4.metric("Campaign ROI", f"{campaign_roi:.1f}%")
        
        st.info(f"💡 By offering a **${sim_offer_cost}** retention package to the top **{sim_target_pct*100:.0f}%** highest-risk accounts, the enterprise preserves **${net_profit_saved:,.0f}** in net revenue that would have otherwise been permanently lost to attrition.")

# ==============================================================================
# 3. CASE STUDY 2: CAR VALUATION (VEDANT - 26MML0045)
# ==============================================================================
elif nav_choice == "🚗 Case Study 2: Car Valuation (Vedant)":
    st.markdown("""
    <div class="main-header">
        <h2 style="margin:0; font-size:24px;">🚗 Question 3: Used Car Valuation & Elasticity Pricing Engine</h2>
        <p style="margin:4px 0 0 0; opacity:0.9;">Student: Vedant (26MML0045) | Dataset: CarDekho Used Cars (8,128 vehicle listings)</p>
    </div>
    """, unsafe_allow_html=True)
    
    car_data = train_car_models()
    clean_df = car_data['clean_df']
    car_models = car_data['models']
    X_test_car = car_data['X_test']
    X_test_scaled_car = car_data['X_test_scaled']
    y_test_car = car_data['y_test']
    car_feature_names = car_data['feature_names']
    car_scaler = car_data['scaler']
    
    tab1, tab2, tab3 = st.tabs([
        "📈 Valuation Performance & Residual Diagnostics",
        "🏷️ Live Algorithmic Price Estimator",
        "📉 Depreciation Dynamics & Powertrain Elasticity"
    ])
    
    # --- TAB 1: Performance ---
    with tab1:
        st.subheader("Holdout Out-of-Sample Performance Benchmark (N=1,626)")
        
        sel_car_model = st.selectbox("Select Regularized Model Architecture:", list(car_models.keys()), index=0)
        reg_model = car_models[sel_car_model]
        
        y_pred_log = reg_model.predict(X_test_scaled_car)
        y_pred_inr = np.exp(y_pred_log)
        y_test_inr = np.exp(y_test_car)
        
        r2 = r2_score(y_test_car, y_pred_log)
        mae_inr = mean_absolute_error(y_test_inr, y_pred_inr) / 100000.0
        med_ae_inr = np.median(np.abs(y_test_inr - y_pred_inr)) / 100000.0
        mape = np.mean(np.abs((y_test_inr - y_pred_inr) / y_test_inr)) * 100.0
        
        c_col1, c_col2, c_col3, c_col4 = st.columns(4)
        c_col1.metric("Out-of-Sample R²", f"{r2:.4f}")
        c_col2.metric("Mean Absolute Error (MAE)", f"₹{mae_inr:.2f} Lakhs")
        c_col3.metric("Median Absolute Error", f"₹{med_ae_inr:.2f} Lakhs")
        c_col4.metric("MAPE (%)", f"{mape:.1f}%")
        
        p1, p2 = st.columns(2)
        with p1:
            # Parity Scatter Plotly
            parity_df = pd.DataFrame({
                'Actual_Lakhs': y_test_inr.values / 100000.0,
                'Pred_Lakhs': y_pred_inr / 100000.0
            })
            fig_par = px.scatter(
                parity_df[parity_df['Actual_Lakhs'] <= 35],
                x='Actual_Lakhs', y='Pred_Lakhs', opacity=0.4,
                title="Valuation Parity (Actual vs. Predicted in ₹ Lakhs)",
                labels={'Actual_Lakhs': 'Actual Selling Price (₹ Lakhs)', 'Pred_Lakhs': 'Predicted Resale Price (₹ Lakhs)'}
            )
            fig_par.add_trace(go.Scatter(x=[0, 35], y=[0, 35], mode='lines', name='y = x Identity', line=dict(color='red', dash='dash', width=2)))
            st.plotly_chart(fig_par, use_container_width=True)
            
        with p2:
            # Residuals vs Fitted
            residuals = y_test_car.values - y_pred_log
            fig_res = px.scatter(
                x=y_pred_log, y=residuals, opacity=0.4,
                title="Residuals vs. Fitted Values (Homoscedasticity Test)",
                labels={'x': 'Fitted Log Selling Price', 'y': 'Residual Error [ln(₹)]'}
            )
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_res, use_container_width=True)
            
    # --- TAB 2: Live Valuation Estimator ---
    with tab2:
        st.subheader("Algorithmic Used Car Resale Pricing Calculator")
        st.write("Specify technical parameters to generate a market-clearing secondary valuation with confidence bounds.")
        
        v_col1, v_col2, v_col3 = st.columns(3)
        with v_col1:
            inp_age = st.slider("Vehicle Age (Years):", 1, 15, 5)
            inp_km = st.slider("Kilometers Driven (Odometer):", 5000, 200000, 50000, 5000)
            inp_fuel = st.selectbox("Powertrain Fuel:", ["Diesel", "Petrol", "CNG", "LPG"])
        with v_col2:
            inp_trans = st.selectbox("Transmission:", ["Manual", "Automatic"])
            inp_seller = st.selectbox("Seller Type:", ["Individual", "Dealer", "Trustmark Dealer"])
            inp_seats = st.slider("Seating Capacity:", 4, 8, 5)
        with v_col3:
            inp_power = st.slider("Max Engine Power (bhp):", 35.0, 250.0, 85.0, 5.0)
            inp_engine = st.slider("Displacement (cc):", 796.0, 3000.0, 1248.0, 50.0)
            inp_mileage = st.slider("Rated Mileage (kmpl):", 10.0, 30.0, 19.5, 0.5)
            
        # Build synthetic row
        car_row = {col: 0.0 for col in car_feature_names}
        car_row['km_driven'] = float(inp_km)
        car_row['seats'] = float(inp_seats)
        car_row['vehicle_age'] = float(inp_age)
        car_row['mileage_kmpl'] = float(inp_mileage)
        car_row['engine_cc'] = float(inp_engine)
        car_row['max_power_bhp'] = float(inp_power)
        
        # Interactions
        car_row['km_per_year'] = inp_km / (inp_age + 0.5)
        car_row['power_displacement_density'] = inp_power / (inp_engine / 1000.0)
        car_row['engine_age_interaction'] = (inp_engine * inp_age) / 1000.0
        car_row['efficiency_power_ratio'] = inp_mileage / (inp_power + 1e-3)
        car_row['log_odometer'] = np.log1p(inp_km)
        
        # Dummy flags
        if f"fuel_{inp_fuel}" in car_row:
            car_row[f"fuel_{inp_fuel}"] = 1.0
        if f"seller_type_{inp_seller}" in car_row:
            car_row[f"seller_type_{inp_seller}"] = 1.0
        if f"transmission_{inp_trans}" in car_row:
            car_row[f"transmission_{inp_trans}"] = 1.0
            
        inp_car_df = pd.DataFrame([car_row])[car_feature_names]
        inp_car_scaled = pd.DataFrame(car_scaler.transform(inp_car_df), columns=car_feature_names)
        
        ridge_model = car_models['Ridge Regression (L2, alpha=6.38)']
        pred_log_val = ridge_model.predict(inp_car_scaled)[0]
        pred_price_lakhs = np.exp(pred_log_val) / 100000.0
        
        # 95% Confidence Bounds
        rmse_log = 0.2982
        lower_lakhs = np.exp(pred_log_val - 1.96 * rmse_log) / 100000.0
        upper_lakhs = np.exp(pred_log_val + 1.96 * rmse_log) / 100000.0
        
        st.markdown(f"""
        <div style="background:#f0f7ff; padding:18px; border-left:6px solid #1f77b4; border-radius:6px; margin-top:12px;">
            <h3 style="margin:0; color:#0d3b66;">Estimated Resale Price: ₹{pred_price_lakhs:.2f} Lakhs</h3>
            <p style="margin:4px 0 0 0; font-size:14px; color:#555;">95% Market Confidence Interval: <b>₹{lower_lakhs:.2f} Lakhs – ₹{upper_lakhs:.2f} Lakhs</b> | Odometer Utilization Intensity: <b>{car_row['km_per_year']:,.0f} km/year</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Dynamic depreciation trajectory
        ages = np.arange(1, 16)
        traj_prices = []
        for a in ages:
            c_copy = car_row.copy()
            c_copy['vehicle_age'] = float(a)
            c_copy['km_per_year'] = inp_km / (a + 0.5)
            c_copy['engine_age_interaction'] = (inp_engine * a) / 1000.0
            t_df = pd.DataFrame([c_copy])[car_feature_names]
            t_scaled = pd.DataFrame(car_scaler.transform(t_df), columns=car_feature_names)
            traj_prices.append(np.exp(ridge_model.predict(t_scaled)[0]) / 100000.0)
            
        fig_traj = go.Figure()
        fig_traj.add_trace(go.Scatter(x=ages, y=traj_prices, mode='lines+markers', name='Fitted Depreciation Trajectory', line=dict(color='#1f77b4', width=3)))
        fig_traj.add_trace(go.Scatter(x=[inp_age], y=[pred_price_lakhs], mode='markers', name=f'Current Vehicle (Age {inp_age}y)', marker=dict(color='red', size=12)))
        fig_traj.update_layout(
            title=f"Lifecycle Price Trajectory for Current Configuration across Age",
            xaxis_title="Vehicle Age (Years)", yaxis_title="Resale Valuation (₹ Lakhs)"
        )
        st.plotly_chart(fig_traj, use_container_width=True)
        
    # --- TAB 3: Depreciation & Diesel Premium ---
    with tab3:
        st.subheader("Market Elasticities: Depreciation & Diesel Resale Premium")
        
        dp1, dp2 = st.columns(2)
        with dp1:
            # Depreciation breakdown
            dep_ages = np.arange(1, 16)
            P0 = 11.84
            k = 0.1164
            dep_curve = P0 * np.exp(-k * dep_ages)
            
            fig_dep = go.Figure()
            fig_dep.add_trace(go.Scatter(x=dep_ages, y=dep_curve, mode='lines+markers', name='P(t) = 11.84 · exp(-0.116·t)', line=dict(color='#ff7f0e', width=3)))
            fig_dep.update_layout(
                title="Empirical Secondary Automotive Depreciation Curve",
                xaxis_title="Vehicle Age (Years)", yaxis_title="Median Valuation (₹ Lakhs)"
            )
            st.plotly_chart(fig_dep, use_container_width=True)
            st.caption("📌 **First-Year Cliff**: ~18-20% drop from invoice, followed by a continuous ~10.5% annual depreciation.")
            
        with dp2:
            # Diesel vs Petrol Across Odometer Tiers
            fuel_tiers = ['0-30k km', '30k-60k km', '60k-90k km', '90k-120k km', '120k+ km']
            diesel_vals = [7.5, 6.2, 5.3, 4.5, 3.8]
            petrol_vals = [5.2, 4.3, 3.7, 3.1, 2.5]
            
            fig_fuel = go.Figure(data=[
                go.Bar(name='Diesel Powertrain', x=fuel_tiers, y=diesel_vals, marker_color='#1f77b4'),
                go.Bar(name='Petrol Powertrain', x=fuel_tiers, y=petrol_vals, marker_color='#2ca02c')
            ])
            fig_fuel.update_layout(
                barmode='group',
                title="Diesel vs. Petrol Resale Valuation across Odometer Usage Tiers",
                xaxis_title="Odometer Mileage Tier", yaxis_title="Median Resale Price (₹ Lakhs)"
            )
            st.plotly_chart(fig_fuel, use_container_width=True)
            st.caption("📌 **Diesel Resale Premium**: Diesel cars maintain an empirical **₹1.2L - ₹2.3L premium** over petrol.")

# ==============================================================================
# 4. DOCUMENTATION
# ==============================================================================
elif nav_choice == "📖 Project Documentation":
    st.markdown("""
    <div class="main-header">
        <h2 style="margin:0; font-size:24px;">📖 VIT ML Mini-Project Reference Documentation</h2>
        <p style="margin:4px 0 0 0; opacity:0.9;">Academic & Technical Documentation for M.Tech MACSE504</p>
    </div>
    """, unsafe_allow_html=True)
    
    doc_path = find_data_file("PROJECT_EXPLANATION.md")
    if os.path.exists(doc_path):
        with open(doc_path, "r", encoding="utf-8") as f:
            doc_content = f.read()
        st.markdown(doc_content)
    else:
        st.warning("PROJECT_EXPLANATION.md file not found in directory.")
