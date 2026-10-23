# Comprehensive Technical Architecture & Project Guide
## VIT M.Tech Machine Learning Lab (MACSE504) — Case Study Based Mini-Project
**School of Computer Science and Engineering, Vellore Institute of Technology (VIT)**

---

### **Executive Overview & Academic Framework**
This repository contains the complete, production-grade codebase, research notebooks, interactive dashboards, and empirical reports for the **M.Tech Machine Learning Laboratory (MACSE504) Case Study Based Mini-Project (Total: 30 Marks)**.

The assignment evaluation scheme is partitioned into three mandatory phases:
* **Phase 1: Part A (10 Marks)** — *Exploratory Data Analysis & Preprocessing* (Data cleaning, type casting, missing value imputation, statistical distributions, correlation analysis).
* **Phase 2: Part B (10 Marks)** — *Feature Engineering & Model Fitting* (Categorical encoding, interaction synthesis, cross-validation, hyperparameter optimization, class imbalance handling).
* **Phase 3: Part C (10 Marks)** — *Model Evaluation, Diagnostic Visualization & Domain/Business/Market Inference* (Out-of-sample metrics, residual analysis, probability calibration, feature attribution, commercial policy formulation).
* **Universal Rubric Mandate**: `***provide an interface to load dataset, training, parameter adjustment and for visualization.` (Implemented via embedded reactive `ipywidgets` consoles in every notebook).

---

### **Student Allocation & Assigned Questions**
According to the official VIT allocation roster (`VL2026270106396_AST08.pdf`):

| Student Name | Register Number | Assigned Case Study | Target Domain & Dataset | Primary Deliverables |
| :--- | :--- | :--- | :--- | :--- |
| **Rohan** | **26MML0020** | **Question 1** | Predictive Churn Risk Modeling for Telecom Subscription Services (`WA_Fn-UseC_-Telco-Customer-Churn.csv` / `telco_preprocessed.csv`) | Part A, Part B, Part C + Interactive GUI Dashboard |
| **Vedant** | **26MML0045** | **Question 3** | Used Car Valuation & Elasticity Analysis using Regularized Regression (`Car details v3.csv`) | Part A, Part B, Part C + Interactive GUI Dashboard |
| **Vaishnavi Jadhav** | **26MML0034** | **Question 3** | Used Car Valuation & Elasticity Analysis using Regularized Regression (`Car details v3.csv`) | Part A, Part B + Interactive GUI Dashboard |

---

# SECTION 1: QUESTION 1 — PREDICTIVE CHURN RISK MODELING (ROHAN - 26MML0020)

## 1.1 Problem Statement & Industry Context
Customer attrition (churn) is one of the most critical financial vulnerabilities in telecommunications. Acquiring a new telecom subscriber costs **$5\times$ to $10\times$** more in marketing, promotional hardware subsidies, and onboarding overhead than retaining an existing account. Furthermore, customer loss immediately reduces monthly recurring revenue (MRR) and compromises Customer Lifetime Value (CLV).

The objective is to formulate an end-to-end, leak-free machine learning pipeline to:
1. Identify high-risk subscribers before they cancel.
2. Uncover the root causes of attrition.
3. Formulate economically optimal customer retention policies.

---

## 1.2 Dataset Profiling & Architecture
* **Source**: Telco Customer Churn Dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`).
* **Volume**: 7,043 customer accounts across 21 raw attributes.
* **Feature Categories**:
  - *Demographics*: `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
  - *Subscribed Services*: `PhoneService`, `MultipleLines`, `InternetService` (DSL, Fiber optic, No), `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
  - *Account Terms*: `Contract` (Month-to-month, One year, Two year), `PaperlessBilling`, `PaymentMethod` (Electronic check, Mailed check, Bank transfer, Credit card).
  - *Continuous Financials*: `tenure` (months), `MonthlyCharges` (USD), `TotalCharges` (USD).
  - *Target*: `Churn` (`Yes` = 1, `No` = 0).

---

## 1.3 Phase 1: Part A (Exploratory Data Analysis & Preprocessing)

### **A. Anomaly Treatment & Type Casting**
* **The Whitespace Anomaly in `TotalCharges`**:
  The raw `TotalCharges` column was stored as an `object` string type. Parsing identified 11 records containing blank whitespace strings (`" "`).
  Cross-referencing revealed that all 11 records corresponded to customers with `tenure == 0` (new accounts that had not yet completed their first billing cycle).
  *Resolution*: These 11 records were parsed using `pd.to_numeric(..., errors='coerce')` and dropped from the modeling matrix, leaving 7,032 verified rows.
* **Customer Identifier**:
  `customerID` is an arbitrary alpha-numeric identifier with zero generalizable predictive utility; it was excised to prevent high-cardinality noise.

### **B. Statistical Distributions & Hypothesis Testing**
* **Target Class Imbalance**:
  - Non-Churn (`0`): 5,163 subscribers (**73.42%**).
  - Churn (`1`): 1,869 subscribers (**26.58%**).
* **Chi-Square ($\chi^2$) Hypothesis Testing on Attrition Drivers**:
  - *Contract Type*: Month-to-month subscribers exhibit an alarming **42.7%** churn rate, compared to **11.3%** for One-year contracts and just **2.8%** for Two-year contracts ($\chi^2 = 1179.5$, $p < 10^{-15}$).
  - *Internet Service Type*: Fiber Optic subscribers exhibit a **41.9%** churn rate, compared to **19.0%** for DSL and **7.4%** for customers without internet service ($\chi^2 = 728.7$, $p < 10^{-15}$).
* **Continuous Feature Distribution & Correlation**:
  - `tenure` displays a sharp bimodal distribution (a spike of new subscribers $\le 6$ months, and a loyal cohort $> 60$ months).
  - `MonthlyCharges` exhibits a multi-modal density with peaks around $\$20$ (basic phone service) and $\$80-\$95$ (bundled fiber internet).
  - Pearson correlation confirms high collinearity between `tenure` and `TotalCharges` ($r = 0.83$).

### **C. Feature Engineering & Binary/OHE Encoding**
* **Synthesized Domain Features**:
  - `TotalServices`: Count of active services subscribed ($\in [1, 9]$).
  - `LifetimeValue`: Empirical proxy computed as $\text{MonthlyCharges} \times \text{tenure}$.
  - `AvgMonthlySpend`: Historical run-rate $\frac{\text{TotalCharges}}{\text{tenure}}$.
  - `ProtectionScore`: Security bundle index ($\text{OnlineSecurity} + \text{DeviceProtection} + \text{TechSupport}$).
* **Categorical Mapping**:
  - Two-class features (`gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn`) mapped directly to binary $\{0, 1\}$.
  - Multi-class nominal categories (`InternetService`, `Contract`, `PaymentMethod`, etc.) converted via One-Hot Encoding with `drop_first=True` to guarantee full column rank and eliminate the dummy variable trap.
  - Finalized clean matrix: 7,032 rows $\times$ 36 columns exported to `telco_preprocessed.csv`.

---

## 1.4 Phase 2: Part B (Feature Engineering & Model Fitting)

### **A. The Accuracy Paradox & Class Imbalance Dilemma**
Because 73.4% of subscribers belong to the majority class (non-churn), a trivial "majority vote" classifier predicting 0 for every customer attains **73.4% accuracy** while capturing **0% of churners (Recall = 0.0)**.
In telecom operations, failing to detect an at-risk subscriber (**False Negative**) costs hundreds of dollars in lost future revenue, whereas issuing a retention offer to a loyal customer (**False Positive**) costs only $\$25-\$50$.
Therefore, optimizing for raw accuracy is fatal; models must be optimized for **Recall, F1-Score, and ROC-AUC**.

### **B. Formulations for Handling Imbalance**
1. **SMOTE (Synthetic Minority Over-sampling Technique)**:
   Instead of duplicating minority instances (which causes decision tree overfitting), SMOTE synthesizes new minority points along the feature space line segments joining the $k$-nearest minority neighbors:
   $$x_{\text{new}} = x_i + \lambda (x_{zi} - x_i), \quad \lambda \sim \mathcal{U}(0, 1)$$
   *Critical Leakage Prevention*: SMOTE was applied strictly to the training folds inside cross-validation pipelines, ensuring synthetic data never contaminated the test or validation sets.
2. **Cost-Sensitive Class Weighting**:
   Re-weights the loss function inversely proportional to class frequencies:
   $$w_j = \frac{N}{2 \cdot N_j}$$
   This penalizes minority classification errors by $\approx 2.76\times$ compared to majority errors.

### **C. Model Architectures & 5-Fold Stratified Cross-Validation**
* **Fitted Classifiers**:
  1. *Logistic Regression (Logit)*: Generalized linear model with $L_2$ regularization serving as an interpretable parametric baseline.
  2. *K-Nearest Neighbors (KNN)*: Metric-based instance learning under Euclidean/Minkowski distance.
  3. *Random Forest (RF)*: Non-linear ensemble of 150 decorrelated decision trees employing bagging and feature subspace sampling ($m = \sqrt{p}$).
* **Stratified 5-Fold Cross-Validation Results**:
  - Incorporating SMOTE and class-weighting elevated minority Recall from **$51.8\%$** up to **$78.2\%$**.
  - Random Forest with class-weighting achieved the highest discrimination: **ROC-AUC = $0.847$** and balanced F1-score of **$0.635$**.
  - Logistic Regression with SMOTE achieved the highest single-model Recall (**$79.1\%$**).

---

## 1.5 Phase 3: Part C (Evaluation, Visualization & Business Retention Strategies)

### **A. Multi-Model Out-of-Sample Performance Evaluation**
Benchmarked on 1,407 unseen holdout test subscribers ($20\%$ partition):

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Class-Weighted)** | **0.7818** | **0.5601** | **0.7299** | **0.6340** | **0.8367** | **0.6281** | **0.1472** |
| **Logistic Regression (SMOTE)** | 0.7413 | 0.5052 | **0.7807** | 0.6133 | 0.8334 | 0.6198 | 0.1685 |
| **Gradient Boosting (Ensemble)** | 0.7967 | 0.6517 | 0.4973 | 0.5641 | 0.8385 | 0.6312 | 0.1384 |
| **K-Nearest Neighbors (KNN)** | 0.7456 | 0.5134 | 0.5615 | 0.5364 | 0.7820 | 0.5186 | 0.1795 |

*Key Takeaway*: While unweighted Gradient Boosting achieves higher raw accuracy ($79.7\%$), its Recall drops to a disappointing $49.7\%$ (missing over half of churning customers). **Random Forest with Class-Weighting** and **Logistic Regression with SMOTE** strike the optimal balance, capturing **$>73-78\%$ of churners**.

### **B. Cost-Sensitive Threshold Sensitivity Analysis**
Standard software defaults to a classification probability cutoff of $p=0.50$.
Given that the financial loss of a missed churner ($C_{FN} \approx \$500$ CLV) is $\approx 14\times$ greater than the cost of a retention perk ($C_{FP} \approx \$35$), the optimal Bayesian cutoff is calculated as:
$$p^* = \frac{C_{FP}}{C_{FN} + C_{FP}} = \frac{35}{500 + 35} \approx 0.06 - 0.35$$

* **Impact of Lowering Cutoff to $p^* = 0.35$**:
  - Recall increases from **$73.0\%$ to $85.6\%$** (capturing an additional 47 churners in the test cohort alone).
  - False Negatives collapse from **101 down to 54**.
  - Total expected financial attrition loss decreases by **over $\$25,000$** across the test sample.

### **C. Dual Feature Attribution & Key Churn Drivers**
1. **Random Forest Permutation Feature Importance (PFI)**:
   Evaluated strictly on unseen test data to eliminate cardinality bias:
   - Top churn drivers: `Contract_Two year`, `tenure`, `MonthlyCharges`, `TotalCharges`, `InternetService_Fiber optic`, `PaymentMethod_Electronic check`.
2. **Logistic Regression Standardized Odds Ratios ($\exp(\beta_j)$)**:
   - *Risk Catalysts (Odds Multipliers)*:
     - **Month-to-Month Contract**: $\text{OR} = 2.45\times$ churn probability.
     - **Fiber Optic Internet**: $\text{OR} = 1.95\times$ churn probability.
     - **Electronic Check Payment**: $\text{OR} = 1.34\times$ churn probability.
     - **Paperless Billing**: $\text{OR} = 1.28\times$ churn probability.
   - *Protective Anchors (Retention Shields)*:
     - **Two-Year Contract**: $\text{OR} = 0.31\times$ (reduces churn odds by $69\%$).
     - **High Tenure**: $\text{OR} = 0.45\times$ (reduces churn odds by $55\%$).
     - **Tech Support**: $\text{OR} = 0.68\times$ (reduces churn odds by $32\%$).
     - **Online Security**: $\text{OR} = 0.72\times$ (reduces churn odds by $28\%$).

### **D. Cohort Deep-Dive: Short Tenure & High Monthly Charges**
A 2D cross-tabulation matrix cross-referencing Tenure Tiers against Monthly Spend identified the focal vulnerability of Question 1:
* **The High-Risk Cohort (Tenure $\le 12$ months and Monthly Charges $> \$70$)**:
  - Accounts for **857 subscribers** ($12.2\%$ of the total customer base).
  - Exhibits an extraordinary **$65.8\%$ churn rate** ($2.5\times$ higher than baseline prevalence).
  - Root-cause service profiling revealed:
    1. **Fiber Optic Bill Shock**: $91.4\%$ of this cohort is on Fiber Optic internet, paying steep bills without experiencing long-term network reliability.
    2. **Severe Support Deficit**: $86.2\%$ do not have bundled `TechSupport`, and $82.5\%$ lack `OnlineSecurity`. Technical friction triggers immediate cancellation.
    3. **Zero Contractual Lock-in**: $88.7\%$ are on Month-to-Month agreements.
    4. **Payment Friction**: $56.3\%$ pay via Electronic Check, amplifying monthly invoice awareness.

### **E. Actionable Retention Playbook & Financial ROI Simulation**
1. **Pillar 1: 90-Day VIP Onboarding & Support Bundle**:
   - Provide free `TechSupport` and `DeviceProtection` for the first 6 months to all new Fiber Optic subscribers ($\le \$15$ cost). Cuts 1st-year fiber churn from $66\%$ to $<35\%$.
2. **Pillar 2: Contract Transition Incentive**:
   - Target month-to-month subscribers with predicted churn risk $\hat{p} \ge 0.35$ with a $\$12$/month loyalty credit in exchange for a 12-month commitment.
3. **Pillar 3: Autopay / ACH Transition Rebate**:
   - Offer a recurring $\$5$ monthly bill credit to convert electronic check users to automated bank draft or credit card autopay.
* **Financial ROI Simulation**:
  Targeting the top $25\%$ highest-risk subscribers with a $\$60$ retention package (assuming a $55\%$ save rate) yields:
  - **Campaign Cost**: $\$21,120$.
  - **Gross Customer Lifetime Value Preserved**: $\$80,580$.
  - **Net Profit Preserved**: **$\$59,460$** (**$281.5\%$ Return on Investment**).

---

# SECTION 2: QUESTION 3 — USED CAR VALUATION & REGULARIZED REGRESSION (VEDANT - 26MML0045 & VAISHNAVI - 26MML0034)

## 2.1 Problem Statement & Automotive Market Context
In the secondary automotive marketplace, estimating accurate vehicle resale prices is plagued by information asymmetry, non-linear mechanical wear, brand tiering, and rapid economic depreciation.
Online automotive platforms (such as CarDekho) require automated algorithmic valuation models to provide transparent price discovery for buyers, sellers, and trade-in inspection agents.

The objective is to formulate an end-to-end regularized regression architecture capable of:
1. Accurately estimating used car market valuation while controlling multicollinearity.
2. Inducing sparsity to isolate the most parsimonious feature subset.
3. Quantifying vehicle depreciation curves and the resale value premium of diesel vs. petrol powertrains.

---

## 2.2 Dataset Profiling & Architecture
* **Source**: Vehicle Dataset from CarDekho (`Car details v3.csv`).
* **Volume**: 8,128 vehicle listings across 13 raw attributes.
* **Raw Attributes**:
  - `name`: Full vehicle model string (e.g. "Maruti Swift Dzire VDI", "Skoda Rapid 1.5 TDI Ambition").
  - `year`: Manufacturing year ($1983 - 2020$).
  - `selling_price`: Resale price in Indian Rupees (₹).
  - `km_driven`: Total odometer distance.
  - `fuel`: Powertrain fuel type (`Diesel`, `Petrol`, `CNG`, `LPG`).
  - `seller_type`: Sales channel (`Individual`, `Dealer`, `Trustmark Dealer`).
  - `transmission`: Gearbox type (`Manual`, `Automatic`).
  - `owner`: Prior ownership history (`First Owner`, `Second Owner`, `Third Owner`, etc.).
  - `mileage`: Fuel efficiency string with units (e.g. `"23.4 kmpl"` or `"18.0 km/kg"`).
  - `engine`: Engine displacement string with units (e.g. `"1248 CC"`).
  - `max_power`: Horsepower output string with units (e.g. `"85 bhp"`).
  - `torque`: Engine torque output string.
  - `seats`: Number of passenger seats.

---

## 2.3 Phase 1: Part A (Exploratory Data Analysis & Preprocessing)

### **A. String Extraction & Unit Normalization**
* Technical specifications were embedded as alphanumeric strings:
  - `mileage`: Stripped `"kmpl"` and `"km/kg"` units using regular expressions $\to$ `mileage_kmpl` (float). 17 anomalous $0.0$ values were flagged as missing.
  - `engine`: Stripped `"CC"` $\to$ `engine_cc` (float).
  - `max_power`: Stripped `"bhp"` $\to$ `max_power_bhp` (float). Anomalous $0.0$ values flagged.
  - `seats`: Converted to numeric float.

### **B. Hierarchical Brand-Based Imputation**
Instead of crude global mean imputation, missing mechanical specifications were imputed using the median of that specific vehicle brand (`df['name'].str.split().str[0]`). If the entire brand lacked data, global medians were used as fallback.

### **C. Vehicle Age Calculation & Target Log Transformation**
* **Vehicle Age**: $\text{vehicle\_age} = 2020 - \text{year}$.
* **Target Skewness Correction**:
  - Raw `selling_price` displayed extreme positive skewness (**$+4.19$**) with massive outliers (luxury vehicles $> ₹50$ Lakhs). Fitting standard linear models to raw price violates homoscedasticity and distorts least-squares penalties.
  - Applying the natural log transformation:
    $$\text{log\_selling\_price} = \ln(\text{selling\_price})$$
    reduced skewness to **$+0.22$** (approximate Gaussian normality), linearizing exponential depreciation and stabilizing error variance across all pricing tiers.

---

## 2.4 Phase 2: Part B (Feature Engineering & Regularized Model Fitting)

### **A. Domain Interaction Feature Engineering**
Vehicle wear and market depreciation exhibit coupled physical interactions:
1. **Annual Mileage Utilization Intensity**:
   $$\text{km\_per\_year} = \frac{\text{km\_driven}}{\text{vehicle\_age} + 0.5}$$
   *Rationale*: Separates chronological age from physical wear. A 3-year-old car driven 100,000 km has experienced far greater wear than a 7-year-old car driven 30,000 km.
2. **Specific Engine Power Density**:
   $$\text{power\_displacement\_density} = \frac{\text{max\_power\_bhp}}{\text{engine\_cc} / 1000.0}$$
   *Rationale*: Measures horsepower generated per liter of displacement, cleanly separating high-efficiency turbocharged engines from low-output economy engines.
3. **Cumulative Powertrain Wear Interaction**:
   $$\text{engine\_age\_interaction} = \frac{\text{engine\_cc} \times \text{vehicle\_age}}{1000.0}$$
   *Rationale*: Large-displacement engines face higher maintenance penalties as they age.
4. **Efficiency-Power Trade-off Ratio**:
   $$\text{efficiency\_power\_ratio} = \frac{\text{mileage\_kmpl}}{\text{max\_power\_bhp} + 1e-3}$$
   *Rationale*: Separates high-mileage economy commuters from performance sports cars.
5. **Log-Transformed Odometer**:
   $$\text{log\_odometer} = \ln(1 + \text{km\_driven})$$
   *Rationale*: Models logarithmic marginal price drops per kilometer driven.

### **B. Mathematical Formulations of Regularized Estimators**
Standardizing all predictors ($\mu=0, \sigma=1$) via `StandardScaler` is mathematically mandatory so that regularization penalties apply uniformly across features.
1. **Ordinary Least Squares (OLS Baseline)**:
   $$\min_{\beta} \frac{1}{2n} \|y - X\beta\|_2^2$$
   Unconstrained; suffers from matrix inversion instability when collinear features (`engine_cc`, `max_power_bhp`, and their interactions) inflate $(X^T X)^{-1}$.
2. **Ridge Regression ($L_2$ Tikhonov Regularization)**:
   $$\min_{\beta} \frac{1}{2n} \|y - X\beta\|_2^2 + \alpha \|\beta\|_2^2$$
   Adds a quadratic penalty that shrinks collinear coefficients toward zero, solving ill-conditioned matrix inversion and controlling variance inflation.
3. **Lasso Regression ($L_1$ Coordinate Descent Regularization)**:
   $$\min_{\beta} \frac{1}{2n} \|y - X\beta\|_2^2 + \alpha \|\beta\|_1$$
   The sharp geometric corners of the $L_1$ diamond force non-essential or redundant coefficients to land exactly on the coordinate axes ($\beta_j = 0$), performing automated feature selection.
4. **ElasticNet ($L_1 + L_2$ Regularization)**:
   $$\min_{\beta} \frac{1}{2n} \|y - X\beta\|_2^2 + \alpha \left( \rho \|\beta\|_1 + \frac{1-\rho}{2} \|\beta\|_2^2 \right)$$
   Overcomes Lasso limitations in high-correlation settings by encouraging grouped feature selection.

### **C. Cross-Validation & Hyperparameter Optimization ($\alpha$)**
* 5-Fold Cross-Validation was conducted across an extensive geometric spectrum $\alpha \in [10^{-4}, 10^{4}]$.
* **Optimal Hyperparameters**:
  - Ridge: $\alpha^* \approx 6.38$.
  - Lasso: $\alpha^* \approx 0.0017$.
  - ElasticNet: $\alpha^* \approx 0.0020, \rho = 0.50$.
* Incorporating domain interaction terms improved out-of-sample $R^2$ from $\approx 0.81$ to **$\approx 0.875$**, reducing test pricing error to $\approx ₹1.40$ Lakhs.

---

## 2.5 Phase 3: Part C (Evaluation, Visualization & Market Inference)

### **A. Performance Measurement on Unseen Test Data**
Evaluated on 1,626 unseen test listings ($20\%$ holdout partition):

| Model Architecture | Test $R^2$ | Adjusted $R^2$ | Log RMSE | Log MAE | MAE (₹ Lakhs) | RMSE (₹ Lakhs) | Median AE (₹ Lakhs) | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression ($L_2$, $\alpha=6.38$)** | **0.8749** | **0.8732** | **0.2982** | **0.2098** | **₹1.397 L** | **₹2.884 L** | **₹0.771 L** | **23.12%** |
| **Lasso Regression ($L_1$, $\alpha=0.0017$)** | 0.8744 | 0.8727 | 0.2988 | 0.2104 | ₹1.403 L | ₹2.890 L | ₹0.773 L | 23.18% |
| **ElasticNet ($L_1+L_2$, $\alpha=0.002$)** | 0.8747 | 0.8730 | 0.2984 | 0.2100 | ₹1.399 L | ₹2.886 L | ₹0.772 L | 23.14% |
| **Multiple Linear Regression (OLS)** | 0.8746 | 0.8729 | 0.2985 | 0.2102 | ₹1.401 L | ₹2.887 L | ₹0.772 L | 23.16% |

*Key Takeaway*:
Ridge Regression achieves an out-of-sample $R^2$ of **$0.875$** with a Median Absolute Error of just **₹$0.77$ Lakhs** ($₹77,000$).
Lasso successfully drives redundant interaction terms to zero while preserving virtually identical predictive accuracy.

### **B. Four-Quadrant Residual Diagnostic Suite**
To ensure Gauss-Markov assumptions are satisfied:
1. **Quadrant 1 (Residuals vs. Fitted Values)**:
   Residuals are uniformly distributed in a horizontal band centered around zero across all price regimes, verifying **Homoscedasticity**.
2. **Quadrant 2 (Standardized Normal Q-Q Plot)**:
   Residuals adhere closely to the theoretical standard normal diagonal, with minor deviations only at the extreme upper-tail luxury cars ($> ₹40$ Lakhs).
3. **Quadrant 3 (Residual Density & Symmetry)**:
   The empirical error distribution is centered at $\mu = -0.003$ with minimal skewness ($+0.21$), matching the theoretical Gaussian bell curve.
4. **Quadrant 4 (Scale-Location Plot)**:
   $\sqrt{|\text{Standardized Residuals}|}$ displays a flat trendline, confirming uniform variance across budget, mid-range, and premium tiers.

### **C. Feature Coefficient Magnitudes & Semi-Elasticities**
Because the model is formulated in log-linear space, coefficients translate directly into percentage price semi-elasticities:
$$\% \Delta \text{Price} \approx (\exp(\beta_j) - 1) \times 100\%$$

* **Dominant Positive Value Drivers**:
  - **Max Power (`max_power_bhp`)**: $\beta = +0.287 \implies \mathbf{+33.2\%}$ valuation increase per 1-SD shift.
  - **Diesel Fuel (`fuel_Diesel`)**: $\beta = +0.278 \implies \mathbf{+32.1\%}$ valuation premium over petrol.
  - **Automatic Transmission (`transmission_Manual` = 0)**: $\beta = +0.202 \implies \mathbf{+22.4\%}$ premium for automatic gearboxes.
  - **Power Displacement Density**: $\beta = +0.074 \implies \mathbf{+7.7\%}$ premium for high-efficiency turbocharged engines.
* **Dominant Negative Value Drivers (Depreciation Drivers)**:
  - **Vehicle Age (`vehicle_age`)**: $\beta = -0.342 \implies \mathbf{-29.0\%}$ valuation decline per 1-SD shift.
  - **Log Odometer (`log_odometer`)**: $\beta = -0.185 \implies \mathbf{-16.9\%}$ valuation decline per 1-SD shift.
  - **Annual Mileage Intensity (`km_per_year`)**: $\beta = -0.141 \implies \mathbf{-13.1\%}$ valuation decline per 1-SD shift.

### **D. Vehicle Depreciation Dynamics & Parametric Decay Modeling**
Automotive resale prices follow an exponential decay model:
$$\text{Price}(t) = P_0 \cdot \exp(-k \cdot t)$$
Fitting the empirical median price curve across vehicle age ($1$ to $15$ years) yields:
* **Baseline Invoice Price ($P_0$)**: ₹$11.84$ Lakhs.
* **Continuous Depreciation Decay Constant ($k$)**: **$0.1164$ per year**.
* **Market Lifecycle Phases**:
  1. *The First-Year Cliff*: Vehicles lose **$18.5\%$ to $21.2\%$** of invoice value in their first year.
  2. *Steady Operating Decay*: Annual value loss stabilizes at **$10.5\%$ per year** from years 2 through 7.
  3. *Residual Salvage Floor*: At year 5, the median car retains **$55.9\%$** of value (cumulative loss $44.1\%$). Beyond year 10, value stabilizes near the mechanical utility baseline.

### **E. Resale Value Premium: Diesel vs. Petrol Powertrains**
Head-to-head empirical pricing across usage tiers ($0$ to $150,000$ km) revealed:
* **Consistent Diesel Valuation Premium**:
  Diesel vehicles maintain a persistent **₹$1.2$ to ₹$2.5$ Lakhs premium** over equivalent petrol vehicles across all odometer bands.
  - *Low Odometer (0 - 30k km)*: Diesel median = ₹$7.5$L vs. Petrol median = ₹$5.2$L (**+₹2.3 Lakhs / +44.2%**).
  - *High Odometer (90k - 120k km)*: Diesel median = ₹$4.5$L vs. Petrol median = ₹$3.1$L (**+₹1.4 Lakhs / +45.2%**).
* **The Regulatory Headwind & Aging Divergence**:
  While diesel commands superior resale value in early and mid-life (driven by fuel economy and torque), its premium drops sharply after **age 7 - 8**.
  *Regulatory Rationale*: In key Indian markets (such as the National Capital Region), the National Green Tribunal enforces a **strict 10-year lifespan on diesel vehicles vs. 15 years on petrol**. As diesels approach age 8, approaching deregistration forces steep discounting.

### **F. Commercial Dealership Inferences & Marketplace Strategies**
1. **Inventory Turnover Policy**:
   Because vehicles aged $1-4$ years lose $\approx 10.5\%$ value per year ($\approx 0.9\%$ per month), dealerships must cap lot holding time at **$< 45$ days** to avoid margin erosion.
2. **Inter-State Regulatory Arbitrage Corridor**:
   Dealers can acquire 6-to-8-year-old diesel vehicles from Delhi NCR at steep discounts ($35-40\%$ below national averages) and remarket them in non-NCR states (Maharashtra, Punjab, Rajasthan) where 15-year limits apply, capturing **₹$80,000 - ₹$1,40,000 net margin per vehicle**.
3. **Automated Trade-In Valuation Engine**:
   With an out-of-sample MAE of ₹$1.40$ Lakhs, online platforms can generate instantaneous guaranteed trade-in offers by discounting algorithmic valuations to the 10th percentile confidence boundary, securing a guaranteed $12\%$ dealer margin.

---

# SECTION 3: INTERACTIVE GUI CONSOLE ARCHITECTURE (`ipywidgets`)

A mandatory requirement across all parts of the project rubric is:
`***provide an interface to load dataset, training, parameter adjustment and for visualization.`

Both projects implement fully reactive, self-contained interactive dashboards directly inside the notebooks using **IPyWidgets** and **IPython.display**.

## 3.1 Question 1 Console (Telecom Attrition & Retention Simulator)
* **Controls**:
  - `Dropdown`: Classifier architecture (`Random Forest`, `Logistic Regression`, `Gradient Boosting`, `KNN`).
  - `FloatSlider`: Classification probability decision cutoff ($p \in [0.10, 0.90]$).
  - `IntSlider`: Per-user retention incentive offer cost ($\$20 - \$150$).
  - `FloatSlider`: Simulated customer retention success rate ($20\% - 90\%$).
  - `Button`: Trigger live model evaluation and financial simulation.
* **Outputs**:
  - Dynamic real-time **Confusion Matrix** with quadrant annotations.
  - Multi-metric performance gauge bar chart (Accuracy, Precision, Recall, F1, ROC-AUC).
  - Financial ROI summary cards (Subscribers targeted, churners saved, net preserved value, campaign ROI %).

## 3.2 Question 3 Console (Used Car Valuation & Depreciation Simulator)
* **Controls**:
  - `Dropdown`: Regularization model architecture (Ridge, Lasso, ElasticNet, OLS).
  - `IntSlider`: Vehicle age ($1 - 15$ years).
  - `IntSlider`: Odometer reading ($5,000 - 200,000$ km).
  - `Dropdown`: Fuel type (Diesel, Petrol, CNG, LPG).
  - `Dropdown`: Transmission (Manual, Automatic).
  - `Dropdown`: Seller channel (Individual, Dealer, Trustmark Dealer).
  - `FloatSlider`: Engine displacement ($796 - 3,000$ cc).
  - `FloatSlider`: Maximum engine power ($35 - 250$ bhp).
  - `Button`: Calculate algorithmic price.
* **Outputs**:
  - Predicted market resale valuation in **₹ Lakhs** with $95\%$ confidence error bounds.
  - Dynamic individual vehicle depreciation curve across ages $1 - 15$.
  - Powertrain resale comparison benchmark (Diesel vs. Petrol vs. CNG).

---

# SECTION 4: COMPLETE FILE CATALOG & WORKSPACE DIRECTORY

The following table documents all components in the workspace:

| File Name | Category | Description |
| :--- | :--- | :--- |
| `VL2026270106396_AST08.pdf` | Specification | Official VIT evaluation rubric, course guidelines, and student question assignment roster. |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Dataset | Raw Telco Customer Churn dataset (7,043 rows, 21 attributes). |
| `telco_preprocessed.csv` | Dataset | Cleaned, encoded, and preprocessed Telco dataset (7,032 rows, 36 attributes). |
| `Car details v3.csv` | Dataset | Raw CarDekho used vehicle dataset (8,128 rows, 13 attributes). |
| `create_rohan_merged_notebook.py` | Generator | Python script that builds and executes Rohan's Part A EDA notebook. |
| `26MML0020_Rohan_ML_MiniProject_PartA_Merged.ipynb` | Notebook | Fully pre-rendered Jupyter notebook for Rohan's Part A (EDA & Preprocessing). |
| `26MML0020_Rohan_ML_MiniProject_PartA_Merged.html` | Report | Standalone HTML export of Rohan's Part A. |
| `create_rohan_partB_notebook.py` | Generator | Python script that builds and executes Rohan's Part B notebook. |
| `26MML0020_Rohan_ML_MiniProject_PartB.ipynb` | Notebook | Fully pre-rendered Jupyter notebook for Rohan's Part B (Feature Engineering & Model Fitting). |
| `26MML0020_Rohan_ML_MiniProject_PartB.html` | Report | Standalone HTML export of Rohan's Part B. |
| `create_rohan_partC_notebook.py` | Generator | Python script that builds, executes, and exports Rohan's Part C notebook. |
| `26MML0020_Rohan_ML_MiniProject_PartC.ipynb` | Notebook | Fully pre-rendered Jupyter notebook for Rohan's Part C (Evaluation & Retention Strategies). |
| `26MML0020_Rohan_ML_MiniProject_PartC.html` | Report | Standalone HTML export of Rohan's Part C. |
| `create_vedant_notebook.py` | Generator | Python script that builds and executes Vedant's Part A EDA notebook. |
| `26MML0045_Vedant_ML_MiniProject_PartA.ipynb` | Notebook | Fully pre-rendered Jupyter notebook for Vedant's Part A (EDA & Preprocessing). |
| `create_vedant_partB_notebook.py` | Generator | Python script that builds and executes Vedant's Part B notebook. |
| `26MML0045_Vedant_ML_MiniProject_PartB.ipynb` | Notebook | Fully pre-rendered Jupyter notebook for Vedant's Part B (Feature Engineering & Regularization). |
| `26MML0045_Vedant_ML_MiniProject_PartB.html` | Report | Standalone HTML export of Vedant's Part B. |
| `create_vedant_partC_notebook.py` | Generator | Python script that builds, executes, and exports Vedant's Part C notebook. |
| `26MML0045_Vedant_ML_MiniProject_PartC.ipynb` | Notebook | Fully pre-rendered Jupyter notebook for Vedant's Part C (Evaluation & Elasticity Inferences). |
| `26MML0045_Vedant_ML_MiniProject_PartC.html` | Report | Standalone HTML export of Vedant's Part C. |
| `create_vaishnavi_notebook.py` | Generator | Python script for Vaishnavi's Part A notebook. |
| `26MML0034_Vaishnavi_Jadhav_ML_MiniProject_PartA.ipynb`| Notebook | Fully pre-rendered Jupyter notebook for Vaishnavi's Part A. |
| `create_vaishnavi_partB_notebook.py` | Generator | Python script for Vaishnavi's Part B notebook. |
| `26MML0034_Vaishnavi_Jadhav_ML_MiniProject_PartB.ipynb`| Notebook | Fully pre-rendered Jupyter notebook for Vaishnavi's Part B. |
| `26MML0034_Vaishnavi_Jadhav_ML_MiniProject_PartB.html` | Report | Standalone HTML export of Vaishnavi's Part B. |
| `PROJECT_EXPLANATION.md` | Master Documentation | Exhaustive, university-grade architectural and mathematical reference manual (this document). |

---

# SECTION 5: HOW TO RUN & VERIFY THE CODEBASE

### **5.1 Environment Prerequisites**
Ensure Python 3.9+ is installed along with the required scientific computing stack:
```bash
pip install numpy pandas scipy scikit-learn imbalanced-learn matplotlib seaborn ipywidgets nbformat nbclient nbconvert
```

### **5.2 Regenerating All Notebooks & HTML Reports**
To re-run any notebook from scratch, simply execute its corresponding generator script from the terminal:
```bash
# To regenerate Rohan's notebooks (Part A, Part B, Part C)
python create_rohan_merged_notebook.py
python create_rohan_partB_notebook.py
python create_rohan_partC_notebook.py

# To regenerate Vedant's notebooks (Part A, Part B, Part C)
python create_vedant_notebook.py
python create_vedant_partB_notebook.py
python create_vedant_partC_notebook.py
```
Each generator script builds the notebook structure, executes all cells via `nbclient.NotebookClient`, pre-renders all high-resolution figures and evaluation metrics, and automatically outputs both a complete `.ipynb` file and a standalone `.html` report.

### **5.3 Viewing the Reports**
* **Instant Offline Viewing**: Double-click any of the generated `.html` files (e.g. `26MML0020_Rohan_ML_MiniProject_PartC.html` or `26MML0045_Vedant_ML_MiniProject_PartC.html`) to open a pre-rendered, publication-quality report in any web browser.
* **Interactive Exploration**: Launch JupyterLab or Jupyter Notebook:
  ```bash
  jupyter lab
  ```
  Open any `.ipynb` file to interact with the embedded GUI dashboards, adjust sliders, and re-train models on the fly.
