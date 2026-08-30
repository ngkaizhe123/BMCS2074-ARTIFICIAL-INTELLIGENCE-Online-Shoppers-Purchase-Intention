# Online Shoppers Purchase Intention Prediction System

## 📌 Project Overview

This project applies **Supervised Machine Learning** techniques to predict whether an online shopping session will
result in a completed purchase based on customer browsing behaviour and session characteristics.

The system features an interactive **Streamlit Web Application** that allows users to explore the dataset, compare
different machine learning models, and perform real-time purchase intention prediction.

Three classification models are implemented and evaluated:

* Extreme Gradient Boosting (XGBoost) with PSO hyperparameter optimization
* Support Vector Machine (SVM) with optimal threshold tuning
* K-Nearest Neighbors (KNN) with Random Forest ensemble

The project compares the performance of each model using standard evaluation metrics including **Accuracy, Precision,
Recall, F1-score, and ROC-AUC**, and includes SHAP (SHapley Additive exPlanations) for model interpretability.

---

## 🎯 Project Objectives

The system aims to:

* Predict whether an online visitor is likely to complete a purchase.
* Compare the performance of multiple supervised learning algorithms.
* Provide interpretable prediction results for marketing decision support using SHAP explanations.
* Demonstrate how customer browsing behaviour influences purchase intention.
* Handle imbalanced data using SMOTE (Synthetic Minority Over-sampling Technique).

---

## 🚀 System Features

### 1. Purchase Intention Prediction

Predict whether an online shopping session will result in a purchase.

Output:

* Purchase Prediction (Yes / No)
* Purchase Probability (%)
* Confidence Level
* Feature importance insights

---

### 2. Model Comparison

Compare different classification algorithms using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC-AUC
* Optimal threshold analysis

Models included:

* XGBoost with PSO optimization
* SVM with threshold tuning
* KNN-RF Ensemble

---

### 3. Decision Support Dashboard

The prediction results can be further interpreted into business-friendly insights such as:

* Purchase Probability
* Customer Priority
* Engagement Level
* Top Influencing Factors
* Marketing Recommendation

These outputs are generated using prediction probabilities together with business rules to assist marketing
decision-making.

---

### 4. Model Interpretability

SHAP (SHapley Additive exPlanations) analysis provides:

* Feature importance rankings
* Individual prediction explanations
* Beeswarm plots showing feature impact distribution
* Waterfall plots for single predictions

---

## 📊 Dataset

**Dataset: Online Shoppers Purchasing Intention Dataset**

The dataset contains customer browsing behaviour collected during online shopping sessions.

Example features include:

* Administrative
* Administrative_Duration
* Informational
* Informational_Duration
* ProductRelated
* ProductRelated_Duration
* BounceRates
* ExitRates
* PageValues
* SpecialDay
* Month
* OperatingSystems
* Browser
* Region
* TrafficType
* VisitorType
* Weekend

Target Variable:

* **Revenue**

    * TRUE = Purchase
    * FALSE = No Purchase

---

## ⚙️ Python Configuration and Setup

### Prerequisites

* Python 3.12
* pip package manager

### Installation Steps

1. **Clone or download the project repository**

2. **Navigate to the project directory**
   ```bash
   cd BMCS2074-ARTIFICIAL-INTELLIGENCE-Online-Shoppers-Purchase-Intention
   ```

3. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

4. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Required Packages

The project requires the following Python packages:

* `matplotlib==3.10.9` - Data visualization
* `numpy==2.0.0` - Numerical computing
* `pandas==3.0.3` - Data manipulation and analysis
* `seaborn==0.13.2` - Statistical data visualization
* `scipy==1.16.0` - Scientific computing
* `imbalanced-learn==0.14.2` - SMOTE for handling imbalanced data
* `scikit-learn==1.9.0` - Machine learning algorithms
* `xgboost==3.3.0` - Gradient boosting framework
* `streamlit==1.58.0` - Web application framework
* `joblib==1.5.3` - Model persistence
* `shap==0.51.0` - Model interpretability

### Running the Application

Start the Streamlit web application:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

---

## 📂 Project Structure

```text
online_shoppers_purchase_prediction/
│
├── data/
│   └── online_shoppers_intention.csv
│
├── src/
│   ├── __init__.py
│   ├── utils.py                  # Utility functions for model evaluation, metrics, SHAP
│   ├── data_preprocessing.py     # Data cleaning and preprocessing pipeline
│   ├── eda.py                    # Exploratory data analysis functions
│   ├── model_visualize.py        # Model visualization and comparison
│   └── ui_theme.py               # Streamlit UI theming
│
├── models/
│   ├── __init__.py
│   ├── xgboost_model.py          # XGBoost with PSO optimization
│   ├── svm_model.py              # SVM with threshold tuning
│   └── knn_rf_ensemble_model.py  # KNN with Random Forest ensemble
│
├── saved_models/
│   ├── xgboost_pso.pkl
│   ├── svm_model.pkl
│   └── knn_rf_ensemble_model.pkl
│
├── pages/
│   ├── 1_Data_Exploration.py     # Data exploration and EDA
│   ├── 2_Model_Comparison.py     # Model performance comparison
│   └── 3_Live_Prediction.py      # Real-time prediction interface
│
├── report_assets/
│   ├── plots/                    # Generated plots and visualizations
│   │   ├── eda/                  # Exploratory data analysis plots
│   │   └── model comparison/     # Model evaluation plots
│   └── threshold_analysis/       # Threshold optimization results
│
├── .devcontainer/
│   └── devcontainer.json         # Development container configuration
│
├── .github/
│   └── workflows/
│       └── ci.yml                # CI configuration
│
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 📈 Model Evaluation

The implemented models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC-AUC
* Optimal threshold analysis

### Model Features

* **XGBoost**: Particle Swarm Optimization (PSO) for hyperparameter tuning
* **SVM**: Precision-Recall threshold optimization
* **KNN-RF**: Ensemble approach combining KNN and Random Forest

### SHAP Analysis

All models include SHAP explanations for:

* Global feature importance
* Individual prediction interpretation
* Feature impact visualization

The best-performing model is selected based on its overall classification performance and interpretability.

---

## 💡 Business Value

This system can support e-commerce platforms by:

* Identifying visitors who are likely to make a purchase.
* Assisting marketing teams in prioritising potential customers.
* Providing personalised marketing recommendations.
* Supporting data-driven business decision making.
* Offering interpretable insights for explainable AI compliance.

---

## 👨‍💻 Development Team

* **NG KAI ZHE** – XGBoost model & Streamlit architecture
* **TAN YONG SHENG** – SVM model
* **YAU SOON HAN** – KNN model & feature engineering
