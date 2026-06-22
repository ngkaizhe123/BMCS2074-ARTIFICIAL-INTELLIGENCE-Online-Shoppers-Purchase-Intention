# Customer Purchase Prediction System

## 📌 Project Overview

This project applies **Supervised Machine Learning** techniques to predict customer purchasing behavior. It features an
interactive **Streamlit Web Application** to showcase and compare the performance of three different models: **KNN**,
**SVM**, and **ANN** in real-time.

The system provides two core capabilities:

1. **Classification Task**: Predicts whether a customer will make a purchase or not (`0`: No Purchase, `1`: Purchase).
2. **Regression Task**: Predicts the **Total Purchase Amount (or Quantity)** based on customer profiles such as salary,
   age, gender, number of past purchases, and time spent on the website.

---

## 📂 Project Structure

```text
customer_purchase_prediction/
│
├── data/
│   └── online_shoppers_intention.csv  # Raw dataset
│
├── src/
│   ├── __init__.py
│   ├── utils.py                       # Shared tools (plotting, model loading)
│   └── data_preprocessing.py          # Unified data cleaning & feature engineering
│
├── models/                            # Source code for model training and exporting
│   ├── __init__.py
│   ├── ann_model.py
│   ├── svm_model.py
│   └── knn_model.py
│
├── saved_models/                      # Trained model artifacts (.h5 or .pkl)
│   ├── ann_model.h5
│   ├── svm_model.pkl
│   └── knn_model.pkl
│
├── pages/                             # Streamlit GUI Multi-page App
│   ├── 1_📊_Data_Exploration.py
│   ├── 2_📈_Model_Comparison.py
│   └── 3_🔮_Live_Prediction.py
│
├── main.py                            # Streamlit main entry point
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation