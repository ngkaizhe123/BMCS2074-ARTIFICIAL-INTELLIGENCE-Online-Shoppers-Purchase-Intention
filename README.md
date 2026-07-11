# Online Shoppers Purchase Intention Prediction System

## 📌 Project Overview

This project applies **Supervised Machine Learning** techniques to predict whether an online shopping session will
result in a completed purchase based on customer browsing behaviour and session characteristics.

The system features an interactive **Streamlit Web Application** that allows users to explore the dataset, compare
different machine learning models, and perform real-time purchase intention prediction.

Three classification models are implemented and evaluated:

* Artificial Neural Network (ANN)
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)

The project compares the performance of each model using standard evaluation metrics including **Accuracy, Precision,
Recall, and F1-score**.

---

## 🎯 Project Objectives

The system aims to:

* Predict whether an online visitor is likely to complete a purchase.
* Compare the performance of multiple supervised learning algorithms.
* Provide interpretable prediction results for marketing decision support.
* Demonstrate how customer browsing behaviour influences purchase intention.

---

## 🚀 System Features

### 1. Purchase Intention Prediction

Predict whether an online shopping session will result in a purchase.

Output:

* Purchase Prediction (Yes / No)
* Purchase Probability (%)
* Confidence Level

---

### 2. Model Comparison

Compare different classification algorithms using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Models included:

* ANN
* SVM
* KNN

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

## 📂 Project Structure

```text
online_shoppers_purchase_prediction/
│
├── data/
│   └── online_shoppers_intention.csv
│
├── src/
│   ├── __init__.py
│   ├── utils.py
│   └── data_preprocessing.py
│
├── models/
│   ├── __init__.py
│   ├── ann_model.py
│   ├── svm_model.py
│   └── knn_model.py
│
├── saved_models/
│   ├── ann_model.h5
│   ├── svm_model.pkl
│   └── knn_model.pkl
│
├── pages/
│   ├── 1_📊_Data_Exploration.py
│   ├── 2_📈_Model_Comparison.py
│   └── 3_🔮_Purchase_Prediction.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 📈 Model Evaluation

The implemented models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

The best-performing model is selected based on its overall classification performance.

---

## 💡 Business Value

This system can support e-commerce platforms by:

* Identifying visitors who are likely to make a purchase.
* Assisting marketing teams in prioritising potential customers.
* Providing personalised marketing recommendations.
* Supporting data-driven business decision making.
