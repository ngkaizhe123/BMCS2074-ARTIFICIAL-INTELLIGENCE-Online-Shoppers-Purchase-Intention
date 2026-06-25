import streamlit as st

# 1. set the page config (must be the first line)
st.set_page_config(
    page_title="Customer Purchase Prediction System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. set the main title of the website
st.title("🛍️ Customer Purchase Prediction System")
st.markdown("---")  # break line

# 3. set the project overview
st.markdown("""
### 📌 Project Background (Project Overview)
Welcome to our machine learning end-of-term project exhibition platform!
**Supervised Machine Learning**, based on user behavior data on e-commerce websites (such as browsing duration, whether it's a weekend, bounce rate, etc.), predict whether the user will ultimately make a purchase.

### 🎯 Objectives (Objectives)
* **Binary Classification Task:** Predict customer purchase behavior (`0`: not purchase, `1`: purchase).
* **Multi-Model Comparison:** Compete with three classic machine learning algorithms and compare their performance.

---

### 🚀 Sidebar Navigation Guide (Navigation)
Click on the menu bar on the left side of the screen to experience our system:
* **📊 1. Data Exploration:** View the structure and data distribution of the dataset we used.
* **📈 2. Model Comparison:** View the evaluation scores of the **ANN**, **SVM**, **KNN** three models (Accuracy, F1-score, etc.).
* **🔮 3. Live Prediction:** Enter simulated data personally and call the AI model loaded in the background to test whether it will predict you to purchase!

---

### 👨‍💻 Development Team (Team Members)
* **NG KAI ZHE** - Responsible for ANN neural network model and Streamlit architecture setup
* **TAN YONG SHENG** - Responsible for SVM model building
* **YAU SOON HAN** - Responsible for KNN model building and feature engineering
""")
