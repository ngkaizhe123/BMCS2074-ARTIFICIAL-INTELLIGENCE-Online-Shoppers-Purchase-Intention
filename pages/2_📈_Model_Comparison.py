import os

import joblib
import streamlit as st

from src.data_preprocessing import get_cleaned_data
from src.utils import get_evaluation_report_df

st.set_page_config(page_title="Model comparison", page_icon="📈")
st.title("Supervised Machine Learning performance comparison (Model Comparison)")
st.markdown(
    "This page shows the performance of **ANN**, **SVM**, **KNN** models in the test set."
)

# 2. Get the test set data (to test the models)
# We only need the test set (X_test_scaled, y_test), so we use _ to omit the other variables
_, X_test_scaled, _, y_test = get_cleaned_data(
    filepath="data/customer_purchase_data.csv"
)

# 3. Check if the model file exists and load the evaluation
ann_path = "saved_models/ann_model.pkl"

if os.path.exists(ann_path):
    st.subheader("ANN (Artificial Neural Network) Evaluation")

    # Load the model you trained and saved earlier
    ann_model = joblib.load(ann_path)

    # Let the model make predictions
    ann_pred = ann_model.predict(X_test_scaled)

    # Call the new function we just wrote in utils to get the table
    ann_df = get_evaluation_report_df(y_test, ann_pred)

    # Use Streamlit to render the beautiful interactive table (st.dataframe)
    # st.table(ann_df) also can be used，st.dataframe supports highlighting and scrolling
    st.dataframe(ann_df, use_container_width=True)

else:
    st.warning(
        "Cannot find the ANN model! Please train and save the model by running `python models/ann_model.py` in the background."
    )

# Note: Once your teammates save the SVM and KNN models, you can directly copy the code inside the if statement above,
# change ann_path to svm_path, and you can display their tables side by side on the webpage!
