from models.xgboost_model import train_xgboost
from src.data_preprocessing import preprocess_data
from src.utils import save_model, evaluate_model, print_metrics

# Data preprocessing
X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
    "data/raw/online_shoppers_intention.csv"
)

# Train model
model = train_xgboost(X_train, y_train)

metrics = evaluate_model(model, X_test, y_test)

print_metrics(metrics)

# Save model and preprocessor
save_model(model, "saved_models/xgboost_model.pkl")
save_model(preprocessor, "saved_models/preprocessor.pkl")
