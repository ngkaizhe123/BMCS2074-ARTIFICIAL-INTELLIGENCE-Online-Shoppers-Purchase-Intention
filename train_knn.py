from models.knn_model import train_knn
from src.data_preprocessing import preprocess_data
from src.utils import save_model, evaluate_model, print_metrics

# Data preprocessing (80% train / 20% test)
X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
    "data/raw/online_shoppers_intention.csv"
)

# Train model
model = train_knn(X_train, y_train)

metrics = evaluate_model(model, X_test, y_test)
print_metrics(metrics)

# Save model
save_model(model, "saved_models/knn_model.pkl")
