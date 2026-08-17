from .knn_rf_ensemble_model import train_knn_rf_ensemble
from .svm_model import train_svm
from .xgboost_model import train_xgboost

train_knn = train_knn_rf_ensemble

__all__ = ["train_knn", "train_knn_rf_ensemble", "train_svm", "train_xgboost"]
