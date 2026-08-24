from .fsvm import FuzzySVM
from .knn_rf_ensemble_model import train_knn_rf_ensemble
from .mkl_svm import HybridKernelSVC
from .svm_model import train_svm
from .xgboost_model import train_xgboost

train_knn = train_knn_rf_ensemble

__all__ = [
    "FuzzySVM",
    "HybridKernelSVC",
    "train_knn",
    "train_knn_rf_ensemble",
    "train_svm",
    "train_xgboost",
]
