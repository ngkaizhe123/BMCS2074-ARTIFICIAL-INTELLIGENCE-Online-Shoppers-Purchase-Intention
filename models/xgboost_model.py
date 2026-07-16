from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier


def train_xgboost(X_train, y_train):
    xgb = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", random_state=42
    )

    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.3],
        "scale_pos_weight": [scale_pos_weight],
    }

    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        scoring="f1",
        cv=5,
        n_iter=30,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )

    random_search.fit(X_train, y_train)

    return random_search.best_estimator_
