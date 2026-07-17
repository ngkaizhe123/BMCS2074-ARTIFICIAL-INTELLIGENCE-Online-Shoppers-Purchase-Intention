from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


def train_knn(X_train, y_train):

    knn = KNeighborsClassifier()

    param_grid = {
        # try odd k values to avoid tie votes in binary classification
        "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        # 'uniform' = all k neighbors vote equally
        # 'distance' = closer neighbors get more voting weight
        "weights": ["uniform", "distance"],
        # 1 = Manhattan distance, 2 = Euclidean distance
        "p": [1, 2],
    }

    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        scoring="f1",  # dataset is imbalanced (few purchases), f1 > accuracy
        cv=5,
        verbose=2,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    print(f"Best KNN params: {grid_search.best_params_}")
    print(f"Best CV F1 score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_
