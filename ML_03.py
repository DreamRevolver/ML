import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier


data = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")
X = data.drop(columns=["Diabetes_012"])
y = data["Diabetes_012"]
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y_min, y_max = y_train.min(), y_train.max()
learning_rates = [0.01, 0.05, 0.1]
depths = [4, 6, 8]
iterations = [500, 1000, 1500]
results = []

for lr in learning_rates:
    for depth in depths:
        for iter_count in iterations:
            print(f"Training model with parameters: Learning Rate={lr}, Depth={depth}, Iterations={iter_count}")
            model = CatBoostClassifier(
                learning_rate=lr,
                depth=depth,
                iterations=iter_count,
                verbose=0,
                random_seed=42,
                auto_class_weights="Balanced"
            )

            # Train the model
            model.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = model.predict(X_test)

            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy:.4f}")

            # Store the results
            results.append([lr, depth, iter_count, accuracy])

results_df = pd.DataFrame(results, columns=["Learning Rate", "Depth", "Iterations", "Accuracy"])
results_df.to_csv("catboost_classification_results.csv", index=False)
print(results_df)
