import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Завантажуємо датасет
data = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")

# Переводимо цільову змінну 'Diabetes_012' в категоріальну змінну (0, 1, 2)
y = data["Diabetes_012"]

# Використовуємо лише числові стовпці для моделювання
X = data.drop(columns=["Diabetes_012"])

# Перетворюємо категоріальні змінні на dummy-перемінні (one-hot encoding)
X = pd.get_dummies(X, drop_first=True)

# Поділяємо дані на три набори: навчальний (70%), валідаційний (20%) та тестувальний (10%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

# Створюємо кастомну логістичну регресію
class LogisticRegressionCustom:
    def __init__(self, learning_rate=0.01, epochs=1000, threshold=0.5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0

        for epoch in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return [1 if i >= self.threshold else 0 for i in y_predicted]

# Запуск моделювання з різними гіперпараметрами
results = []
X_train_np, X_val_np, X_test_np = X_train.to_numpy(), X_val.to_numpy(), X_test.to_numpy()

for learning_rate in [0.01, 0.1, 1]:
    for epochs in [500, 1000, 1500]:
        for threshold in [0.4, 0.5, 0.6]:
            # Навчання моделі
            print(f"Start of modeling with parameters: Learning rate: {learning_rate}, Epochs: {epochs}, Threshold: {threshold}")
            model = LogisticRegressionCustom(learning_rate=learning_rate, epochs=epochs, threshold=threshold)
            model.fit(X_train_np, y_train.to_numpy())

            # Прогнозування на валідаційному наборі
            y_pred_val = model.predict(X_val_np)
            accuracy = accuracy_score(y_val, y_pred_val)

            # Збереження результатів
            results.append([learning_rate, epochs, threshold, accuracy])

# Створення DataFrame для збереження результатів
results_df = pd.DataFrame(results, columns=["Learning Rate", "Epochs", "Threshold", "Accuracy"])

# Збереження результатів у файл .csv
results_df.to_csv("logistic_regression_results.csv", index=False)

# Виведення результатів
print("Результати експериментів з різними гіперпараметрами:")
print(results_df)
