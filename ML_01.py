import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from itertools import product

# Шлях до файлу
file_path = 'diabetes_012_health_indicators_BRFSS2015.csv'
df = pd.read_csv(file_path)

df = df.dropna()

X = df.drop(columns=['Diabetes_012'])  # Всі стовпці, крім Diabetes_012
y = df['Diabetes_012']  # Цільова змінна

# Стандартизація даних
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Розбиття на тренувальну, валідаційну та тестову вибірки
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42, stratify=y_temp)

# Списки гіперпараметрів для пошуку
C_values = [0.1, 1, 10]
penalties = ['l1', 'l2']
max_iters = [100, 500, 1000]
warm_starts = [True, False]

# Список всіх комбінацій гіперпараметрів
hyperparameter_combinations = list(product(C_values, penalties, max_iters, warm_starts))

# Список для збереження результатів
results = []

# Перебір всіх комбінацій гіперпараметрів
for C, penalty, max_iter, warm_start in hyperparameter_combinations:
    print(f"Start of modeling with parameters: C={C}, penalty={penalty}, max_iter={max_iter}, warm_start={warm_start}")

    # Створення моделі з поточними гіперпараметрами
    model = LogisticRegression(C=C, penalty=penalty, max_iter=max_iter, warm_start=warm_start, solver='liblinear',
                               random_state=42)

    # Навчання моделі
    model.fit(X_train, y_train)

    # Оцінка точності на валідаційних даних
    y_valid_pred = model.predict(X_valid)
    accuracy_val = accuracy_score(y_valid, y_valid_pred)

    # Додавання результату до списку
    results.append([C, penalty, max_iter, warm_start, accuracy_val])

# Перетворення результатів на DataFrame
results_df = pd.DataFrame(results, columns=['C', 'Penalty', 'Max_Iter', 'Warm_Start', 'Accuracy'])

results_df.to_csv("results.csv", index=False)

print(results_df)
