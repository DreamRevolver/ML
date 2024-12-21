import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

data = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")

X = data.drop(columns=["Diabetes_012"])
y = data["Diabetes_012"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
layer_configs = [(64, 32), (128, 64), (64, 64, 32)]
learning_rates = [0.001, 0.01]
batch_sizes = [32, 64]
epochs = 50
results = []
for layers in layer_configs:
    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"Навчання моделі: Layers: {layers}, Learning Rate: {lr}, Batch Size: {batch_size}")

            print('Building model...')
            model = Sequential()
            model.add(Input(shape=(X_train.shape[1],)))
            for layer_size in layers:
                model.add(Dense(layer_size, activation='relu'))
            model.add(Dense(3, activation='softmax'))
            print('Finished building model...')
            print('Compiling model...')
            optimizer = Adam(learning_rate=lr)
            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            print('Finished compiling model...')
            print('Teaching model...')
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            print('Finished teaching model...')
            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
            results.append([layers, lr, batch_size, test_loss, test_accuracy])
results_df = pd.DataFrame(results, columns=["Layer Configs", "Learning Rate", "Batch Size", "Test Loss", "Test Accuracy"])
results_df.to_csv("diabetes_nn_results.csv", index=False)
print("Результати збережено у файл 'diabetes_nn_results.csv'")
