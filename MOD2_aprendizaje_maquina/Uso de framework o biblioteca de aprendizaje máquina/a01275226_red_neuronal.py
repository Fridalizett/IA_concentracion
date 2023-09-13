# -*- coding: utf-8 -*-
"""A01275226_Red_Neuronal.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xYbGlOjXIh31nLLxtt_WiIEbpkq6AUty

Frida Lizett Zavala Pérez

A01275226

# Red Neuronal

### Preparación de datos
Primero, debes cargar y preparar tus datos. Las características están en X y etiquetas en y, dividimos los datos en conjuntos de entrenamiento, validación y prueba. Además, normalizar los datos es una buena práctica para el entrenamiento de redes neuronales.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# Cargar tus datos
filename= 'new_df.csv'
data = pd.read_csv(filename)

y = data.diabetes
X = data.drop('diabetes', axis = 1)

# Dividir los datos en conjuntos de entrenamiento, validación y prueba
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

"""## Creación del modelo

Red neuronal usando TensorFlow y Keras
"""

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # clasificación binaria
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

"""## Entrenamiento del modelo
Se entrena el modelo con los datos de entrenamiento y validación.

En este punto se pueden aplicar las técnicas de regularización, para reducir overfitting en caso de que sea necesario.
"""

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

"""## Evaluación del modelo
Se evalua el modelo con el conjunto de prueba y se observan las métricas del rendimiento
"""

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Precisión en el conjunto de prueba:", test_accuracy)

"""## Predicciones con nuevos datos (datos de test)
Se generan las predicciones con los datos de test y se visualizan las respuestas con el porcentaje de exactitud y una gráfica que compara el numero de predicciones correctas o incorrectas.
"""

# Obtén las predicciones
# Obtén las predicciones
# Convierte las predicciones a valores binarios (0 o 1) utilizando un umbral de decisión (por ejemplo, 0.5)

# Hacer predicciones en nuevos datos
predictions = model.predict(X_test)

threshold = 0.5
binary_predictions = (predictions > threshold).astype(int)

# Compara las predicciones binarias con los valores reales
comparison = (binary_predictions == y_test.values.reshape(-1, 1))

# Calcula la precisión de las predicciones en el conjunto de prueba
accuracy = np.mean(comparison)
print(f"Precisión en el conjunto de prueba: {accuracy * 100:.2f}%")

# También puedes contar los casos correctos e incorrectos
correct_predictions = np.sum(comparison)
incorrect_predictions = len(comparison) - correct_predictions
print(f"Predicciones correctas: {correct_predictions}")
print(f"Predicciones incorrectas: {incorrect_predictions}")


#Grafica los valores predichos y los valores reales

# Etiquetas para las barras
labels = ['Correcto', 'Incorrecto']

# Valores correspondientes a las barras
values = [correct_predictions, incorrect_predictions]

# Colores para las barras
colors = ['green', 'red']

# Crea un gráfico de barras
plt.bar(labels, values, color=colors)

# Agrega etiquetas al gráfico
plt.title('Predicciones Correctas e Incorrectas')
plt.xlabel('Categoría')
plt.ylabel('Cantidad de Predicciones')
plt.xticks(rotation=0)

# Muestra el gráfico
plt.show()

import seaborn as sns
from sklearn.metrics import confusion_matrix
# Calcular la matriz de confusión
confusion = confusion_matrix(y_test, binary_predictions)

# Visualizar la matriz de confusión utilizando Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", square=True,
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicciones')
plt.ylabel('Valores Reales')
plt.title('Matriz de Confusión')
plt.show()

"""## Análisis del sesgo y varianza"""

# Gráfico de curvas de aprendizaje
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.show()