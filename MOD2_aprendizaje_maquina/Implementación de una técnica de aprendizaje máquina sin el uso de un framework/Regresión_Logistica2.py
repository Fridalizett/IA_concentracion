''' Frida Lizett Zavala Pérez 
    A01275226

    Regresión Logistica
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



'''Limpieza del data set'''

#cargar datos desde archivo
data = pd.read_csv("diabetes_prediction_dataset.csv")

print(data.info())
#Finding null values
print(data.isnull().sum())
#gender column
data.gender.value_counts()

#Drop a other de género para evitar conflicto en las predicciones
data = data.drop(data[data['gender'] == 'Other'].index)
print(data.gender.value_counts())

data['smoking_history'] = data['smoking_history'].replace('ever', 'used_to')
data['smoking_history'] = data['smoking_history'].replace('not current', 'used_to')
data['smoking_history'] = data['smoking_history'].replace('former', 'used_to')
data['smoking_history'] = data['smoking_history'].replace('No Info', 'no')
data['smoking_history'] = data['smoking_history'].replace('never', 'no')
data['smoking_history'] = data['smoking_history'].replace('current', 'yes')
print(data.smoking_history.value_counts())


#One Hot Encoding
sex = pd.get_dummies(data['gender'])
new_df = pd.concat([data.drop('gender', axis = 1), sex], axis = 1)

smoke = pd.get_dummies(data['smoking_history'])
new_df = pd.concat([new_df.drop('smoking_history', axis = 1), smoke], axis = 1)

print(new_df.head())

# Guardar el conjunto de datos modificado en un nuevo archivo CSV
new_df.to_csv('new_df.csv', index=False)

data = pd.read_csv('new_df.csv')



"""INICIA LA CLASIFICACIÓN CON REGRESIÓN LOGISTICA"""

y = data.diabetes
X = data.drop('diabetes', axis = 1)

# Dividir el dataset en conjuntos de entrenamiento y prueba
#se dividen con una proporción de 80% y 20% con slices para train y test respectivamente
X_train = X[0:79986]
X_test = X[79986:99983] #numpy slices of an array

y_train = y[0:79986]
y_test = y[79986:99983] #numpy slices of an array



# Estandarizar las características del dataset a partir del promedio y la desviacion estandar
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Agregar un término de sesgo (bias) a las características
X_train_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# Inicializar parámetros
theta = np.zeros(X_train_bias.shape[1])

# Definir la función sigmoide (regresión logistica) pone los valores entre 0 y 1
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Definir la función de costo (entropía cruzada)
def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    cost = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

# Gradiente descendente
def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = []
    for _ in range(num_iterations):
        z = np.dot(X, theta)
        predictions = sigmoid(z)
        error = y - predictions
        gradient = np.dot(X.T, error) / m
        theta += learning_rate * gradient
        cost_history.append(np.mean(np.abs(error)))
    
    return theta, cost_history

# Hiperparámetros del algoritmo
learning_rate = 0.03
num_iterations = 1000

# Entrenar el modelo
trained_theta, cost_history = gradient_descent(X_train_bias, y_train, theta, learning_rate, num_iterations)

# Calcular las predicciones en el conjunto de prueba
y_pred = sigmoid(np.dot(X_test_bias, trained_theta))
y_pred_classes = (y_pred >= 0.5).astype(int)

# Calcular el error absoluto entre predicciones y valores reales
error = np.abs(y_pred_classes - y_test)

# Graficar el ERROR en función de las iteraciones
plt.plot(cost_history)
plt.xlabel('Iteraciones')
plt.ylabel('Error')
plt.title('Error en función de las iteraciones')
plt.show()

# Imprimir valores calculados y reales

for i in range(len(y_test)):
    print(f"Valor calculado: {y_pred_classes[i]}, Valor real: {y_test.values[i]}")

"""Gráfica de los valores reales y valores predichos"""

# errores calculados
correct_predictions = len(error) - np.sum(error)
incorrect_predictions = np.sum(error)

# Crear una lista de etiquetas para las barras
labels = ['Predicciones Correctas', 'Predicciones Incorrectas']

# Crear una lista de valores para las barras
values = [correct_predictions, incorrect_predictions]

# Crear la gráfica de barras
plt.bar(labels, values, color=['green', 'red'])
plt.xlabel('Tipo de Predicción')
plt.ylabel('Cantidad de Predicciones')
plt.title('Predicciones Correctas vs. Incorrectas')

# Mostrar la gráfica
plt.show()

"""Calcular el accuracy del modelo"""

# Calcular y mostrar el accuracy manualmente
accuracy = correct_predictions / len(y_test) * 100
print(f'Accuracy del modelo: {accuracy:.2f}%')