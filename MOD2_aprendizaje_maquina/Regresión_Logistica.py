''' Frida Lizett Zavala Pérez 
    A01275226

    Regresión Logistica
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cargar el conjunto de datos Iris desde un archivo  
# Se definen los nommbres para las columnas
columns = ["sepal length","sepal width","petal length","petal width", "class"]
data = pd.read_csv('iris_modificado.csv',names = columns)

#se mezclan los datos del dataset debido a que como se dividirán en los datos de test y prueba 
#se requiere que se encuenten datos de todas las clases, inicialmente se encuentran en orden
data = data.sample(frac=1, random_state=42)#mezclar los datos del dataset

#se dividen los datos para X y Y
X = data[["sepal length","sepal width","petal length","petal width"]]
y = data[["class"]]

# Dividir el dataset en conjuntos de entrenamiento y prueba
#se dividen con una proporción de 80% y 20% con slices para train y test respectivamente
X_train = X[0:120]
X_test = X[120:151] #numpy slices of an array

y_train = y[0:120]
y_test = y[120:151] #numpy slices of an array


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
learning_rate = 0.01
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
    print(f"Valor calculado: {y_pred_classes[i]}, Valor real: {y_test[i]}")




