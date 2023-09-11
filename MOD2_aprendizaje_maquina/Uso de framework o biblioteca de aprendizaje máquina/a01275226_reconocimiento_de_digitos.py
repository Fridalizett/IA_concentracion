
"""A01275226_Reconocimiento_de_digitos.ipynb

A01275226
Frida Lizett Zavala Pérez

# Reconocimiento de digitos


Este ejemplo de red neuronal para clasificación, es una adaptación de un ejemplo mostrado en el siguiente libro.
[Chollet, F. (2017). Deep learning with python. Manning Publications.](https://www.manning.com/books/deep-learning-with-python)
"""

from keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers

import matplotlib.pyplot as plt

"""## Recuperar los dígitos de la librería mnist"""

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

for i in range(10):
    digit = train_images[i]
    plt.imshow(digit, cmap=plt.cm.binary)
    plt.show()
print("Tamaño de test_images:")
print(len(test_images))

"""## Creación del modelo"""

#secuencial
'''model = models.Sequential([
  layers.Dense(512, activation='relu'),
  layers.Dense(256, activation='relu'),
  layers.Dense(10, activation='softmax')
])'''

#API
input_tensor = layers.Input(shape=(784))
x1 = layers.Dense(512, activation ='relu')(input_tensor)
x2 = layers.Dense(256, activation ='relu')(x1)
output_tensor = layers.Dense(10, activation ='softmax')(x2)

model = models.Model(inputs = input_tensor, outputs = output_tensor)

"""## Definir las funciones para
- Optimización
- Pérdida (Loss function)
- Metrica de éxito de la red neuronal
"""

model.compile(optimizer='rmsprop',                    #Root Mean Square Propogation 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

"""## Cambiar la forma de los datos de 28x28 a vecores de 784 y normalizar sus valores de [0,255] a [0, 1]"""
print("Train images (shape):")
print (train_images.shape)

copy_test_images = test_images  #hacemos una copia para poder imprimir los digitos originales después
train_images = train_images.reshape((len(train_images), 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((len(test_images), 28 * 28))
test_images = test_images.astype('float32') / 255

print("Train images (shape después de cambiar y normalizar los datos):")
print (train_images.shape)

x = range(0, 784)
digit = train_images[0]
plt.bar(x, digit)
plt.show()

"""## Realizar el aprendizaje con
- 128 lotes
- 5 Épocas
"""

print("number of batches",len(train_images)/128)
model.fit(train_images, train_labels, epochs=5, batch_size=128)

"""## Probar el modelo"""

test_digits = test_images[0:10]
predictions = model.predict(test_digits)

print("Probando el modelo")
for i in range(10):
    digit = copy_test_images[i]
    plt.imshow(digit, cmap=plt.cm.binary)
    plt.show()
    print("Predictions vector: ", predictions[i])
    print("Most probable: ",predictions[i].argmax())
    print("True value:" ,  test_labels[i])
    print("Probability: ", predictions[i][predictions[i].argmax()])
    print("")

"""## Probar el modelo con todos los datos de test"""
print("Probando el moddelo con todos los datos de test")
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

