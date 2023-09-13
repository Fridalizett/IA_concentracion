

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

data['smoking_history'] = data['smoking_history'].replace('ever', 'used_to_smoke')
data['smoking_history'] = data['smoking_history'].replace('not current', 'used_to_smoke')
data['smoking_history'] = data['smoking_history'].replace('former', 'used_to_smoke')
data['smoking_history'] = data['smoking_history'].replace('No Info', 'dont_smoke')
data['smoking_history'] = data['smoking_history'].replace('never', 'dont_smoke')
data['smoking_history'] = data['smoking_history'].replace('current', 'smoke')
print(data.smoking_history.value_counts())


#One Hot Encoding
sex = pd.get_dummies(data['gender'])
new_df = pd.concat([data.drop('gender', axis = 1), sex], axis = 1)

smoke = pd.get_dummies(data['smoking_history'])
new_df = pd.concat([new_df.drop('smoking_history', axis = 1), smoke], axis = 1)

print(new_df.head())

# Guardar el conjunto de datos modificado en un nuevo archivo CSV
new_df.to_csv('new_df.csv', index=False)