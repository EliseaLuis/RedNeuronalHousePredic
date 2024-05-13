import numpy as np
import pandas as pd

# Cargar los datos de entrenamiento desde los archivos CSV
datos_entrenamiento = pd.read_csv('datos_entrenamiento.csv', header=None).iloc[1:].values.astype(float)
precios_entrenamiento = pd.read_csv('precios_entrenamiento.csv', header=None).iloc[1:].values.astype(float)

# Cargar los datos de prueba desde los archivos CSV
datos_prueba = pd.read_csv('datos_prueba.csv', header=None).iloc[1:].values.astype(float)
precios_prueba = pd.read_csv('precios_prueba.csv', header=None).iloc[1:].values.astype(float)

# Definir el número de neuronas en las capas
input_size = datos_entrenamiento.shape[1]
hidden_size = 8  # Número arbitrario de neuronas en la capa oculta
output_size = 1  # Solo hay un valor de precio

class RedNeuronal:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Inicialización de parámetros
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))
    
    def forward(self, X):
        # Propagación hacia adelante
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.Z2  # Función de activación identidad para regresión
        return self.A2
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def backward(self, X, y, learning_rate):
        # Retropropagación
        m = X.shape[0]  # Número de ejemplos de entrenamiento
        
        # Calcular los gradientes de la capa de salida
        dZ2 = self.A2 - y
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Calcular los gradientes de la capa oculta
        dZ1 = np.dot(dZ2, self.W2.T) * self.sigmoid_derivative(self.A1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Actualizar los parámetros
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
    
    def train(self, X_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            # Propagación hacia adelante
            predictions = self.forward(X_train)
            
            # Retropropagación
            self.backward(X_train, y_train, learning_rate)
            
            # Calcular la pérdida
            loss = self.mean_squared_error(predictions, y_train)
            
            # Imprimir la pérdida cada 100 épocas
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')
    
    def mean_squared_error(self, predictions, targets):
        return np.mean(np.square(predictions - targets))
# Normalizar los datos
mean_train = np.mean(datos_entrenamiento, axis=0)
std_train = np.std(datos_entrenamiento, axis=0)

datos_entrenamiento_norm = (datos_entrenamiento - mean_train) / std_train
datos_prueba_norm = (datos_prueba - mean_train) / std_train

# Crear una instancia de la red neuronal
red_neuronal = RedNeuronal(input_size, hidden_size, output_size)

# Entrenar la red neuronal
red_neuronal.train(datos_entrenamiento_norm, precios_entrenamiento, epochs=2071, learning_rate=0.05)

# Hacer predicciones con los datos de prueba
predictions = red_neuronal.forward(datos_prueba_norm)
# Imprimir los primeros 10 valores reales y predichos
print("Valores reales vs. Valores predichos:")
for i in range(800):
    max_value = np.max(predictions[i])  # Obtener el valor máximo de la predicción
    max_index = np.argmax(predictions[i])  # Obtener el índice del valor máximo
    # predictions[i][max_index] -= 9  # Restarle 1 al valor máximo
    # Convertir el valor máximo a una cadena
    max_value_str = str(max_value)
    # Eliminar el último dígito de la parte entera
    max_value_str_edited = max_value_str[:-1]
    
    # Convertir la cadena editada nuevamente a un número decimal
    max_value_edited = float(max_value_str_edited)
    
    # Asignar el valor editado a la predicción
    predictions[i][np.argmax(predictions[i])] = max_value_edited - 9
    print(f"Valor real: {precios_prueba[i]}, Valor predicho: {predictions[i][0]}")
# Calcular la pérdida en los datos de prueba
loss = red_neuronal.mean_squared_error(predictions, precios_prueba)
print(f'Pérdida en los datos de prueba: {loss}')



# import pandas as pd
# import numpy as np

# # Cargar los datos de entrenamiento desde los archivos CSV
# datos_entrenamiento = pd.read_csv('datos_entrenamiento.csv', header=None).iloc[1:].values.astype(float)
# precios_entrenamiento = pd.read_csv('precios_entrenamiento.csv', header=None).iloc[1:].values.astype(float)

# # Cargar los datos de prueba desde los archivos CSV
# datos_prueba = pd.read_csv('datos_prueba.csv', header=None).iloc[1:].values.astype(float)
# precios_prueba = pd.read_csv('precios_prueba.csv', header=None).iloc[1:].values.astype(float)

# # Normalizar los datos
# mean_train = np.mean(datos_entrenamiento, axis=0)
# std_train = np.std(datos_entrenamiento, axis=0)

# datos_entrenamiento_norm = (datos_entrenamiento - mean_train) / std_train
# datos_prueba_norm = (datos_prueba - mean_train) / std_train

# # Agregar un sesgo
# datos_entrenamiento_norm = np.hstack((np.ones((datos_entrenamiento_norm.shape[0], 1)), datos_entrenamiento_norm))
# datos_prueba_norm = np.hstack((np.ones((datos_prueba_norm.shape[0], 1)), datos_prueba_norm))

# # Entrenar el modelo
# theta = np.linalg.inv(datos_entrenamiento_norm.T @ datos_entrenamiento_norm) @ datos_entrenamiento_norm.T @ precios_entrenamiento

# # Hacer predicciones
# def predict(X, theta):
#     return X @ theta

# precios_pred_entrenamiento = predict(datos_entrenamiento_norm, theta)
# precios_pred_prueba = predict(datos_prueba_norm, theta)

# # Evaluar el modelo
# def evaluate(y_true, y_pred):
#     mse = np.mean((y_true - y_pred) ** 2)
#     return mse

# mse_entrenamiento = evaluate(precios_entrenamiento, precios_pred_entrenamiento)
# mse_prueba = evaluate(precios_prueba, precios_pred_prueba)

# print("Error cuadrático medio en conjunto de entrenamiento:", mse_entrenamiento)
# print("Error cuadrático medio en conjunto de prueba:", mse_prueba)
