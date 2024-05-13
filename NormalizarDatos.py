import csv
import numpy as np
import pandas as pd
datos_entrenamiento = []
precios_entrenamiento = []
datos_prueba = []
precios_prueba = []
# Leer el archivo CSV y seleccionar las columnas relevantes, incluido el precio
def leer_datos(nombre_archivo):
    datos = []
    precios = []
    with open(nombre_archivo, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Reemplazar valores faltantes con 0
            for key in row:
                if row[key] == '':
                    row[key] = '0'
            # Seleccionar las columnas relevantes y convertirlas a flotantes
            dato = [float(row['Codigo Postal']), float(row['Recamaras']), float(row['Banos']), 
                    float(row['Estacionamientos']), float(row['NoPisos']), float(row['Edad']), 
                    float(row['Area Terreno']), float(row['Area Construida'])]
            precio = float(row['Precio'])
            datos.append(dato)
            precios.append(precio)
    return datos, precios

# Normalizar los datos para que estén en la misma escala
def normalizar_datos(datos, precios):

    datos = np.array(datos)

    # Dividir los datos en entrenamiento y pruebas
    entrenamiento = int(0.7 * len(datos))
    datos_entrenamiento = datos[:entrenamiento]
    datos_prueba = datos[entrenamiento:]

    # Normalizar los datos
    datos_entrenamiento = (datos_entrenamiento - datos_entrenamiento.mean(axis=0)) / datos_entrenamiento.std(axis=0)
    datos_prueba = (datos_prueba - datos_prueba.mean(axis=0)) / datos_prueba.std(axis=0)

    return datos_entrenamiento, datos_prueba, precios[:entrenamiento], precios[entrenamiento:]

# Obtener los datos y normalizarlos

datos, precios = leer_datos('caracteristicas_inmuebles.csv')

datos_entrenamiento, datos_prueba, precios_entrenamiento, precios_prueba = normalizar_datos(datos, precios)

# Convertir los datos a DataFrames
df_entrenamiento = pd.DataFrame(datos_entrenamiento)
df_prueba = pd.DataFrame(datos_prueba)

# Guardar los datos normalizados en archivos CSV
df_entrenamiento.to_csv('datos_entrenamiento.csv', index=False)
df_prueba.to_csv('datos_prueba.csv', index=False)

# Guardar los precios en archivos CSV
pd.DataFrame(precios_entrenamiento).to_csv('precios_entrenamiento.csv', index=False)
pd.DataFrame(precios_prueba).to_csv('precios_prueba.csv', index=False)



# import csv
# import numpy as np
# import pandas as pd

# datos_entrenamiento = []
# precios_entrenamiento = []
# datos_prueba = []
# precios_prueba = []

# # Leer el archivo CSV y seleccionar las columnas relevantes, incluido el precio
# def leer_datos(nombre_archivo):
#     datos = []
#     precios = []
#     with open(nombre_archivo, 'r', encoding='utf-8') as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             # Reemplazar valores faltantes con 0
#             for key in row:
#                 if row[key] == '':
#                     row[key] = '0'
#             # Seleccionar las columnas relevantes y convertirlas a flotantes
#             dato = [float(row['Codigo Postal']), float(row['Recamaras']), float(row['Banos']), 
#                     float(row['Estacionamientos']), float(row['NoPisos']), float(row['Edad']), 
#                     float(row['Area Terreno']), float(row['Area Construida'])]
#             precio = float(row['Precio'])
#             datos.append(dato)
#             precios.append(precio)
#     return datos, precios

# # Obtener los datos sin normalizar
# datos, precios = leer_datos('caracteristicas_inmuebles.csv')

# # Dividir los datos en entrenamiento y prueba
# entrenamiento = int(0.7 * len(datos))
# datos_entrenamiento = datos[:entrenamiento]
# datos_prueba = datos[entrenamiento:]
# precios_entrenamiento = precios[:entrenamiento]
# precios_prueba = precios[entrenamiento:]

# # Convertir los datos a DataFrames
# df_entrenamiento = pd.DataFrame(datos_entrenamiento)
# df_prueba = pd.DataFrame(datos_prueba)

# # Guardar los datos en archivos CSV
# df_entrenamiento.to_csv('datos_entrenamiento.csv', index=False)
# df_prueba.to_csv('datos_prueba.csv', index=False)

# # Guardar los precios en archivos CSV
# pd.DataFrame(precios_entrenamiento).to_csv('precios_entrenamiento.csv', index=False)
# pd.DataFrame(precios_prueba).to_csv('precios_prueba.csv', index=False)
