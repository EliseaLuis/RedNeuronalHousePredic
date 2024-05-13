
# import requests
# from bs4 import BeautifulSoup as bs
# import random 
# import time
# import pandas as pd
# import numpy as np
# from selenium import webdriver
# from selenium.webdriver.common.by import By 
# from selenium.webdriver.common.keys import Keys
# import undetected_chromedriver as uc 
# import re

# browser =uc.Chrome()
# # Leer el archivo CSV con las URLs de los inmuebles
# df_urls = pd.read_csv('urls_inmuebles.csv')
# # Listas para almacenar las características de los inmuebles
# titulos = []
# codigos_postales = []
# precios = []
# recamaras = []
# banos = []
# estacionamientos = []
# NoPisos = []
# Edad = []
# areaTerreno = []
# areaConstruida = []
# tamañoJardin = []

# for url in df_urls['URLs']:
#     browser.get(url)
#     browser.implicitly_wait(10)
#     html = browser.page_source 
#     soup = bs(html, 'lxml')
#     titulo = soup.find('h1', {'class': 'sc-7dd64cc4-1 jTpNGO'}).text.split(',')[1]
#     titulos.append(titulo)
#     codigoPostal = soup.find('span', {'itemprop': 'postalCode'}).text
#     codigos_postales.append(codigoPostal)
#     precio = int(soup.find('div', {'class': 'sc-82b87f44-1 koodIS price-text'}).text.replace('$', '').replace('MXN', '').replace(',', ''))
#     precios.append(precio)
#     descriptions = soup.find_all(class_='description-text')
#     descriptionsData = soup.find_all(class_='description-number')
#     descriptions.pop(0)
#     values = []
#     for data in descriptionsData:
#         data_text = data.text.strip()
#         numbers = re.findall(r'\d+', data_text)
#         if numbers:
#             value = int(numbers[0])
#         else:
#             value = 0
#         values.append(value)
#     recamaras.append(values[1] if len(values) > 1 else None)
#     banos.append(values[2] if len(values) > 2 else None)
#     estacionamientos.append(values[3] if len(values) > 3 else None)
#     NoPisos.append(values[4] if len(values) > 4 else None)
#     Edad.append(values[5] if len(values) > 5 else None)
#     areaTerreno.append(values[6] if len(values) > 6 else None)
#     areaConstruida.append(values[7] if len(values) > 7 else None)
#     tamañoJardin.append(values[8] if len(values) > 8 else None)
#     time.sleep(4)  # Agrega un retraso para evitar sobrecargar el servidor

# browser.quit()

# # Crear un DataFrame con las características de los inmuebles
# df_caracteristicas = pd.DataFrame({
#     'Titulo': titulos,
#     'Codigo Postal': codigos_postales,
#     'Precio': precios,
#     'Recamaras': recamaras,
#     'Baños': banos,
#     'Estacionamientos': estacionamientos,
#     'NoPisos': NoPisos,
#     'Edad': Edad,
#     'Area Terreno': areaTerreno,
#     'Area Construida': areaConstruida,
#     'Tamaño Jardin': tamañoJardin
# })

# # Guardar el DataFrame en un archivo CSV
# df_caracteristicas.to_csv('caracteristicas_inmuebles.csv', index=False)

# print("Archivo CSV creado exitosamente.")





import requests
from bs4 import BeautifulSoup as bs
import random 
import time
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By 
from selenium.webdriver.common.keys import Keys
import undetected_chromedriver as uc 
import re

browser = uc.Chrome()
# Leer el archivo CSV con las URLs de los inmuebles
df_urls = pd.read_csv('urls_inmuebles.csv')
# Listas para almacenar las características de los inmuebles
titulos = []
codigos_postales = []
precios = []
recamaras = []
banos = []
estacionamientos = []
NoPisos = []
Edad = []
areaTerreno = []
areaConstruida = []
tamañoJardin = []

for url in df_urls['URLs']:
    browser.get(url)
    browser.implicitly_wait(10)
    html = browser.page_source 
    soup = bs(html, 'lxml')
    
    try:
        titulo = soup.find('h1', {'class': 'sc-7dd64cc4-1 jTpNGO'}).text.split(',')[1]
    except AttributeError:
        titulo = "Titulo no disponible"
    titulos.append(titulo)
    
    try:
        codigoPostal = soup.find('span', {'itemprop': 'postalCode'}).text
    except AttributeError:
        codigoPostal = "CP no disponible"
    codigos_postales.append(codigoPostal)
    
    try:
        precio = int(soup.find('div', {'class': 'sc-82b87f44-1 koodIS price-text'}).text.replace('$', '').replace('MXN', '').replace(',', ''))
    except (AttributeError, ValueError):
        precio = np.nan  # Otra forma de indicar que el valor no está disponible
    precios.append(precio)
    
    descriptions = soup.find_all(class_='description-text')
    if descriptions:  # Verificar si la lista no está vacía
        descriptions.pop(0)
    descriptionsData = soup.find_all(class_='description-number')
    values = []
    for data in descriptionsData:
        data_text = data.text.strip()
        numbers = re.findall(r'\d+', data_text)
        if numbers:
            value = int(numbers[0])
        else:
            value = 0
        values.append(value)
    
    recamaras.append(values[1] if len(values) > 1 else None)
    banos.append(values[2] if len(values) > 2 else None)
    estacionamientos.append(values[3] if len(values) > 3 else None)
    NoPisos.append(values[4] if len(values) > 4 else None)
    Edad.append(values[5] if len(values) > 5 else None)
    areaTerreno.append(values[6] if len(values) > 6 else None)
    areaConstruida.append(values[7] if len(values) > 7 else None)
    tamañoJardin.append(values[8] if len(values) > 8 else None)
    time.sleep(4)  # Agrega un retraso para evitar sobrecargar el servidor

browser.quit()

# Crear un DataFrame con las características de los inmuebles
df_caracteristicas = pd.DataFrame({
    'Titulo': titulos,
    'Codigo Postal': codigos_postales,
    'Precio': precios,
    'Recamaras': recamaras,
    'Baños': banos,
    'Estacionamientos': estacionamientos,
    'NoPisos': NoPisos,
    'Edad': Edad,
    'Area Terreno': areaTerreno,
    'Area Construida': areaConstruida,
    'Tamaño Jardin': tamañoJardin
})

# Guardar el DataFrame en un archivo CSV
df_caracteristicas.to_csv('caracteristicas_inmuebles.csv', index=False)

print("Archivo CSV creado exitosamente.")

