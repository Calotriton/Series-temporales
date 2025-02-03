# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 19:59:17 2025

@author: User
"""

# PREPARACIÓN DE LOS DATOS #
# Cargamos las librerías a utilizar
import numpy as np
import pandas as pd
import os
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from tabulate import tabulate
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm


# Establecemos el directorio de trabajo
os.chdir(r'C:\Users\User\Documents\Master Data Science\Mineria de datos y modelizacion predictiva\Series temporales\Tarea')

# Cargamos el archivo CSV
df = pd.read_csv("temp_anomaly.csv", delimiter=';')

# Damos formato de fecha
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
print(df.head())

# Indexamos la serie de datos de anomalía térmica 
# Cambiamos las comas por puntos y convertimos las columnas a numéricas
df['AnomalyOcean'] = df['AnomalyOcean'].str.replace(',', '.').astype(float)
df['AnomalyLand'] = df['AnomalyLand'].str.replace(',', '.').astype(float)

# Indexamos 'Year' y seleccionamos ambas anomalías
df.set_index('Year', inplace=True)

# Ploteamos ambas series
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['AnomalyOcean'], label='Anomalía térmica oceánica')
plt.plot(df.index, df['AnomalyLand'], label='Anomalía térmica terrestre')

# Añadimos etiquetas y título
plt.xlabel('Año')
plt.ylabel('Anomalía térmica global')
plt.title('Anomalías térmicas oceánicas y terrestres (1850-2023)')
plt.legend()
plt.grid(True)
plt.show()

#Realizamos la descomposición estacional OCEÁNICA por el método aditivo
# Descomposición estacional (modelo aditivo) con un periodo anual (12 meses)
decomposition_ocean = seasonal_decompose(df['AnomalyOcean'], model='additive', period=12)
print(decomposition_ocean.seasonal[:12])

# Graficar los componentes
plt.rc("figure", figsize=(12, 8))
decomposition_ocean.plot()
plt.show()

#Realizamos la descomposición estacional TERRESTRE por el método aditivo
# Descomposición estacional (modelo aditivo) con un periodo anual (12 meses)
decomposition_land = seasonal_decompose(df['AnomalyLand'], model='additive', period=12)
print(decomposition_land.seasonal[:12])
# Graficar los componentes
plt.rc("figure", figsize=(12, 8))
decomposition_land.plot()
plt.show()


# Cálculo de la serie ajustada estacionalmente (restando el componente estacional)
S_Ajustada_Est = df['AnomalyOcean'] - decomposition_ocean.seasonal

# Gráfico de la serie original, la tendencia y la serie ajustada estacionalmente
plt.figure(figsize=(12, 8))

plt.plot(df['AnomalyOcean'], label='Datos', color='gray')
plt.plot(decomposition_ocean.trend, label='Tendencia', color='blue')
plt.plot(S_Ajustada_Est, label='Estacionalmente ajustada', color='red')

plt.xlabel('Fecha')
plt.ylabel('Anomalía Oceánica')
plt.title('Anomalía Oceánica: Serie Original, Tendencia y Ajustada Estacionalmente')
plt.legend()
plt.show()


# División de la serie en train y test
train = df['AnomalyOcean'][:-10]  # Todos los datos excepto las últimas 10 observaciones
test = df['AnomalyOcean'][-10:]   # Las últimas 10 observaciones

# Graficar la serie train y test
plt.figure(figsize=(12, 8))
plt.plot(train, label='Train', color='gray')
plt.plot(test, label='Test', color='green')

plt.xlabel('Fecha')
plt.ylabel('Anomalía térmica oceánica')
plt.title('División de la serie en train y test')
plt.legend()
plt.show()



# Selección de método de alisado
# 1. Método de alisado doble de Holt 
model_holt = Holt(train, initialization_method="estimated").fit()
fcast_holt = model_holt.forecast(10)  

# Gráfico 
plt.figure(figsize=(12, 8))
plt.plot(train, label='Train', color='gray')
plt.plot(test, label='Test', color='green')
plt.plot(model_holt.fittedvalues, label='Suavizado (Holt)', color='blue')
plt.plot(fcast_holt, label='Predicciones (Holt)', color='red')
plt.xlabel('Fecha')
plt.ylabel('Anomalía térmica oceánica')
plt.title('Método doble de Holt')
plt.legend()
plt.show()

# Parámetros 
print("Parámetros del método doble de Holt:")
print(model_holt.params_formatted)


# 2. Método de tendencia amortiguada
model_damped = Holt(train, damped_trend=True, initialization_method="estimated").fit()
fcast_damped = model_damped.forecast(10)  

# Gráfico
plt.figure(figsize=(12, 8))
plt.plot(train, label='Train', color='gray')
plt.plot(test, label='Test', color='green')
plt.plot(model_damped.fittedvalues, label='Suavizado (Holt Damped)', color='blue')
plt.plot(fcast_damped, label='Predicciones (Holt Damped)', color='red')
plt.xlabel('Fecha')
plt.ylabel('Anomalía térmica oceánica')
plt.title('Método de tendencia amortiguada')
plt.legend()
plt.show()

# Parámetros 
print("Parámetros del método de tendencia amortiguada:")
print(model_damped.params_formatted)

# Comparación de Predicciones
# Tabla con predicciones de cada modelo
predicciones = {
    'Fecha': test.index,
    'Real': test.values,
    'Holt': fcast_holt.values,
    'Holt Damped': fcast_damped.values
}

print("Tabla de Predicciones:")
print(tabulate(predicciones, headers="keys", tablefmt="fancy_grid"))



# Comparación de error cuadrático medio (RMSE) de los modelos
# RMSE para Holt
rmse_holt = np.sqrt(mean_squared_error(test, fcast_holt))
print("RMSE (Holt):", rmse_holt)

# RMSE para Holt Damped
rmse_damped = np.sqrt(mean_squared_error(test, fcast_damped))
print("RMSE (Holt Damped):", rmse_damped)


# Representación de los correlogramas
# Graficar ACF y PACF
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plot_acf(train, lags=30, ax=plt.gca())
plt.title('Función de autocorrelación (ACF)')

plt.subplot(2, 1, 2)
plot_pacf(train, lags=30, ax=plt.gca())
plt.title('Función de autocorrelación parcial (PACF)')
plt.show()

# Aplicación del modelo ARIMA (1, 1, 0)
# Diferenciamos la serie
diferencias = train.diff().dropna()

# Ajustamos el modelo ARIMA 
model_arima = ARIMA(train, order=(1, 1, 0))  
results_arima = model_arima.fit()
print(results_arima.summary())

# Verificamos los residuos
residuals = results_arima.resid

# Graficamos ACF y PACF de los residuos
plot_acf(residuals, lags=30)
plot_pacf(residuals, lags=30)
plt.show()

# Mostramos las gráficas de los residuos del modelo
results_arima.plot_diagnostics(figsize=(12, 8))
plt.show()

# Ajustamos el modelo ARIMA automático (sin estacionalidad)
auto_model = pm.auto_arima(
    train,
    seasonal=False,            # No hay componente estacional
    trace=True,                # Muestra los modelos probados
    error_action='ignore',     # Ignora errores en ajustes
    suppress_warnings=True,    # Omite advertencias
    stepwise=True              # Método paso a paso para eficiencia
)
print(auto_model.summary())

auto_model.plot_diagnostics(figsize=(12, 8))
plt.show()


# Calculamos la anomalía térmica de los próximos 37 años y su intervalo de confianza
train = train.asfreq('YS')
best_arima = ARIMA(train, order=(1, 1, 3))
resultados_a = best_arima.fit()
prediciones_a = resultados_a.get_forecast(steps=37)
predi_test_a = prediciones_a.predicted_mean
intervalos_confianza_a = prediciones_a.conf_int()

# Mostramos las predicciones y los intervalos de confianza
print("Predicciones:")
print(predi_test_a)
print("\nIntervalos de Confianza:")
print(intervalos_confianza_a)

# Graficamos las predicciones junto a la serie original
plt.figure(figsize=(12, 8))
plt.plot(train, label='Datos observados (Train)', color='gray')
plt.plot(test, label='Datos reales (Test)', color='green')
plt.plot(predi_test_a, label='Predicciones', color='red')
plt.fill_between(intervalos_confianza_a.index, intervalos_confianza_a.iloc[:, 0], intervalos_confianza_a.iloc[:, 1], color='pink', alpha=0.3, label='Intervalo de confianza (95%)')
plt.xlabel('Año')
plt.ylabel('Anomalía térmica oceánica')
plt.title('Predicciones del modelo ARIMA(1, 1, 3)')
plt.legend()
plt.show()



