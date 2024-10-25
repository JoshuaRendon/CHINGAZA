# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:25:40 2024

@author: Joshua Rendon Algecira
"""
# Importar las librerias
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pyhomogeneity as hg
# https://github.com/mmhs013/pyHomogeneity/blob/master/Examples/Example_pyHomogeneity.ipynb
#Número de Dios
ND = (1+5**0.5)/2

# Ruta del archivo combinado
ruta_combi = r'D:\Semillero\2024\Precipitacion\Resultados\Datos\Combinado_diario2.csv'
# Ruta para las curvas de doble masa
ruta_curvas = r'D:\Semillero\2024\Precipitacion\Resultados\Graficas\Doble_masa'
#Ruta para el df mensual
ruta_mensual = r'D:\Semillero\2024\Precipitacion\Resultados\Datos'
# Ruta para guardar el petit
ruta_petit = r'D:\Semillero\2024\Precipitacion\Resultados\Datos\petit_anual.csv'
# Ruta para guardar el snht
ruta_snht = r'D:\Semillero\2024\Precipitacion\Resultados\Datos\snht_anual.csv'
# Ruta para guardar el buishand
ruta_buishand = r'D:\Semillero\2024\Precipitacion\Resultados\Datos\buishand_anual.csv'
# Ruta para guardar los h de los test
ruta_test = r'D:\Semillero\2024\Precipitacion\Resultados\Datos\Test_Homo_anual.csv'
# Leer el csv del combinado
df = pd.read_csv(ruta_combi, encoding='latin1')
# Convertir la columna 'Fecha' a formato 
df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True)#, errors='coerce')
# Asegurarse de que 'Fecha' sea el índice para poder usar resample
df.set_index('Fecha', inplace=True)
# Contar los NaN por mes
nan_mensual = df.resample('MS').apply(lambda x: x.isna().sum())
# Sumar los valores por mes
df_mensual = df.resample('MS').sum()
# Reemplazar la suma por NaN en los meses que tienen más de 6 NaN en cualquier columna
df_mensual[nan_mensual > 6] = np.nan

# Gráficas de Doble masa
# Nuevo DataFrame para almacenar los resultados acumulados
df_acumulado = pd.DataFrame(index=df_mensual.index) 
# Hallar el acumulado de cada estación
for column in df_mensual.columns:
    df2 = df_mensual
    df2 = df2.dropna(subset = column)
    mean_filas = df2.mean(axis=1).cumsum()
    mean_filas.name = column # Calcular precipitación promedio acumulada
    df_acumulado[column] = mean_filas  # Añadir la serie acumulada al nuevo DataFrame



# Poner la columna mes en el df
df_mensual['Mes'] = df_mensual.index.month
# Crear los data frames
petit = []
snht = []
buishand = []
# Hacer el ciclo for en el df_mensual
for estacion in df_mensual.columns[:-1]: 
    datos = df_mensual[[estacion,'Mes']]
    datos['Acumulado'] = datos[estacion].cumsum()
    #datos = datos.dropna(subset=estacion)
    datos['Referencia'] = df_acumulado[estacion]
    datos_ano = datos.resample('YE').sum()
    # Hacer el Petit Test
    petit_res = hg.pettitt_test(datos_ano[estacion], alpha=0.01, sim = 20000)
    petit.append([estacion,petit_res.h,petit_res.cp,petit_res.p,petit_res.U,petit_res.avg.mu1,petit_res.avg.mu2])
    # Test de snht
    snht_res = hg.snht_test(datos_ano[estacion], alpha = 0.01, sim = 20000)
    snht.append([estacion,snht_res.h,snht_res.cp,snht_res.p,snht_res.T,snht_res.avg.mu1,snht_res.avg.mu2])
    # Test de Buishand
    buishand_res = hg.buishand_u_test(datos_ano[estacion],alpha = 0.01, sim = 20000)
    buishand.append([estacion,buishand_res.h,buishand_res.cp,buishand_res.p,buishand_res.U,buishand_res.avg.mu1,buishand_res.avg.mu2])
    # Crear la figura
    #plt.figure(figsize=(10,10/ND))
    #sns.regplot(x=datos['Referencia'], y=datos['Acumulado'], marker="x", color=".65", line_kws=dict(color="cornflowerblue"))
    #plt.title(f'Double mass analysis {estacion}')
    #plt.ylabel('Acumulated (mm)')
    #plt.xlabel('reference cumulative (mm)')
    #plt.grid(True)
    #plt.show()
    # Mostrar la gráfica
    #ruta_doble = os.path.join(ruta_curvas, f'CDM_{estacion}.png')
    #plt.savefig(ruta_doble, dpi = 350)
    #plt.close()
# Convertir la lista de los petit en un DataFrame
petit_df = pd.DataFrame(petit, columns=['estacion','h', 'cp', 'p', 'U', 'mu1','mu2']) 
petit_df.to_csv(ruta_petit,index=False)
# Convertir la lista de los snht en un DataFrame
snht_df = pd.DataFrame(snht, columns=['estacion','h', 'cp', 'p', 'T', 'mu1','mu2']) 
snht_df.to_csv(ruta_snht,index=False)
# Convertir la lista de los buishand en un DataFrame
buishand_df = pd.DataFrame(snht, columns=['estacion','h', 'cp', 'p', 'U', 'mu1','mu2']) 
buishand_df.to_csv(ruta_buishand,index=False)
# Df combinado con los True o False
df_test = []
df_test = pd.DataFrame({
    'estacion': petit_df['estacion'],
    'petit_h': petit_df['h'],
    'snht_h': snht_df['h'],
    'buishand_h': buishand_df['h']
})
df_test.to_csv(ruta_test,index=False)
