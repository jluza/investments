# -*- coding: utf-8 -*-
import time
import pandas as pd
import numpy as np
import os
start = time.ctime()
print(start)

# =============================================================================
# Definimos la fecha de esta edicion del investments. La misma debe corresponder a la carpeta donde se encuentran los archivos descargados.
# La fecha se setea siempre para el viernes de la semana en la que se trabaja.
# =============================================================================
path = r'D:\Dropbox (MPD)\Analytics Argentina\Non-Billable projects\Investment Tracker\2019-10-23'
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path) # Selecciona la carpeta a trabajar

import funcionesInvestments as fi

# =============================================================================
# La funcion Preparador levanta los archivos descargados desde Crunchbase.com y se encarga principalmente de acomodar la informacion para el proceso, 
# eliminando duplicados para chequear que no haya transacciones repetidas en versiones anteriores del investments. La funcion devuelve dos objetos:
#     - El primero se trata de las transacciones y las companias que seran clasificadas y sobre las que se recopilara informacion.
#     - El segundo es el historico de clasificaciones de companias sobre el cual el algoritmo aprendera para predecir
#       las nuevas observaciones.
# =============================================================================

datos, database, investments, class_hist = fi.Preparador(path)  # PUEDO SAMPLEAR fi.Preparador(sample = 1)

# =============================================================================
# La funcion Clasificar utiliza un algoritmo Balanced Random Forest para categorizar los nuevos datos entrenandose con la base de datos historica 
# del investments, que contiene clasificaciones anteriores. 
# Para hacer esto, el programa aprende de las clasificaciones historicas anteriores (B2B o Fintech) a partir de las instancias de datos de las
# categorias asignadas por Crunchbase (Category) y de la descripcion provista por el sitio (Description).
# 
# La funcion nos devuelve el mismo dataframe que usamos como input, excepto que nos agrega tres columnas nuevas:
#
#     - Prediction: esta columna nos dice si la compania puede ser un area de interes para el investments (B2B, B2C, Fintech), 
#                   donde 1 = interes y 0 = no interes.
#                   
#     - Prob. of being rejected: la columna indica la probabilidad que el algoritmo asigna a esa observacion de que la compania no sea de interes,
#                                dado lo visto en Category y Description.
#                                
#     - Prob. of being of interest: la columna indica la probabilidad asignada por el algoritmo de que la compania sea B2B o Fintech, 
#                                   dado lo observado en Category y Description.
#          
# --------------- Prob. of being rejected + Prob. of being of interest = 1 -----------------------------
#
# =============================================================================

clasificado = fi.Clasificar(database, datos, path)
clasificado['Investee'].fillna(clasificado['Acquiree Name'], inplace=True)
class_hist['Investee'].fillna(class_hist['Acquiree Name'], inplace=True)

test = clasificado.merge(class_hist, how = 'left', on = 'Investee')[['Investee','Category.1','Area of Focus']].replace({0:np.nan}).drop_duplicates(keep='first')


clasificado = clasificado.merge(test, how = 'left', on = 'Investee')
clasificado.drop_duplicates(inplace = True)
clasificado.to_excel('clasificado.xlsx', index = None)
confirmation = input('PROGRAMA EN PAUSA: CLASIFICAR LAS EMPRESAS, GUARDAR Y LUEGO TOCAR ENTER, SIN MOVER EL ARCHIVO CLASIFICADO FUERA DE LA CARPETA \n')
clasificado = pd.read_excel('clasificado.xlsx')

# =============================================================================
# """ Arreglo el tema de las columnas corridas para las acquisitions """
# rows_to_shift = clasificado.loc[clasificado['Money Raised'].str.contains(r'http') == True].index
# clasificado.iloc[rows_to_shift,6:12] = clasificado.iloc[rows_to_shift,6:12].shift(axis = 1)
# 
# =============================================================================

class_history = pd.concat([class_hist, clasificado])
class_history.to_excel('class_history.xlsx', index = None)


ascrapear = clasificado.loc[clasificado['Category.1'] != 'Rejected'].drop(['Prediction', 'Prob. of being rejected','Prob. of being of interest'] , axis = 1)
ascrapear = ascrapear.drop_duplicates().reset_index(drop=True)



htmls_deals = fi.Scrapeardeals(ascrapear)
deals, orgs = fi.ParseandoHTMLdeals(htmls_deals)
htmls_orgs = fi.ScrapearORGS(orgs)
data_companias = fi.ParsearHTMLOrganizaciones(htmls_orgs)
master, chequeo, companias = fi.Estructurador(deals, data_companias, ascrapear)


chequeo.to_excel('Chequeo.xlsx')

master.to_excel('aMasterData.xlsx')
#master = pd.read_excel('aMasterData.xlsx', index_col = 0)
chinainvestments = master.loc[(master['Investor Region'] == 'China') | (master['Investor Region'] == 'Hong Kong')]
chinainvestments.to_excel('aChinaInvestments.xlsx')

indiainvestments = master.loc[master['Investor Region'] == 'India']
indiainvestments.to_excel('IndiaInvestments.xlsx')

end = time.ctime()
print(end)