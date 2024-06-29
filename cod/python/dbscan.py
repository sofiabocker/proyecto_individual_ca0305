#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sofiabocker
"""

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer, load_diabetes, load_linnerud

class MetodoDBSCAN():
    
    # Constructor
    def __init__(self, df):
        '''
        Inicializa un objeto de la clase con los parámetros dados.
    
        Parameters
        ----------
        df : pd.DataFrame
            Dataset de sklearn.datasets al que se le va a aplicar el método
    
        Returns
        -------
        None
        '''
        self.__df = df
    
    # Get
    @property
    def df(self):
        '''
        Obtiene el Dataset de sklearn.datasets en el que se va a trabajar
    
        Returns
        -------
        df : sklearn.datasets
            Dataset de sklearn.datasets.
        '''
        return self.__df
    
    # Set 
    @df.setter
    def df(self, nuevo_df):
        '''
        Establece un nuevo dataframe.
    
        Parameters
        ----------
        df : sklearn.datasets
            El nuevo dataset de sklearn.datasets.
    
        Returns
        -------
        None
        '''
        self.__df = nuevo_df
    
    # Str
    def __str__(self):
        '''
        Devuelve una representación de cadena del objeto.
    
        Returns
        -------
        __str__ : str
            Una cadena que representa el objeto con los valores actuales de sus atributos.
        '''
        return f'DataFrame : {self.__df}'
    
    # Métodos
    def tabla_resumen(self):
        '''Genera una tabla resumen del dataset de sklearn.datasets con el que se va a trabajar.

        Parameters
        --------------
        None
    
        Returns
        -------------
        resumen: pd.DataFrame
                Tabla resumen del dataset de sklearn.datasets.
        '''
        X, y = self.__df.data, self.__df.target
        dataset = pd.DataFrame(X, columns = self.__df.feature_names)
        dataset['target'] = y
        resumen = dataset.describe()
        return resumen
    
    def clusters(self, eps, min_samples):
        '''Realiza el clustering con DBSCAN y devuelve el DataFrame con las etiquetas de los clusters.

        Parameters
        --------------
        eps : float
            Distancia máxima entre dos muestras para que una sea considerada como en el vecindario de la otra.
        min_samples : int
            Número de muestras (o peso total) en un vecindario para que un punto sea considerado como un punto central.
    
        Returns
        -------------
        dataset : pd.DataFrame
            DataFrame con las etiquetas de los clusters.
        '''
        # Cargar el dataset
        X = self.__df.data
        
        # Encontrar los clusters
        dbscan = DBSCAN(eps = eps, min_samples = min_samples)
        dbscan.fit_predict(X)
        
        # Convertir a un dataframe
        dataset = pd.DataFrame(X, columns = self.__df.feature_names)
        dataset['cluster'] = dbscan.labels_
        return dataset
    
    def graficar(self, eps, min_samples):
        '''Genera un pairplot del dataset con las etiquetas de los clusters de DBSCAN.

        Parameters
        --------------
        eps : float
            Distancia máxima entre dos muestras para que una sea considerada como en el vecindario de la otra.
        min_samples : int
            Número de muestras (o peso total) en un vecindario para que un punto sea considerado como un punto central.
    
        Returns
        -------------
        None
        '''
        dataset = self.clusters(eps, min_samples)
        
        # Graficar
        pairplot = sns.pairplot(dataset, hue = 'cluster', palette = 'Accent')
        pairplot.fig.suptitle(f"Pairplot con DBSCAN Clusters", y = 1.02)
        plt.show()
        
    def mejor_cluster(self):
        '''Calcula el silhouette score para diferentes cantidades de clusters

        Parameters
        --------------
        None
    
        Returns
        -------------
        None
        '''
        
        # Cargar el dataset
        X = self.__df.data
        
        eps_values = np.arange(0.3, 1.2, 0.1)
        min_samples_values = range(2, 10)
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                resultado = dbscan.fit_predict(X)
                
                # Calcular Silhouette Score solo si hay más de un cluster
                if len(set(resultado)) > 1:
                    silueta = silhouette_score(X, resultado)
                    print(f"eps: {eps}, min_samples: {min_samples}, Silhouette Score: {silueta}")
                else:
                    print(f"eps: {eps}, min_samples: {min_samples}, Silhouette Score: No clusters found")
        
        return()
        
    def silhouette(self, eps, min_samples):
        '''Calcula y devuelve el Silhouette Score del clustering con DBSCAN.

        Parameters
        --------------
        clusters : int
            Número de clusters.
    
        Returns
        -------------
        silueta: str
            Silhouette Score del clustering.
        '''
        # Cargar el dataset
        X = self.__df.data
        
        # Encontrar los clusters
        dbscan = DBSCAN(eps = eps, min_samples = min_samples)
        resultado = dbscan.fit_predict(X)
        
        silueta = silhouette_score(X, resultado)
        


        
        return(f'Silhouette Score DBSCAN: {silueta}')
        