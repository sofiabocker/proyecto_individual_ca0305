#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sofiabocker
"""

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import seaborn as sns
import pandas as pd

from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer, load_diabetes, load_linnerud

class MetodoGaussian():
    
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
    
    def clusters(self, components):
        '''Realiza el clustering con GaussianMixture y devuelve el DataFrame con las etiquetas de los clusters.

        Parameters
        --------------
        components : int
            Número de componentes de la mezcla.
        random : int
            Estado aleatorio para la inicialización de GaussianMixture.
    
        Returns
        -------------
        dataset : pd.DataFrame
            DataFrame con las etiquetas de los clusters.
        '''
        # Cargar el dataset
        X = self.__df.data
        
        # Encontrar los clusters
        gm = GaussianMixture(n_components = components)
        gm.fit(X)
        labels = gm.predict(X)
        
        # Convertir a un dataframe
        dataset = pd.DataFrame(X, columns = self.__df.feature_names)
        dataset['cluster'] = labels
        return dataset
    
    def graficar(self, components):
        '''Genera un pairplot del dataset con las etiquetas de los clusters de GaussianMixture.

        Parameters
        --------------
        components : int
            Número de componentes de la mezcla.
        random : int
            Estado aleatorio para la inicialización de GaussianMixture.
    
        Returns
        -------------
        None
        '''
        dataset = self.clusters(components)
        
        # Graficar
        pairplot = sns.pairplot(dataset, hue = 'cluster', palette = 'Accent')
        pairplot.fig.suptitle(f"Pairplot con GaussianMixture Clusters", y=1.02)
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
        
        for n_components in range(2, 11):
            gm = GaussianMixture(n_components = n_components)
            resultado = gm.fit_predict(X)
            silueta = silhouette_score(X, resultado)
            print(f"Cantidad de clusters: {n_components}, Silhouette Score: {silueta}")
            
        return()
        
    def silhouette(self, components):
        '''Calcula y devuelve el Silhouette Score del clustering con GaussianMixture.

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
        gm = GaussianMixture(n_components = components)
        gm.fit(X)
        resultado = gm.predict(X)
        
        silueta = silhouette_score(X, resultado)
        
        return(f'Silhouette Score GaussianMixture: {silueta}')
    
