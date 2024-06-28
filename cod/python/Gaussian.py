#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 16:54:24 2024

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
    
    def clusters(self, components, random):
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
        X = self.__df.data
        gm = GaussianMixture(n_components = components, random_state = random)
        gm.fit(X)
        labels = gm.predict(X)
        dataset = pd.DataFrame(X, columns = self.__df.feature_names)
        dataset['cluster'] = labels
        return dataset
    
    def graficar(self, components, random):
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
        # Cargar el dataset
        X, y = self.__df.data, self.__df.target
        
        # Encontrar los clusters
        gm = GaussianMixture(n_components = components, random_state = random)
        gm.fit(X)
        pred = gm.predict(X)
        
        # Convertir a un dataframe
        df = pd.DataFrame(X, columns = self.__df.feature_names)
        df['cluster'] = pred
        
        # Graficar
        pairplot = sns.pairplot(df, hue = 'cluster', palette = 'Accent')
        pairplot.fig.suptitle(f"Pairplot con GaussianMixture Clusters", y=1.02)
        plt.show()
        
    def silhouette(self, components, random):
        '''

        Parameters
        --------------
        
    
        Returns
        -------------
        
        '''
        # Cargar el dataset
        X, y = self.__df.data, self.__df.target
        
        # Encontrar los clusters
        gm = GaussianMixture(n_components = components, random_state = random)
        gm.fit(X)
        resultado = gm.predict(X)
        
        silueta = silhouette_score(X, resultado)
        
        return(f'Silhouette Score GaussianMixture: {silueta}')
    
    
#iris_data = load_iris()
#metodo = MetodoGaussian(iris_data)
#print(metodo.tabla_resumen())

#clustered_data_gm = metodo.clusters(3, 42)
#print(clustered_data_gm.head())
#metodo.graficar(3, 42)

#silueta_gm = metodo.silhouette(3, 42)
#print(silueta_gm)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

