import pandas as pd
import numpy as np
import sys
from scipy.stats import zscore

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.exceptions import NotFittedError

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

import joblib

import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.path_file = self.config.get('data_path', None)
        self.sep = self.config.get('sep', ',')
        self.delete_columns = self.config.get('delete_columns')
        self.split = self.config.get('split')
        self.k = self.config.get('k_features')

        self.label_encoder = LabelEncoder()
        self.numeric_imputer = SimpleImputer(strategy = self.config.get("numeric_imputer"))
        self.categorical_imputer = SimpleImputer(strategy = self.config.get("categorical_imputer"))

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.feature_selector = SelectKBest(score_func = f_regression)
        self.transformers = {}
        self.missing_threshold = self.config.get('missing_threshold', 0.1)
        self.target = self.config.get('target_column', None)
        self.model_type = self.config.get('model_type', None)
        self.threshold_outlier = self.config.get('threshold_outlier', 3)
        self.lower_percentile = self.config.get('lower_percentile', 5)
        self.upper_percentile = self.config.get('upper_percentile', 95)
        self.seed = 11 # semilla para la separacion de los datos

        self.balance_threshold = self.config.get('balance_thershold',0.5)
        self.balance_method = self.config.get('balance_method', None)

        if self.balance_method == 'over_sampling':
            self.sampler = SMOTE()
        elif self.balance_method == 'under_sampling':
            self.sampler = RandomUnderSampler()
        else:
            self.sampler = None

        self.unprocessed_columns = [] # columnas no procesadas por no cumplir criterios de nulos

    # Función para cargar los datos y hacer depuración. 
    def load_dataset(self):
        print("---------------------------------------------------")
        sys.stdout.flush()
        print("--------------- Carga de datos -------------------")
        sys.stdout.flush()
        print("---------------------------------------------------")
        sys.stdout.flush()
        try:
            self.df= pd.read_csv(self.path_file, sep = self.sep)
            self.df.columns = self.df.columns.str.strip()
            self.df.columns = self.df.columns.str.replace(' ', '_')

            print("Cantidad de registros cargados: ", self.df.shape[0])
            sys.stdout.flush()
            print("Cantidad de columnas cargadas: ", self.df.shape[1])
            sys.stdout.flush()
            print("Eliminar conlumnas indicadas por el usuario: ", self.delete_columns)
            sys.stdout.flush()
            self.df.drop(columns = self.delete_columns, inplace = True)
            
            print("Eliminar conlumnas con valores unicos, todos los valores diferentes y duplicados: ")
            sys.stdout.flush()
            # Columnas con un único valor.
            unique_columns = self.df.columns[self.df.nunique() == 1]
            self.df.drop(columns = unique_columns, inplace = True)

            # Columnas con todos los valores diferentes.
            unique_columns = self.df.columns[self.df.nunique() == len(self.df)]
            self.df.drop(columns = unique_columns, inplace = True)

            # Valores duplicados
            self.df.drop_duplicates(inplace = True)
            self.df.reset_index(drop = True, inplace = True) 

            print("Cantidad de datos nuevos ", self.df.shape)
            sys.stdout.flush()    

            # Selección de la variable objetivo
            target_column = self.target
            if target_column is None:
                raise ValueError("Variable objetivo no identificada en los parametros")
            
            # Asignación de 'y' varialbe objetivo y 'X' variables predictora 
            self.y = self.df[target_column]
            self.X = self.df.drop(columns=[target_column])

            # Identificar variables numéricas y categóricas
            self.numeric_columns = self.X.select_dtypes(include=['number','float64','float64','int32','int64']).columns
            self.categorical_columns = self.X.select_dtypes(include=['object', 'category']).columns

            return self.X, self.y

        except Exception as e:
            print("Error cargando el dataset:", e)

    # Función para realizar un analisis descriptivo y una visualización de los datos
    def descriptive_analysis(self):
        print(" Información de los datos: ")
        sys.stdout.flush() 
        print(self.df.info())
        sys.stdout.flush() 

        # Análisis descriptivo de variables numéricas
        if self.numeric_columns is not None:
            numeric_data = self.df[self.numeric_columns]
            if not numeric_data.empty:
                print("Análisis descriptivo de variables numéricas: ")
                sys.stdout.flush() 
                print(numeric_data.describe())
                sys.stdout.flush() 

                # Visualización de variables numéricas
                print("Visualización de variables numéricas: ")
                sys.stdout.flush() 
                numeric_data.hist(figsize=(12, 8))
                plt.tight_layout()
                plt.show()
        
        # Análisis descriptivo de variables categóricas
        if self.categorical_columns is not None:
            categorical_data = self.df[self.categorical_columns]
            if not categorical_data.empty: 
                print("Análisis descriptivo de variables categóricas: ")
                sys.stdout.flush() 
                print(categorical_data.describe())
                sys.stdout.flush() 

                # Visualización de variables categóricas
                print("Visualización de variables categóricas:")
                sys.stdout.flush() 
                num_plots = len(self.categorical_columns)
                num_groups = (num_plots + 5) // 6  # Calcular el número de grupos de 6
                for group in range(num_groups):
                    start_index = group * 6
                    end_index = min(start_index + 6, num_plots)
                    num_variables = end_index - start_index
                    num_rows = (num_variables + 1) // 3
                    fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows*5), squeeze=False)
                    for i in range(start_index, end_index):
                        row_index = (i - start_index) // 3
                        col_index = (i - start_index) % 3
                        col = self.categorical_columns[i]
                        sns.countplot(data=categorical_data, x=col, hue=col, palette='viridis', ax=axes[row_index, col_index], legend=False)
                        axes[row_index, col_index].set_title(f"Distribución de {col}")
                        axes[row_index, col_index].tick_params(axis='x', rotation=45)
                    plt.tight_layout()
                    plt.show()
                    sys.stdout.flush()

    # Función para separar un % de los datos para realizar predicciones después de crear el modelo
    def split_data_for_predictions(self, save_path):

        print("Separación de datos para predecir: ")
        sys.stdout.flush()  
        # Seleccionar datos aleatorios
        np.random.seed(self.seed)
        num_rows_to_predict = int(len(self.df) * self.split)
        random_indices = np.random.choice(self.df.index, num_rows_to_predict, replace=False)
        prediction_data = self.df.loc[random_indices]#??

        # Eliminar los datos seleccionados del DataFrame original
        self.df.drop(random_indices, inplace=True)
        self.df.reset_index(drop=True, inplace=True)#??

        # Guardar los datos para predicciones en un nuevo archivo CSV
        try:
            prediction_data.to_csv(save_path, index=False)
            print(f"Datos para predicciones guardados en '{save_path}'")
            sys.stdout.flush() 
        except Exception as e:
            print("Error al guardar los datos para predicciones:", e)
            sys.stdout.flush() 

    # Función para remover los datos atipicos a través del z_score
    def remove_outliers_zscore(self):
        print("Eliminar valores atipicos")
        sys.stdout.flush()
        # Calcular z-scores para las columnas numéricas
        z_scores = zscore(self.X[self.numeric_columns])

        # Identificar filas con valores atípicos
        outlier_rows = (np.abs(z_scores) > self.threshold_outlier).any(axis=1)

        # Eliminar filas con valores atípicos
        self.X = self.X[~outlier_rows]
        self.y = self.y[~outlier_rows]

        self.X.reset_index(drop=True, inplace=True)
        self.y.reset_index(drop=True, inplace=True)
        print("Cantidad de datos nuevos ", self.X.shape, self.y.shape)
        sys.stdout.flush()   

        return self.X, self.y
    
    # Función para entrenar los transformadores: Imputar, Escalar, Codificar
    def fit(self):
        print("---------------------------------------------------")
        sys.stdout.flush()
        print("------------ Creando transformadores  -------------")
        sys.stdout.flush()
        print("---------------------------------------------------")
        sys.stdout.flush()#

        print("Imputar datos numéricos.")
        sys.stdout.flush()
        numeric_data = self.X[self.numeric_columns]
        # Verificar si hay datos faltantes en las variables numéricas
        if numeric_data.isnull().any().any():
            # Verificar si el porcentaje de datos faltantes es menor que el umbral
            if numeric_data.isnull().mean().mean() < self.missing_threshold:
                # Ajustar el imputador a todas las variables numéricas
                self.numeric_imputer.fit(numeric_data)
                # Guardar el imputador en el diccionario de transformadores
                self.transformers['numeric_imputer'] = self.numeric_imputer
            else:
                print(f"Al menos una de las columnas numéricas tiene un porcentaje de datos faltantes mayor al {self.missing_threshold * 100}%")
            
        else:
            print("No hay datos faltantes en las columnas numéricas, no se requiere imputación.")

        print("Imputar datos categóricos.")
        sys.stdout.flush()
        categorical_data = self.X[self.categorical_columns]
        # Verificar si hay datos faltantes en las variables categóricas
        if categorical_data.isnull().any().any():
            # Verificar si el porcentaje de datos faltantes es menor que el umbral
            if categorical_data.isnull().mean().mean() < self.missing_threshold:
                # Ajustar el imputador a todas las variables categóricas
                self.categorical_imputer.fit(categorical_data)
                # Guardar el imputador en el diccionario de transformadores
                self.transformers['categorical_imputer'] = self.categorical_imputer
            else:
                print(f"Al menos una de las columnas categóricas tiene un porcentaje de datos faltantes mayor al {self.missing_threshold * 100}%")
        else:
            print("No hay datos faltantes en las columnas categóricas, no se requiere imputación.")

        print("Escalar datos numéricas.")
        sys.stdout.flush()
        # Escalar variables numéricas
        self.scaler_X.fit(self.X[self.numeric_columns])
        self.transformers['scaler_X'] = self.scaler_X

        # Si el modelo es de regresión también se escala la variable objetivo
        if self.model_type == 'Regression':
            self.y = np.array(self.y)
            self.scaler_y.fit(self.y.reshape(-1, 1))
            self.transformers['scaler_y'] = self.scaler_y

        if self.model_type == 'Classification':
            self.label_encoder.fit_transform(self.y)
            self.transformers['label_encoder']  = self.label_encoder

        print("Codificar datos categóricos.")
        sys.stdout.flush()
        # Codificar variables categóricas
        self.one_hot_encoder.fit(self.X[self.categorical_columns])
        self.transformers['one_hot_encoder'] = self.one_hot_encoder
        
    # Función para aplicar las transformaciones a los datos.
    def transform(self):
        print("---------------------------------------------------")
        sys.stdout.flush()
        print("---------- Aplicando transformadores  -------------")
        sys.stdout.flush()
        print("---------------------------------------------------")
        sys.stdout.flush()

        print("Imputar datos nulos.")
        sys.stdout.flush()
        # Imputar datos nulos en variables numéricas
        if 'numeric_imputer' in self.transformers:
            self.X[self.numeric_columns] = self.transformers['numeric_imputer'].transform(self.X[self.numeric_columns])
   
        # Imputar datos nulos en variables categóricas
        if 'categorical_imputer' in self.transformers:
            self.X[self.categorical_columns] = self.transformers['categorical_imputer'].transform(self.X[self.categorical_columns])


        print("Escalar datos numéricos.")
        sys.stdout.flush()
        self.X[self.numeric_columns] = self.transformers['scaler_X'].transform(self.X[self.numeric_columns])
        if self.model_type == 'Regression':
            self.y = np.array(self.y)
            self.y = self.transformers['scaler_y'].transform(self.y.reshape(-1, 1)).ravel()
        
        print("Codificar datos categóricos.")
        sys.stdout.flush()
        encoded_features = self.transformers['one_hot_encoder'].transform(self.X[self.categorical_columns])
        encoded_feature_names = self.one_hot_encoder.get_feature_names_out(input_features=self.categorical_columns)
        encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoded_feature_names)
        self.X.drop(columns=self.categorical_columns, inplace=True)
        self.X = pd.concat([self.X, encoded_df], axis=1)
  
        # Si el modelo es de clasificación realizar Balanceo de datos
        if self.model_type == "Classification":
            print("Codificación de variable a predecir.")
            sys.stdout.flush()
            self.y = pd.Series(self.transformers['label_encoder'].transform(self.y))
            # Mostramos las etiquetas codificadas
            print("Etiquetas codificadas:", self.y)

            # Mostramos el mapeo de etiquetas originales a códigos numéricos
            print("Mapeo de etiquetas originales a códigos numéricos:")
            for label, code in zip(self.transformers['label_encoder'].classes_, self.transformers['label_encoder'].transform(self.transformers['label_encoder'].classes_)):
                print(f"{label}: {code}")

            print("Balanceo de datos: ")
            sys.stdout.flush()
            print(f"Datos balanceados usando {self.balance_method} con {self.sampler}")
            sys.stdout.flush()   
            print(f"cantidad clases antes del balanceo: {self.y.value_counts()}")
            sys.stdout.flush()   
            X_resampled, y_resampled = self.sampler.fit_resample(self.X, self.y)
            print("y_resampled")
            sys.stdout.flush()
            self.X = X_resampled
            self.y = y_resampled  
            print(f"cantidad clases despues del balanceo: {self.y.value_counts()}")
            sys.stdout.flush()  
        
        return self.X
    
    # Función para seleccionar las caracteristicas mas representativas
    def select_features(self):
        print("Selección de características: ")
        sys.stdout.flush()
        # Ajustar el objeto SelectKBest al conjunto de datos
        n_features = int(self.X.shape[1]*self.k)
        print('Cantidad caracteristicas a seleccionar: ', n_features)
        sys.stdout.flush()
        print('Cantidad caracteristicas inicial: ', self.X.shape[1])
        sys.stdout.flush()
        self.feature_selector.k = n_features
        self.feature_selector.fit(self.X, self.y)

        # Obtener las características más representativas
        selected_features = self.X.iloc[:, self.feature_selector.get_support()]
        self.X = selected_features

        print('Caracteristicas seleccionadas: ',self.X.columns)
        sys.stdout.flush() 

        # Guargar caracteristicas seleccionadas en los transformadores
        self.transformers['feature_selector'] = self.X.columns
        return self.X
    
    # Función para obtener 'y' varialbe objetivo y 'X' variables predictoras
    def get_processed_dataframe(self):
        return self.X , self.y

    # Función para guardar los transformadores.
    def save_transformers(self, filename):
        print("Guardando transformadores: ")
        sys.stdout.flush() 

        try:
            # Guarda el diccionario en un archivo usando joblib
            joblib.dump(self.transformers, filename)
            print(f"Las transformaciones se guardaron en '{filename}'.")
        except Exception as e:
            print(f"Error al guardar las transformaciones: {e}")

    # Función para cargar los transformadores.
    def load_transformers(self, filename):
        print("Cargando transformadores: ")
        sys.stdout.flush()

        try:
            # Carga el diccionario del archivo usando joblib
            transformers = joblib.load(filename)
            print(f"Las transformaciones se cargaron desde '{filename}'.")
            return transformers
        except Exception as e:
            print(f"Error al cargar las transformacioens: {e}")

    # Función para aplicar los transformadores a los datos a predecir
    def apply_transformers(self, transformers, X):
        print("Aplicando transformadores: ")
        sys.stdout.flush()

        # Obtener columnas numericas y categoricas.
        numeric_columns = X.select_dtypes(include=['number']).columns
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns

        # Revisar los transformadores, de acuerdo a las llaves aplicar sobre los datos
        for name, transformer in transformers.items():
            if 'numeric_imputer' in name:
                X[numeric_columns] = transformers['numeric_imputer'].transform(X[numeric_columns])
            elif 'categorical_imputer' in name:
               X[categorical_columns] = transformers['categorical_imputer'].transform(X[categorical_columns])
            elif name == 'scaler_X':
                X[numeric_columns] = transformer.transform(X[numeric_columns])
            elif name == 'one_hot_encoder':
                encoded_feature_names = []
                for i, column in enumerate(categorical_columns):
                    encoded_feature_names.extend([f"{column}_{category}" for category in transformer.categories_[i]])

                encoded_features = transformer.transform(X[categorical_columns])
                encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoded_feature_names)
                X.drop(columns=categorical_columns, inplace=True)
                X = pd.concat([X, encoded_df], axis=1)
            elif name == 'feature_selector':
                X = X[transformers['feature_selector']]

        return X