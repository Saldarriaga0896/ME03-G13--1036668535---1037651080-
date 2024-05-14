import pandas as pd
import json
import sys
from src.PreprocessingModule import DataPreprocessor

def load_params(config_file):
  with open(config_file, 'r') as f:
      config = json.load(f)
  return config

config_file = 'config.json'
config = load_params(config_file)
function = config.get("function")
model_type = config.get("model_type")
target_column = config.get("target_column")

path_transforms = 'data/' +config.get('project_name')
path_predict = 'data/predict_' + config.get('project_name')

#---------------------------------------------------------------#
#---------------------- Cargar instancias ----------------------#
#---------------------------------------------------------------#
preprocessor = DataPreprocessor(config)

print("------- Preprocesamiento de los datos -------------")

# Carga del dataset y configuraciones
preprocessor.load_dataset()
# Tratamiento columnas categóricas
preprocessor.transform_categorical_columns()
# Descripción de los datos
#preprocessor.descriptive_analysis()
# Guardar un porcentaje de datos para predicciones
preprocessor.split_data_for_predictions(path_predict)
# Eliminar datos atipicos de las variables numericas
preprocessor.remove_outliers_zscore()
# Ajustar el preprocesador a los datos
preprocessor.fit()
# Transformar los datos de entrenamiento
preprocessor.transform()
# Seleccion de caracteristicas representativas
preprocessor.select_features()
# Guardar transformadores
preprocessor.save_transformers(path_transforms)

# Obtener las variables predictoras "X" y a predecir "y" procesadas.
X,y = preprocessor.get_processed_dataframe()

print(X,y)