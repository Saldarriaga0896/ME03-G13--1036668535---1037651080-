# Proyecto de Generación de Modelos de Machine Learning Supervisado

Este proyecto permite generar modelos de Machine Learning supervisado para problemas de regresión y clasificación.

## Instrucciones de uso

1. **Abrir el notebook en Google Colab:**
   - Abrir el notebook llamado ME03_G13_[1036668535]_[1037651080].ipynb, ubicado en la raíz de este repositorio, usando Google Colaboratory.

2. **Clonar repositorio e instalar dependencias:**
   - Una vez abierto el notebook en Google Colab, ejecutar en orden las dos primeras sentencias
   - !git clone https://github.com/Saldarriaga0896/ME03_G13_1036668535_1037651080.git (Para clonar el repositorio en el ambiente de Colab)
   - !pip install -r /content/ME03_G13_1036668535_1037651080/requirements.txt (Para instalar las librerías necesarias)
     
3. **Configuración del Proyecto:**
   Ajusta los parámetros del proyecto en el archivo de configuración "config.json".
   - Configuración dataset a procesar:
      - `project_name`: Nombre del proyecto para guardar e identificar las transformaciones / modelos / datos.
      - `data_path`: Ruta del archivo de datos.
      - `sep`: Separador del csv ",", ";", "-", "|", etc.
      - `target_column`: Nombre de la columna objetivo.
      - `split`: Porcentaje de separación del dataset inicial para posteriormente hacer predicciones.

   - Preprocesamiento:
      - `missing_threshold`: Umbral de decision de datos que pueden ser nulos para realizar imputacion, valores de (0.0 a 1.0)
      - `numeric_imputer`: Estrategia con la que se van a reemplazar los datos faltantes numéricos nulos ("mean", "median" ,          "most_frequent")
      - `categorical_imputer`: Estrategia con la que se van a reemplazar los datos faltantes categóricos ("most_frequent","unknown")
      - `threshold_categorical`: Cantidad de valores diferentes permitidos por columna categórica
      - `threshold_outlier`: Cantidad en desviació de estandar para identificar datos atipicos
      - `balance_method`: Modo de balanceo de datos : over-sampling o under-sampling
      - `balance_thershold`: Umbral de decision para balancear los datos, valores entre 0.0 y 1.0
      - `k_features`: Porcentaje de caractaristicas representativas a seleccionar para entrenar el modelo. 
      - `delete_columns`: Columnas que el usuario identifique que se puedan eliminar.

   - Entrenamiento:
        - `model_type`: Tipo de modelo a realizar : Rregression o  Classification
        - `function`: Función a utilizar para el entrenamiento y generación del modelo o predicción consumiendo el modelo generado.        "training","predict".
        - `n_jobs`: Cantidad de procesamiento a utilizar de la maquina . -1 para tomar todos los recursos.
        - `cv`: Validacion cruzada. 
        - `scoring_regression`: Métrica a utilizar para seleccionar el mejor modelo en Modelos de Regresión 
        - `scoring_classification`: Métrica a utilizar para seleccionar el mejor modelo en Modelos de Clasificación.
        - `random_state`: Semilla para generar los modelos. 
        - `models_regression`: Modelos de regresión que van a competir. Puede activar o desactivar con las banderas true o false
        - `params_regression`: Hiperparametros de cada modelo. Puede ser modificados los rangos
        - `models_classification`: Modelos de clasificacion que van a competir. Puede activar o desactivar con las banderas true o false.
        - `params_classification`: Hiperparametros de cada modelo. Puede ser modificados los rangos.

4. **Ejecutar los pasos del notebook:**
   - Una vez configurado, ejecuta los pasos siguientes del notebook para obtener los resultados preprocesados (También se puede ejecutar el notebook completo).

## ACLARACIONES:
    1. El módulo no incluye tratamiento de fechas. Se recomienda separar en diferentes columnas los datos (año, mes, día, hora, etc)
    2. Caracteristicas compuestas, por ejemplo: Distancia = [15Km,20.000mts, 12Km] - deben indicarse la columna la unidad y los datos serán los valores . Distancia(KM) = [15,20,12]
    3. Textos como oraciones, frases, articulos , etc. Solo se aceptaran categorías claras. 
    4. Los columnas con grandes desbalanceos entre caracteristicas o nulos . Se dejarán aparte en una lista para que el usuario defina que preprocesamiento aplicar para dichas columnas o si desea eliminarlas. 
