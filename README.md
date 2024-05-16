# Proyecto de Generación de Modelos de Machine Learning Supervisado

Este proyecto permite generar modelos de Machine Learning supervisado para problemas de regresión y clasificación.

## Instrucciones de uso

1. **Crear un Entorno Virtual:**
   - Se recomienda utilizar una versión de Python 3.9.6. Puedes crear un entorno virtual siguiendo las instrucciones en [este enlace](https://docs.python.org/es/3/tutorial/venv.html).

2. **Instalar Dependencias:**
   - Una vez activado el entorno virtual, instala las dependencias del archivo `requirements.txt` utilizando el siguiente comando:
     ```bash
     pip install -r requirements.txt
     ```
3. **Configuración del Proyecto:**
   Ajusta los parámetros del proyecto en el archivo de configuración "config.json".
   - Configuración dataset a procesar:
      - `project_name`: Nombre del proyecto para guardar e identificar las transformaciones / modelos / datos.
      - `model_type`: Tipo de modelo a realizar : Rregression o  Classification
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

4. **Ejecutar el Archivo `main.py`:**
   - Una vez configurado, ejecuta el archivo `main.py` para comenzar el proceso de generación de modelos.

## ACLARACIONES:
    1. Tratamiento de fechas.
    2. Caracteristicas compuestas Ejemplo: Distancia = [15Km,20.000mts, 12Km] - deben indicarse la columna la unidad y los datos serán los valores . Distancia(KM) = [15,20,12]
    3. Textos como oraciones, fraces, articulos , etc. Solo se aceptaran categorías claras. 
    4. Los columnas con grandes desbalanceos entre caracteristicas o nulos . Se dejarán aparte en una lista para que el usuario defina que preprocesamiento aplicar para dichas columnas o si desea eliminarlas. 