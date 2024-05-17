# Proyecto de Generación de Modelos de Machine Learning Supervisado

Este proyecto permite generar modelos de Machine Learning supervisado para problemas de regresión y clasificación.

## Instrucciones de uso

1. **Abrir el notebook en Google Colaboratory:**
   - Abrir el notebook llamado "ME03_G13_[1036668535]_[1037651080].ipynb" en Google Colab
2. **Leer las recomendaciones sobre las características de los DataSet**
3. **Clonar el repositorio:**
   - Ejecutar la celda que clona el repositorio en el ambiente de Google Colab:
  
     !git clone https://github.com/Saldarriaga0896/ME03_G13_1036668535_1037651080.git

4. **Instalar Dependencias:**
   - Instalar las dependencias del archivo `requirements.txt` ejecutando la celda que contiene el siguiente código:
     
     !pip install -r /content/ME03_G13_1036668535_1037651080/requirements.txt

   - Cuando se hayan instalado las dependencias, Google Colab solicitará que reinicie la sesión para aplicar los cambios correctamente.
     
5. **Configuración del Proyecto:**
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

6. **Ejecutar las celdas siguientes:**
   - Una vez configurado el proyecto, ejecuta las celdas siguientes en Google Colab. Se ejecuta a partir del paso 3.3 indicado en el notebook.
   - En un momento de la ejecución del paso 3.6, la función check_abnormal_columns solicita al usuario ingresar un valor 1 o 2 para continuar la ejecución o detener respectivamente.
  
7. **Obtener el DataSet Procesado**:
   - Una vez haya finalizado la ejecución el dataset resultado se encontrará en la carpeta "/content/ME03_G13_1036668535_1037651080/data/" ubicada en Colaboratoy y este tendrá el nombre de 'project_name'__processed.csv (project_name se puede modificar en la configuración del proyecto).

