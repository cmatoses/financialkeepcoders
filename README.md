# Financial KeepCoders 
## Transformando las finanzas con IA  
**Bootcamp Big Data, IA y ML - KeepCoding**

---

### Descripción del proyecto
Este proyecto tiene como objetivo aplicar técnicas avanzadas de Machine Learning y Deep Learning para analizar y predecir movimientos en los mercados financieros. Utilizando una combinación de modelos predictivos, procesamiento de datos y estrategias de trading, hemos creado un sistema (versión 1.0) para tomar decisiones de inversión basadas en datos históricos y modelos estadísticos.

El flujo de trabajo incluye la descarga de datos de BigQuery, el preprocesamiento de esos datos financieros, la selección de tickers, la implementación de modelos de aprendizaje automático y la validación de las predicciones mediante backtesting en datos históricos.

---

### Tecnologías utilizadas
- **Python**: Lenguaje principal para el desarrollo del proyecto.
- **Google Cloud BigQuery**: Fuente de datos para la extracción de grandes volúmenes de datos financieros.
- **Pandas, NumPy**: Manipulación y análisis de datos.
- **Scikit-learn**: Modelos tradicionales de Machine Learning como Random Forest.
- **TensorFlow, Keras**: Implementación de redes neuronales como LSTM, GRU y Conv1D.
- **Matplotlib, Seaborn**: Visualización de datos.
- **Flask**: Para la creación de la API.
- **FinBERT, NLTK**: Generación de noticias sintéticas y análisis de sentimiento.

---

### Cómo ejecutar el proyecto 

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/usuario/financialkeepcoders.git
   cd financialkeepcoders
   ```
2. **Instalar las dependencias**: 
   ```bash
   pip install -r requirements.txt
   ```

3. **Colocar el archivo CSV en el directorio correspondiente**:
Si se tiene el archivo .csv de datos descargado (que se especifica más abajo), se colocará en el mismo directorio y dentro de "variables.env" especificaremos la variable DATA_PATH hacia a esa ruta.

4. Ejecutar el archivo Python: 
   ```bash
   python ml_modelos_final.py
   ```
**Opcional**: Ejecutar la aplicación Flask: 
Si prefieres ejecutar la aplicación Flask, más abajo se especifica cómo.

---

### Puntos clave del proyecto a nivel de modelado

1. **Preprocesamiento de datos**
   - Carga de datos desde Google BigQuery o un archivo CSV local, según disponibilidad. Este es el directorio en Drive con el archivo "gold_main.csv" principal: [Enlace al archivo .csv](https://drive.google.com/drive/folders/1IdonUB38UQC8TSGIrdPwrMCv5eThVB5d?usp=sharing)
Esto agiliza la descarga de datos, pero podemos conectarnos directamente a BigQuery, el código comprobará si el archivo está en la carpeta o no.
**NOTA IMPORTANTE**: Al hacer este repositorio público, las credenciales (bigquery_credentials.json) de la carpeta conf no funcionan por motivos de seguridad. Tenemos en local el archivo válido con las credenciales para poder conectarse a BigQuery (nos la podéis pedir).
**Si se decide descargar este .csv, solo con clonar el repositorio e instalar las dependencias, se puede ejecutar sin problemas el archivo ml_modelos_final.py o la aplicación Flask.**
   - Filtrado de tickers por volumen, sector y cluster.
   - Normalización y escalado de las características mediante `MinMaxScaler`.

2. **Modelos predictivos**
   - **Modelos de Clasificación de ML**: Random Forest para la clasificación de señales de compra y venta.
   - **Modelos de Regresión de ML**: Predicción de precios mediante Random Forest Regressor.
   - **Deep Learning con LSTM y GRU**: Captura de patrones a largo plazo en series temporales.
   - **Conv1D**: Detección de patrones locales en secuencias financieras.
   - **Modelos Híbridos**: Combinación de LSTM y RF para aprovechar tanto patrones locales como globales.

3. **Validación Walk-Forward**
   - Se utilizó la técnica de validación walk-forward para asegurar la consistencia temporal en las predicciones.
   - Esta técnica permite evaluar la capacidad del modelo para generalizar a datos futuros no vistos.
   - De las 4 acciones seleccionadas, hacemos training con las 3 primeras y testeamos con la última.

4. **Backtesting**
   - Implementación de un sistema de backtesting que evalúa el rendimiento de las estrategias de trading basadas en las predicciones de los modelos.
   - Cálculo de métricas financieras como rentabilidad, drawdown, y ratio de Sharpe.

5. **Escalabilidad y personalización**
   - El sistema permite la selección personalizada de sectores, clusters y tickers, ajustándose a las necesidades del usuario.
   - Posibilidad de seleccionar tickers de forma aleatoria o manual en el archivo `variables.env`.

6. **Análisis de sentimiento**
   - Generación de noticias sintéticas mediante NLTK y análisis de sentimiento financiero con FinBERT. Esto nos ha permitido usar la técnica Transfer Learning.
   - Los datos textuales complementan los modelos, añadiendo una dimensión cualitativa al análisis financiero.
  
7. **Aplicación Flask**
   - Hemos añadido valor al proyecto con Flask, convirtiendo el código de algoritmos en una herramienta más accesible e interactiva.
   - La aplicación **Flask** permite que el proyecto sea más accesible al usuario final mediante una interfaz web simple que interactúa con los modelos y los datos subyacentes. Con Flask, se ha creado una **API** que expone funcionalidades clave del sistema en tiempo real, permitiendo al usuario:

      - Ejecutar consultas a BigQuery y/o cargar datos financieros sin interactuar directamente con el código.
      - Seleccionar el sector y cluster que quiere predecir (limitado a 4 acciones), lo que le aporta personalización y flexibilidad a la hora de realizar análisis o predicciones.
      - Correr los modelos predictivos de manera sencilla, facilitando la integración de este sistema con otras aplicaciones o procesos.
      - Acceder a los resultados del backtesting y las predicciones, obteniendo el mejor modelo y la predicción para el día siguiente.

   #### Cómo ejecutar la aplicación Flask
   
   1. **Clonar el repositorio**:
      ```bash
      git clone https://github.com/usuario/financialkeepcoders.git
      cd financialkeepcoders
      cd financial_keepcoders_app
      ```
   2. **Instalar las dependencias en el entorno virtual**:
      ```bash
      pip install -r requirements_flask.txt
      ```
   3. **Configurar las variables de entorno**:
   Crea un archivo "variables_flask.env" en el directorio raíz del proyecto y configura las siguientes variables:
   - DATA_PATH: Ruta del archivo CSV local para los datos financieros que se indica más arriba. (Si no lo tienes la aplicación se conectará directamente a BigQuery para descargarlo, tardará un poco más).

   
   4. **Ejecutar la aplicación Flask**:
   Inicia la aplicación Flask desde el archivo principal app.py:
      ```bash
      python app.py
      ```
      
   5. **Acceder a la aplicación**:
   Una vez que Flask esté en funcionamiento, podrás acceder a la aplicación en tu navegador en (http://localhost:5000) o interactuar con la API para obtener predicciones y ejecutar análisis financieros.
   

---

### Conclusiones

- **Robustez del modelo**: A pesar de que los resultados en esta versión inicial no son destacables (resultados bajos en regresión y moderados en clasificación), hemos comprobado que es posible diseñar una estrategia de inversión basada en estas predicciones que pueda superar al benchmark. El éxito radica en la correcta combinación de las predicciones y la estrategia.
- **Evaluación realista**: La implementación de la validación walk-forward, junto con el backtesting, permite que el sistema evalúe las predicciones de manera precisa en entornos de mercado reales, simulando condiciones auténticas y proporcionando una visión clara del rendimiento.
- **Flexibilidad**: El sistema ofrece una gran capacidad de personalización, permitiendo ajustar el análisis financiero según sectores, clusters y tickers. Esto facilita la adaptación a distintas estrategias de inversión y necesidades del usuario.


### Limitaciones y futuras mejoras
- **Costes de transacción y liquidez**: No se han tenido en cuenta comisiones y demás intereses de mercado.
- **Ampliación del análisis**: Aumentar el número de tickers simultáneos para mejorar la representatividad.
- **Optimización de hiperparámetros**: Uso de técnicas automáticas como Grid Search o Bayesian Optimization pueden mejorar el rendimiento (aunque a priori comprobamos que no).
- **Mejora de la generalización**: Aunque podemos cargar modelos preentrenados, sería bueno poder entrenar un número mayor de acciones para conseguir la ansiada generalización, que a nivel computacional nos ha sido imposible implementar.
- **Hacer Fine Tuning**: En relación al anterior punto, el siguiente paso podría incluir hacer Fine Tuning con estos modelos preentrenados, donde podamos congelar capas y adaptar los hiperparámetros a nuestro problema para intentar lograr una predicción mejor y, por tanto, generalización.

---


### Agradecimientos

Gracias por todo. Ha sido un año lleno de aprendizajes y retos a cual más apasionante. 

¡Ha sido genial!

**Carles Matoses Miret**  
**Sara Díaz-P. Martín**


