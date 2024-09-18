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

### Puntos clave del proyecto

1. **Preprocesamiento de datos**
   - Carga de datos desde Google BigQuery o un archivo CSV local, según disponibilidad. Este es el directorio en Drive con el archivo .csv principal: [Enlace al archivo .csv] (https://drive.google.com/drive/folders/1IdonUB38UQC8TSGIrdPwrMCv5eThVB5d?usp=sharing)
Esto agiliza la descarga de datos, pero podemos conectarnos directamente a BigQuery, el código comprobará si el archivo está en la carpeta o no.
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
   - Generación de noticias sintéticas mediante NLTK y análisis de sentimiento financiero con FinBERT.
   - Los datos textuales complementan los modelos, añadiendo una dimensión cualitativa al análisis financiero.

---

### Conclusiones

- **Robustez del modelo**: Aunque los resultados no son impresionantes (malos para regresión y medios para clasificación), hemos visto que en base a esas predicciones podemos crear nuestra propia estrategia de inversión y que sea ganadora batiendo al benchmark. En la correcta combinación está el éxito.
- **Evaluación realista**: Gracias a la validación walk-forward y el backtesting, el sistema ofrece una evaluación precisa de las predicciones en contextos de mercado reales.
- **Flexibilidad**: El sistema permite la personalización del análisis financiero según sectores, clusters y tickers, facilitando su adaptabilidad a diferentes estrategias de inversión.

### Limitaciones y futuras mejoras
- **Costes de transacción y liquidez**: No se han tenido en cuenta comisiones y demás intereses de mercado.
- **Ampliación del análisis**: Aumentar el número de tickers simultáneos para mejorar la representatividad.
- **Optimización de hiperparámetros**: Uso de técnicas automáticas como Grid Search o Bayesian Optimization pueden mejorar el rendimiento (aunque a priori comprobamos que no).
- **Mejora de la generalización**: Aunque podemos cargar modelos preentrenados, sería bueno poder entrenar un número mayor de acciones para conseguir la ansiada generalización, que a nivel computacional nos ha sido imposible implementar.

---


### Agradecimientos

Gracias por todo. Ha sido un año lleno de aprendizajes y retos a cual más apasionante. ¡Ha sido genial!

**Carles Matoses Miret**  
**Sara Díaz-P. Martín**


