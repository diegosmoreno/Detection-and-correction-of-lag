# Detection-and-correction-of-lag

# Detecting and Correcting Data Misalignment in Continuous Glucose Monitoring Forecast Evaluation

## Descripción

Este repositorio contiene el código desarrollado para el Trabajo Fin de Máster titulado **"Detecting and Correcting Data Misalignment in Continuous Glucose Monitoring Forecast Evaluation"**.

El objetivo del proyecto es **desarrollar una metodología sólida para detectar y corregir los desajustes sistemáticos o time lags entre los resultados de las predicciones de los niveles de glucosa en sangre y las mediciones reales correspondientes, garantizando así una evaluación realista del rendimiento del modelo de predicción**.

## Estructura del repositorio

```text
├── scripts/        # Código utilizado para la consecución de resultados
├── README.md       # Documentación principal
```

## Requisitos

* Python 3.x
* Las librerías necesarias se incluyen en cada uno de los scripts


## Scripts
En la carpeta de scripts se incluyen dos archivos de código en Python para cada una de las metodologías propuestas y detalladas en la memoria del TFM. Uno de los scripts lleva a cabo las labores de preprocesamiento y aplica la función CCF o DTW, global o en tiempo real. Este script devuelve añade básicamente al conjunto de datos de entrada la serie de predicciones corregida y valores del lag detectado. En cada script se explica detalladamente el código.

El otro script incluido para cada técnica sirve para calcular las métricas de evaluación empleando la serie de predicciones sin corregir y corregida.

## Ejecución

Cada uno de los scripts está listo para ejecutarse directamente, tan solo es necesario elegir el archivo de datos de entrada que va a ser analizado.

```bash
python3 scripts/file.py
```

## Datos

Por protección de datos, no se incluyen los datasets de entrada empleados. La descripción de estos datasets y sus referencias se incluyen en el capítulo 3 de la memoria.

## Resultados

Los resultados parciales tampoco se incluyen por protección de datos. Los resultados finales pueden consultarse completamente en la sección de resultados de la memoria del TFM, así como en los anexos de esta.

## Autor

* Diego Sebastián Moreno Ceacero
* Máster en DATA SCIENCE AND COMPUTER ENGINEERING
* Universidad DE GRANADA

## Licencia

Este repositorio se ha creado con fines académicos como parte del Trabajo Fin de Máster.
