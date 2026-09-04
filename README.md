# 🩺 Melanoma Detector AI

Clasificación binaria de lesiones de piel (**Es melanoma** vs **No es melanoma**) mediante **Regresión Logística**, con un aplicativo web interactivo construido en **Streamlit**.

## 🚀 Demo / Deploy

App desplegada en Streamlit Community Cloud: [https://melanoma-detector.streamlit.app](https://melanoma-detector.streamlit.app)

## 📋 Descripción del Proyecto

Este proyecto aplica **Machine Learning** para clasificar lesiones de piel determinando si **Es melanoma** (Maligno) o **No es melanoma** (Benigno) a partir de imágenes dérmicas. Cumple el ciclo completo de un proyecto de Inteligencia Artificial:

1. **Análisis exploratorio de datos (EDA)**
2. **Preprocesamiento y normalización de características** (vectorización de píxeles 32x32x3 y `StandardScaler`)
3. **Entrenamiento con Regresión Logística y Función Sigmoide**
4. **Evaluación de desempeño** (Matriz de confusión TP, TN, FP, FN y Curva ROC / AUC-ROC)
5. **Análisis de coeficientes de influencia espacial**
6. **Despliegue interactivo en Streamlit**

## 🖥️ Interfaz del Aplicativo Web (Streamlit)

La interfaz cuenta con 3 pestañas principales:

1. **🔍 Clasificar:** Carga de imágenes (JPG, JPEG, PNG) y predicción con etiquetas **"Es melanoma"** o **"No es melanoma"**, mostrando probabilidades, nivel de confianza y ubicación del score en la función sigmoide.
2. **📊 Métricas de Evaluación:** Visualización cuantitativa de **Accuracy (84.30%)**, **Precision (86.72%)**, **Recall (81.00%)**, **F1-Score (83.76%)** y **AUC-ROC (91.97%)**, junto con la **Matriz de Confusión (TP, TN, FP, FN)** y la **Curva ROC**.
3. **📐 Modelado Matemático y Entrenamiento:** Explicación técnica de la función Sigmoide $\sigma(z) = \frac{1}{1 + e^{-z}}$, la función de pérdida Log Loss, el optimizador L-BFGS, y el mapa de calor para el **análisis de coeficientes** por píxel.

## 📊 Resultados del Modelo (Test: 2,000 imágenes)

| Métrica | Valor | Descripción / Interpretación |
|---|:---:|---|
| **Exactitud (Accuracy)** | **84.30%** | Proporción total de diagnósticos correctos |
| **Precisión (Precision)** | **86.72%** | Porcentaje de verdaderos melanomas entre las alertas |
| **Sensibilidad / Exhaustividad (Recall)** | **81.00%** | Porcentaje de melanomas reales detectados |
| **Puntuación F1 (F1-Score)** | **83.76%** | Balance armónico entre Precisión y Recall |
| **Área Bajo la Curva (AUC-ROC)** | **91.97%** | Capacidad discriminativa global del modelo |

### Matriz de Confusión

- **Verdaderos Positivos (TP):** 810 (Casos de melanoma detectados correctamente como "Es melanoma")
- **Verdaderos Negativos (TN):** 876 (Lesiones benignas detectadas correctamente como "No es melanoma")
- **Falsos Positivos (FP):** 124 (Lesiones benignas clasificadas como "Es melanoma")
- **Falsos Negativos (FN):** 190 (Casos de melanoma clasificados como "No es melanoma")

## 🗂️ Estructura del Repositorio

```
melanoma-detector/
├── app.py                 # Aplicación web Streamlit (Clasificación, Métricas y Modelado)
├── train_model.py         # Script de entrenamiento, evaluación y generación de gráficos
├── requirements.txt       # Dependencias de Python
├── model/                 # Archivos serializados del modelo
│   ├── melanoma_logistic_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── model_config.pkl
├── plots/                 # Gráficos del EDA, coeficientes y evaluación
└── docs/                  # Documentación del informe técnico
```

## 🧠 Sobre el Modelo

- **Características:** Píxeles de imágenes redimensionadas a 32×32 RGB (vector de 3,072 características $x_1, \dots, x_{3072}$).
- **Preprocesamiento:** `StandardScaler` ($\mu = 0, \sigma = 1$).
- **Modelo:** `LogisticRegression` de scikit-learn con `solver='lbfgs'`, `max_iter=1000` y `C=1.0`.
- **Clasificaciones:**
  - `0` (Benigno): **No es melanoma**
  - `1` (Maligno): **Es melanoma**

## 🖥️ Ejecución Local

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. (Opcional) Entrenar el modelo

```bash
python train_model.py
```

### 3. Ejecutar la aplicación

```bash
streamlit run app.py
```

Abre el navegador en `http://localhost:8501`.

## ⚠️ Aviso

Esta herramienta es **educativa y de apoyo al diagnóstico**. **No reemplaza la evaluación de un dermatólogo profesional.** Ante cualquier sospecha de lesión dérmica, consulta inmediatamente a un especialista.

## 👥 Integrantes del Equipo

- **Suárez Condori Juan Gabriel**
- **Ramos Ticahuanca Gianella Alexandra**
- **Jesus del Aguila Garcia**
- **Mendoza Torres Lincol Jhon**
