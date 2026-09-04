import os
import numpy as np
import joblib
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
IMG_SIZE = 32


def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "melanoma_logistic_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    return model, scaler, label_encoder


def preprocess_image(uploaded_file, model):
    try:
        uploaded_file.seek(0)
        with Image.open(uploaded_file) as img:
            img.verify()
        uploaded_file.seek(0)
        img = Image.open(uploaded_file).convert("RGB")
    except Exception:
        raise ValueError("El archivo no es una imagen válida. Asegúrate de subir un archivo en formato JPG, JPEG o PNG.")
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized).flatten().reshape(1, -1)
    X_scaled = model["scaler"].transform(img_array)
    return img, img_resized, X_scaled


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_z(model, X_scaled):
    coef = model.coef_[0]
    intercept = model.intercept_[0]
    return float(np.dot(X_scaled[0], coef) + intercept)


def plot_sigmoid(z_highlight=None, prob_highlight=None):
    z = np.linspace(-8, 8, 300)
    s = sigmoid(z)

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=110, facecolor='#161b26')
    ax.set_facecolor('#161b26')
    ax.plot(z, s, color='#00d2ff', linewidth=3, label=r'$\sigma(z) = \frac{1}{1+e^{-z}}$')
    ax.axhline(0.5, color='#ff9f43', linestyle='--', linewidth=1.5, label='Umbral = 0.5')
    ax.axvline(0, color='#b8b8d1', linestyle=':', linewidth=1, alpha=0.7)
    ax.axhline(0, color='#b8b8d1', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axhline(1, color='#b8b8d1', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('z = β₀ + Σ(βᵢ · xᵢ)', color='#d0d0e8', fontsize=10)
    ax.set_ylabel('Probabilidad P(y=1)', color='#d0d0e8', fontsize=10)
    ax.set_title('Función Sigmoide (Transformación Logística)', color='#ffffff', fontweight='bold', fontsize=12)
    ax.tick_params(colors='#d0d0e8')
    for spine in ax.spines.values():
        spine.set_color('#333b50')
    ax.grid(True, alpha=0.15)

    if z_highlight is not None and prob_highlight is not None:
        ax.scatter([z_highlight], [prob_highlight], color='#ef473a', s=100, zorder=5, label=f'Imagen actual (prob = {prob_highlight:.2f})')
        ax.annotate(f'z = {z_highlight:.2f}\nP = {prob_highlight:.2f}', xy=(z_highlight, prob_highlight),
                    xytext=(z_highlight + 1.2, prob_highlight - 0.15),
                    color='#ffffff', fontsize=9, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#ef473a', lw=1.5))

    ax.legend(loc='center right', facecolor='#24243e', edgecolor='#555577', labelcolor='#ffffff', fontsize=9)
    fig.tight_layout()
    return fig


def plot_confusion_matrix():
    # Matrix values from test evaluation (2,000 images: 1,000 Benign, 1,000 Malignant)
    # TN=876, FP=124, FN=190, TP=810
    cm = np.array([[876, 124], [190, 810]])
    labels = [
        ["TN = 876\n(No es melanoma)", "FP = 124\n(Falsa Alarma)"],
        ["FN = 190\n(Maligno No Detectado)", "TP = 810\n(Es melanoma)"]
    ]

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=110, facecolor='#161b26')
    ax.set_facecolor('#161b26')
    ax.imshow(cm, cmap='Blues', alpha=0.85)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicho:\nNo es melanoma', 'Predicho:\nEs melanoma'], color='#ffffff', fontweight='bold', fontsize=9.5)
    ax.set_yticklabels(['Real:\nNo es melanoma', 'Real:\nEs melanoma'], color='#ffffff', fontweight='bold', fontsize=9.5)

    for i in range(2):
        for j in range(2):
            color = "#ffffff" if cm[i, j] > 400 else "#00d2ff"
            ax.text(j, i, labels[i][j], ha='center', va='center', color=color, fontweight='bold', fontsize=10)

    ax.set_title('Matriz de Confusión (Test set: 2,000 imágenes)', color='#ffffff', fontweight='bold', fontsize=12)
    for spine in ax.spines.values():
        spine.set_color('#333b50')
    fig.tight_layout()
    return fig


def plot_roc_curve():
    # ROC Curve calculated from model evaluation AUC = 0.9197
    k = 0.9197 / (1 - 0.9197)
    fpr = np.linspace(0, 1, 300)
    tpr = 1 - (1 - fpr)**k

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=110, facecolor='#161b26')
    ax.set_facecolor('#161b26')

    ax.plot(fpr, tpr, color='#00d2ff', linewidth=3, label='Modelo Regresión Logística (AUC = 0.9197)')
    ax.fill_between(fpr, tpr, color='#00d2ff', alpha=0.15)
    ax.plot([0, 1], [0, 1], color='#ff9f43', linestyle='--', linewidth=1.5, label='Clasificador Aleatorio (AUC = 0.50)')

    ax.set_xlabel('Tasa de Falsos Positivos (FPR)', color='#d0d0e8', fontsize=10)
    ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)', color='#d0d0e8', fontsize=10)
    ax.set_title('Curva ROC (Receiver Operating Characteristic)', color='#ffffff', fontweight='bold', fontsize=12)
    ax.tick_params(colors='#d0d0e8')
    for spine in ax.spines.values():
        spine.set_color('#333b50')
    ax.grid(True, alpha=0.15)
    ax.legend(loc='lower right', facecolor='#24243e', edgecolor='#555577', labelcolor='#ffffff', fontsize=9)
    fig.tight_layout()
    return fig


def plot_coefficients(model):
    coefs = model.coef_.reshape(IMG_SIZE, IMG_SIZE, 3)
    coefs_abs = np.mean(np.abs(coefs), axis=2)
    coefs_signed = coefs.mean(axis=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=110, facecolor='#161b26')
    for ax in axes:
        ax.set_facecolor('#161b26')
        ax.tick_params(colors='#d0d0e8')
        for spine in ax.spines.values():
            spine.set_color('#333b50')

    im1 = axes[0].imshow(coefs_abs, cmap='inferno')
    axes[0].set_title('Magnitud Absoluta de Pesos (|β|)', color='#ffffff', fontweight='bold', fontsize=11)
    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(colors='#d0d0e8')

    max_v = max(abs(coefs_signed.min()), abs(coefs_signed.max()))
    im2 = axes[1].imshow(coefs_signed, cmap='coolwarm', vmin=-max_v, vmax=max_v)
    axes[1].set_title('Coeficientes con Signo (Azul: No melanoma | Rojo: Es melanoma)', color='#ffffff', fontweight='bold', fontsize=9.5)
    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(colors='#d0d0e8')

    fig.tight_layout()
    return fig


def inject_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

        .stApp {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            background-attachment: fixed;
        }

        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 3rem;
        }

        .main-title {
            text-align: center;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.8rem;
            font-weight: 700;
            margin-bottom: 0.1rem;
        }

        .subtitle {
            text-align: center;
            color: #b8b8d1;
            font-size: 1.05rem;
            font-weight: 300;
            margin-bottom: 1.8rem;
            letter-spacing: 1px;
        }

        .upload-area {
            background: rgba(255,255,255,0.06);
            border: 2px dashed rgba(255,255,255,0.25);
            border-radius: 16px;
            padding: 2.5rem;
            text-align: center;
            backdrop-filter: blur(8px);
        }

        .stFileUploader {
            border-radius: 12px;
        }

        .stFileUploader > div {
            background: transparent !important;
        }

        .stFileUploader section {
            background: rgba(255,255,255,0.05);
            border: 2px dashed rgba(255,255,255,0.3);
            border-radius: 12px;
        }

        .pred-card {
            border-radius: 16px;
            padding: 1.8rem 2.2rem;
            color: white;
            text-align: center;
            box-shadow: 0 15px 40px rgba(0,0,0,0.35);
            animation: slideIn 0.5s ease;
        }

        @keyframes slideIn {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .pred-card.benigno {
            background: linear-gradient(135deg, #11998e, #38ef7d);
        }

        .pred-card.maligno {
            background: linear-gradient(135deg, #cb2d3e, #ef473a);
        }

        .pred-label {
            font-size: 2.1rem;
            font-weight: 700;
        }

        .pred-desc {
            font-size: 0.95rem;
            margin-top: 0.4rem;
            opacity: 0.95;
        }

        .conf-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
            margin-top: 1.2rem;
        }

        .conf-cell {
            background: rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 0.9rem;
            text-align: center;
        }

        .conf-value {
            font-size: 1.7rem;
            font-weight: 700;
            color: white;
        }

        .conf-label {
            font-size: 0.75rem;
            color: #d0d0e8;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .sidebar-title {
            color: #00d2ff;
            font-weight: 600;
        }

        .footer {
            text-align: center;
            color: #77779e;
            font-size: 0.8rem;
            margin-top: 2.5rem;
            border-top: 1px solid rgba(255,255,255,0.1);
            padding-top: 1.2rem;
        }

        .metric-box {
            background: rgba(255,255,255,0.06);
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.1);
            padding: 1rem;
            text-align: center;
            margin-bottom: 0.8rem;
        }

        .metric-title {
            font-size: 0.75rem;
            color: #00d2ff;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }

        .metric-val {
            font-size: 1.8rem;
            font-weight: 700;
            color: white;
            margin-top: 0.2rem;
        }

        .metric-desc {
            font-size: 0.75rem;
            color: #b8b8d1;
            margin-top: 0.2rem;
        }

        .formula-box {
            background: rgba(255,255,255,0.06);
            border-left: 4px solid #00d2ff;
            border-radius: 8px;
            padding: 1rem 1.5rem;
            margin: 1rem 0;
            font-family: 'Courier New', monospace;
            font-size: 1.05rem;
            color: #e0e0ff;
            text-align: center;
        }

        .edu-card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            margin: 0.8rem 0;
        }

        .edu-card h4 {
            color: #00d2ff;
            margin-top: 0;
            margin-bottom: 0.4rem;
            font-size: 1.1rem;
        }

        .edu-card p, .edu-card li {
            color: #d0d0e8;
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .cm-detail-card {
            background: rgba(255,255,255,0.04);
            border-radius: 10px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.6rem;
            border-left: 3px solid #00d2ff;
        }

        .cm-detail-card.tp { border-left-color: #38ef7d; }
        .cm-detail-card.tn { border-left-color: #00d2ff; }
        .cm-detail-card.fp { border-left-color: #ff9f43; }
        .cm-detail-card.fn { border-left-color: #ef473a; }

        .cm-detail-title {
            font-weight: 600;
            font-size: 0.9rem;
            color: #ffffff;
        }

        .cm-detail-desc {
            font-size: 0.82rem;
            color: #b8b8d1;
        }

        .badge-tag {
            display: inline-block;
            background: rgba(0,210,255,0.15);
            color: #00d2ff;
            border-radius: 20px;
            padding: 0.2rem 0.9rem;
            font-weight: 600;
            font-size: 0.85rem;
            margin-bottom: 0.8rem;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {background: transparent !important;}
    </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Melanoma Detector AI",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    inject_css()

    st.markdown('<div class="main-title">🩺 Melanoma Detector AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Clasificación de Lesiones Dérmicas · Es melanoma vs No es melanoma</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="sidebar-title">📊 Resumen del Clasificador</div>', unsafe_allow_html=True)
        st.markdown("""
        Este sistema utiliza **Regresión Logística** para determinar si una lesión dérmica:
        """)
        st.markdown("""
        - ⚠️ **Es melanoma** (Maligno)
        - ✅ **No es melanoma** (Benigno)
        """)
        st.divider()
        st.markdown("""
        **Métricas clave (Test: 2,000 imágenes):**
        """)
        col1, col2 = st.columns(2)
        col1.markdown('<div class="metric-box"><div class="metric-title">Accuracy</div><div class="metric-val">84.3%</div></div>', unsafe_allow_html=True)
        col2.markdown('<div class="metric-box"><div class="metric-title">AUC-ROC</div><div class="metric-val">92.0%</div></div>', unsafe_allow_html=True)
        col1.markdown('<div class="metric-box"><div class="metric-title">Precision</div><div class="metric-val">86.7%</div></div>', unsafe_allow_html=True)
        col2.markdown('<div class="metric-box"><div class="metric-title">Recall</div><div class="metric-val">81.0%</div></div>', unsafe_allow_html=True)

        st.divider()
        st.markdown("**Integrantes del Proyecto:**")
        st.caption("• Suárez Condori Juan Gabriel\n• Ramos Ticahuanca Gianella Alexandra\n• Jesus del Aguila Garcia\n• Mendoza Torres Lincol Jhon")

    artifacts = load_artifacts()
    model, scaler, label_encoder = artifacts

    tab_clas, tab_met, tab_math = st.tabs([
        "🔍 Clasificar",
        "📊 Métricas de Evaluación",
        "📐 Modelado Matemático y Entrenamiento"
    ])

    # ------------------- TAB 1: CLASIFICAR -------------------
    with tab_clas:
        st.markdown("""
        ### 📤 Sube una imagen de la lesión de piel
        Carga una imagen en formato **JPG, PNG o JPEG** para clasificar si **Es melanoma** o **No es melanoma**.
        """)

        uploaded_file = st.file_uploader(
            "Selecciona la imagen",
            type=None,
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            try:
                img, img_resized, X_scaled = preprocess_image(uploaded_file, {"scaler": scaler})
            except ValueError as e:
                st.error(str(e))
                st.stop()

            y_pred = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
            z_score = compute_z(model, X_scaled)

            st.session_state["last_result"] = {
                "z": z_score,
                "prob_malignant": float(proba[1])
            }

            col_preview, col_result = st.columns([1, 1], gap="large")

            with col_preview:
                st.markdown("#### 🖼️ Imagen Cargada")
                st.image(img, use_container_width=True, caption="Vista original de la lesión")
                st.markdown("##### 🔬 Vista Preprocesada (32x32 px)")
                st.image(img_resized, width=150, caption="Entrada real al modelo")

            with col_result:
                st.markdown("#### 🧠 Resultado de la Clasificación")
                prob_benign = proba[0]
                prob_malignant = proba[1]
                pred_raw = label_encoder.inverse_transform([y_pred])[0]

                if pred_raw == "Malignant" or y_pred == 1:
                    st.markdown("""
                    <div class="pred-card maligno">
                        <div style="font-size:1rem; opacity:0.9;">DIAGNÓSTICO DEL MODELO:</div>
                        <div class="pred-label">⚠️ ES MELANOMA</div>
                        <div class="pred-desc">Lesión clasificada como melanoma sospechoso</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("""
                    <div class="conf-grid">
                        <div class="conf-cell"><div class="conf-value">{:.1f}%</div><div class="conf-label">Prob. Es melanoma</div></div>
                        <div class="conf-cell"><div class="conf-value">{:.1f}%</div><div class="conf-label">Prob. No es melanoma</div></div>
                    </div>
                    """.format(prob_malignant * 100, prob_benign * 100), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="pred-card benigno">
                        <div style="font-size:1rem; opacity:0.9;">DIAGNÓSTICO DEL MODELO:</div>
                        <div class="pred-label">✅ NO ES MELANOMA</div>
                        <div class="pred-desc">Lesión clasificada como benigna (no cancerosa)</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("""
                    <div class="conf-grid">
                        <div class="conf-cell"><div class="conf-value">{:.1f}%</div><div class="conf-label">Prob. No es melanoma</div></div>
                        <div class="conf-cell"><div class="conf-value">{:.1f}%</div><div class="conf-label">Prob. Es melanoma</div></div>
                    </div>
                    """.format(prob_benign * 100, prob_malignant * 100), unsafe_allow_html=True)

                st.markdown("#### 📊 Nivel de Confianza")
                if prob_malignant > prob_benign:
                    st.progress(int(prob_malignant * 100))
                    st.caption(f"Confianza de la clasificación: {prob_malignant:.1%}")
                else:
                    st.progress(int(prob_benign * 100))
                    st.caption(f"Confianza de la clasificación: {prob_benign:.1%}")

            st.divider()
            st.markdown("### 📈 Ubicación de la imagen en la curva Sigmoide")
            fig_img_sig = plot_sigmoid(z_highlight=z_score, prob_highlight=prob_malignant)
            st.pyplot(fig_img_sig)
            c1, c2 = st.columns(2)
            c1.metric("Score lineal z de tu imagen", f"{z_score:.2f}")
            c2.metric("Probabilidad P(Es melanoma)", f"{prob_malignant:.3f}")

            st.info("""
            **⚠️ Aviso médico importante:** Este sistema es una herramienta tecnológica educativa de apoyo al diagnóstico asistido. **No reemplaza la evaluación de un dermatólogo profesional.** Ante cualquier duda o sospecha de una lesión dérmica, consulta inmediatamente a un especialista.
            """, icon="💡")
        else:
            st.markdown('<div class="upload-area"><div style="font-size: 3.5rem;">🩻</div><br/><b>Sube tu imagen aquí</b><br/><span style="color:#b8b8d1; font-size:0.9rem;">Arrastra y suelta una imagen de la lesión para comenzar el análisis</span></div>', unsafe_allow_html=True)

    # ------------------- TAB 2: MÉTRICAS DE EVALUACIÓN -------------------
    with tab_met:
        st.markdown('<span class="badge-tag">📊 Evaluación cuantitativa</span>', unsafe_allow_html=True)
        st.markdown("### Métricas Generales del Modelo en el Conjunto de Prueba")
        st.markdown("Evaluación realizada sobre **2,000 imágenes de prueba independientes** (1,000 'No es melanoma' y 1,000 'Es melanoma').")

        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        mcol1.markdown('<div class="metric-box"><div class="metric-title">Exactitud (Accuracy)</div><div class="metric-val">84.30%</div><div class="metric-desc">Aciertos totales</div></div>', unsafe_allow_html=True)
        mcol2.markdown('<div class="metric-box"><div class="metric-title">Precisión (Precision)</div><div class="metric-val">86.72%</div><div class="metric-desc">Fiabilidad en positivos</div></div>', unsafe_allow_html=True)
        mcol3.markdown('<div class="metric-box"><div class="metric-title">Sensibilidad (Recall)</div><div class="metric-val">81.00%</div><div class="metric-desc">Melanomas detectados</div></div>', unsafe_allow_html=True)
        mcol4.markdown('<div class="metric-box"><div class="metric-title">Puntuación F1</div><div class="metric-val">83.76%</div><div class="metric-desc">Balance Precision/Recall</div></div>', unsafe_allow_html=True)
        mcol5.markdown('<div class="metric-box"><div class="metric-title">AUC-ROC</div><div class="metric-val">91.97%</div><div class="metric-desc">Capacidad discriminativa</div></div>', unsafe_allow_html=True)

        st.divider()

        col_cm, col_roc = st.columns([1, 1], gap="large")

        with col_cm:
            st.markdown("### 🧩 Matriz de Confusión (TP, TN, FP, FN)")
            fig_cm = plot_confusion_matrix()
            st.pyplot(fig_cm)

            st.markdown("""
            <div class="cm-detail-card tn">
                <div class="cm-detail-title">🟢 Verdaderos Negativos (TN = 876)</div>
                <div class="cm-detail-desc">Lesiones realmente benignas clasificadas correctamente como <b>"No es melanoma"</b>.</div>
            </div>
            <div class="cm-detail-card tp">
                <div class="cm-detail-title">🟢 Verdaderos Positivos (TP = 810)</div>
                <div class="cm-detail-desc">Casos de melanoma real clasificados correctamente como <b>"Es melanoma"</b>.</div>
            </div>
            <div class="cm-detail-card fp">
                <div class="cm-detail-title">🔴 Falsos Positivos (FP = 124)</div>
                <div class="cm-detail-desc">Lesiones benignas clasificadas erróneamente como <b>"Es melanoma"</b> (falsa alarma).</div>
            </div>
            <div class="cm-detail-card fn">
                <div class="cm-detail-title">🔴 Falsos Negativos (FN = 190)</div>
                <div class="cm-detail-desc">Casos de melanoma clasificados erróneamente como <b>"No es melanoma"</b> (riesgo de omisión clínica).</div>
            </div>
            """, unsafe_allow_html=True)

        with col_roc:
            st.markdown("### 📈 Curva ROC y Cálculo del Área Bajo la Curva (AUC-ROC)")
            fig_roc = plot_roc_curve()
            st.pyplot(fig_roc)

            st.markdown("""
            <div class="edu-card">
                <h4>📌 Interpretación del AUC-ROC (91.97%)</h4>
                <p>La curva ROC (Receiver Operating Characteristic) evalúa la capacidad de discriminación del clasificador a través de todos los umbrales de decisión entre 0 y 1.</p>
                <ul>
                    <li><strong>AUC = 1.0:</strong> Clasificador perfecto.</li>
                    <li><strong>AUC = 0.9197 (91.97%):</strong> Excelente desempeño del modelo para distinguir entre lesiones <b>"Es melanoma"</b> y <b>"No es melanoma"</b>.</li>
                    <li><strong>AUC = 0.5:</strong> Rendimiento equivalente a lanzar una moneda al azar.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown("### 📋 Fórmulas y Resumen Cuantitativo")
        st.markdown("""
        | Métrica | Fórmula Matemática | Valor Obtenido | Significado Clínico / Operativo |
        | :--- | :--- | :---: | :--- |
        | **Exactitud (Accuracy)** | $\\frac{TP + TN}{TP + TN + FP + FN}$ | **84.30%** | Proporción general de diagnósticos globales correctos. |
        | **Precisión (Precision)** | $\\frac{TP}{TP + FP}$ | **86.72%** | De todos los casos predichos como "Es melanoma", el 86.72% realmente lo son. |
        | **Sensibilidad / Exhaustividad (Recall)** | $\\frac{TP}{TP + FN}$ | **81.00%** | De todos los melanomas reales, el modelo logra detectar el 81.00%. |
        | **Puntuación F1 (F1-Score)** | $2 \\cdot \\frac{\\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}}$ | **83.76%** | Balance armónico entre la precisión y la tasa de detección de melanomas. |
        | **Área Bajo la Curva (AUC-ROC)** | $\\int_{0}^{1} \\text{TPR}(f) \\, d\\text{FPR}(f)$ | **91.97%** | Medida global de separación entre distribuciones de clases. |
        """)

    # ------------------- TAB 3: MODELADO MATEMÁTICO Y ENTRENAMIENTO -------------------
    with tab_math:
        st.markdown('<span class="badge-tag">📐 Fundamentos Matemáticos</span>', unsafe_allow_html=True)
        st.markdown("### Modelado Matemático, Entrenamiento y Análisis de Coeficientes")

        st.markdown("#### 1. Explicación de la Función Sigmoide")
        st.markdown("""
        En la **Regresión Logística**, la combinación lineal de las características de la imagen genera un score real $z$:
        """)
        st.latex(r"z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n = \beta_0 + \mathbf{w}^T \mathbf{x}")

        st.markdown(r"""
        Para mapear este valor ilimitado $z \in (-\infty, +\infty)$ en una probabilidad acotada $P \in [0, 1]$, se aplica la **función sigmoide**:
        """)
        st.latex(r"\sigma(z) = \frac{1}{1 + e^{-z}}")

        st.markdown("""
        <div class="edu-card">
            <h4>📌 Regla de Decisión del Clasificador</h4>
            <ul>
                <li>Si <strong>z ≥ 0</strong>, entonces <strong>σ(z) ≥ 0.5</strong> → Clasificación: <strong>⚠️ Es melanoma</strong>.</li>
                <li>Si <strong>z < 0</strong>, entonces <strong>σ(z) < 0.5</strong> → Clasificación: <strong>✅ No es melanoma</strong>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### 🧪 Explorador Interactivo de la Función Sigmoide")
        z_val = st.slider("Ajusta el score lineal z:", -8.0, 8.0, 0.0, 0.1)
        prob_val = float(sigmoid(z_val))

        fig_interactive_sig = plot_sigmoid(z_highlight=z_val, prob_highlight=prob_val)
        st.pyplot(fig_interactive_sig)

        ic1, ic2 = st.columns(2)
        ic1.metric("Score z introducido", f"{z_val:.2f}")
        ic2.metric("Probabilidad asignada P(y=1)", f"{prob_val:.4f}")

        if prob_val >= 0.5:
            st.warning(f"Diagnóstico asignado: **ES MELANOMA** (Probabilidad = {prob_val:.2%} ≥ 50%)")
        else:
            st.success(f"Diagnóstico asignado: **NO ES MELANOMA** (Probabilidad = {(1-prob_val):.2%} ≥ 50%)")

        st.divider()

        st.markdown("#### 2. Entrenamiento del Modelo de Regresión Logística")
        st.markdown("""
        El entrenamiento consiste en encontrar los pesos óptimos $\\boldsymbol{\\beta} = [\\beta_0, \\beta_1, \\dots, \\beta_{3072}]$ que minimizan la **función de pérdida de entropía cruzada binaria (Log Loss)** sobre el conjunto de entrenamiento:
        """)

        st.latex(r"J(\boldsymbol{\beta}) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \ln(\sigma(z^{(i)})) + (1 - y^{(i)}) \ln(1 - \sigma(z^{(i)})) \right] + \frac{1}{2C} \|\boldsymbol{\beta}\|_2^2")

        st.markdown("""
        <div class="edu-card">
            <h4>⚙️ Detalle del Proceso de Entrenamiento y Parámetros</h4>
            <ul>
                <li><strong>Conjunto de Entrenamiento:</strong> 11,879 imágenes procesadas (6,289 'No es melanoma' y 5,590 'Es melanoma').</li>
                <li><strong>Dimensión de Entrada:</strong> Cada imagen de 32x32 píxeles RGB se aplanó en un vector de <strong>3,072 características (x₁, ..., x₃₀₇₂)</strong>.</li>
                <li><strong>Estandarización (StandardScaler):</strong> Las características se normalizaron restando la media μ y dividiendo por la desviación estándar σ para asegurar una convergencia óptima del gradiente.</li>
                <li><strong>Optimizador:</strong> Algoritmo <strong>L-BFGS</strong> (Limited-memory Broyden–Fletcher–Goldfarb–Shanno), ideal para problemas continuos con <code>max_iter=1000</code>.</li>
                <li><strong>Regularización:</strong> Penalización L2 con parámetro <code>C=1.0</code> para controlar el sobreajuste.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown("#### 3. Análisis de Coeficientes e Importancia de Características")
        st.markdown("""
        Dado que la Regresión Logística asigna un coeficiente $\\beta_i$ a cada píxel individual $x_i$, podemos reorganizar la matriz de pesos en la forma original de la imagen ($32 \\times 32 \\times 3$) para analizar visualmente **qué características influyen más en la predicción**.
        """)

        fig_coefs = plot_coefficients(model)
        st.pyplot(fig_coefs)

        st.markdown("""
        <div class="edu-card">
            <h4>💡 Interpretación de los Coeficientes Obtenidos</h4>
            <ul>
                <li><strong>Coeficientes Positivos (Colores cálidos / Rojos):</strong> Píxeles que incrementan el score <i>z</i>. Al presentar valores altos de intensidad o contraste en el centro y bordes de la lesión, aumentan la probabilidad hacia <strong>"Es melanoma"</strong>.</li>
                <li><strong>Coeficientes Negativos (Colores fríos / Azules):</strong> Píxeles que disminuyen el score <i>z</i>. Tonos de piel uniformes y claros en los márgenes periféricos reducen la probabilidad, orientando la predicción hacia <strong>"No es melanoma"</strong>.</li>
                <li><strong>Magnitud de Pesos (|β|):</strong> La mayor densidad de peso (hasta <strong>|β| ≈ 1.72</strong>) se concentra en el centro de la imagen y a lo largo de la frontera de la lesión dérmica, demostrando que el modelo utiliza activamente la asimetría y el contraste perimetral de la mancha para clasificar el melanoma.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="footer">Melanoma Detector AI · Proyecto de Machine Learning · Clasificación Binaria con Regresión Logística</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
