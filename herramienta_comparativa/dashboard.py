"""
Dashboard de Comparación de Arquitecturas para Detección de Defectos Industriales
TFG 2025-26 - Vision Transformers vs CNNs

Este dashboard documenta el histórico completo de experimentación y permite
comparar las diferentes arquitecturas bajo condiciones equivalentes.
"""

import streamlit as st
import json
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from pathlib import Path

# --- CONFIGURACIÓN ---
st.set_page_config(
    layout="wide", 
    page_title="TFG - Comparativa Arquitecturas",
    page_icon="🔬",
    initial_sidebar_state="expanded"
)

# Ruta base de la herramienta (herramienta_comparativa/) - permite export completo e independiente
TOOL_ROOT = Path(__file__).parent
DATA_PATH = TOOL_ROOT / "data"
METADATA_FILE = DATA_PATH / "experiments_metadata.json"

# Raíz de datos empaquetados para la vista «Visualizaciones» (predicciones, GT, raw/)
VISUALIZATION_DATA_ROOT = DATA_PATH / "images_selected_for_visualize"

# --- ESTILOS CSS ---
st.markdown("""
<style>
    /* Cards de métricas */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-card h2 { margin: 0; font-size: 2.5em; }
    .metric-card p { margin: 5px 0 0 0; opacity: 0.9; }
    
    /* Cards de fase */
    .phase-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .phase-card.fase1 { border-color: #3498db; }
    .phase-card.fase2 { border-color: #e74c3c; }
    .phase-card.fase3 { border-color: #2ecc71; }
    
    /* Info boxes */
    .info-box {
        background: #e8f4f8;
        border-left: 4px solid #17a2b8;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
    
    /* Architecture cards */
    .arch-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 10px 0;
    }
    .arch-card h4 {
        margin-top: 0;
        color: #495057;
    }
    
    /* Tabs mejorados */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { 
        height: 50px; 
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 0 20px;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #667eea !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNCIONES DE CARGA DE DATOS ---

@st.cache_data
def load_metadata():
    """Carga los metadatos de experimentos"""
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def load_experiment_results(exp_path):
    """Carga los resultados de evaluación de un experimento"""
    json_files = ["test_evaluation_results_comparable.json", "test_evaluation_results.json"]
    for jf in json_files:
        full_path = DATA_PATH / exp_path / jf
        if full_path.exists():
            with open(full_path, 'r') as f:
                return json.load(f)
    return None

@st.cache_data
def load_threshold_analysis(exp_path):
    """Carga los resultados de análisis de thresholds si están disponibles"""
    threshold_dir = DATA_PATH / exp_path / "threshold_analysis"
    if not threshold_dir.exists():
        return None
    
    results = {}
    for threshold in [0.75, 0.90]:
        json_file = threshold_dir / f"test_evaluation_results_comparable_th{threshold:.2f}.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                results[threshold] = json.load(f)
    
    return results if results else None

@st.cache_data
def load_training_history(exp_path):
    """Carga el historial de entrenamiento"""
    history_path = DATA_PATH / exp_path / "training_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            return json.load(f)
    return None

def get_best_epoch_info(exp_path, criterion="val_loss"):
    """Obtiene información sobre el mejor epoch"""
    history = load_training_history(exp_path)
    if not history:
        return None
    
    if criterion == "val_loss":
        # Encontrar época con menor val_loss
        best_epoch = min(history, key=lambda x: x.get('val_loss', float('inf')))
        return {
            "epoch": best_epoch.get('epoch', 'N/A'),
            "val_loss": best_epoch.get('val_loss', 'N/A'),
            "criterion": "Menor pérdida de validación"
        }
    return None

def get_training_images(exp_path):
    """Obtiene las imágenes de métricas de entrenamiento"""
    metrics_path = DATA_PATH / exp_path / "training_metrics"
    if not metrics_path.exists():
        return {}
    
    mapping = {}
    for f in sorted(metrics_path.iterdir()):
        if f.suffix.lower() in ['.png', '.jpg']:
            name_lower = f.name.lower()
            if "total_loss" in name_lower or "loss_total" in name_lower:
                mapping["Total Loss"] = f
            elif "classifier" in name_lower or "classification" in name_lower:
                mapping["Classification Loss"] = f
            elif "box_reg" in name_lower or "bbox" in name_lower:
                mapping["Box Regression Loss"] = f
            elif "learning_rate" in name_lower or "lr" in name_lower:
                mapping["Learning Rate"] = f
            elif "components" in name_lower:
                mapping["Loss Components"] = f
            elif "metrics" in name_lower or "training" in name_lower:
                mapping["Training Metrics"] = f
            elif "map" in name_lower:
                mapping["Validation mAP"] = f
    return mapping

def clean_class_name(name):
    """Normaliza nombres de clases"""
    if name.upper() in ["CLASS_6", "NORMAL"]:
        return "NORMAL"
    return name.upper()

def get_visualization_images(exp_path, num_images=30):
    """Obtiene las imágenes de visualización de un experimento."""
    vis_path = DATA_PATH / exp_path / "visualizations_test"
    if not vis_path.exists():
        return []
    
    # Obtener todas las imágenes PNG
    images = sorted(vis_path.glob("*.png"))
    return images[:num_images]

def get_all_results_df(metadata, include_thresholds=False):
    """Crea un DataFrame con todos los resultados"""
    rows = []
    for phase_id, experiments in metadata["experiments"].items():
        for exp_id, exp_info in experiments.items():
            results = load_experiment_results(exp_info["path"])
            if results:
                score_threshold = results.get("score_threshold", 0.15)
                
                row = {
                    "ID": exp_id,
                    "Nombre": exp_info["name"],
                    "Arquitectura": exp_info["architecture"],
                    "Resolución": exp_info["resolution"],
                    "Épocas": exp_info["epochs"],
                    "Fase": exp_info["phase"],
                    "Score Threshold": score_threshold,
                    "mAP": results.get("mAP", 0),
                    "is_best": exp_info.get("is_best", False),
                    "path": exp_info["path"]
                }
                # Añadir AP por clase
                for cls, val in results.get("AP_per_class", {}).items():
                    row[f"AP_{clean_class_name(cls)}"] = val
                # Añadir Precision por clase
                for cls, val in results.get("precision_per_class", {}).items():
                    row[f"Precision_{clean_class_name(cls)}"] = val
                # Añadir Recall por clase
                for cls, val in results.get("recall_per_class", {}).items():
                    row[f"Recall_{clean_class_name(cls)}"] = val
                rows.append(row)
                
                # Si incluir thresholds y el modelo tiene análisis de thresholds
                if include_thresholds and exp_info.get("threshold_analysis", {}).get("available", False):
                    threshold_results = load_threshold_analysis(exp_info["path"])
                    if threshold_results:
                        for th, th_results in threshold_results.items():
                            row_th = {
                                "ID": f"{exp_id}_th{th:.2f}",
                                "Nombre": exp_info['name'],
                                "Arquitectura": exp_info["architecture"],
                                "Resolución": exp_info["resolution"],
                                "Épocas": exp_info["epochs"],
                                "Fase": exp_info["phase"],
                                "Score Threshold": th,
                                "mAP": th_results.get("mAP", 0),
                                "is_best": False,
                                "path": exp_info["path"]
                            }
                            # Añadir AP por clase
                            for cls, val in th_results.get("AP_per_class", {}).items():
                                row_th[f"AP_{clean_class_name(cls)}"] = val
                            # Añadir Precision por clase
                            for cls, val in th_results.get("precision_per_class", {}).items():
                                row_th[f"Precision_{clean_class_name(cls)}"] = val
                            # Añadir Recall por clase
                            for cls, val in th_results.get("recall_per_class", {}).items():
                                row_th[f"Recall_{clean_class_name(cls)}"] = val
                            rows.append(row_th)
    return pd.DataFrame(rows)

# --- VISTAS DEL DASHBOARD ---

def render_home(metadata):
    """Vista 1: Inicio - Contexto del Proyecto"""
    st.title("🔬 Detección de Defectos Industriales")
    st.markdown("### Comparativa de Arquitecturas: Vision Transformers vs CNNs")
    
    st.markdown("---")
    
    # Resumen ejecutivo
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>0.785</h2>
            <p>mAP Mejor Modelo (DEIMv2)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
            <h2>8</h2>
            <p>Experimentos Realizados</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);">
            <h2>3</h2>
            <p>Arquitecturas Evaluadas</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Contexto de la investigación
    st.markdown("### 📋 Contexto de la Investigación")
    
    st.markdown("""
    Este proyecto de investigación evalúa y compara diferentes arquitecturas de deep learning para la 
    **detección de defectos en componentes industriales**. El objetivo principal es determinar qué tipo 
    de arquitectura ofrece mejor rendimiento para este problema específico de visión por computador.
    """)
    
    st.markdown("""
    <div class="info-box">
    <strong>Dataset utilizado:</strong> Conjunto curado de imágenes industriales con 6 tipos de defectos:
    NORMAL (sin defectos), DEFORMACIONES, ROTURA/FRACTURA, RAYONES/ARAÑAZOS, PERFORACIONES y CONTAMINACIÓN.
    El dataset presenta alta variabilidad en iluminación, escalas y tipos de superficies.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Metodología de experimentación
    st.markdown("### 🔬 Metodología de Experimentación")
    
    st.markdown("""
    La experimentación se ha dividido en **4 fases principales**:
    
    1. **Fase 1 (Octubre 2025)**: Establecer líneas base con arquitecturas CNN clásicas (ResNet-18, EfficientNet-B0)
    2. **Fase 2 (Noviembre 2025)**: Explorar Vision Transformers (DEIMv2) con diferentes configuraciones
    3. **Fase 3 (Diciembre 2025)**: Validar resultados entrenando CNNs con la misma resolución que los ViTs
    4. **Fase 4 (Diciembre 2025)**: Validación de robustez del mejor modelo con score thresholds altos
    
    Para cada experimento, el **mejor checkpoint** se selecciona según:
    - **CNNs (ResNet/EfficientNet)**: Menor pérdida de validación (val_loss)
    - **ViTs (DEIMv2)**: Mayor mAP@0.5 en el conjunto de validación
    """)
    
    st.markdown("---")
    
    # Arquitecturas evaluadas
    st.markdown("### 🏗️ Arquitecturas Evaluadas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="arch-card">
        <h4>🔵 ResNet-18</h4>
        <p><strong>Tipo:</strong> CNN (Red Neuronal Convolucional)</p>
        <p><strong>Año:</strong> 2015 (Microsoft Research)</p>
        <p><strong>Características:</strong></p>
        <ul>
            <li>18 capas con conexiones residuales</li>
            <li>Bias inductivo fuerte (localidad espacial)</li>
            <li>Convergencia rápida (~50 epochs)</li>
            <li>11M parámetros</li>
        </ul>
        <p><strong>Detector:</strong> Faster R-CNN</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="arch-card">
        <h4>🟣 EfficientNet-B0</h4>
        <p><strong>Tipo:</strong> CNN con escalado compuesto</p>
        <p><strong>Año:</strong> 2019 (Google Brain)</p>
        <p><strong>Características:</strong></p>
        <ul>
            <li>Escalado balanceado de profundidad/anchura</li>
            <li>Optimizada para resoluciones 224-380px</li>
            <li>Muy eficiente en parámetros</li>
            <li>5M parámetros</li>
        </ul>
        <p><strong>Detector:</strong> Faster R-CNN</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="arch-card" style="border: 2px solid #e74c3c;">
        <h4>🔴 DEIMv2 (Vision Transformer)</h4>
        <p><strong>Tipo:</strong> Transformer para detección en tiempo real</p>
        <p><strong>Año:</strong> 2024-2025</p>
        <p><strong>Características:</strong></p>
        <ul>
            <li>Backbone DINOv3 (ViT preentrenado)</li>
            <li>Atención global desde el inicio</li>
            <li>Convergencia lenta (~150-200 epochs)</li>
            <li>~17M parámetros</li>
        </ul>
        <p><strong>Detector:</strong> DEIM Decoder</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sección especial sobre DEIMv2
    st.markdown("### 🎯 Enfoque en DEIMv2: Estado del Arte en Detección en Tiempo Real")
    
    st.markdown("""
    **DEIMv2** es una arquitectura de detección de objetos en tiempo real que combina:
    
    - **DINOv3**: Backbone de Vision Transformer preentrenado con auto-supervisión en grandes datasets
    - **DEIM (Dense Enhanced Image Matching)**: Framework de entrenamiento optimizado para DETRs
    - **Spatial Tuning Adapter (STA)**: Convierte la salida de escala única de DINOv3 en features multi-escala
    
    Según el paper científico de DEIMv2, esta arquitectura logra resultados estado del arte en el benchmark COCO:
    - **DEIMv2-S**: Primer modelo sub-10M en superar 50 AP en COCO
    - **DEIMv2-X**: 57.8 AP con solo 50.3M parámetros
    
    La clave del éxito de los Vision Transformers en detección es su capacidad de capturar **relaciones 
    espaciales globales** desde las primeras capas, a diferencia de las CNNs que construyen el contexto 
    gradualmente a través de convoluciones locales.
    """)
    
    st.markdown("""
    <div class="success-box">
    <strong>Resultado clave:</strong> En nuestro problema de detección de defectos industriales, 
    DEIMv2 alcanzó un mAP de 0.785, superando significativamente a las arquitecturas CNN que 
    obtuvieron máximos de 0.08-0.16.
    </div>
    """, unsafe_allow_html=True)


def render_timeline(metadata):
    """Vista 2: Línea Temporal de Experimentación"""
    st.title("📜 Línea Temporal de Experimentación")
    st.markdown("Evolución cronológica del proceso de investigación")
    
    st.markdown("---")
    
    phases = metadata["phases"]
    
    for phase_id, phase_info in phases.items():
        phase_num = phase_id[-1]
        
        if phase_num == "1":
            icon, color = "🔵", "#3498db"
        elif phase_num == "2":
            icon, color = "🔴", "#e74c3c"
        elif phase_num == "3":
            icon, color = "🟢", "#2ecc71"
        else:
            icon, color = "🟡", "#f39c12"
        
        st.markdown(f"""
        <div class="phase-card" style="border-color: {color};">
            <h3>{icon} {phase_info['name']}</h3>
            <p><strong>Período:</strong> {phase_info['date_range']}</p>
            <p><strong>Descripción:</strong> {phase_info['description']}</p>
            <p><strong>Motivación:</strong> {phase_info['motivation']}</p>
            <p><strong>Conclusión:</strong> <em>{phase_info['conclusion']}</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Determinar qué experimentos mostrar
        if phase_num == "1":
            phase_key = "fase1_baseline"
        elif phase_num == "2":
            phase_key = "fase2_vit"
        elif phase_num == "3":
            phase_key = "fase3_comparacion_justa"
        else:
            phase_key = None  # Fase 4 no tiene experimentos individuales, es análisis del mejor modelo
        
        phase_experiments = metadata["experiments"].get(phase_key, {})
        
        if phase_experiments:
            cols = st.columns(len(phase_experiments))
            for idx, (exp_id, exp_info) in enumerate(phase_experiments.items()):
                with cols[idx]:
                    results = load_experiment_results(exp_info["path"])
                    mAP = results.get("mAP", 0) if results else 0
                    
                    badge = "⭐" if exp_info.get("is_best") else ""
                    st.metric(
                        label=f"{exp_info['name']} {badge}",
                        value=f"{mAP:.3f}",
                        delta=f"{exp_info['resolution']}"
                    )
        
        st.markdown("---")


def render_explorer(metadata):
    """Vista 3: Explorador de Experimentos"""
    st.title("🔬 Explorador de Experimentos")
    st.markdown("Análisis detallado de cada experimento")
    
    # Selector de experimento
    all_experiments = {}
    for phase_id, experiments in metadata["experiments"].items():
        for exp_id, exp_info in experiments.items():
            all_experiments[exp_info["name"]] = (exp_id, exp_info)
    
    selected_name = st.selectbox(
        "Selecciona un experimento:",
        options=list(all_experiments.keys()),
        index=list(all_experiments.keys()).index("DEIMv2 @ 1024px (300 epochs)") if "DEIMv2 @ 1024px (300 epochs)" in all_experiments else 0
    )
    
    exp_id, exp_info = all_experiments[selected_name]
    
    st.markdown("---")
    
    # Info del experimento
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Configuración")
        
        config_data = {
            "Arquitectura": exp_info['architecture'],
            "Backbone": exp_info['backbone'],
            "Detector": exp_info['detector'],
            "Resolución": exp_info['resolution'],
            "Épocas totales": exp_info['epochs'],
            "Batch Size": exp_info['batch_size'],
            "Learning Rate": exp_info['learning_rate'],
            "Optimizer": exp_info['optimizer']
        }
        
        for key, value in config_data.items():
            st.markdown(f"**{key}:** {value}")
        
        # Información del mejor checkpoint
        st.markdown("---")
        st.markdown("### 🏆 Mejor Checkpoint")
        
        best_epoch = exp_info.get("best_epoch")
        if best_epoch:
            st.markdown(f"**Época:** {best_epoch} de {exp_info['epochs']}")
        else:
            # Intentar obtener de training_history
            best_info = get_best_epoch_info(exp_info["path"])
            if best_info:
                st.markdown(f"**Época:** {best_info['epoch']} de {exp_info['epochs']}")
                st.markdown(f"**Val Loss:** {best_info['val_loss']:.4f}" if isinstance(best_info['val_loss'], float) else "")
        
        st.markdown(f"**Criterio:** {exp_info['best_checkpoint_criterion']}")
        
        if exp_info.get("notes"):
            st.markdown("---")
            st.markdown(f"""
            <div class="info-box">
            <strong>Notas:</strong> {exp_info['notes']}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Resultados de Evaluación")
        
        # Si el modelo tiene análisis de thresholds, permitir seleccionar threshold
        selected_threshold = None
        threshold_results_available = None
        
        if exp_info.get("threshold_analysis", {}).get("available", False):
            threshold_results_available = load_threshold_analysis(exp_info["path"])
            
            st.markdown("""
            <div class="info-box">
            <strong>⚠️ Este modelo tiene análisis de robustez disponible:</strong> Puedes seleccionar diferentes score thresholds 
            para ver cómo varían las métricas. El threshold por defecto es 0.15 (evaluación original).
            </div>
            """, unsafe_allow_html=True)
            
            threshold_options = {
                "0.15 (Original)": 0.15,
                "0.75 (Alto)": 0.75,
                "0.90 (Muy Alto)": 0.90
            }
            
            selected_threshold_label = st.selectbox(
                "🎯 Selecciona Score Threshold:",
                options=list(threshold_options.keys()),
                index=0,
                help="Threshold más bajo (0.15) incluye más detecciones. Thresholds más altos (0.75, 0.90) son más estrictos y solo incluyen detecciones de alta confianza."
            )
            selected_threshold = threshold_options[selected_threshold_label]
            
            # Cargar resultados según threshold seleccionado
            if selected_threshold == 0.15:
                results = load_experiment_results(exp_info["path"])
            else:
                if threshold_results_available and selected_threshold in threshold_results_available:
                    results = threshold_results_available[selected_threshold]
                else:
                    results = load_experiment_results(exp_info["path"])
        else:
            results = load_experiment_results(exp_info["path"])
        
        if results:
            # Mostrar threshold usado
            score_threshold = results.get('score_threshold', 'N/A')
            threshold_badge = f"🎯 Score Threshold: **{score_threshold}**" if score_threshold != 'N/A' else ""
            
            if threshold_badge:
                st.markdown(f"<div style='padding: 10px; background-color: #e8f4f8; border-radius: 5px; margin-bottom: 15px;'>{threshold_badge}</div>", unsafe_allow_html=True)
            
            # mAP principal
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("mAP@0.5", f"{results['mAP']:.4f}")
            with col_m2:
                st.metric("Imágenes Test", results.get('num_test_images', 'N/A'))
            
            # Selector de métrica
            metric_type = st.radio(
                "Selecciona métrica a visualizar:",
                ["AP (Average Precision)", "Precision", "Recall"],
                horizontal=True
            )
            
            if metric_type == "AP (Average Precision)":
                data_dict = results.get("AP_per_class", {})
                title = "Average Precision por Clase"
                color_scale = "RdYlGn"
            elif metric_type == "Precision":
                data_dict = results.get("precision_per_class", {})
                title = "Precision por Clase"
                color_scale = "Blues"
            else:
                data_dict = results.get("recall_per_class", {})
                title = "Recall por Clase"
                color_scale = "Oranges"
            
            if data_dict:
                df_metric = pd.DataFrame({
                    "Clase": [clean_class_name(k) for k in data_dict.keys()],
                    "Valor": list(data_dict.values())
                })
                
                # Añadir información de threshold al título si está disponible
                title_with_threshold = title
                if score_threshold != 'N/A':
                    title_with_threshold = f"{title} (Score Threshold: {score_threshold})"
                
                fig = px.bar(df_metric, x="Clase", y="Valor", 
                           title=title_with_threshold,
                           color="Valor",
                           color_continuous_scale=color_scale,
                           text=df_metric["Valor"].apply(lambda x: f"{x:.3f}"))
                fig.update_traces(textposition="outside")
                fig.update_layout(showlegend=False, yaxis_range=[0, 1.1])
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No se encontraron resultados de evaluación")
    
    # Análisis de thresholds (solo para el mejor modelo DEIMv2)
    if exp_info.get("threshold_analysis", {}).get("available", False) and threshold_results_available:
        st.markdown("---")
        st.markdown("### 🔬 Análisis Comparativo de Robustez (Todos los Thresholds)")
        
        threshold_results = threshold_results_available
        # Cargar resultados originales
        original_results = load_experiment_results(exp_info["path"])
        
        if threshold_results:
            st.markdown("""
            <div class="info-box">
            <strong>Validación de Robustez:</strong> Este modelo fue evaluado con score thresholds progresivamente más estrictos 
            (0.75 y 0.90) para validar que la precision perfecta observada no es un artefacto del threshold bajo. 
            Ver <strong>Fase 4</strong> en la documentación para más detalles.
            </div>
            """, unsafe_allow_html=True)
            
            # Crear DataFrame comparativo
            thresholds_data = []
            
            # Añadir threshold original
            thresholds_data.append({
                "Threshold": 0.15,
                "mAP": original_results.get("mAP", 0),
                "Precision (promedio)": np.mean(list(original_results.get("precision_per_class", {}).values())),
                "Recall (promedio)": np.mean(list(original_results.get("recall_per_class", {}).values()))
            })
            
            # Añadir thresholds altos
            for th, th_results in sorted(threshold_results.items()):
                thresholds_data.append({
                    "Threshold": th,
                    "mAP": th_results.get("mAP", 0),
                    "Precision (promedio)": np.mean(list(th_results.get("precision_per_class", {}).values())),
                    "Recall (promedio)": np.mean(list(th_results.get("recall_per_class", {}).values()))
                })
            
            df_thresholds = pd.DataFrame(thresholds_data)
            
            # Gráfico de evolución de mAP
            st.markdown("#### Evolución de mAP por Score Threshold")
            fig_map = px.line(df_thresholds, x="Threshold", y="mAP", 
                            markers=True, title="mAP vs Score Threshold",
                            labels={"Threshold": "Score Threshold", "mAP": "mAP@0.5"})
            fig_map.update_traces(line=dict(width=3), marker=dict(size=10))
            fig_map.add_hline(y=0.70, line_dash="dash", line_color="green", 
                            annotation_text="Límite de excelencia (0.70)")
            fig_map.update_layout(height=400)
            st.plotly_chart(fig_map, use_container_width=True)
            
            # Tabla comparativa
            st.markdown("#### Tabla Comparativa de Métricas")
            st.dataframe(df_thresholds.style.format({
                "Threshold": "{:.2f}",
                "mAP": "{:.4f}",
                "Precision (promedio)": "{:.4f}",
                "Recall (promedio)": "{:.4f}"
            }), use_container_width=True, hide_index=True)
            
            # Análisis por clase
            st.markdown("#### Análisis de AP por Clase")
            
            classes = list(original_results.get("AP_per_class", {}).keys())
            threshold_values = [0.15] + sorted(threshold_results.keys())
            
            ap_data = []
            for cls in classes:
                for th in threshold_values:
                    if th == 0.15:
                        ap_val = original_results.get("AP_per_class", {}).get(cls, 0)
                    else:
                        ap_val = threshold_results[th].get("AP_per_class", {}).get(cls, 0)
                    
                    ap_data.append({
                        "Clase": clean_class_name(cls),
                        "Threshold": th,
                        "AP": ap_val
                    })
            
            df_ap = pd.DataFrame(ap_data)
            fig_ap = px.bar(df_ap, x="Clase", y="AP", color="Threshold", 
                          barmode="group", title="AP por Clase y Threshold",
                          color_discrete_map={0.15: "#e74c3c", 0.75: "#3498db", 0.90: "#2ecc71"})
            fig_ap.update_layout(height=500, yaxis_range=[0, 1.1])
            st.plotly_chart(fig_ap, use_container_width=True)
            
            # Conclusiones
            st.markdown("""
            <div class="success-box">
            <strong>Conclusión del Análisis:</strong> El modelo mantiene excelente rendimiento (mAP > 0.70) 
            incluso con thresholds muy estrictos (0.90), y preserva precision perfecta (1.0) en todas las clases. 
            Esto confirma que el modelo está bien entrenado, es robusto y no muestra signos de overfitting.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Los archivos de análisis de thresholds no están disponibles")
    
    # Imágenes de entrenamiento
    st.markdown("---")
    st.markdown("### 📈 Métricas de Entrenamiento")
    images = get_training_images(exp_info["path"])
    
    if images:
        tabs = st.tabs(list(images.keys()))
        for tab, (metric_name, img_path) in zip(tabs, images.items()):
            with tab:
                st.image(Image.open(img_path), use_container_width=True)
    else:
        st.info("No hay imágenes de métricas de entrenamiento disponibles para este experimento")


def render_comparison(metadata):
    """Vista 4: Comparativa Final"""
    st.title("📊 Comparativa de Arquitecturas")
    st.markdown("Análisis comparativo entre todas las arquitecturas evaluadas")
    
    st.markdown("---")
    
    # Selector para incluir análisis de thresholds
    include_thresholds = st.checkbox(
        "🔬 Incluir análisis de thresholds (DEIMv2 con th=0.75 y th=0.90)",
        value=False,
        help="Si está marcado, se incluirán las evaluaciones del mejor modelo DEIMv2 con diferentes score thresholds en la comparativa"
    )
    
    df = get_all_results_df(metadata, include_thresholds=include_thresholds)
    
    if df.empty:
        st.error("No se encontraron datos para comparar")
        return
    
    st.markdown("---")
    
    # Selector de comparación
    comparison_type = st.radio(
        "Tipo de comparación:",
        ["📋 Todos los experimentos", "🏆 Mejores por arquitectura", "🎯 Comparación a 1024x1024"],
        horizontal=True
    )
    
    if comparison_type == "🏆 Mejores por arquitectura":
        # Mejores: ResNet-18 @ 1024, EfficientNet nativa, DEIMv2 300ep
        best_ids = ["resnet18_1024", "efficientnet_nativa", "deimv2_1024_300ep"]
        df_filtered = df[df["ID"].isin(best_ids)].copy()
        title_suffix = " (Mejores modelos por arquitectura)"
    elif comparison_type == "🎯 Comparación a 1024x1024":
        # Solo los que usan 1024x1024
        ids_1024 = ["resnet18_1024", "efficientnet_1024", "deimv2_1024_300ep"]
        df_filtered = df[df["ID"].isin(ids_1024)].copy()
        title_suffix = " (Resolución 1024x1024)"
    else:
        df_filtered = df.copy()
        title_suffix = ""
    
    st.markdown("---")
    
    # Gráfico de mAP - CORREGIDO
    st.markdown(f"### mAP Global{title_suffix}")
    
    # Añadir información de threshold al nombre si está disponible
    if "Score Threshold" in df_filtered.columns:
        df_filtered_display = df_filtered.copy()
        df_filtered_display["Nombre_Display"] = df_filtered_display.apply(
            lambda row: f"{row['Nombre']} [th={row['Score Threshold']:.2f}]" 
            if row.get("Score Threshold", 0.15) != 0.15 or "th=" in str(row['Nombre']) 
            else row['Nombre'],
            axis=1
        )
    else:
        df_filtered_display = df_filtered.copy()
        df_filtered_display["Nombre_Display"] = df_filtered_display["Nombre"]
    
    # Ordenar por mAP y crear gráfico
    df_sorted = df_filtered_display.sort_values("mAP", ascending=True).reset_index(drop=True)
    
    fig = go.Figure()
    
    # Colores por arquitectura
    color_map = {
        "ResNet-18": "#3498db",
        "EfficientNet-B0": "#9b59b6",
        "DEIMv2": "#e74c3c"
    }
    
    # Track which architectures have been shown in legend
    shown_architectures = set()
    
    for idx, row in df_sorted.iterrows():
        arch = row["Arquitectura"]
        show_in_legend = arch not in shown_architectures
        if show_in_legend:
            shown_architectures.add(arch)
        
        # Color según threshold si es DEIMv2
        marker_color = color_map.get(arch, "#gray")
        if arch == "DEIMv2" and "Score Threshold" in row:
            th = row.get("Score Threshold", 0.15)
            if th == 0.75:
                marker_color = "#c0392b"  # Rojo más oscuro
            elif th == 0.90:
                marker_color = "#8b0000"  # Rojo muy oscuro
        
        fig.add_trace(go.Bar(
            y=[row["Nombre_Display"]],
            x=[float(row["mAP"])],
            orientation='h',
            name=arch,
            marker_color=marker_color,
            text=[f"{row['mAP']:.3f}"],
            textposition="outside",
            showlegend=show_in_legend,
            hovertemplate=f"<b>{row['Nombre_Display']}</b><br>" +
                         f"mAP: {row['mAP']:.4f}<br>" +
                         (f"Score Threshold: {row.get('Score Threshold', 0.15):.2f}<br>" if "Score Threshold" in row else "") +
                         "<extra></extra>"
        ))
    
    fig.update_layout(
        height=max(300, len(df_sorted) * 50),
        xaxis_title="mAP@0.5",
        yaxis_title="",
        barmode='group',
        legend_title="Arquitectura"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Mostrar información de thresholds si están disponibles
    if "Score Threshold" in df_filtered.columns:
        st.info("💡 **Nota:** Los modelos con diferentes score thresholds se muestran con el formato [th=X.XX]. Threshold más bajo (0.15) incluye más detecciones, thresholds más altos (0.75, 0.90) son más estrictos.")
    
    # Tabs para métricas detalladas
    tab_ap, tab_prec, tab_recall, tab_table = st.tabs(["📊 AP por Clase", "🎯 Precision por Clase", "🔍 Recall por Clase", "📋 Tabla Completa"])
    
    with tab_ap:
        ap_cols = [c for c in df_filtered.columns if c.startswith("AP_")]
        if ap_cols:
            df_melted = df_filtered.melt(
                id_vars=["Nombre", "Arquitectura"],
                value_vars=ap_cols,
                var_name="Clase",
                value_name="AP"
            )
            df_melted["Clase"] = df_melted["Clase"].str.replace("AP_", "")
            
            fig = px.bar(
                df_melted,
                x="Clase",
                y="AP",
                color="Nombre",
                barmode="group",
                title="Average Precision por Clase",
                text_auto='.3f'
            )
            fig.update_layout(height=500, yaxis_range=[0, 1.1])
            st.plotly_chart(fig, use_container_width=True)
    
    with tab_prec:
        prec_cols = [c for c in df_filtered.columns if c.startswith("Precision_")]
        if prec_cols:
            df_melted = df_filtered.melt(
                id_vars=["Nombre", "Arquitectura"],
                value_vars=prec_cols,
                var_name="Clase",
                value_name="Precision"
            )
            df_melted["Clase"] = df_melted["Clase"].str.replace("Precision_", "")
            
            fig = px.bar(
                df_melted,
                x="Clase",
                y="Precision",
                color="Nombre",
                barmode="group",
                title="Precision por Clase",
                text_auto='.3f'
            )
            fig.update_layout(height=500, yaxis_range=[0, 1.1])
            st.plotly_chart(fig, use_container_width=True)
    
    with tab_recall:
        recall_cols = [c for c in df_filtered.columns if c.startswith("Recall_")]
        if recall_cols:
            df_melted = df_filtered.melt(
                id_vars=["Nombre", "Arquitectura"],
                value_vars=recall_cols,
                var_name="Clase",
                value_name="Recall"
            )
            df_melted["Clase"] = df_melted["Clase"].str.replace("Recall_", "")
            
            fig = px.bar(
                df_melted,
                x="Clase",
                y="Recall",
                color="Nombre",
                barmode="group",
                title="Recall por Clase",
                text_auto='.3f'
            )
            fig.update_layout(height=500, yaxis_range=[0, 1.1])
            st.plotly_chart(fig, use_container_width=True)
    
    with tab_table:
        display_cols = ["Nombre", "Arquitectura", "Resolución", "Épocas", "mAP"]
        if "Score Threshold" in df_filtered.columns:
            display_cols.insert(-1, "Score Threshold")
        
        # Preparar nombres para mostrar
        df_table = df_filtered[display_cols].copy()
        if "Score Threshold" in df_table.columns:
            df_table["Nombre"] = df_table.apply(
                lambda row: f"{row['Nombre']} [th={row['Score Threshold']:.2f}]" 
                if row.get("Score Threshold", 0.15) != 0.15 or "th=" in str(row['Nombre'])
                else row['Nombre'],
                axis=1
            )
        
        st.dataframe(
            df_table.sort_values("mAP", ascending=False),
            use_container_width=True,
            hide_index=True
        )


def render_conclusions(metadata):
    """Vista 5: Conclusiones"""
    st.title("📝 Conclusiones")
    st.markdown("Resumen de hallazgos del proceso de experimentación")
    
    st.markdown("---")
    
    # Tabla resumen
    st.markdown("### 📊 Tabla Resumen de Resultados")
    
    df = get_all_results_df(metadata)
    summary_ids = ["resnet18_nativa", "resnet18_1024", "efficientnet_nativa", "efficientnet_1024", 
                   "deimv2_640_87ep", "deimv2_1024_80ep", "deimv2_1024_120ep", "deimv2_1024_300ep"]
    
    df_summary = df[df["ID"].isin(summary_ids)][["Nombre", "Arquitectura", "Resolución", "Épocas", "mAP"]]
    df_summary = df_summary.sort_values("mAP", ascending=False)
    
    # Marcar el mejor de cada arquitectura
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Hallazgos principales
    st.markdown("### 🔍 Hallazgos Principales")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Impacto de la Resolución en CNNs")
        st.markdown("""
        | Modelo | Res. Nativa | Res. 1024x1024 | Cambio |
        |--------|-------------|----------------|--------|
        | ResNet-18 | 0.077 | 0.080 | **+3.9%** ✅ |
        | EfficientNet-B0 | 0.162 | 0.122 | **-24.7%** ❌ |
        
        **Observación:** ResNet-18 mejora ligeramente con mayor resolución, 
        pero EfficientNet empeora significativamente. Esto se debe a que 
        EfficientNet está optimizada para resoluciones menores (224-380px).
        """)
    
    with col2:
        st.markdown("#### Impacto de la Resolución y Épocas en ViTs")
        st.markdown("""
        | Configuración | mAP | Mejora vs anterior |
        |---------------|-----|-------------------|
        | 640px, 87ep | 0.499 | baseline |
        | 1024px, 80ep | 0.624 | +25.1% |
        | 1024px, 120ep | 0.766 | +22.8% |
        | 1024px, 300ep | 0.785 | +2.5% |
        
        **Observación:** Los ViTs se benefician enormemente de mayor resolución 
        y entrenamientos más largos. La convergencia óptima se alcanza 
        alrededor del epoch 187.
        """)
    
    st.markdown("---")
    
    # Análisis de robustez con thresholds
    st.markdown("### 🔬 Validación de Robustez (Fase 4)")
    
    # Cargar análisis de thresholds del mejor modelo
    best_deimv2 = None
    for phase_id, experiments in metadata["experiments"].items():
        for exp_id, exp_info in experiments.items():
            if exp_info.get("is_best_deimv2", False):
                best_deimv2 = exp_info
                break
        if best_deimv2:
            break
    
    if best_deimv2 and best_deimv2.get("threshold_analysis", {}).get("available", False):
        threshold_results = load_threshold_analysis(best_deimv2["path"])
        original_results = load_experiment_results(best_deimv2["path"])
        
        if threshold_results and original_results:
            st.markdown("""
            El mejor modelo DEIMv2 fue evaluado con score thresholds progresivamente más estrictos para validar 
            que la precision perfecta observada no es un artefacto del threshold bajo.
            """)
            
            # Crear datos para visualización
            thresholds_data = {
                "Threshold": [0.15, 0.75, 0.90],
                "mAP": [
                    original_results.get("mAP", 0),
                    threshold_results[0.75].get("mAP", 0),
                    threshold_results[0.90].get("mAP", 0)
                ],
                "Precision (promedio)": [
                    np.mean(list(original_results.get("precision_per_class", {}).values())),
                    np.mean(list(threshold_results[0.75].get("precision_per_class", {}).values())),
                    np.mean(list(threshold_results[0.90].get("precision_per_class", {}).values()))
                ],
                "Recall (promedio)": [
                    np.mean(list(original_results.get("recall_per_class", {}).values())),
                    np.mean(list(threshold_results[0.75].get("recall_per_class", {}).values())),
                    np.mean(list(threshold_results[0.90].get("recall_per_class", {}).values()))
                ]
            }
            
            df_robust = pd.DataFrame(thresholds_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Evolución de mAP")
                fig_map = px.line(df_robust, x="Threshold", y="mAP", markers=True,
                                title="mAP vs Score Threshold",
                                labels={"Threshold": "Score Threshold", "mAP": "mAP@0.5"})
                fig_map.update_traces(line=dict(width=3, color="#e74c3c"), marker=dict(size=12))
                fig_map.add_hline(y=0.70, line_dash="dash", line_color="green", 
                                annotation_text="Límite excelencia (0.70)")
                fig_map.update_layout(height=350)
                st.plotly_chart(fig_map, use_container_width=True)
            
            with col2:
                st.markdown("#### Trade-off Precision-Recall")
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(
                    x=df_robust["Recall (promedio)"],
                    y=df_robust["Precision (promedio)"],
                    mode='lines+markers+text',
                    text=[f"th={t:.2f}" for t in df_robust["Threshold"]],
                    textposition="top right",
                    line=dict(width=3, color="#3498db"),
                    marker=dict(size=12)
                ))
                fig_pr.update_layout(
                    title="Precision vs Recall (promedio)",
                    xaxis_title="Recall (promedio)",
                    yaxis_title="Precision (promedio)",
                    height=350,
                    yaxis_range=[0.95, 1.01],
                    xaxis_range=[0.65, 0.9]
                )
                st.plotly_chart(fig_pr, use_container_width=True)
            
            st.markdown("""
            <div class="success-box">
            <strong>Conclusión:</strong> El modelo mantiene excelente rendimiento (mAP > 0.70) incluso con threshold 0.90, 
            y preserva precision perfecta (1.0) en todos los thresholds. Esto confirma robustez y ausencia de overfitting.
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Comparativa arquitectónica
    st.markdown("### 🏗️ Diferencias Arquitectónicas Fundamentales")
    
    st.markdown("""
    | Aspecto | CNNs (ResNet/EfficientNet) | ViTs (DEIMv2) |
    |---------|---------------------------|---------------|
    | **Bias inductivo** | Fuerte (localidad, invarianza a traslación) | Mínimo |
    | **Receptive field** | Local → Global (gradual) | Global desde el inicio |
    | **Convergencia** | Rápida (~50 epochs) | Lenta (~150-200 epochs) |
    | **Sensibilidad a resolución** | Baja (EfficientNet) / Moderada (ResNet) | Alta |
    | **mAP máximo alcanzado** | 0.162 (EfficientNet nativa) | **0.785** (DEIMv2) |
    """)
    
    st.markdown("---")
    
    # Conclusión final
    st.markdown("### 🏆 Conclusión Final")
    
    st.markdown("""
    <div class="success-box">
    <h4>Los Vision Transformers son significativamente superiores para la detección de defectos industriales</h4>
    
    <p>Los experimentos realizados demuestran que:</p>
    <ol>
        <li><strong>DEIMv2 alcanza un mAP de 0.785</strong>, superando por amplio margen a las CNNs</li>
        <li><strong>La diferencia no se debe solo a la resolución</strong>: Incluso con 1024x1024, las CNNs obtienen mAP de 0.08-0.12</li>
        <li><strong>Los ViTs capturan mejor las relaciones espaciales</strong> necesarias para detectar defectos industriales con alta variabilidad</li>
        <li><strong>La convergencia de ViTs requiere más epochs</strong> (~150-200) comparado con CNNs (~50)</li>
    </ol>
    
    <p><strong>Recomendación:</strong> Para problemas de detección de defectos industriales con alta variabilidad 
    visual, los Vision Transformers (específicamente DEIMv2 con backbone DINOv3) son la arquitectura recomendada.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Mejores modelos por arquitectura
    st.markdown("### 🥇 Mejores Modelos por Arquitectura")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="arch-card">
        <h4>🔵 ResNet-18</h4>
        <p><strong>Mejor config:</strong> 1024x1024</p>
        <p><strong>mAP:</strong> 0.080</p>
        <p><strong>Mejora vs nativa:</strong> +3.9%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="arch-card">
        <h4>🟣 EfficientNet-B0</h4>
        <p><strong>Mejor config:</strong> Nativa</p>
        <p><strong>mAP:</strong> 0.162</p>
        <p><strong>Nota:</strong> Empeora con 1024px</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="arch-card" style="border: 2px solid #e74c3c;">
        <h4>🔴 DEIMv2 ⭐</h4>
        <p><strong>Mejor config:</strong> 1024x1024, 300ep</p>
        <p><strong>mAP:</strong> 0.785</p>
        <p><strong>Mejor epoch:</strong> 187</p>
        </div>
        """, unsafe_allow_html=True)


@st.cache_data
def load_predictions_json(architecture):
    """Carga predicciones desde JSON para una arquitectura específica."""
    # 1. Intentar datos exportados (formato plano: data/predictions/{arch}_predictions.json)
    exported_json = DATA_PATH / "predictions" / f"{architecture}_predictions.json"
    if exported_json.exists():
        with open(exported_json, 'r') as f:
            return json.load(f)
    
    # 2. Datos locales empaquetados: data/images_selected_for_visualize/predictions/
    local_predictions_dir = VISUALIZATION_DATA_ROOT / "predictions"
    local_json_path = local_predictions_dir / architecture / "predictions_all.json"
    if local_json_path.exists():
        with open(local_json_path, 'r') as f:
            return json.load(f)
    
    return None


def draw_ground_truth_on_image(img, gt_annotations, category_names):
    """Dibuja ground truth sobre una imagen."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title('Ground Truth', fontsize=16, fontweight='bold', pad=20)
    
    COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']
    
    for ann in gt_annotations:
        x, y, w, h = ann['bbox']
        category_id = ann['category_id']
        
        class_name = category_names.get(category_id, f"Class_{category_id}")
        color = COLORS[category_id % len(COLORS)]
        
        # Dibujar bounding box
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Etiqueta
        ax.text(
            x, y - 5,
            class_name,
            color='white',
            fontsize=10,
            fontweight='bold',
            bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=2)
        )
    
    plt.tight_layout()
    return fig


def draw_predictions_on_image(img, predictions, category_names, score_threshold=0.2):
    """Dibuja predicciones sobre una imagen."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f'Predictions (threshold ≥ {score_threshold:.2f})', fontsize=16, fontweight='bold', pad=20)
    
    COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']
    
    # Filtrar por threshold
    filtered_preds = [p for p in predictions if p['score'] >= score_threshold]
    
    for pred in filtered_preds:
        x, y, w, h = pred['bbox']
        category_id = pred['category_id']
        score = pred['score']
        
        class_name = category_names.get(category_id, f"Class_{category_id}")
        color = COLORS[category_id % len(COLORS)]
        
        # Dibujar bounding box
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Etiqueta con score
        label_text = f"{class_name} {score:.2f}"
        ax.text(
            x, y - 5,
            label_text,
            color='white',
            fontsize=10,
            fontweight='bold',
            bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=2)
        )
    
    plt.tight_layout()
    return fig


@st.cache_data
def load_ground_truth_annotations():
    """Carga anotaciones ground truth desde el archivo COCO del test."""
    # 1. Intentar datos exportados (ground_truth.json filtrado)
    exported_gt = DATA_PATH / "ground_truth.json"
    if exported_gt.exists():
        with open(exported_gt, 'r') as f:
            coco_data = json.load(f)
    else:
        # 2. test.json junto al paquete de visualización (portable)
        bundled_test_json = VISUALIZATION_DATA_ROOT / "test.json"
        # 3. test.json en la raíz de data/ (layout habitual del repo)
        local_test_json = DATA_PATH / "test.json"
        if bundled_test_json.exists():
            with open(bundled_test_json, 'r') as f:
                coco_data = json.load(f)
        elif local_test_json.exists():
            with open(local_test_json, 'r') as f:
                coco_data = json.load(f)
        else:
            return None
    
    # Crear índice de anotaciones por nombre de archivo
    image_name_to_annotations = {}
    image_name_to_info = {}
    
    for img_info in coco_data['images']:
        file_name = img_info['file_name']
        img_id = img_info['id']
        image_name = Path(file_name).stem
        image_name_to_info[image_name] = img_info
    
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        # Buscar imagen por id
        for img_info in coco_data['images']:
            if img_info['id'] == img_id:
                image_name = Path(img_info['file_name']).stem
                if image_name not in image_name_to_annotations:
                    image_name_to_annotations[image_name] = []
                image_name_to_annotations[image_name].append(ann)
                break
    
    return image_name_to_annotations, coco_data['categories']


def render_visualizations(metadata):
    """Vista 6: Visualizaciones Comparativas Dinámicas"""
    st.title("🖼️ Visualizaciones Comparativas Dinámicas")
    st.markdown("Comparación visual interactiva de predicciones entre diferentes modelos con threshold dinámico")
    
    st.markdown("---")
    
    # Rutas: prioridad para datos locales (herramienta independiente y exportable)
    # Por defecto se asume el layout empaquetado bajo data/images_selected_for_visualize/raw/
    RAW_IMAGES_DIR = VISUALIZATION_DATA_ROOT / "raw"
    # 1. Datos exportados (formato: data/images_selected/)
    if (DATA_PATH / "images_selected").exists() and list((DATA_PATH / "images_selected").glob("*")):
        RAW_IMAGES_DIR = DATA_PATH / "images_selected"
        st.info("📦 Usando datos exportados (data/images_selected)")
    # 2. Datos locales integrados (data/images_selected_for_visualize/raw/)
    elif (VISUALIZATION_DATA_ROOT / "raw").exists():
        RAW_IMAGES_DIR = (VISUALIZATION_DATA_ROOT / "raw").resolve()
        st.info("📦 Usando datos integrados (`data/images_selected_for_visualize/raw/`)")
    
    # Verificar que existe la estructura
    if not RAW_IMAGES_DIR.exists():
        # Intentar rutas alternativas (cwd relativo al lanzar streamlit)
        alternative_paths = [
            VISUALIZATION_DATA_ROOT / "raw",
            DATA_PATH / "images_selected",
            Path.cwd() / "data" / "images_selected_for_visualize" / "raw",
            TOOL_ROOT / "data" / "images_selected_for_visualize" / "raw",
        ]
        
        found_alternative = None
        for alt_path in alternative_paths:
            alt_resolved = alt_path.resolve()
            if alt_resolved.exists():
                found_alternative = alt_resolved
                break
        
        if found_alternative:
            RAW_IMAGES_DIR = found_alternative
            st.success(f"✅ Imágenes encontradas en ubicación alternativa: `{RAW_IMAGES_DIR}`")
        else:
            st.error(f"""
            ⚠️ **No se encontró la carpeta de imágenes seleccionadas**
            
            **Ruta buscada (absoluta):** `{RAW_IMAGES_DIR}`
            
            **Directorio de la herramienta:** `{TOOL_ROOT}`
            
            **Directorio de trabajo actual:** `{Path.cwd()}`
            
            **Rutas alternativas probadas:**
            {chr(10).join(f'  - {p.resolve()}' for p in alternative_paths)}
            
            Por favor verifica:
            1. Que existe `data/images_selected_for_visualize/raw/` con imágenes .jpg o .png
            2. Que tienes permisos de lectura
            3. Si usas el formato exportado, que existe `data/images_selected/`
            """)
            return
    
    # Cargar predicciones de cada arquitectura
    predictions_resnet = load_predictions_json("resnet18")
    predictions_efficientnet = load_predictions_json("efficientnet")
    predictions_deimv2 = load_predictions_json("deimv2")
    
    # Debug: mostrar qué predicciones se cargaron
    import os
    if os.getenv("STREAMLIT_DEBUG", "0") == "1":
        st.info(f"""
        **Debug - Predicciones cargadas:**
        - ResNet-18: {'✅' if predictions_resnet else '❌'}
        - EfficientNet: {'✅' if predictions_efficientnet else '❌'}
        - DEIMv2: {'✅' if predictions_deimv2 else '❌'}
        """)
    
    if not any([predictions_resnet, predictions_efficientnet, predictions_deimv2]):
        st.warning("""
        ⚠️ **No se encontraron predicciones JSON**
        
        Para generar predicciones, ejecuta los scripts en modo `--selected-images-mode`:
        
        Las predicciones deben estar en:
        - `data/predictions/{arch}_predictions.json` (formato exportado), o
        - `data/images_selected_for_visualize/predictions/{resnet18,efficientnet,deimv2}/predictions_all.json`
        
        Coloca los JSON en `data/images_selected_for_visualize/predictions/` o genera predicciones con los scripts de visualización (`--selected-images-mode`) apuntando a `data/images_selected_for_visualize/`.
        """)
        return
    
    # Obtener lista de imágenes disponibles
    image_files = sorted(list(RAW_IMAGES_DIR.glob("*.jpg")) + list(RAW_IMAGES_DIR.glob("*.png")))
    
    if not image_files:
        st.warning(f"No se encontraron imágenes en {RAW_IMAGES_DIR}")
        return
    
    image_names = [f.stem for f in image_files]
    
    # ========================================================================
    # SELECTOR DE ARQUITECTURA Y THRESHOLD
    # ========================================================================
    col_arch, col_thresh = st.columns([2, 1])
    
    with col_arch:
        selected_architecture = st.selectbox(
            "Selecciona arquitectura:",
            options=["ResNet-18", "EfficientNet-B0", "DEIMv2"],
            index=2  # DEIMv2 por defecto
        )
    
    with col_thresh:
        score_threshold = st.slider(
            "Score Threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Ajusta el threshold para filtrar predicciones. Threshold más alto muestra solo predicciones más confiables."
        )
    
    # Mapeo de arquitectura a predicciones
    arch_to_predictions = {
        "ResNet-18": predictions_resnet,
        "EfficientNet-B0": predictions_efficientnet,
        "DEIMv2": predictions_deimv2
    }
    
    selected_predictions = arch_to_predictions[selected_architecture]
    
    # Cargar ground truth
    gt_data = load_ground_truth_annotations()
    if gt_data:
        gt_annotations_dict, categories = gt_data
        category_names = {cat['id']: cat.get('unified_category_name', cat.get('name', f"Class_{cat['id']}")) for cat in categories}
    else:
        gt_annotations_dict = None
        category_names = {
            0: "NORMAL",
            1: "DEFORMACIONES",
            2: "ROTURA_FRACTURA",
            3: "RAYONES_ARANAZOS",
            4: "PERFORACIONES",
            5: "CONTAMINACION"
        }
    
    st.markdown("---")
    
    # ========================================================================
    # NAVEGACIÓN DE IMÁGENES CON FLECHAS
    # ========================================================================
    
    # Usar session state para mantener el índice de imagen actual
    if 'current_image_idx' not in st.session_state:
        st.session_state.current_image_idx = 0
    
    # Asegurar que el índice esté en rango
    if st.session_state.current_image_idx >= len(image_names):
        st.session_state.current_image_idx = 0
    if st.session_state.current_image_idx < 0:
        st.session_state.current_image_idx = len(image_names) - 1
    
    # Botones de navegación
    col_nav1, col_nav2, col_nav3, col_nav4 = st.columns([1, 2, 2, 1])
    
    with col_nav1:
        if st.button("◀️ Anterior", use_container_width=True):
            st.session_state.current_image_idx = (st.session_state.current_image_idx - 1) % len(image_names)
            st.rerun()
    
    with col_nav2:
        current_idx = st.session_state.current_image_idx
        st.markdown(f"<div style='text-align: center; padding: 10px;'><strong>Imagen {current_idx + 1} de {len(image_names)}</strong></div>", unsafe_allow_html=True)
    
    with col_nav3:
        selected_image_name = image_names[st.session_state.current_image_idx]
        st.markdown(f"<div style='text-align: center; padding: 10px;'><code>{selected_image_name}</code></div>", unsafe_allow_html=True)
    
    with col_nav4:
        if st.button("Siguiente ▶️", use_container_width=True):
            st.session_state.current_image_idx = (st.session_state.current_image_idx + 1) % len(image_names)
            st.rerun()
    
    st.markdown("---")
    
    # Cargar imagen seleccionada
    selected_image_file = RAW_IMAGES_DIR / f"{selected_image_name}.jpg"
    if not selected_image_file.exists():
        selected_image_file = RAW_IMAGES_DIR / f"{selected_image_name}.png"
    
    if not selected_image_file.exists():
        st.error(f"No se encontró la imagen: {selected_image_name}")
        return
    
    img = Image.open(selected_image_file)
    
    # ========================================================================
    # VISUALIZACIÓN: GROUND TRUTH (IZQUIERDA) Y PREDICCIONES (DERECHA)
    # ========================================================================
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### 📋 Ground Truth")
        
        # Obtener anotaciones ground truth
        if gt_annotations_dict and selected_image_name in gt_annotations_dict:
            gt_anns = gt_annotations_dict[selected_image_name]
            fig_gt = draw_ground_truth_on_image(img, gt_anns, category_names)
            st.pyplot(fig_gt)
            plt.close(fig_gt)
            st.caption(f"**{len(gt_anns)}** anotaciones ground truth")
        else:
            st.image(img, use_container_width=True)
            st.caption("⚠️ No se encontraron anotaciones ground truth")
    
    with col_right:
        st.markdown(f"### 🤖 {selected_architecture} (Threshold: {score_threshold:.2f})")
        
        if selected_predictions and selected_image_name in selected_predictions:
            preds = selected_predictions[selected_image_name]['predictions']
            fig_pred = draw_predictions_on_image(img, preds, category_names, score_threshold)
            st.pyplot(fig_pred)
            plt.close(fig_pred)
            
            # Estadísticas
            filtered_count = len([p for p in preds if p['score'] >= score_threshold])
            st.caption(f"**{filtered_count}** detecciones mostradas (de {len(preds)} totales)")
        else:
            st.warning(f"⚠️ No hay predicciones disponibles para {selected_architecture}")
            st.image(img, use_container_width=True)
    
    # ========================================================================
    # INFORMACIÓN ADICIONAL
    # ========================================================================
    st.markdown("---")
    
    with st.expander("ℹ️ Información sobre la Visualización Dinámica"):
        st.markdown("""
        **Cómo usar:**
        1. Selecciona la arquitectura que quieres visualizar (ResNet-18, EfficientNet-B0 o DEIMv2)
        2. Ajusta el **Score Threshold** con el slider para filtrar predicciones
        3. Usa las flechas ◀️ ▶️ para navegar entre imágenes
        4. Las predicciones se actualizan dinámicamente según el threshold
        
        **Vista:**
        - **Izquierda:** Imagen original con anotaciones Ground Truth
        - **Derecha:** Predicciones del modelo seleccionado con threshold ajustable
        
        **Threshold:**
        - **Threshold bajo (0.0-0.3)**: Muestra todas las predicciones, incluyendo las menos confiables
        - **Threshold medio (0.3-0.6)**: Muestra predicciones con confianza moderada
        - **Threshold alto (0.6-1.0)**: Solo muestra predicciones muy confiables
        
        **Navegación:**
        - Usa los botones ◀️ Anterior y Siguiente ▶️ para cambiar de imagen
        - El contador muestra la imagen actual y el total
        
        **Nota:** Las predicciones se cargan desde archivos JSON generados con threshold 0.20,
        por lo que puedes ajustar dinámicamente el threshold sin re-evaluar el modelo.
        """)


# --- NAVEGACIÓN PRINCIPAL ---

def main():
    # Cargar metadatos
    if not METADATA_FILE.exists():
        st.error("No se encontró el archivo de metadatos. Verifica la estructura de datos.")
        return
    
    metadata = load_metadata()
    
    # Sidebar con navegación
    st.sidebar.title("🧭 Navegación")
    st.sidebar.markdown("---")
    
    pages = {
        "🏠 Inicio": render_home,
        "📜 Línea Temporal": render_timeline,
        "🔬 Explorador": render_explorer,
        "📊 Comparativa": render_comparison,
        "🖼️ Visualizaciones": render_visualizations,
        "📝 Conclusiones": render_conclusions
    }
    
    selection = st.sidebar.radio("Ir a:", list(pages.keys()))
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **TFG 2025-26**  
    *Detección de Defectos Industriales*  
    *Vision Transformers vs CNNs*
    """)
    
    # Renderizar página seleccionada
    pages[selection](metadata)


if __name__ == "__main__":
    main()
