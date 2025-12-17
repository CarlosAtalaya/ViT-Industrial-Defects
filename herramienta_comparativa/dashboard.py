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

# Ruta base de datos
DATA_PATH = Path(__file__).parent / "data"
METADATA_FILE = DATA_PATH / "experiments_metadata.json"

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

def get_all_results_df(metadata):
    """Crea un DataFrame con todos los resultados"""
    rows = []
    for phase_id, experiments in metadata["experiments"].items():
        for exp_id, exp_info in experiments.items():
            results = load_experiment_results(exp_info["path"])
            if results:
                row = {
                    "ID": exp_id,
                    "Nombre": exp_info["name"],
                    "Arquitectura": exp_info["architecture"],
                    "Resolución": exp_info["resolution"],
                    "Épocas": exp_info["epochs"],
                    "Fase": exp_info["phase"],
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
    La experimentación se ha dividido en **3 fases principales**:
    
    1. **Fase 1 (Octubre 2024)**: Establecer líneas base con arquitecturas CNN clásicas (ResNet-18, EfficientNet-B0)
    2. **Fase 2 (Noviembre 2024)**: Explorar Vision Transformers (DEIMv2) con diferentes configuraciones
    3. **Fase 3 (Diciembre 2024)**: Validar resultados entrenando CNNs con la misma resolución que los ViTs
    
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
        else:
            icon, color = "🟢", "#2ecc71"
        
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
        else:
            phase_key = "fase3_comparacion_justa"
        
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
        results = load_experiment_results(exp_info["path"])
        
        if results:
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
                fig = px.bar(df_metric, x="Clase", y="Valor", 
                           title=title,
                           color="Valor",
                           color_continuous_scale=color_scale,
                           text=df_metric["Valor"].apply(lambda x: f"{x:.3f}"))
                fig.update_traces(textposition="outside")
                fig.update_layout(showlegend=False, yaxis_range=[0, 1.1])
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No se encontraron resultados de evaluación")
    
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
    
    df = get_all_results_df(metadata)
    
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
    
    # Ordenar por mAP y crear gráfico
    df_sorted = df_filtered.sort_values("mAP", ascending=True).reset_index(drop=True)
    
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
        
        fig.add_trace(go.Bar(
            y=[row["Nombre"]],
            x=[float(row["mAP"])],
            orientation='h',
            name=arch,
            marker_color=color_map.get(arch, "#gray"),
            text=[f"{row['mAP']:.3f}"],
            textposition="outside",
            showlegend=show_in_legend
        ))
    
    fig.update_layout(
        height=max(300, len(df_sorted) * 50),
        xaxis_title="mAP@0.5",
        yaxis_title="",
        barmode='group',
        legend_title="Arquitectura"
    )
    st.plotly_chart(fig, use_container_width=True)
    
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
        st.dataframe(
            df_filtered[display_cols].sort_values("mAP", ascending=False),
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
