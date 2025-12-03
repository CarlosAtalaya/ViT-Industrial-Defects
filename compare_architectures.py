import streamlit as st
import json
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# --- CONFIGURACIÓN DE RUTAS ---
MODEL_PATHS = {
    "EfficientNet-B0 (FPN)": "models/trainings/best_model_efficientnet",
    "ResNet-18 (FPN)": "models/trainings/best_model_resnet18",
    "ViT+DEIMv2 over DINOV3": "models/trainings/deimv2/300epochs",
}

JSON_FILES = ["test_evaluation_results_comparable.json", "test_evaluation_results.json"]

st.set_page_config(layout="wide", page_title="Analítica de Modelos de Visión")

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0;}
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: white; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.1);}
    .stTabs [aria-selected="true"] { background-color: #eef; color: #337ab7; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

st.title("🔬 Dashboard de Comparación de Arquitecturas")

# --- FUNCIONES DE LÓGICA ---

def clean_class_name(name):
    """Normaliza nombres de clases (ej. Class_6 -> NORMAL)"""
    if name.upper() == "CLASS_6":
        return "NORMAL"
    return name

def load_data(models_dict):
    data_list = []
    config_info = {} # Para guardar thresholds
    
    for model_name, path in models_dict.items():
        json_path = None
        for jf in JSON_FILES:
            full_path = os.path.join(path, jf)
            if os.path.exists(full_path):
                json_path = full_path
                break
        
        if json_path:
            with open(json_path, 'r') as f:
                content = json.load(f)
                
                # Guardamos info de configuración para mostrarla aparte
                config_info[model_name] = {
                    "iou": content.get("iou_threshold", "N/A"),
                    "score": content.get("score_threshold", "N/A"),
                    "imgs": content.get("num_test_images", "N/A")
                }

                # Base entry
                base_entry = {"Model": model_name, "mAP": content.get("mAP", 0)}
                
                # Procesar métricas por clase (AP, Precision, Recall)
                metrics_to_parse = {
                    "AP": content.get("AP_per_class", {}),
                    "Precision": content.get("precision_per_class", {}),
                    "Recall": content.get("recall_per_class", {})
                }

                # Aplanamos para el DataFrame
                row = base_entry.copy()
                
                for metric_type, values_dict in metrics_to_parse.items():
                    for cls, val in values_dict.items():
                        clean_cls = clean_class_name(cls)
                        # Creamos columnas tipo: AP_NORMAL, Precision_ROTURA, etc.
                        row[f"{metric_type}_{clean_cls}"] = val
                
                data_list.append(row)
    
    return pd.DataFrame(data_list), config_info

def get_image_mapping(path):
    """
    Escanea la carpeta training_metrics y crea un mapa simplificado.
    Devuelve un dict: {'total_loss': '1_total_loss.png', ...}
    """
    metrics_path = os.path.join(path, "training_metrics")
    mapping = {}
    if os.path.exists(metrics_path):
        files = sorted([f for f in os.listdir(metrics_path) if f.endswith(('.png', '.jpg'))])
        for f in files:
            # Lógica heurística para identificar de qué trata la imagen
            # Convertimos nombre a minúsculas y quitamos números para buscar keywords
            name_lower = f.lower()
            
            if "total_loss" in name_lower:
                mapping["Total Loss"] = f
            elif "classifier_loss" in name_lower or "classification" in name_lower:
                mapping["Clasificación Loss"] = f
            elif "box_regression" in name_lower or "bbox" in name_lower:
                mapping["Box Regression Loss"] = f
            elif "learning_rate" in name_lower:
                mapping["Learning Rate"] = f
            elif "map" in name_lower and "val" in name_lower:
                mapping["Validation mAP"] = f
            elif "components" in name_lower:
                mapping["Comparativa Componentes Loss"] = f
            elif "ap50" in name_lower:
                mapping["Validation AP50"] = f
                
    return mapping

# --- CARGA DE DATOS ---
df, configs = load_data(MODEL_PATHS)

if df.empty:
    st.error("No se encontraron datos. Revisa las rutas.")
    st.stop()

# --- SECCIÓN 1: CONFIGURACIÓN Y MÉTRICAS GLOBALES ---
st.markdown("### 1. Configuración de Evaluación")
cols = st.columns(len(configs))
for idx, (model, cfg) in enumerate(configs.items()):
    with cols[idx]:
        st.markdown(f"""
        <div class="metric-card">
            <strong>{model}</strong><br>
            <span style="font-size:14px; color:gray">Score Thresh: </span><b>{cfg['score']}</b><br>
            <span style="font-size:14px; color:gray">IoU Thresh: </span><b>{cfg['iou']}</b><br>
            <span style="font-size:14px; color:gray">Imágenes Test: </span>{cfg['imgs']}
        </div>
        """, unsafe_allow_html=True)

st.divider()

# --- SECCIÓN 2: COMPARATIVA DE RENDIMIENTO ---
st.markdown("### 2. Rendimiento Detallado por Clase")

tab_ap, tab_prec, tab_rec = st.tabs(["📊 Average Precision (AP)", "🎯 Precision", "🔍 Recall"])

def plot_metrics(dataframe, metric_prefix, title):
    # Filtrar columnas que empiecen por el prefijo (ej: "AP_")
    cols = [c for c in dataframe.columns if c.startswith(f"{metric_prefix}_")]
    
    if not cols:
        st.warning(f"No hay datos de {title}")
        return

    # Preparar datos para gráfico de barras
    df_melted = dataframe.melt(id_vars=["Model"], value_vars=cols, var_name="Clase", value_name="Score")
    df_melted["Clase"] = df_melted["Clase"].str.replace(f"{metric_prefix}_", "")
    
    # 1. Gráfico de Barras
    fig_bar = px.bar(
        df_melted, x="Clase", y="Score", color="Model", barmode="group",
        title=f"Comparativa de {title} por Clase", text_auto='.2f',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_bar.update_layout(yaxis=dict(range=[0, 1.05]))
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # 2. Radar Chart (Solo si hay más de 2 clases para que tenga sentido)
    categories = list(set(df_melted["Clase"]))
    if len(categories) > 2:
        fig_radar = go.Figure()
        for i, row in dataframe.iterrows():
            r_values = [row[c] for c in cols]
            # Cerrar el radar repitiendo el primer valor
            r_values_closed = r_values + [r_values[0]]
            cats_closed = [c.replace(f"{metric_prefix}_", "") for c in cols] + [cols[0].replace(f"{metric_prefix}_", "")]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=r_values_closed, theta=cats_closed, fill='toself', name=row['Model']
            ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title=f"Perfil de {title} (Radar)")
        st.plotly_chart(fig_radar, use_container_width=True)

with tab_ap:
    plot_metrics(df, "AP", "Average Precision")
with tab_prec:
    plot_metrics(df, "Precision", "Precisión (False Positives)")
    st.caption("*Precision 1.0 significa que cuando el modelo predice esta clase, nunca se equivoca (no hay falsos positivos).*")
with tab_rec:
    plot_metrics(df, "Recall", "Recall (False Negatives)")
    st.caption("*Recall 1.0 significa que el modelo encuentra TODAS las instancias de esta clase (no se deja ninguna).*")

st.divider()

# --- SECCIÓN 3: ANÁLISIS DEL ENTRENAMIENTO ---
st.markdown("### 3. Inspección del Entrenamiento")
st.info("Selecciona el concepto que quieres analizar. La herramienta buscará automáticamente el gráfico correspondiente en cada arquitectura.")

# Definir conceptos que sabemos que existen en tus carpetas
conceptos_disponibles = [
    "Total Loss", 
    "Clasificación Loss", 
    "Box Regression Loss", 
    "Learning Rate",
    "Validation mAP",
    "Comparativa Componentes Loss"
]

selected_concept = st.pills("Selecciona métrica a visualizar:", conceptos_disponibles, selection_mode="single", default="Total Loss")

if selected_concept:
    # Contenedor para las imágenes
    cols = st.columns(len(MODEL_PATHS))
    
    for idx, (model_name, path) in enumerate(MODEL_PATHS.items()):
        mapping = get_image_mapping(path)
        
        with cols[idx]:
            st.markdown(f"**{model_name}**")
            
            if selected_concept in mapping:
                img_filename = mapping[selected_concept]
                img_path = os.path.join(path, "training_metrics", img_filename)
                
                try:
                    image = Image.open(img_path)
                    st.image(image, caption=img_filename, use_container_width=True)
                except Exception as e:
                    st.error(f"Error cargando imagen: {e}")
            else:
                st.warning("⚠️ Gráfico no disponible")
                st.caption(f"No se encontró '{selected_concept}' en {path}")