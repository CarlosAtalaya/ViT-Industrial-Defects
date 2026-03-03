#!/bin/bash
# Setup y lanzamiento del Dashboard de Comparación de Arquitecturas
# TFG 2025-26 - Detección de Defectos Industriales

set -e

# Directorio de la herramienta (donde está este script, un nivel arriba)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$TOOL_DIR/venv"

echo "=============================================="
echo "  Dashboard Comparativa Arquitecturas - TFG  "
echo "=============================================="
echo ""
echo "Directorio de la herramienta: $TOOL_DIR"
echo ""

cd "$TOOL_DIR"

# Crear entorno virtual si no existe
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/3] Creando entorno virtual en $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    echo "      Entorno virtual creado."
else
    echo "[1/3] Entorno virtual encontrado."
fi

# Activar entorno e instalar dependencias
echo "[2/3] Activando entorno e instalando dependencias..."
source "$VENV_DIR/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "      Dependencias instaladas."

# Lanzar dashboard
echo "[3/3] Lanzando dashboard Streamlit..."
echo ""
echo "      El dashboard se abrirá en: http://localhost:8501"
echo "      Pulsa Ctrl+C para detener el servidor."
echo ""

streamlit run dashboard.py
