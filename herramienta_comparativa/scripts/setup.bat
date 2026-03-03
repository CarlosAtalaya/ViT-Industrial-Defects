@echo off
REM Setup y lanzamiento del Dashboard de Comparación de Arquitecturas
REM TFG 2025-26 - Detección de Defectos Industriales

setlocal EnableDelayedExpansion

REM Directorio de la herramienta (donde está este script, un nivel arriba)
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
for %%I in ("%SCRIPT_DIR%\..") do set "TOOL_DIR=%%~fI"
set "VENV_DIR=%TOOL_DIR%\venv"

echo ==============================================
echo   Dashboard Comparativa Arquitecturas - TFG
echo ==============================================
echo.
echo Directorio de la herramienta: %TOOL_DIR%
echo.

cd /d "%TOOL_DIR%"

REM Crear entorno virtual si no existe
if not exist "%VENV_DIR%" (
    echo [1/3] Creando entorno virtual en %VENV_DIR% ...
    python -m venv "%VENV_DIR%"
    echo       Entorno virtual creado.
) else (
    echo [1/3] Entorno virtual encontrado.
)

REM Activar entorno e instalar dependencias
echo [2/3] Activando entorno e instalando dependencias...
call "%VENV_DIR%\Scripts\activate.bat"
python -m pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo       Dependencias instaladas.

REM Lanzar dashboard
echo [3/3] Lanzando dashboard Streamlit...
echo.
echo       El dashboard se abrirá en: http://localhost:8501
echo       Pulsa Ctrl+C para detener el servidor.
echo.

streamlit run dashboard.py

pause
