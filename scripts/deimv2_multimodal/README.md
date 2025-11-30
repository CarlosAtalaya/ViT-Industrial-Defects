# Fase 1: DEIMv2 con DINOv3 (State-of-the-Art)

Este directorio contiene la implementación ganadora del proyecto: un detector de objetos basado en Vision Transformers (DEIMv2) potenciado por el backbone DINOv3.

## 🏆 Resultados Clave (Baseline SOTA)
Este modelo estableció el techo de rendimiento del proyecto, superando a las arquitecturas CNN clásicas.

| Métrica | Valor | Notas |
| :--- | :--- | :--- |
| **mAP (0.50:0.95)** | **0.785** | Rendimiento SOTA en el dataset industrial. |
| **Precision** | **Alta** | Excelente discriminación en clases *Normal* y *Perforaciones*. |
| **Robustez** | **Alta** | Gran capacidad de generalización en texturas complejas. |

## 📂 Documentación Técnica
Para ver los detalles profundos de la arquitectura, configuración de hiperparámetros y análisis de resultados de esta fase, consulta el documento principal:

👉 **[Arquitectura e Implementación DEIMv2](deimv2_arquitetcura_implementacion.md)**

## 🚀 Ejecución
El entrenamiento y evaluación se gestionan mediante el script maestro:
```bash
python3 train_deimv2_industrial.py --config configs/deimv2_industrial_defects.yml

