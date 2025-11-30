# Fase 2: Experimentación Multimodal Progresiva (Opción 3)

Este directorio contiene los experimentos de **Refinamiento Multimodal** aplicados sobre el modelo base de la Fase 1.

## 🔬 Hipótesis de Investigación
Se intentó superar el **mAP de 0.785** del modelo visual puro inyectando descripciones semánticas (texto) para desambiguar clases conflictivas (ej. *Rotura* vs *Rayón*).

### Metodología Implementada
- **Estrategia:** Fine-tuning Progresivo (Residual Learning).
- **Arquitectura:** DEIMv2 + CLIP Text Encoder + Módulo de Fusión Latente.
- **Técnica de Estabilidad:** Inicialización *Zero-Start* ($\alpha=0$) para preservar el conocimiento previo.
- **Entrenamiento:** Congelación del backbone DINOv3; entrenamiento exclusivo de cabeceras y fusión.

## 📉 Resultados y Conclusiones (Post-Mortem)

Tras múltiples iteraciones experimentales, se concluye que **la multimodalidad no aporta mejoras** en este dominio específico.

1.  **Saturación:** El modelo no logró superar el baseline visual (0.785).
2.  **Interferencia:** La inyección de texto introdujo ruido, degradando el mAP en validación (~0.67) a pesar de la reducción del *loss* en entrenamiento (*overfitting*).
3.  **Conclusión Científica:** El backbone visual DINOv3 ya captura la totalidad de la información relevante. La complejidad multimodal es innecesaria para este dataset.

> **Estado:** Experimentación concluida. Se mantiene el modelo de la Fase 1 como la solución óptima para el despliegue.

## 🛠️ Scripts Principales
- `train_auto_phase2.py`: Script de entrenamiento con *Early Stopping* basado en Score Híbrido.
- `analyze_phase2_logs.py`: Herramienta de extracción de métricas y generación de gráficas comparativas.
- `models_utils/`: Módulos de fusión y encoders de texto.