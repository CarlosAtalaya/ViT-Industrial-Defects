#!/usr/bin/env python3
"""
Generador de Gráficas de Entrenamiento - Fase 2
===============================================
Lee el log.txt generado por train.py y produce gráficas comparables
con la documentación de Fase 1.
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

def parse_log(log_path):
    data = {
        'epoch': [],
        'loss': [],
        'lr': [],
        'map': [],
        'ap50': []
    }
    
    print(f"📂 Leyendo log: {log_path}")
    if not os.path.exists(log_path):
        raise FileNotFoundError("No se encontró el archivo log.txt. Ejecuta train.py primero.")

    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                entry = json.loads(line)
                data['epoch'].append(entry['epoch'])
                data['loss'].append(entry['train_loss'])
                data['lr'].append(entry['train_lr'])
                
                # Extraer métricas COCO si existen (lista de 12 floats)
                coco = entry.get('test_coco_eval_bbox', [])
                if coco and len(coco) > 0:
                    data['map'].append(coco[0])      # mAP 0.5:0.95
                    data['ap50'].append(coco[1])     # AP 0.5
                else:
                    data['map'].append(0)
                    data['ap50'].append(0)
            except json.JSONDecodeError:
                continue
                
    return data

def save_plot(x, y, title, ylabel, filename, color='blue', best_type='min'):
    """Genera y guarda una gráfica estándar del proyecto."""
    if not x or not y:
        print(f"⚠️  Saltando gráfica {filename}: Sin datos.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', linewidth=2, color=color, alpha=0.8, label=ylabel)
    
    # Marcar el mejor punto
    if best_type == 'min':
        best_idx = np.argmin(y)
        best_val = y[best_idx]
        color_mark = 'red'
    else: # max
        best_idx = np.argmax(y)
        best_val = y[best_idx]
        color_mark = 'green'
        
    # Solo marcar si tiene sentido (evitar marcar 0 en mAP como mejor si todo es 0)
    if not (best_type == 'max' and best_val == 0):
        plt.plot(x[best_idx], best_val, '*', markersize=15, color=color_mark, 
                 label=f'Best: {best_val:.4f} (Epoch {x[best_idx]})')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✅ Gráfica guardada: {filename}")

def main():
    # Configuración de rutas
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(base_dir, "outputs", "fase2_test_run-5epochs", "log.txt")
    output_metrics_dir = os.path.join(base_dir, "outputs", "fase2_test_run-5epochs", "training_metrics")
    
    os.makedirs(output_metrics_dir, exist_ok=True)
    
    try:
        metrics = parse_log(log_file)
        
        print("\n📊 Generando visualizaciones...")
        
        # 1. Training Loss
        save_plot(metrics['epoch'], metrics['loss'], 
                 "Fase 2: Training Loss (Classification)", "Loss",
                 os.path.join(output_metrics_dir, "2_total_loss.png"),
                 color='tab:red', best_type='min')
        
        # 2. Learning Rate
        save_plot(metrics['epoch'], metrics['lr'], 
                 "Learning Rate Schedule", "Learning Rate",
                 os.path.join(output_metrics_dir, "1_learning_rate.png"),
                 color='tab:orange', best_type='max') # Max irrelevante aquí, solo visual
                 
        # 3. Validation mAP
        # Filtramos ceros para el plot si hay épocas sin validación
        valid_epochs = [e for e, m in zip(metrics['epoch'], metrics['map']) if m > 0]
        valid_maps = [m for m in metrics['map'] if m > 0]
        
        if valid_maps:
            save_plot(valid_epochs, valid_maps,
                     "Validation mAP @ 0.5:0.95", "mAP",
                     os.path.join(output_metrics_dir, "7_validation_map.png"),
                     color='tab:green', best_type='max')
        else:
            print("ℹ️  No hay datos de validación (mAP) para graficar aún.")

        print(f"\n✨ Proceso completado. Gráficas en: {output_metrics_dir}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Asegúrate de haber ejecutado 'train.py' y que 'log.txt' existe.")

if __name__ == "__main__":
    main()