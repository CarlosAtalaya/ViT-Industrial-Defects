#!/usr/bin/env python3
"""
Script para visualizar métricas de entrenamiento (ResNet18/EfficientNet).
Genera gráficas individuales de pérdidas y learning rate.

FORMATO UNIFORMADO: Una gráfica por archivo en carpeta training_metrics/
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np


def plot_individual_metric(epochs, train_data, val_data, 
                          metric_name, ylabel, title, 
                          output_path, use_log_scale=False):
    """
    Genera una gráfica individual para una métrica específica.
    
    Args:
        epochs: Lista de épocas
        train_data: Datos de entrenamiento
        val_data: Datos de validación
        metric_name: Nombre del archivo (sin extensión)
        ylabel: Etiqueta del eje Y
        title: Título de la gráfica
        output_path: Ruta completa donde guardar la imagen
        use_log_scale: Si True, usar escala logarítmica en Y
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot datos
    ax.plot(epochs, train_data, label='Train', marker='o', 
            linewidth=2, markersize=4, alpha=0.8)
    ax.plot(epochs, val_data, label='Validation', marker='s', 
            linewidth=2, markersize=4, alpha=0.8)
    
    # Configuración
    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    if use_log_scale:
        ax.set_yscale('log')
    
    # Marcar mejor valor de validación
    best_idx = np.argmin(val_data)
    best_val = val_data[best_idx]
    best_epoch = epochs[best_idx]
    ax.plot(best_epoch, best_val, 'r*', markersize=15, 
            label=f'Best: {best_val:.4f} (Epoch {best_epoch})')
    ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Guardado: {os.path.basename(output_path)}")


def plot_learning_rate(epochs, lr_data, output_path):
    """Genera gráfica de learning rate."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, lr_data, marker='o', linewidth=2, 
            markersize=4, color='green', alpha=0.8)
    
    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Guardado: {os.path.basename(output_path)}")


def plot_loss_components(epochs, components_dict, output_path):
    """Genera gráfica comparativa de componentes de pérdida."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p']
    
    for i, (name, values) in enumerate(components_dict.items()):
        marker = markers[i % len(markers)]
        ax.plot(epochs, values, label=name, marker=marker, 
                linewidth=2, markersize=4, alpha=0.8)
    
    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('Loss (Train)', fontsize=12)
    ax.set_title('Componentes de Pérdida Durante el Entrenamiento', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Guardado: {os.path.basename(output_path)}")


def plot_training_metrics(history_path, base_output_dir):
    """
    Genera todas las gráficas de métricas de entrenamiento.
    
    Args:
        history_path: Ruta al archivo training_history.json
        base_output_dir: Directorio base (se creará training_metrics/ dentro)
    """
    # Cargar historial
    print(f"\nCargando historial desde: {history_path}")
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Crear directorio de salida
    output_dir = os.path.join(base_output_dir, 'training_metrics')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Guardando gráficas en: {output_dir}\n")
    
    # Extraer métricas
    epochs = [entry['epoch'] for entry in history]
    train_loss = [entry['loss'] for entry in history]
    val_loss = [entry['val_loss'] for entry in history]
    
    train_loss_classifier = [entry['loss_classifier'] for entry in history]
    train_loss_box_reg = [entry['loss_box_reg'] for entry in history]
    train_loss_objectness = [entry['loss_objectness'] for entry in history]
    train_loss_rpn_box_reg = [entry['loss_rpn_box_reg'] for entry in history]
    
    val_loss_classifier = [entry['val_loss_classifier'] for entry in history]
    val_loss_box_reg = [entry['val_loss_box_reg'] for entry in history]
    val_loss_objectness = [entry['val_loss_objectness'] for entry in history]
    val_loss_rpn_box_reg = [entry['val_loss_rpn_box_reg'] for entry in history]
    
    lr = [entry['lr'] for entry in history]
    
    # Generar gráficas individuales
    print("Generando gráficas individuales...")
    
    # 1. Pérdida total
    plot_individual_metric(
        epochs, train_loss, val_loss,
        'total_loss', 'Loss', 'Pérdida Total',
        os.path.join(output_dir, '1_total_loss.png')
    )
    
    # 2. Pérdida del clasificador
    plot_individual_metric(
        epochs, train_loss_classifier, val_loss_classifier,
        'classifier_loss', 'Loss', 'Pérdida del Clasificador',
        os.path.join(output_dir, '2_classifier_loss.png')
    )
    
    # 3. Pérdida de regresión de bbox
    plot_individual_metric(
        epochs, train_loss_box_reg, val_loss_box_reg,
        'box_regression_loss', 'Loss', 'Pérdida de Regresión BBox',
        os.path.join(output_dir, '3_box_regression_loss.png')
    )
    
    # 4. Pérdida de objectness (RPN)
    plot_individual_metric(
        epochs, train_loss_objectness, val_loss_objectness,
        'objectness_loss', 'Loss', 'Pérdida de Objectness (RPN)',
        os.path.join(output_dir, '4_objectness_loss.png')
    )
    
    # 5. Pérdida de RPN box regression
    plot_individual_metric(
        epochs, train_loss_rpn_box_reg, val_loss_rpn_box_reg,
        'rpn_box_reg_loss', 'Loss', 'Pérdida de RPN BBox Regression',
        os.path.join(output_dir, '5_rpn_box_regression_loss.png')
    )
    
    # 6. Learning rate
    plot_learning_rate(
        epochs, lr,
        os.path.join(output_dir, '6_learning_rate.png')
    )
    
    # 7. Componentes de pérdida (comparativa)
    components = {
        'Classifier': train_loss_classifier,
        'Box Regression': train_loss_box_reg,
        'Objectness': train_loss_objectness,
        'RPN Box Reg': train_loss_rpn_box_reg
    }
    plot_loss_components(
        epochs, components,
        os.path.join(output_dir, '7_loss_components_comparison.png')
    )
    
    # Imprimir resumen
    print("\n" + "=" * 80)
    print("RESUMEN DE ENTRENAMIENTO")
    print("=" * 80)
    print(f"Número de épocas: {len(epochs)}")
    print(f"\nPérdida inicial (Train/Val): {train_loss[0]:.4f} / {val_loss[0]:.4f}")
    print(f"Pérdida final (Train/Val): {train_loss[-1]:.4f} / {val_loss[-1]:.4f}")
    print(f"Mejor pérdida de validación: {min(val_loss):.4f} (época {epochs[np.argmin(val_loss)]})")
    print(f"\nLearning rate inicial: {lr[0]:.6f}")
    print(f"Learning rate final: {lr[-1]:.6f}")
    print("=" * 80)


def main(args):
    """Función principal."""
    
    print("=" * 80)
    print("VISUALIZACIÓN DE MÉTRICAS DE ENTRENAMIENTO")
    print("Faster R-CNN (ResNet18/EfficientNet)")
    print("=" * 80)
    
    # Buscar archivo de historial
    if os.path.isfile(args.history_path):
        history_path = args.history_path
        output_dir = os.path.dirname(history_path)
    elif os.path.isdir(args.history_path):
        # Buscar training_history.json en el directorio
        history_path = os.path.join(args.history_path, 'training_history.json')
        output_dir = args.history_path
    else:
        raise ValueError(f"Ruta no válida: {args.history_path}")
    
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"No se encontró: {history_path}")
    
    # Generar gráficas
    plot_training_metrics(history_path, output_dir)
    
    print("\n✅ Visualización completada")
    print(f"📁 Gráficas guardadas en: {os.path.join(output_dir, 'training_metrics/')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualizar métricas de entrenamiento (formato uniformado)'
    )
    
    parser.add_argument(
        '--history-path',
        type=str,
        required=True,
        help='Ruta al archivo training_history.json o directorio que lo contiene'
    )
    
    args = parser.parse_args()
    main(args)