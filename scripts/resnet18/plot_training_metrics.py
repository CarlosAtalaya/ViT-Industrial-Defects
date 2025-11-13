"""
Script para visualizar métricas de entrenamiento.
Genera gráficas de pérdidas y learning rate durante el entrenamiento.
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np


def plot_training_metrics(history_path, output_dir):
    """
    Genera gráficas de métricas de entrenamiento.
    
    Args:
        history_path: Ruta al archivo training_history.json
        output_dir: Directorio donde guardar las gráficas
    """
    # Cargar historial
    with open(history_path, 'r') as f:
        history = json.load(f)
    
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
    
    # Crear figura con múltiples subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Métricas de Entrenamiento - ResNet18 + Faster R-CNN', 
                 fontsize=16, fontweight='bold')
    
    # 1. Pérdida total
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, label='Train Loss', marker='o', linewidth=2)
    ax.plot(epochs, val_loss, label='Val Loss', marker='s', linewidth=2)
    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('Pérdida', fontsize=12)
    ax.set_title('Pérdida Total', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. Pérdida del clasificador
    ax = axes[0, 1]
    ax.plot(epochs, train_loss_classifier, label='Train', marker='o', linewidth=2)
    ax.plot(epochs, val_loss_classifier, label='Val', marker='s', linewidth=2)
    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('Pérdida', fontsize=12)
    ax.set_title('Pérdida del Clasificador', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. Pérdida de regresión de bbox
    ax = axes[0, 2]
    ax.plot(epochs, train_loss_box_reg, label='Train', marker='o', linewidth=2)
    ax.plot(epochs, val_loss_box_reg, label='Val', marker='s', linewidth=2)
    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('Pérdida', fontsize=12)
    ax.set_title('Pérdida de Regresión BBox', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 4. Pérdida de objectness
    ax = axes[1, 0]
    ax.plot(epochs, train_loss_objectness, label='Train', marker='o', linewidth=2)
    ax.plot(epochs, val_loss_objectness, label='Val', marker='s', linewidth=2)
    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('Pérdida', fontsize=12)
    ax.set_title('Pérdida de Objectness (RPN)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 5. Pérdida de RPN box regression
    ax = axes[1, 1]
    ax.plot(epochs, train_loss_rpn_box_reg, label='Train', marker='o', linewidth=2)
    ax.plot(epochs, val_loss_rpn_box_reg, label='Val', marker='s', linewidth=2)
    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('Pérdida', fontsize=12)
    ax.set_title('Pérdida de RPN BBox Reg', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 6. Learning rate
    ax = axes[1, 2]
    ax.plot(epochs, lr, marker='o', linewidth=2, color='green')
    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    # Guardar figura
    save_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Gráfica guardada en: {save_path}")
    
    plt.close()
    
    # Crear segunda figura: comparación de componentes de pérdida
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(epochs, train_loss_classifier, label='Classifier', marker='o', linewidth=2)
    ax.plot(epochs, train_loss_box_reg, label='Box Regression', marker='s', linewidth=2)
    ax.plot(epochs, train_loss_objectness, label='Objectness', marker='^', linewidth=2)
    ax.plot(epochs, train_loss_rpn_box_reg, label='RPN Box Reg', marker='d', linewidth=2)
    
    ax.set_xlabel('Época', fontsize=13)
    ax.set_ylabel('Pérdida (Train)', fontsize=13)
    ax.set_title('Componentes de Pérdida Durante el Entrenamiento', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'loss_components.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Gráfica guardada en: {save_path}")
    
    plt.close()
    
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
    
    print(f"\nCargando historial desde: {history_path}")
    print(f"Guardando gráficas en: {output_dir}")
    
    # Generar gráficas
    plot_training_metrics(history_path, output_dir)
    
    print("\n✓ Visualización completada")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualizar métricas de entrenamiento'
    )
    
    parser.add_argument(
        '--history-path',
        type=str,
        required=True,
        help='Ruta al archivo training_history.json o directorio que lo contiene'
    )
    
    args = parser.parse_args()
    main(args)