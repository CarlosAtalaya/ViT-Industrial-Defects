#!/usr/bin/env python3
"""
Script para visualizar métricas de entrenamiento de DEIMv2.
Lee el log.txt de tensorboard y genera gráficas.
"""

import os
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np


def parse_training_log(log_file):
    """
    Parsea el archivo log.txt de DEIMv2.
    
    Returns:
        dict con listas de métricas por época
    """
    metrics = {
        'epoch': [],
        'train_loss': [],
        'val_map': [],
        'lr': [],
        'loss_mal': [],
        'loss_bbox': [],
        'loss_giou': [],
    }
    
    current_epoch = None
    
    with open(log_file, 'r') as f:
        for line in f:
            # Detectar época
            epoch_match = re.search(r'Epoch: \[(\d+)\]', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            
            # Loss total
            if 'Averaged stats' in line:
                loss_match = re.search(r'loss: [\d.]+\s+\(([\d.]+)\)', line)
                if loss_match and current_epoch is not None:
                    if current_epoch not in metrics['epoch']:
                        metrics['epoch'].append(current_epoch)
                        metrics['train_loss'].append(float(loss_match.group(1)))
                
                # Otras pérdidas
                mal_match = re.search(r'loss_mal: [\d.]+\s+\(([\d.]+)\)', line)
                bbox_match = re.search(r'loss_bbox: [\d.]+\s+\(([\d.]+)\)', line)
                giou_match = re.search(r'loss_giou: [\d.]+\s+\(([\d.]+)\)', line)
                
                if mal_match:
                    metrics['loss_mal'].append(float(mal_match.group(1)))
                if bbox_match:
                    metrics['loss_bbox'].append(float(bbox_match.group(1)))
                if giou_match:
                    metrics['loss_giou'].append(float(giou_match.group(1)))
            
            # mAP de validación
            if 'Average Precision  (AP) @[ IoU=0.50:0.95' in line:
                map_match = re.search(r'= ([\d.]+)', line)
                if map_match:
                    metrics['val_map'].append(float(map_match.group(1)))
            
            # Learning rate
            if 'lr:' in line and 'Epoch:' in line:
                lr_match = re.search(r'lr: ([\d.e-]+)', line)
                if lr_match:
                    lr_val = float(lr_match.group(1))
                    if len(metrics['lr']) < len(metrics['epoch']):
                        metrics['lr'].append(lr_val)
    
    return metrics


def plot_training_metrics(metrics, output_dir):
    """Genera gráficas de métricas."""
    
    epochs = metrics['epoch']
    
    # Figura principal
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Métricas de Entrenamiento - DEIMv2', fontsize=16, fontweight='bold')
    
    # 1. Loss total
    ax = axes[0, 0]
    ax.plot(epochs, metrics['train_loss'], marker='o', linewidth=2, label='Train Loss')
    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('Pérdida', fontsize=12)
    ax.set_title('Pérdida Total', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. mAP validación
    ax = axes[0, 1]
    if len(metrics['val_map']) > 0:
        ax.plot(epochs[:len(metrics['val_map'])], metrics['val_map'], 
                marker='s', linewidth=2, color='green', label='Val mAP')
        ax.set_xlabel('Época', fontsize=12)
        ax.set_ylabel('mAP', fontsize=12)
        ax.set_title('mAP en Validación', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # 3. Componentes de pérdida
    ax = axes[1, 0]
    if len(metrics['loss_mal']) > 0:
        ax.plot(epochs, metrics['loss_mal'], marker='o', linewidth=2, label='MAL Loss')
    if len(metrics['loss_bbox']) > 0:
        ax.plot(epochs, metrics['loss_bbox'], marker='s', linewidth=2, label='BBox Loss')
    if len(metrics['loss_giou']) > 0:
        ax.plot(epochs, metrics['loss_giou'], marker='^', linewidth=2, label='GIoU Loss')
    
    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('Pérdida', fontsize=12)
    ax.set_title('Componentes de Pérdida', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 4. Learning rate
    ax = axes[1, 1]
    if len(metrics['lr']) > 0:
        ax.plot(epochs[:len(metrics['lr'])], metrics['lr'], 
                marker='o', linewidth=2, color='purple')
        ax.set_xlabel('Época', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar
    save_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Gráfica guardada en: {save_path}")
    
    plt.close()
    
    # Imprimir resumen
    print("\n" + "="*80)
    print("RESUMEN DE ENTRENAMIENTO")
    print("="*80)
    print(f"Número de épocas: {len(epochs)}")
    
    if len(metrics['train_loss']) > 0:
        print(f"\nPérdida inicial: {metrics['train_loss'][0]:.4f}")
        print(f"Pérdida final: {metrics['train_loss'][-1]:.4f}")
    
    if len(metrics['val_map']) > 0:
        best_map = max(metrics['val_map'])
        best_epoch = metrics['val_map'].index(best_map)
        print(f"\nMejor mAP: {best_map:.4f} (época {epochs[best_epoch]})")
    
    if len(metrics['lr']) > 0:
        print(f"\nLearning rate inicial: {metrics['lr'][0]:.6f}")
        print(f"Learning rate final: {metrics['lr'][-1]:.6f}")
    
    print("="*80)


def main(args):
    """Función principal."""
    
    print("="*80)
    print("VISUALIZACIÓN DE MÉTRICAS - DEIMV2")
    print("="*80)
    
    # Buscar log file
    if os.path.isfile(args.log_path):
        log_file = args.log_path
        output_dir = os.path.dirname(log_file)
    elif os.path.isdir(args.log_path):
        log_file = os.path.join(args.log_path, 'log.txt')
        output_dir = args.log_path
    else:
        raise ValueError(f"Ruta no válida: {args.log_path}")
    
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"No se encontró log.txt en: {log_file}")
    
    print(f"\nCargando log desde: {log_file}")
    
    # Parsear métricas
    metrics = parse_training_log(log_file)
    
    if len(metrics['epoch']) == 0:
        print("⚠️  No se encontraron métricas en el log")
        return
    
    # Generar gráficas
    plot_training_metrics(metrics, output_dir)
    
    print("\n✓ Visualización completada")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualizar métricas de DEIMv2')
    
    parser.add_argument('--log-path', type=str, required=True,
                       help='Ruta al log.txt o directorio que lo contiene')
    
    args = parser.parse_args()
    main(args)