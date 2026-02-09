#!/usr/bin/env python3
"""
Script para visualizar métricas de entrenamiento de DEIMv2.
Lee el log.txt en formato JSON y genera gráficas individuales.

FORMATO UNIFORMADO: Una gráfica por archivo en carpeta training_metrics/
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np


def parse_training_log(log_file):
    """
    Parsea el archivo log.txt de DEIMv2 (formato JSON por línea).
    
    Returns:
        dict con listas de métricas por época
    """
    metrics = {
        'epoch': [],
        'train_lr': [],
        'train_loss': [],
        'train_loss_mal': [],
        'train_loss_bbox': [],
        'train_loss_giou': [],
        'train_loss_fgl': [],
        'val_map': [],
        'val_ap50': [],
        'val_ap75': [],
    }
    
    print(f"\nParsing log file: {log_file}")
    
    with open(log_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # Extraer época
                epoch = data.get('epoch', None)
                if epoch is None:
                    continue
                
                metrics['epoch'].append(epoch)
                
                # Métricas de entrenamiento
                metrics['train_lr'].append(data.get('train_lr', 0))
                metrics['train_loss'].append(data.get('train_loss', 0))
                metrics['train_loss_mal'].append(data.get('train_loss_mal', 0))
                metrics['train_loss_bbox'].append(data.get('train_loss_bbox', 0))
                metrics['train_loss_giou'].append(data.get('train_loss_giou', 0))
                metrics['train_loss_fgl'].append(data.get('train_loss_fgl', 0))
                
                # Métricas de validación (test_coco_eval_bbox)
                # Formato: [mAP, AP50, AP75, ...]
                coco_eval = data.get('test_coco_eval_bbox', [])
                if len(coco_eval) >= 3:
                    metrics['val_map'].append(coco_eval[0])
                    metrics['val_ap50'].append(coco_eval[1])
                    metrics['val_ap75'].append(coco_eval[2])
                else:
                    metrics['val_map'].append(0)
                    metrics['val_ap50'].append(0)
                    metrics['val_ap75'].append(0)
                
            except json.JSONDecodeError as e:
                print(f"⚠️  Error parsing line {line_num}: {e}")
                continue
    
    print(f"✅ Parsed {len(metrics['epoch'])} epochs")
    
    # Verificar que hay datos
    if len(metrics['epoch']) == 0:
        raise ValueError("No se encontraron datos de épocas en el log")
    
    return metrics


def plot_single_metric(epochs, data, ylabel, title, output_path, 
                       color='blue', use_log_scale=False, mark_best=True, file_format='pdf'):
    """
    Genera una gráfica individual para una métrica.
    
    Args:
        epochs: Lista de épocas
        data: Datos a plotear
        ylabel: Etiqueta del eje Y
        title: Título de la gráfica
        output_path: Ruta donde guardar
        color: Color de la línea
        use_log_scale: Si True, usar escala logarítmica
        mark_best: Si True, marcar mejor valor
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, data, color=color, linewidth=2, 
            marker='o', markersize=4, alpha=0.8)
    
    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if use_log_scale:
        ax.set_yscale('log')
    
    # Marcar mejor valor (mínimo para loss, máximo para mAP)
    if mark_best and len(data) > 0 and max(data) > 0:
        if 'mAP' in title or 'AP' in title:
            best_idx = np.argmax(data)
            best_val = data[best_idx]
            marker_color = 'green'
            label = f'Best: {best_val:.4f} (Epoch {epochs[best_idx]})'
        else:
            best_idx = np.argmin(data)
            best_val = data[best_idx]
            marker_color = 'red'
            label = f'Best: {best_val:.4f} (Epoch {epochs[best_idx]})'
        
        ax.plot(epochs[best_idx], best_val, '*', 
                color=marker_color, markersize=15, label=label)
        ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    # Asegurar que la extensión del archivo coincida con el formato
    if not output_path.endswith(f'.{file_format}'):
        output_path = output_path.rsplit('.', 1)[0] + f'.{file_format}'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', format=file_format)
    plt.close()
    
    print(f"  ✓ Guardado: {os.path.basename(output_path)}")


def plot_multiple_metrics(epochs, metrics_dict, ylabel, title, output_path, file_format='pdf'):
    """Genera gráfica comparativa de múltiples métricas."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'd', 'v', 'p']
    
    for i, (name, data) in enumerate(metrics_dict.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        ax.plot(epochs, data, label=name, color=color, 
                linewidth=2, marker=marker, markersize=4, alpha=0.8)
    
    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Guardado: {os.path.basename(output_path)}")


def plot_training_metrics(log_path, base_output_dir, file_format='pdf'):
    """
    Genera todas las gráficas de métricas.
    
    Args:
        log_path: Ruta al log.txt
        base_output_dir: Directorio base (se creará training_metrics/ dentro)
    """
    # Parsear log
    metrics = parse_training_log(log_path)
    epochs = metrics['epoch']
    
    # Crear directorio de salida
    output_dir = os.path.join(base_output_dir, 'training_metrics')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGuardando gráficas en: {output_dir}\n")
    
    print("Generando gráficas individuales...")
    
    # 1. Learning Rate
    plot_single_metric(
        epochs, metrics['train_lr'],
        'Learning Rate', 'Learning Rate Schedule',
        os.path.join(output_dir, f'1_learning_rate.{file_format}'),
        color='green', use_log_scale=True, mark_best=False, file_format=file_format
    )
    
    # 2. Loss Total
    plot_single_metric(
        epochs, metrics['train_loss'],
        'Loss', 'Pérdida Total de Entrenamiento',
        os.path.join(output_dir, f'2_total_loss.{file_format}'),
        color='red', mark_best=True, file_format=file_format
    )
    
    # 3. Loss MAL (Classification)
    plot_single_metric(
        epochs, metrics['train_loss_mal'],
        'Loss', 'Pérdida de Clasificación (MAL)',
        os.path.join(output_dir, f'3_classification_loss.{file_format}'),
        color='blue', mark_best=True, file_format=file_format
    )
    
    # 4. Loss BBox (Regression)
    plot_single_metric(
        epochs, metrics['train_loss_bbox'],
        'Loss', 'Pérdida de Regresión BBox',
        os.path.join(output_dir, f'4_bbox_regression_loss.{file_format}'),
        color='orange', mark_best=True, file_format=file_format
    )
    
    # 5. Loss GIoU
    plot_single_metric(
        epochs, metrics['train_loss_giou'],
        'Loss', 'Pérdida GIoU',
        os.path.join(output_dir, f'5_giou_loss.{file_format}'),
        color='purple', mark_best=True, file_format=file_format
    )
    
    # 6. Loss Focal (FGL)
    plot_single_metric(
        epochs, metrics['train_loss_fgl'],
        'Loss', 'Pérdida Focal Loss',
        os.path.join(output_dir, f'6_focal_loss.{file_format}'),
        color='brown', mark_best=True, file_format=file_format
    )
    
    # 7. Validation mAP
    if len(metrics['val_map']) > 0 and max(metrics['val_map']) > 0:
        plot_single_metric(
            epochs, metrics['val_map'],
            'mAP', 'Validation mAP @0.5:0.95',
            os.path.join(output_dir, f'7_validation_map.{file_format}'),
            color='green', mark_best=True, file_format=file_format
        )
    
    # 8. Validation AP50
    if len(metrics['val_ap50']) > 0 and max(metrics['val_ap50']) > 0:
        plot_single_metric(
            epochs, metrics['val_ap50'],
            'AP@0.5', 'Validation AP @0.5',
            os.path.join(output_dir, f'8_validation_ap50.{file_format}'),
            color='blue', mark_best=True, file_format=file_format
        )
    
    # 9. Validation AP75
    if len(metrics['val_ap75']) > 0 and max(metrics['val_ap75']) > 0:
        plot_single_metric(
            epochs, metrics['val_ap75'],
            'AP@0.75', 'Validation AP @0.75',
            os.path.join(output_dir, f'9_validation_ap75.{file_format}'),
            color='red', mark_best=True, file_format=file_format
        )
    
    # 10. Comparación de losses desglosadas
    loss_components = {
        'MAL (Classification)': metrics['train_loss_mal'],
        'BBox Regression': metrics['train_loss_bbox'],
        'GIoU': metrics['train_loss_giou'],
        'Focal Loss': metrics['train_loss_fgl']
    }
    plot_multiple_metrics(
        epochs, loss_components,
        'Loss', 'Comparación de Componentes de Pérdida',
        os.path.join(output_dir, f'10_loss_components_comparison.{file_format}'),
        file_format=file_format
    )
    
    # 11. Comparación de métricas de validación
    if len(metrics['val_map']) > 0 and max(metrics['val_map']) > 0:
        val_metrics = {
            'mAP @0.5:0.95': metrics['val_map'],
            'AP @0.5': metrics['val_ap50'],
            'AP @0.75': metrics['val_ap75']
        }
        plot_multiple_metrics(
            epochs, val_metrics,
            'COCO mAP', 'Métricas de Validación',
            os.path.join(output_dir, f'11_validation_metrics_comparison.{file_format}'),
            file_format=file_format
        )
    
    # Imprimir resumen
    print("\n" + "="*80)
    print("RESUMEN DE ENTRENAMIENTO")
    print("="*80)
    print(f"Total de épocas: {len(epochs)}")
    print(f"Learning rate final: {metrics['train_lr'][-1]:.2e}")
    print(f"Loss final: {metrics['train_loss'][-1]:.4f}")
    
    if len(metrics['val_map']) > 0 and max(metrics['val_map']) > 0:
        best_idx = np.argmax(metrics['val_map'])
        print(f"\nMejor validación:")
        print(f"  Época: {epochs[best_idx]}")
        print(f"  mAP @0.5:0.95: {metrics['val_map'][best_idx]:.4f}")
        print(f"  AP @0.5: {metrics['val_ap50'][best_idx]:.4f}")
        print(f"  AP @0.75: {metrics['val_ap75'][best_idx]:.4f}")
    
    print("="*80)


def main(args):
    """Función principal."""
    print("="*80)
    print("VISUALIZACIÓN DE MÉTRICAS DE ENTRENAMIENTO")
    print("DEIMv2 + DINOv3")
    print("="*80)
    
    # Verificar archivo
    if not os.path.exists(args.log_path):
        print(f"\n❌ ERROR: No se encontró el archivo: {args.log_path}")
        return
    
    # Directorio de salida
    output_dir = os.path.dirname(args.log_path) if not args.output_dir else args.output_dir
    
    # Generar gráficas
    try:
        plot_training_metrics(args.log_path, output_dir, file_format=args.format)
        print(f"\n✅ Visualización completada")
        print(f"📁 Gráficas guardadas en: {os.path.join(output_dir, 'training_metrics/')}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualizar métricas de entrenamiento de DEIMv2 (formato uniformado)'
    )
    
    parser.add_argument('--log-path', type=str, required=True,
                       help='Ruta al archivo log.txt')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directorio de salida (default: mismo que log.txt)')
    parser.add_argument('--format', type=str, default='pdf',
                       choices=['pdf', 'png', 'svg'],
                       help='Formato de salida para las gráficas (default: pdf)')
    
    args = parser.parse_args()
    main(args)