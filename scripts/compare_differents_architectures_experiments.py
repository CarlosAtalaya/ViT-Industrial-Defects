"""
Script para comparar resultados entre múltiples experimentos.
Útil para comparar ResNet-18 vs otros backbones (EfficientNet, ViT, etc.)
"""

import os
import json
import argparse
from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt


def load_experiment_results(experiment_dir: str) -> Dict:
    """
    Carga los resultados de un experimento.
    
    Args:
        experiment_dir: Directorio del experimento
    
    Returns:
        Dict con métricas del experimento
    """
    results = {
        'name': os.path.basename(experiment_dir),
        'path': experiment_dir
    }
    
    # Cargar configuración
    config_path = os.path.join(experiment_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            results['config'] = config
            results['epochs'] = config.get('epochs', 'N/A')
            results['batch_size'] = config.get('batch_size', 'N/A')
            results['lr'] = config.get('lr', 'N/A')
    
    # Cargar historial de entrenamiento
    history_path = os.path.join(experiment_dir, 'training_history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
            last_epoch = history[-1]
            results['final_train_loss'] = last_epoch.get('loss', None)
            results['final_val_loss'] = last_epoch.get('val_loss', None)
            results['best_val_loss'] = min(e['val_loss'] for e in history)
            results['training_time'] = sum(e.get('time', 0) for e in history)
    
    # Cargar resultados de evaluación en test
    eval_path = os.path.join(experiment_dir, 'test_evaluation_results.json')
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            eval_results = json.load(f)
            results['mAP'] = eval_results.get('mAP', None)
            results['AP_per_class'] = eval_results.get('AP_per_class', {})
            results['precision_per_class'] = eval_results.get('precision_per_class', {})
            results['recall_per_class'] = eval_results.get('recall_per_class', {})
    
    # Cargar información del checkpoint
    checkpoint_path = os.path.join(experiment_dir, 'checkpoints', 'best_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        results['checkpoint_exists'] = True
    else:
        results['checkpoint_exists'] = False
    
    return results


def create_comparison_table(experiments: List[Dict]) -> pd.DataFrame:
    """
    Crea una tabla comparativa de experimentos.
    
    Args:
        experiments: Lista de resultados de experimentos
    
    Returns:
        DataFrame con la comparación
    """
    data = []
    
    for exp in experiments:
        row = {
            'Experimento': exp['name'],
            'Épocas': exp.get('epochs', 'N/A'),
            'Batch Size': exp.get('batch_size', 'N/A'),
            'Learning Rate': exp.get('lr', 'N/A'),
            'Train Loss (final)': f"{exp.get('final_train_loss', 0):.4f}" if exp.get('final_train_loss') else 'N/A',
            'Val Loss (final)': f"{exp.get('final_val_loss', 0):.4f}" if exp.get('final_val_loss') else 'N/A',
            'Val Loss (best)': f"{exp.get('best_val_loss', 0):.4f}" if exp.get('best_val_loss') else 'N/A',
            'mAP': f"{exp.get('mAP', 0):.4f}" if exp.get('mAP') else 'N/A',
            'Tiempo (min)': f"{exp.get('training_time', 0)/60:.1f}" if exp.get('training_time') else 'N/A'
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def create_comparison_plots(experiments: List[Dict], output_dir: str):
    """
    Crea gráficas comparativas entre experimentos.
    
    Args:
        experiments: Lista de resultados de experimentos
        output_dir: Directorio donde guardar las gráficas
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filtrar experimentos que tienen mAP
    exps_with_map = [e for e in experiments if e.get('mAP') is not None]
    
    if not exps_with_map:
        print("No hay experimentos con mAP para comparar")
        return
    
    # Gráfica 1: Comparación de mAP
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = [e['name'] for e in exps_with_map]
    maps = [e['mAP'] for e in exps_with_map]
    
    bars = ax.bar(range(len(names)), maps, color='steelblue', alpha=0.7)
    ax.set_xlabel('Experimento', fontsize=12)
    ax.set_ylabel('mAP', fontsize=12)
    ax.set_title('Comparación de mAP entre Experimentos', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Agregar valores sobre las barras
    for bar, value in zip(bars, maps):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_map.png'), dpi=150)
    plt.close()
    
    # Gráfica 2: Comparación de AP por clase
    # Obtener todas las clases
    all_classes = set()
    for exp in exps_with_map:
        all_classes.update(exp.get('AP_per_class', {}).keys())
    all_classes = sorted(all_classes)
    
    if all_classes:
        fig, ax = plt.subplots(figsize=(14, 7))
        
        x = range(len(all_classes))
        width = 0.8 / len(exps_with_map)
        
        for i, exp in enumerate(exps_with_map):
            aps = [exp.get('AP_per_class', {}).get(cls, 0) for cls in all_classes]
            offset = (i - len(exps_with_map)/2) * width + width/2
            ax.bar([xi + offset for xi in x], aps, width, 
                   label=exp['name'], alpha=0.7)
        
        ax.set_xlabel('Categoría', fontsize=12)
        ax.set_ylabel('Average Precision (AP)', fontsize=12)
        ax.set_title('Comparación de AP por Categoría', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(all_classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison_ap_per_class.png'), dpi=150)
        plt.close()
    
    # Gráfica 3: Pérdidas de validación (best)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    exps_with_val = [e for e in experiments if e.get('best_val_loss') is not None]
    if exps_with_val:
        names = [e['name'] for e in exps_with_val]
        val_losses = [e['best_val_loss'] for e in exps_with_val]
        
        bars = ax.bar(range(len(names)), val_losses, color='coral', alpha=0.7)
        ax.set_xlabel('Experimento', fontsize=12)
        ax.set_ylabel('Validation Loss (best)', fontsize=12)
        ax.set_title('Comparación de Mejor Pérdida de Validación', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Agregar valores sobre las barras
        for bar, value in zip(bars, val_losses):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison_val_loss.png'), dpi=150)
        plt.close()
    
    print(f"\nGráficas de comparación guardadas en: {output_dir}")


def main(args):
    """Función principal."""
    
    print("=" * 80)
    print("COMPARACIÓN DE EXPERIMENTOS")
    print("=" * 80)
    
    # Buscar todos los experimentos en el directorio
    if args.experiment_dirs:
        experiment_dirs = args.experiment_dirs
    else:
        # Buscar automáticamente en results/training
        base_dir = 'results/training'
        if not os.path.exists(base_dir):
            print(f"Error: No se encontró el directorio {base_dir}")
            return
        
        experiment_dirs = [
            os.path.join(base_dir, d) 
            for d in os.listdir(base_dir) 
            if os.path.isdir(os.path.join(base_dir, d))
        ]
    
    if not experiment_dirs:
        print("No se encontraron experimentos para comparar")
        return
    
    print(f"\nEncontrados {len(experiment_dirs)} experimentos:")
    for exp_dir in experiment_dirs:
        print(f"  - {exp_dir}")
    
    # Cargar resultados de cada experimento
    print("\nCargando resultados...")
    experiments = []
    for exp_dir in experiment_dirs:
        try:
            results = load_experiment_results(exp_dir)
            experiments.append(results)
            print(f"  ✓ {results['name']}")
        except Exception as e:
            print(f"  ✗ Error cargando {exp_dir}: {e}")
    
    if not experiments:
        print("\nNo se pudieron cargar experimentos")
        return
    
    # Crear tabla comparativa
    print("\n" + "=" * 80)
    print("TABLA COMPARATIVA")
    print("=" * 80)
    df = create_comparison_table(experiments)
    print("\n" + df.to_string(index=False))
    
    # Guardar tabla
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, 'comparison_table.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nTabla guardada en: {csv_path}")
    
    # Crear gráficas comparativas
    print("\n" + "=" * 80)
    print("GENERANDO GRÁFICAS COMPARATIVAS")
    print("=" * 80)
    create_comparison_plots(experiments, output_dir)
    
    # Resumen de métricas por clase
    print("\n" + "=" * 80)
    print("RESUMEN DE MÉTRICAS POR CLASE")
    print("=" * 80)
    
    for exp in experiments:
        if exp.get('mAP'):
            print(f"\n{exp['name']}:")
            print(f"  mAP: {exp['mAP']:.4f}")
            if exp.get('AP_per_class'):
                print("  AP por clase:")
                for cls, ap in sorted(exp['AP_per_class'].items()):
                    print(f"    {cls}: {ap:.4f}")
    
    print("\n" + "=" * 80)
    print("✓ Comparación completada")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Comparar resultados entre múltiples experimentos'
    )
    
    parser.add_argument(
        '--experiment-dirs',
        type=str,
        nargs='+',
        help='Lista de directorios de experimentos a comparar'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/comparison',
        help='Directorio donde guardar la comparación'
    )
    
    args = parser.parse_args()
    
    # Instalar pandas si no está disponible
    try:
        import pandas as pd
    except ImportError:
        print("pandas no está instalado. Instalando...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'pandas'])
        import pandas as pd
    
    main(args)