#!/usr/bin/env python3
"""
Script para recalcular métricas de evaluación con un score threshold más alto.

Este script toma las detecciones ya generadas (test_detections_filtered.json)
y recalcula las métricas filtrando por un score threshold más alto, sin necesidad
de re-evaluar el modelo completo.

Útil para verificar si un modelo está realmente bien o está overfitteado cuando
los resultados con threshold bajo muestran precision perfecta.
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def load_detections(detections_file):
    """Carga las detecciones desde el archivo JSON."""
    print(f"📂 Cargando detecciones desde: {detections_file}")
    with open(detections_file, 'r') as f:
        detections = json.load(f)
    print(f"   ✅ Cargadas {len(detections)} detecciones")
    return detections


def filter_by_score_threshold(detections, score_threshold):
    """Filtra detecciones por score threshold."""
    print(f"\n🔍 Filtrando detecciones con score >= {score_threshold}...")
    
    filtered = [det for det in detections if det.get('score', 0) >= score_threshold]
    
    print(f"   ✅ Detecciones después del filtro: {len(filtered)} (de {len(detections)})")
    print(f"   📊 Reducción: {len(detections) - len(filtered)} detecciones eliminadas")
    
    return filtered


def calculate_metrics_per_class(coco_gt, results, iou_threshold=0.5):
    """
    Calcula AP, Precision y Recall por clase.
    Compatible con formato de baselines CNN.
    """
    if len(results) == 0:
        print("⚠️  No hay detecciones, retornando métricas vacías")
        return {}
    
    # Evaluar con COCO API
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    
    # Extraer métricas por clase
    metrics_per_class = {}
    
    for cat_id, cat_info in enumerate(coco_gt.cats.values()):
        cat_name = cat_info['name']
        
        # AP por clase (usando eval_imgs)
        # Índice IoU: 0 = 0.5, 1 = 0.55, ..., 9 = 0.95
        iou_idx = int((iou_threshold - 0.5) / 0.05) if iou_threshold >= 0.5 else 0
        iou_idx = min(iou_idx, 9)  # Asegurar que no exceda el rango
        
        # Extraer precision/recall de COCO eval
        precision = coco_eval.eval['precision'][iou_idx, :, cat_id, 0, 2]  # [recall_thresholds]
        recall_vals = coco_eval.eval['recall'][iou_idx, cat_id, 0, 2]  # scalar
        
        # AP es el área bajo la curva precision-recall
        ap = np.mean(precision[precision > -1]) if len(precision[precision > -1]) > 0 else 0.0
        
        # Precision y Recall al threshold 0.5 IoU
        prec = np.max(precision[precision > -1]) if len(precision[precision > -1]) > 0 else 0.0
        rec = float(recall_vals) if recall_vals > -1 else 0.0
        
        metrics_per_class[cat_name] = {
            'AP': float(ap),
            'precision': float(prec),
            'recall': float(rec)
        }
    
    return metrics_per_class


def evaluate_with_iou_threshold(coco_gt, results, iou_threshold=0.5, score_threshold=0.15):
    """
    Evalúa con IoU threshold específico (compatible con baselines).
    """
    if len(results) == 0:
        return {
            'mAP': 0.0,
            'AP50': 0.0,
            'AP75': 0.0
        }, {}
    
    # Evaluación estándar COCO
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.iouThrs = np.array([iou_threshold])  # Solo evaluar IoU específico
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extraer métricas
    metrics = {
        'mAP': float(coco_eval.stats[0]),  # AP @ IoU=threshold
        'AP50': float(coco_eval.stats[1]) if len(coco_eval.stats) > 1 else float(coco_eval.stats[0]),
        'AP75': float(coco_eval.stats[2]) if len(coco_eval.stats) > 2 else float(coco_eval.stats[0])
    }
    
    # Métricas por clase
    per_class = calculate_metrics_per_class(coco_gt, results, iou_threshold)
    
    return metrics, per_class


def print_results(metrics, per_class, category_names, iou_threshold, score_threshold):
    """Imprime resultados en formato compatible con baselines."""
    print("\n" + "="*80)
    print("RESULTADOS DE EVALUACIÓN - TEST SET (RECALCULADO)")
    print("="*80)
    
    print(f"\n{'Parámetros de Evaluación:':<30}")
    print(f"  IoU Threshold:  {iou_threshold}")
    print(f"  Score Threshold: {score_threshold}")
    
    print(f"\n{'Métrica Global':<30} {'Valor':<10}")
    print("-"*80)
    print(f"{'mAP @ IoU=' + str(iou_threshold):<30} {metrics['mAP']:.4f}")
    
    print(f"\n{'Métricas por Clase':<30} {'AP':<10} {'Precision':<12} {'Recall':<10}")
    print("-"*80)
    
    for cat_name, cat_metrics in per_class.items():
        print(f"{cat_name:<30} {cat_metrics['AP']:.4f}    {cat_metrics['precision']:.4f}      {cat_metrics['recall']:.4f}")
    
    print("="*80)


def save_results(metrics, per_class, results, output_dir, num_test_images, 
                iou_threshold, score_threshold):
    """
    Guarda resultados en formato COMPATIBLE con baselines.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Formato JSON idéntico a ResNet/EfficientNet
    results_dict = {
        'mAP': metrics['mAP'],
        'iou_threshold': iou_threshold,
        'score_threshold': score_threshold,
        'num_test_images': num_test_images,
        'AP_per_class': {name: m['AP'] for name, m in per_class.items()},
        'precision_per_class': {name: m['precision'] for name, m in per_class.items()},
        'recall_per_class': {name: m['recall'] for name, m in per_class.items()}
    }
    
    # Guardar métricas con sufijo indicando el threshold
    results_file = os.path.join(output_dir, f'test_evaluation_results_comparable_th{score_threshold:.2f}.json')
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✅ Resultados guardados en: {results_file}")
    
    # Guardar detecciones filtradas
    detections_file = os.path.join(output_dir, f'test_detections_filtered_th{score_threshold:.2f}.json')
    with open(detections_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Detecciones guardadas en: {detections_file}")


def main(args):
    """Función principal."""
    print("="*80)
    print("RECÁLCULO DE MÉTRICAS CON SCORE THRESHOLD ALTO")
    print("="*80)
    print(f"\n📊 Score threshold objetivo: {args.score_threshold}")
    print(f"📊 IoU threshold: {args.iou_threshold}")
    
    # Cargar detecciones
    detections = load_detections(args.detections_file)
    
    # Filtrar por score threshold
    filtered_detections = filter_by_score_threshold(detections, args.score_threshold)
    
    # Cargar ground truth
    print(f"\n📂 Cargando anotaciones de test desde: {args.test_ann_file}")
    coco_gt = COCO(args.test_ann_file)
    print(f"   ✅ Cargadas {len(coco_gt.getImgIds())} imágenes de test")
    
    # Calcular métricas
    print(f"\n🔬 Calculando métricas...")
    metrics, per_class = evaluate_with_iou_threshold(
        coco_gt=coco_gt,
        results=filtered_detections,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold
    )
    
    # Obtener nombres de categorías
    category_names = {cat['id']: cat['name'] for cat in coco_gt.dataset['categories']}
    
    # Imprimir resultados
    print_results(metrics, per_class, category_names, args.iou_threshold, args.score_threshold)
    
    # Guardar resultados
    output_dir = os.path.dirname(args.detections_file)
    save_results(
        metrics=metrics,
        per_class=per_class,
        results=filtered_detections,
        output_dir=output_dir,
        num_test_images=len(coco_gt.getImgIds()),
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold
    )
    
    print("\n✅ Recálculo completado")
    
    # Comparación con threshold original si existe
    original_results_file = os.path.join(output_dir, 'test_evaluation_results_comparable.json')
    if os.path.exists(original_results_file):
        print("\n" + "="*80)
        print("COMPARACIÓN CON THRESHOLD ORIGINAL (0.15)")
        print("="*80)
        
        with open(original_results_file, 'r') as f:
            original = json.load(f)
        
        print(f"\n{'Métrica':<30} {'Original (0.15)':<20} {'Nuevo ({})':<20} {'Cambio':<15}")
        print("-"*85)
        print(f"{'mAP':<30} {original['mAP']:<20.4f} {metrics['mAP']:<20.4f} {metrics['mAP'] - original['mAP']:+.4f}")
        
        print(f"\n{'AP por Clase':<30} {'Original (0.15)':<20} {'Nuevo ({})':<20} {'Cambio':<15}")
        print("-"*85)
        for cat_name in category_names.values():
            orig_ap = original['AP_per_class'].get(cat_name, 0.0)
            new_ap = per_class.get(cat_name, {}).get('AP', 0.0)
            print(f"{cat_name:<30} {orig_ap:<20.4f} {new_ap:<20.4f} {new_ap - orig_ap:+.4f}")
        
        print(f"\n{'Precision por Clase':<30} {'Original (0.15)':<20} {'Nuevo ({})':<20} {'Cambio':<15}")
        print("-"*85)
        for cat_name in category_names.values():
            orig_prec = original['precision_per_class'].get(cat_name, 0.0)
            new_prec = per_class.get(cat_name, {}).get('precision', 0.0)
            print(f"{cat_name:<30} {orig_prec:<20.4f} {new_prec:<20.4f} {new_prec - orig_prec:+.4f}")
        
        print(f"\n{'Recall por Clase':<30} {'Original (0.15)':<20} {'Nuevo ({})':<20} {'Cambio':<15}")
        print("-"*85)
        for cat_name in category_names.values():
            orig_rec = original['recall_per_class'].get(cat_name, 0.0)
            new_rec = per_class.get(cat_name, {}).get('recall', 0.0)
            print(f"{cat_name:<30} {orig_rec:<20.4f} {new_rec:<20.4f} {new_rec - orig_rec:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Recalcular métricas de evaluación con score threshold más alto',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Recalcular con threshold 0.75
  python recalculate_metrics_with_threshold.py \\
    --detections-file outputs/deimv2_1024_300epochs/test_detections_filtered.json \\
    --test-ann-file ../../curated_dataset_splitted_20251101_provisional_1st_version/test/test.json \\
    --score-threshold 0.75

  # Recalcular con threshold 0.5
  python recalculate_metrics_with_threshold.py \\
    --detections-file outputs/deimv2_1024_300epochs/test_detections_filtered.json \\
    --test-ann-file ../../curated_dataset_splitted_20251101_provisional_1st_version/test/test.json \\
    --score-threshold 0.5
        """
    )
    
    parser.add_argument('--detections-file', type=str, required=True,
                       help='Ruta al archivo test_detections_filtered.json con las detecciones')
    parser.add_argument('--test-ann-file', type=str, required=True,
                       help='Archivo de anotaciones de test (COCO format)')
    parser.add_argument('--score-threshold', type=float, default=0.75,
                       help='Score threshold para filtrar detecciones (default: 0.75)')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold para evaluación (default: 0.5)')
    
    args = parser.parse_args()
    main(args)

