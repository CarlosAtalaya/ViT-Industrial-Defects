#!/usr/bin/env python3
"""
Script de evaluación mejorado para DEIMv2.
Genera métricas COMPARABLES con baselines ResNet-18 y EfficientNet.

Mejoras:
- Score threshold configurable
- IoU threshold configurable  
- AP/Precision/Recall por clase
- Formato JSON idéntico a baselines
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Agregar DEIMv2 al path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent if 'scripts' in str(SCRIPT_DIR) else SCRIPT_DIR.parent
DEIMV2_PATH = PROJECT_ROOT / "DEIMv2"
sys.path.insert(0, str(DEIMV2_PATH))

from engine.core import YAMLConfig


def load_model_and_config(checkpoint_path, config_path, device):
    """Carga modelo y configuración desde checkpoint."""
    print(f"\nCargando configuración desde: {config_path}")
    cfg = YAMLConfig(config_path)
    
    print(f"Cargando checkpoint desde: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Crear modelo
    model = cfg.model
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    print(f"✅ Modelo cargado (época {checkpoint.get('epoch', 'N/A')})")
    
    return model, cfg


@torch.no_grad()
def evaluate_on_test(model, cfg, device, ann_file, img_folder, score_threshold=0.15):
    """
    Evalúa modelo en conjunto de test.
    
    Args:
        score_threshold: Filtrar detecciones por debajo de este score (default: 0.15)
    """
    # Cargar dataset de test
    print(f"\nCargando dataset de test...")
    print(f"  Imágenes: {img_folder}")
    print(f"  Anotaciones: {ann_file}")
    print(f"  Score threshold: {score_threshold}")
    
    coco_gt = COCO(ann_file)
    
    # Realizar inferencia
    print("\n🚀 Realizando inferencia en conjunto de test...")
    results = []
    
    img_ids = sorted(coco_gt.getImgIds())
    
    # Importar transforms
    import torchvision.transforms as T
    from PIL import Image
    
    transform = T.Compose([
        T.Resize([1024, 1024]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for i, img_id in enumerate(img_ids):
        if (i + 1) % 50 == 0:
            print(f"  Procesadas {i+1}/{len(img_ids)} imágenes...")
        
        # Cargar imagen
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(img_folder, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        
        # Transformar
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Predicción
        outputs = model(img_tensor)

        # Extraer predicciones del formato DEIMv2
        pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
        pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 4]

        # Obtener scores y labels
        scores = pred_logits.softmax(-1).max(-1)[0]  
        labels = pred_logits.softmax(-1).argmax(-1)

        # Desnormalizar boxes y convertir formato
        img_h, img_w = img_info['height'], img_info['width']

        boxes = pred_boxes.cpu()
        boxes[:, 0] *= img_w  # cx
        boxes[:, 1] *= img_h  # cy
        boxes[:, 2] *= img_w  # w
        boxes[:, 3] *= img_h  # h

        # Convertir de [cx, cy, w, h] a [x, y, w, h] para COCO
        boxes_coco = boxes.clone()
        boxes_coco[:, 0] -= boxes_coco[:, 2] / 2  
        boxes_coco[:, 1] -= boxes_coco[:, 3] / 2

        scores = scores.cpu()
        labels = labels.cpu()

        # CRÍTICO: Filtrar por score threshold y clase válida
        for box, score, label in zip(boxes_coco, scores, labels):
            if label < 6 and score >= score_threshold:  # Solo clases 0-5 y score > threshold
                results.append({
                    'image_id': int(img_id),
                    'category_id': int(label),
                    'bbox': box.tolist(),
                    'score': float(score)
                })
    
    print(f"✅ Inferencia completada: {len(results)} detecciones (score >= {score_threshold})")
    
    return results, coco_gt


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
        # Filtrar evaluaciones para esta categoría e IoU específico
        # Índice IoU: 0 = 0.5, 1 = 0.55, ..., 9 = 0.95
        iou_idx = int((iou_threshold - 0.5) / 0.05) if iou_threshold >= 0.5 else 0
        
        # Extraer precision/recall de COCO eval
        precision = coco_eval.eval['precision'][iou_idx, :, cat_id, 0, 2]  # [recall_thresholds]
        recall_vals = coco_eval.eval['recall'][iou_idx, cat_id, 0, 2]  # scalar
        
        # AP es el área bajo la curva precision-recall
        ap = np.mean(precision[precision > -1])  # Ignorar -1 (no evaluado)
        
        # Precision y Recall al threshold 0.5 IoU
        # Usar máximos para aproximar valores "típicos"
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
    print("RESULTADOS DE EVALUACIÓN - TEST SET")
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
    
    # Guardar métricas
    results_file = os.path.join(output_dir, 'test_evaluation_results_comparable.json')
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✅ Resultados guardados en: {results_file}")
    
    # Guardar detecciones completas
    detections_file = os.path.join(output_dir, 'test_detections_filtered.json')
    with open(detections_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Detecciones guardadas en: {detections_file}")


def main(args):
    """Función principal de evaluación."""
    print("="*80)
    print("EVALUACIÓN DEIMV2 - DETECCIÓN DE DEFECTOS INDUSTRIALES")
    print("MODO: Comparable con baselines CNN")
    print("="*80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Cargar modelo
    model, cfg = load_model_and_config(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=device
    )
    
    # Evaluar en test con score threshold
    results, coco_gt = evaluate_on_test(
        model=model,
        cfg=cfg,
        device=device,
        ann_file=args.test_ann_file,
        img_folder=args.test_img_folder,
        score_threshold=args.score_threshold
    )
    
    # Calcular métricas con IoU threshold específico
    metrics, per_class = evaluate_with_iou_threshold(
        coco_gt=coco_gt,
        results=results,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold
    )
    
    # Obtener nombres de categorías
    category_names = {cat['id']: cat['name'] for cat in coco_gt.dataset['categories']}
    
    # Imprimir resultados
    print_results(metrics, per_class, category_names, args.iou_threshold, args.score_threshold)
    
    # Guardar resultados
    output_dir = os.path.dirname(args.checkpoint)
    save_results(
        metrics=metrics,
        per_class=per_class,
        results=results,
        output_dir=output_dir,
        num_test_images=len(coco_gt.getImgIds()),
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold
    )
    
    print("\n✅ Evaluación completada")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluar DEIMv2 con métricas comparables')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Ruta al checkpoint del modelo')
    parser.add_argument('--config', type=str, required=True,
                       help='Ruta al archivo de configuración')
    parser.add_argument('--test-img-folder', type=str, required=True,
                       help='Directorio con imágenes de test')
    parser.add_argument('--test-ann-file', type=str, required=True,
                       help='Archivo de anotaciones de test')
    parser.add_argument('--score-threshold', type=float, default=0.15,
                       help='Threshold de confianza para filtrar detecciones (default: 0.15)')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold para evaluación (default: 0.5)')
    
    args = parser.parse_args()
    main(args)