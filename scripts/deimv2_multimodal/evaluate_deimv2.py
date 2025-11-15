#!/usr/bin/env python3
"""
Script de evaluación para DEIMv2 en detección de defectos industriales.
Calcula métricas COCO mAP en conjunto de test.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Agregar DEIMv2 al path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
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
def evaluate_on_test(model, cfg, device, ann_file, img_folder):
    """Evalúa modelo en conjunto de test usando COCO API."""
    
    # Cargar dataset de test
    print(f"\nCargando dataset de test...")
    print(f"  Imágenes: {img_folder}")
    print(f"  Anotaciones: {ann_file}")
    
    coco_gt = COCO(ann_file)
    
    # Realizar inferencia directamente sin dataloader
    print("\n🚀 Realizando inferencia en conjunto de test...")
    results = []
    
    img_ids = sorted(coco_gt.getImgIds())
    
    # Importar transforms
    import torchvision.transforms as T
    from PIL import Image
    
    transform = T.Compose([
        T.Resize([640, 640]),
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
        pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 4] en formato [cx, cy, w, h] normalizado

        # Obtener scores y labels
        scores = pred_logits.softmax(-1).max(-1)[0]  # Máxima probabilidad
        labels = pred_logits.softmax(-1).argmax(-1)   # Clase predicha

        # Desnormalizar boxes y convertir de [cx, cy, w, h] a [x1, y1, x2, y2]
        img_h, img_w = img_info['height'], img_info['width']

        boxes = pred_boxes.cpu()
        boxes[:, 0] *= img_w  # cx
        boxes[:, 1] *= img_h  # cy
        boxes[:, 2] *= img_w  # w
        boxes[:, 3] *= img_h  # h

        # Convertir de [cx, cy, w, h] a [x, y, w, h] para COCO
        boxes_coco = boxes.clone()
        boxes_coco[:, 0] -= boxes_coco[:, 2] / 2  # x = cx - w/2
        boxes_coco[:, 1] -= boxes_coco[:, 3] / 2  # y = cy - h/2

        scores = scores.cpu()
        labels = labels.cpu()

        for box, score, label in zip(boxes_coco, scores, labels):
            # Filtrar clase background (típicamente la última)
            if label < 6:  # Solo las 6 clases de defectos (0-5)
                results.append({
                    'image_id': int(img_id),
                    'category_id': int(label),
                    'bbox': box.tolist(),
                    'score': float(score)
                })
    
    print(f"✅ Inferencia completada: {len(results)} detecciones")
    
    # Evaluar con COCO API
    print("\n📊 Calculando métricas COCO mAP...")
    
    if len(results) == 0:
        print("⚠️  No hay detecciones. mAP = 0.0")
        return {
            'mAP': 0.0,
            'AP50': 0.0,
            'AP75': 0.0,
            'mAP_small': 0.0,
            'mAP_medium': 0.0,
            'mAP_large': 0.0
        }, results
    
    # Guardar resultados temporalmente
    coco_dt = coco_gt.loadRes(results)
    
    # Evaluar
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extraer métricas
    metrics = {
        'mAP': float(coco_eval.stats[0]),           # AP @ IoU=0.50:0.95
        'AP50': float(coco_eval.stats[1]),          # AP @ IoU=0.50
        'AP75': float(coco_eval.stats[2]),          # AP @ IoU=0.75
        'mAP_small': float(coco_eval.stats[3]),     # AP small
        'mAP_medium': float(coco_eval.stats[4]),    # AP medium
        'mAP_large': float(coco_eval.stats[5]),     # AP large
        'AR_max1': float(coco_eval.stats[6]),       # AR @ maxDets=1
        'AR_max10': float(coco_eval.stats[7]),      # AR @ maxDets=10
        'AR_max100': float(coco_eval.stats[8]),     # AR @ maxDets=100
    }
    
    return metrics, results


def print_results(metrics, category_names):
    """Imprime resultados de forma legible."""
    print("\n" + "="*80)
    print("RESULTADOS DE EVALUACIÓN - TEST SET")
    print("="*80)
    
    print(f"\n{'Métrica':<30} {'Valor':<10}")
    print("-"*80)
    print(f"{'mAP @ IoU=0.50:0.95':<30} {metrics['mAP']:.4f}")
    print(f"{'AP @ IoU=0.50':<30} {metrics['AP50']:.4f}")
    print(f"{'AP @ IoU=0.75':<30} {metrics['AP75']:.4f}")
    print(f"{'mAP (small objects)':<30} {metrics['mAP_small']:.4f}")
    print(f"{'mAP (medium objects)':<30} {metrics['mAP_medium']:.4f}")
    print(f"{'mAP (large objects)':<30} {metrics['mAP_large']:.4f}")
    print(f"{'AR @ maxDets=100':<30} {metrics['AR_max100']:.4f}")
    print("="*80)


def save_results(metrics, results, output_dir, num_test_images):
    """Guarda resultados en JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar métricas
    results_file = os.path.join(output_dir, 'test_evaluation_results.json')
    summary = {
        'num_test_images': num_test_images,
        'num_detections': len(results),
        'metrics': metrics
    }
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Resultados guardados en: {results_file}")
    
    # Guardar detecciones completas
    detections_file = os.path.join(output_dir, 'test_detections.json')
    with open(detections_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Detecciones guardadas en: {detections_file}")


def main(args):
    """Función principal de evaluación."""
    
    print("="*80)
    print("EVALUACIÓN DEIMV2 - DETECCIÓN DE DEFECTOS INDUSTRIALES")
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
    
    # Evaluar en test
    metrics, results = evaluate_on_test(
        model=model,
        cfg=cfg,
        device=device,
        ann_file=args.test_ann_file,
        img_folder=args.test_img_folder
    )
    
    # Imprimir resultados
    coco_gt = COCO(args.test_ann_file)
    category_names = {cat['id']: cat['name'] for cat in coco_gt.dataset['categories']}
    print_results(metrics, category_names)
    
    # Guardar resultados
    output_dir = os.path.dirname(args.checkpoint)
    save_results(metrics, results, output_dir, len(coco_gt.getImgIds()))
    
    print("\n✅ Evaluación completada")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluar DEIMv2 en test set')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Ruta al checkpoint del modelo')
    parser.add_argument('--config', type=str, required=True,
                       help='Ruta al archivo de configuración')
    parser.add_argument('--test-img-folder', type=str, required=True,
                       help='Directorio con imágenes de test')
    parser.add_argument('--test-ann-file', type=str, required=True,
                       help='Archivo de anotaciones de test')
    
    args = parser.parse_args()
    main(args)
