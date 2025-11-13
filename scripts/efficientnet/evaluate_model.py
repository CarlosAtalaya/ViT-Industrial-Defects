"""
Script de evaluación para detección de defectos industriales.
Calcula métricas mAP (mean Average Precision) en conjunto de test.
"""

import os
import json
import argparse
from typing import Dict, List
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from dataset_industrial_defects_loader import (
    IndustrialDefectsDataset,
    collate_fn,
    get_transform
)
from train_efficientnet_fasterrcnn import get_model_efficientnet_fasterrcnn


def calculate_iou(box1, box2):
    """
    Calcula IoU (Intersection over Union) entre dos bounding boxes.
    
    Args:
        box1, box2: [x_min, y_min, x_max, y_max]
    
    Returns:
        IoU score (float)
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Área de intersección
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Áreas de las cajas
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # IoU
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0
    
    return iou


def compute_ap(recalls, precisions):
    """
    Calcula Average Precision usando el método de interpolación de 11 puntos.
    
    Args:
        recalls: lista de recall values
        precisions: lista de precision values
    
    Returns:
        Average Precision
    """
    # Agregar sentinel values
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    # Compute precision envelope
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    
    # Integrar área bajo la curva
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap


def evaluate_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    num_classes: int,
    iou_threshold: float = 0.5
) -> Dict:
    """
    Calcula métricas mAP (mean Average Precision).
    
    Args:
        predictions: Lista de predicciones por imagen
        ground_truths: Lista de ground truths por imagen
        num_classes: Número de clases (sin contar background)
        iou_threshold: Umbral de IoU para considerar TP
    
    Returns:
        Dict con métricas por clase y mAP global
    """
    # Organizar por clase
    class_predictions = defaultdict(list)  # {class_id: [(score, image_id, box), ...]}
    class_ground_truths = defaultdict(list)  # {class_id: [(image_id, box), ...]}
    class_gt_count = defaultdict(int)  # {class_id: count}
    
    # Procesar ground truths
    for img_id, gt in enumerate(ground_truths):
        for box, label in zip(gt['boxes'], gt['labels']):
            class_id = label.item()
            if class_id == 0:  # Skip background
                continue
            class_ground_truths[class_id].append((img_id, box.cpu().numpy()))
            class_gt_count[class_id] += 1
    
    # Procesar predicciones
    for img_id, pred in enumerate(predictions):
        for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
            class_id = label.item()
            if class_id == 0:  # Skip background
                continue
            class_predictions[class_id].append((score.item(), img_id, box.cpu().numpy()))
    
    # Calcular AP por clase
    aps = {}
    precisions_per_class = {}
    recalls_per_class = {}
    
    for class_id in range(1, num_classes):  # Skip background (class 0)
        if class_id not in class_predictions:
            aps[class_id] = 0.0
            precisions_per_class[class_id] = 0.0
            recalls_per_class[class_id] = 0.0
            continue
        
        # Ordenar predicciones por score (descendente)
        predictions_class = sorted(
            class_predictions[class_id],
            key=lambda x: x[0],
            reverse=True
        )
        
        # Marcar ground truths como detectados
        gt_detected = defaultdict(lambda: defaultdict(bool))
        
        tp = np.zeros(len(predictions_class))
        fp = np.zeros(len(predictions_class))
        
        for idx, (score, img_id, pred_box) in enumerate(predictions_class):
            # Obtener ground truths de esta imagen y clase
            gts_for_image = [
                (gt_box, gt_idx)
                for gt_idx, (gt_img_id, gt_box) in enumerate(class_ground_truths[class_id])
                if gt_img_id == img_id
            ]
            
            if len(gts_for_image) == 0:
                fp[idx] = 1
                continue
            
            # Calcular IoU con todos los GTs de esta imagen
            ious = [calculate_iou(pred_box, gt_box) for gt_box, _ in gts_for_image]
            max_iou = max(ious)
            max_iou_idx = np.argmax(ious)
            gt_idx = gts_for_image[max_iou_idx][1]
            
            if max_iou >= iou_threshold:
                if not gt_detected[img_id][gt_idx]:
                    tp[idx] = 1
                    gt_detected[img_id][gt_idx] = True
                else:
                    fp[idx] = 1  # Ya detectado (duplicate detection)
            else:
                fp[idx] = 1
        
        # Calcular precision y recall acumulativos
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / max(class_gt_count[class_id], 1)
        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-6)
        
        # Calcular AP
        ap = compute_ap(recalls, precisions)
        aps[class_id] = ap
        
        # Guardar precision y recall finales
        precisions_per_class[class_id] = precisions[-1] if len(precisions) > 0 else 0.0
        recalls_per_class[class_id] = recalls[-1] if len(recalls) > 0 else 0.0
    
    # Calcular mAP
    mean_ap = np.mean(list(aps.values())) if aps else 0.0
    
    return {
        'mAP': mean_ap,
        'AP_per_class': aps,
        'precision_per_class': precisions_per_class,
        'recall_per_class': recalls_per_class
    }


@torch.no_grad()
def evaluate_model(model, data_loader, device, category_id_to_name, score_threshold=0.5):
    """
    Evalúa el modelo y retorna predicciones y ground truths.
    
    Args:
        model: Modelo a evaluar
        data_loader: DataLoader del conjunto de test
        device: Device (cuda/cpu)
        category_id_to_name: Mapeo de IDs a nombres de categorías
        score_threshold: Umbral de confianza para filtrar predicciones
    
    Returns:
        predictions, ground_truths
    """
    model.eval()
    
    predictions = []
    ground_truths = []
    
    print("\nRealizando inferencia en conjunto de test...")
    for images, targets in tqdm(data_loader, desc="Evaluando"):
        images = list(image.to(device) for image in images)
        
        # Inferencia
        outputs = model(images)
        
        # Procesar predicciones
        for output, target in zip(outputs, targets):
            # Filtrar por score threshold
            keep_indices = output['scores'] >= score_threshold
            
            pred = {
                'boxes': output['boxes'][keep_indices],
                'labels': output['labels'][keep_indices],
                'scores': output['scores'][keep_indices]
            }
            predictions.append(pred)
            
            # Ground truth
            gt = {
                'boxes': target['boxes'],
                'labels': target['labels']
            }
            ground_truths.append(gt)
    
    return predictions, ground_truths


def print_evaluation_results(metrics, category_id_to_name):
    """Imprime resultados de evaluación de forma legible."""
    print("\n" + "=" * 80)
    print("RESULTADOS DE EVALUACIÓN")
    print("=" * 80)
    
    print(f"\n{'Clase':<25} {'AP':<10} {'Precision':<12} {'Recall':<10}")
    print("-" * 80)
    
    for class_id, ap in sorted(metrics['AP_per_class'].items()):
        class_name = category_id_to_name.get(class_id, f"Unknown_Class_{class_id}")
        precision = metrics['precision_per_class'].get(class_id, 0.0)
        recall = metrics['recall_per_class'].get(class_id, 0.0)
        print(f"{class_name:<25} {ap:<10.4f} {precision:<12.4f} {recall:<10.4f}")
    
    print("-" * 80)
    print(f"{'mAP (mean Average Precision)':<25} {metrics['mAP']:.4f}")
    print("=" * 80)


def main(args):
    """Función principal de evaluación."""
    
    print("=" * 80)
    print("EVALUACIÓN DE DETECCIÓN DE DEFECTOS INDUSTRIALES")
    print("=" * 80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Cargar dataset de test
    print("\nCargando dataset de test...")
    test_dataset = IndustrialDefectsDataset(
        root_dir=args.dataset_path,
        split='test',
        transforms=get_transform(train=False)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Número de imágenes de test: {len(test_dataset)}")
    print(f"Número de batches: {len(test_loader)}")
    
    # Cargar modelo
    print(f"\nCargando modelo desde: {args.checkpoint}")
    
    num_classes = len(test_dataset.categories) + 1  # +1 para background
    model = get_model_efficientnet_fasterrcnn(
        num_classes=num_classes,
        pretrained_backbone=False
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Modelo cargado (época {checkpoint['epoch']})")
    
    # Evaluar
    predictions, ground_truths = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        category_id_to_name=test_dataset.category_id_to_name,
        score_threshold=args.score_threshold
    )
    
    # Calcular métricas
    print("\nCalculando métricas mAP...")
    metrics = evaluate_map(
        predictions=predictions,
        ground_truths=ground_truths,
        num_classes=num_classes,
        iou_threshold=args.iou_threshold
    )
    
    # Imprimir resultados
    print_evaluation_results(metrics, test_dataset.category_id_to_name)
    
    # Guardar resultados
    output_dir = os.path.dirname(args.checkpoint)
    results_file = os.path.join(output_dir, 'test_evaluation_results.json')
    
    # Convertir métricas a formato serializable
    results = {
        'mAP': float(metrics['mAP']),
        'iou_threshold': args.iou_threshold,
        'score_threshold': args.score_threshold,
        'num_test_images': len(test_dataset),
        'AP_per_class': {
            test_dataset.category_id_to_name.get(k, f"Class_{k}"): float(v) 
            for k, v in metrics['AP_per_class'].items()
        },
        'precision_per_class': {
            test_dataset.category_id_to_name.get(k, f"Class_{k}"): float(v)
            for k, v in metrics['precision_per_class'].items()
        },
        'recall_per_class': {
            test_dataset.category_id_to_name.get(k, f"Class_{k}"): float(v)
            for k, v in metrics['recall_per_class'].items()
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResultados guardados en: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluar modelo de detección en conjunto de test'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Ruta al checkpoint del modelo'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='curated_dataset_splitted_20251101_provisional_1st_version',
        help='Ruta al dataset'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Número de workers para DataLoader'
    )
    parser.add_argument(
        '--score-threshold',
        type=float,
        default=0.5,
        help='Umbral de confianza para filtrar predicciones'
    )
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.5,
        help='Umbral de IoU para considerar True Positive'
    )
    
    args = parser.parse_args()
    main(args)