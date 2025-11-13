"""
Script de diagnóstico para analizar problemas de bajo mAP.
Revisa predicciones, distribución de clases y posibles problemas.
"""

import torch
import json
import argparse
from collections import Counter, defaultdict

from dataset_industrial_defects_loader import IndustrialDefectsDataset, get_transform, collate_fn
from torch.utils.data import DataLoader
from train_efficientnet_fasterrcnn import get_model_efficientnet_fasterrcnn


@torch.no_grad()
def diagnose_model(args):
    """Diagnostica problemas del modelo."""
    
    print("=" * 80)
    print("DIAGNÓSTICO DEL MODELO")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cargar dataset
    test_dataset = IndustrialDefectsDataset(
        root_dir=args.dataset_path,
        split='test',
        transforms=get_transform(train=False)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Cargar modelo
    num_classes = len(test_dataset.categories) + 1
    model = get_model_efficientnet_fasterrcnn(num_classes=num_classes, pretrained_backbone=False)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"\nModelo cargado: época {checkpoint['epoch']}")
    print(f"Número de clases: {num_classes}")
    print(f"IDs de categorías en dataset: {list(test_dataset.category_id_to_name.keys())}")
    print(f"Nombres: {list(test_dataset.category_id_to_name.values())}")
    
    # Analizar predicciones
    print("\n" + "=" * 80)
    print("ANALIZANDO PREDICCIONES")
    print("=" * 80)
    
    all_predictions = []
    all_scores = []
    predicted_classes = []
    gt_classes = []
    
    num_images_with_detections = 0
    num_images_without_detections = 0
    
    for i, (images, targets) in enumerate(test_loader):
        images = list(image.to(device) for image in images)
        
        # Predicción
        outputs = model(images)
        
        for output, target in zip(outputs, targets):
            # Ground truth
            gt_labels = target['labels'].cpu().numpy()
            for label in gt_labels:
                gt_classes.append(label)
            
            # Predicciones
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()
            
            all_scores.extend(scores)
            
            if len(scores) > 0:
                num_images_with_detections += 1
                
                # Filtrar por threshold
                keep = scores >= args.score_threshold
                filtered_labels = labels[keep]
                filtered_scores = scores[keep]
                
                for label, score in zip(filtered_labels, filtered_scores):
                    predicted_classes.append(label)
                    all_predictions.append({
                        'label': int(label),
                        'score': float(score)
                    })
            else:
                num_images_without_detections += 1
        
        if (i + 1) % 50 == 0:
            print(f"  Procesadas {i+1}/{len(test_loader)} imágenes...")
    
    # Estadísticas
    print("\n" + "=" * 80)
    print("ESTADÍSTICAS")
    print("=" * 80)
    
    print(f"\nImágenes con detecciones: {num_images_with_detections}/{len(test_dataset)}")
    print(f"Imágenes sin detecciones: {num_images_without_detections}/{len(test_dataset)}")
    
    print(f"\nTotal de predicciones (antes de threshold): {len(all_scores)}")
    print(f"Total de predicciones (después de threshold={args.score_threshold}): {len(all_predictions)}")
    
    # Distribución de scores
    if len(all_scores) > 0:
        import numpy as np
        scores_array = np.array(all_scores)
        print(f"\nDistribución de scores:")
        print(f"  Min: {scores_array.min():.4f}")
        print(f"  Max: {scores_array.max():.4f}")
        print(f"  Media: {scores_array.mean():.4f}")
        print(f"  Mediana: {np.median(scores_array):.4f}")
        print(f"  Scores > 0.5: {(scores_array > 0.5).sum()}")
        print(f"  Scores > 0.3: {(scores_array > 0.3).sum()}")
        print(f"  Scores > 0.1: {(scores_array > 0.1).sum()}")
    
    # Clases predichas
    print(f"\n{'='*80}")
    print("DISTRIBUCIÓN DE CLASES PREDICHAS (score >= {})".format(args.score_threshold))
    print(f"{'='*80}")
    
    pred_counter = Counter(predicted_classes)
    for class_id, count in pred_counter.most_common():
        class_name = test_dataset.category_id_to_name.get(class_id, f"Unknown_{class_id}")
        print(f"  Clase {class_id} ({class_name}): {count} predicciones")
    
    if len(pred_counter) == 0:
        print("  ⚠️  NO HAY PREDICCIONES CON SCORE >= {}".format(args.score_threshold))
        print("  Prueba reducir el --score-threshold")
    
    # Ground truth
    print(f"\n{'='*80}")
    print("DISTRIBUCIÓN DE CLASES EN GROUND TRUTH")
    print(f"{'='*80}")
    
    gt_counter = Counter(gt_classes)
    for class_id, count in gt_counter.most_common():
        class_name = test_dataset.category_id_to_name.get(class_id, f"Unknown_{class_id}")
        print(f"  Clase {class_id} ({class_name}): {count} anotaciones")
    
    # Diagnóstico
    print(f"\n{'='*80}")
    print("DIAGNÓSTICO")
    print(f"{'='*80}")
    
    if num_images_without_detections > len(test_dataset) * 0.5:
        print("\n⚠️  PROBLEMA: Más del 50% de imágenes sin detecciones")
        print("Posibles causas:")
        print("  - Modelo no converge bien (revisar training loss)")
        print("  - Score threshold muy alto (prueba 0.1 o 0.3)")
        print("  - Learning rate inadecuado")
    
    if len(pred_counter) == 0:
        print("\n⚠️  PROBLEMA: NO hay predicciones con score >= {}".format(args.score_threshold))
        print("Solución: Reducir score_threshold a 0.1 o 0.3")
    
    if len(pred_counter) > 0 and len(pred_counter) < len(test_dataset.categories) / 2:
        print(f"\n⚠️  PROBLEMA: Solo predice {len(pred_counter)} de {len(test_dataset.categories)} clases")
        print("Posibles causas:")
        print("  - Dataset desbalanceado (algunas clases tienen pocas imágenes)")
        print("  - Modelo necesita más épocas de entrenamiento")
        print("  - Loss weighting para clases minoritarias")
    
    if len(all_scores) > 0:
        max_score = max(all_scores)
        if max_score < 0.7:
            print(f"\n⚠️  PROBLEMA: Score máximo muy bajo ({max_score:.4f})")
            print("El modelo no está confiado en sus predicciones")
            print("Posibles causas:")
            print("  - Modelo no entrenó suficientes épocas")
            print("  - Datos de train/test muy diferentes")
            print("  - Necesita más data augmentation")
    
    # Recomendaciones
    print(f"\n{'='*80}")
    print("RECOMENDACIONES")
    print(f"{'='*80}")
    
    print("\n1. EVALUAR CON SCORE THRESHOLD MÁS BAJO:")
    print("   python evaluate_model.py --checkpoint ... --score-threshold 0.3")
    
    print("\n2. REVISAR TRAINING LOSS:")
    print("   python plot_training_metrics.py --history-path ...")
    
    print("\n3. VISUALIZAR PREDICCIONES:")
    print("   python visualize_predictions.py --checkpoint ... --score-threshold 0.3")
    
    print("\n4. SI EL MODELO NO CONVERGE:")
    print("   - Entrenar más épocas (30-50 en lugar de 20)")
    print("   - Reducir learning rate (0.001 en lugar de 0.005)")
    print("   - Usar data augmentation más fuerte")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diagnosticar problemas del modelo')
    
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint del modelo')
    parser.add_argument('--dataset-path', type=str, required=True, help='Ruta al dataset')
    parser.add_argument('--score-threshold', type=float, default=0.5, help='Score threshold')
    
    args = parser.parse_args()
    
    diagnose_model(args)