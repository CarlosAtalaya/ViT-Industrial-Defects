#!/usr/bin/env python3
"""
Script para visualizar predicciones de DEIMv2.
Genera imágenes con bounding boxes de predicciones y ground truth.
"""

import os
import sys
import argparse
import random
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

# Agregar DEIMv2 al path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEIMV2_PATH = PROJECT_ROOT / "DEIMv2"
sys.path.insert(0, str(DEIMV2_PATH))

from engine.core import YAMLConfig


# Colores para visualización
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']


def load_model_and_config(checkpoint_path, config_path, device):
    """Carga modelo y configuración."""
    cfg = YAMLConfig(config_path)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = cfg.model
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    return model, cfg


def denormalize_image(image_tensor):
    """Desnormaliza imagen normalizada con ImageNet stats."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    return image


@torch.no_grad()
def visualize_image(
    model,
    image_path,
    ground_truth_boxes,
    ground_truth_labels,
    category_names,
    device,
    cfg,
    score_threshold=0.5,
    save_path=None
):
    """Visualiza predicciones en una imagen."""
    
    # Cargar y preprocesar imagen
    from PIL import Image
    import torchvision.transforms as T
    
    img = Image.open(image_path).convert('RGB')
    orig_size = img.size  # (width, height)
    
    # Aplicar transformaciones
    transform = T.Compose([
        T.Resize([640, 640]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Predicción
    outputs = model(img_tensor)

    # Extraer del formato DEIMv2
    pred_logits = outputs['pred_logits'][0]
    pred_boxes_norm = outputs['pred_boxes'][0]

    # Scores y labels
    scores = pred_logits.softmax(-1).max(-1)[0].cpu()
    labels = pred_logits.softmax(-1).argmax(-1).cpu()

    # Filtrar por score y clase válida
    keep = (scores >= score_threshold) & (labels < 6)
    scores = scores[keep]
    labels = labels[keep]
    pred_boxes_norm = pred_boxes_norm[keep]

    # Desnormalizar boxes [cx, cy, w, h] -> [x1, y1, x2, y2]
    orig_h, orig_w = orig_size[1], orig_size[0]
    pred_boxes = pred_boxes_norm.cpu().clone()
    pred_boxes[:, 0] *= orig_w  # cx
    pred_boxes[:, 1] *= orig_h  # cy
    pred_boxes[:, 2] *= orig_w  # w
    pred_boxes[:, 3] *= orig_h  # h

    # [cx, cy, w, h] -> [x1, y1, x2, y2]
    boxes_xyxy = pred_boxes.clone()
    boxes_xyxy[:, 0] -= boxes_xyxy[:, 2] / 2  # x1
    boxes_xyxy[:, 1] -= boxes_xyxy[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes_xyxy[:, 0] + pred_boxes[:, 2]  # x2
    boxes_xyxy[:, 3] = boxes_xyxy[:, 1] + pred_boxes[:, 3]  # y2

    pred_boxes = boxes_xyxy.numpy()
    pred_labels = labels.numpy()
    pred_scores = scores.numpy()
    
    # Crear visualización
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Ground Truth
    ax1 = axes[0]
    ax1.imshow(img)
    ax1.set_title('Ground Truth', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    for box, label in zip(ground_truth_boxes, ground_truth_labels):
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h
        
        class_name = category_names.get(label, f"Class_{label}")
        color = COLORS[label % len(COLORS)]
        
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax1.add_patch(rect)
        
        ax1.text(
            x1, y1 - 5,
            class_name,
            color='white',
            fontsize=10,
            fontweight='bold',
            bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=2)
        )
    
    # Predictions
    ax2 = axes[1]
    ax2.imshow(img)
    ax2.set_title(f'Predictions (score ≥ {score_threshold})', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        class_name = category_names.get(label, f"Class_{label}")
        color = COLORS[label % len(COLORS)]
        
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax2.add_patch(rect)
        
        label_text = f"{class_name} {score:.2f}"
        ax2.text(
            x1, y1 - 5,
            label_text,
            color='white',
            fontsize=10,
            fontweight='bold',
            bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=2)
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main(args):
    """Función principal de visualización."""
    
    print("="*80)
    print("VISUALIZACIÓN DE PREDICCIONES - DEIMV2")
    print("="*80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Cargar modelo
    print(f"\nCargando modelo...")
    model, cfg = load_model_and_config(args.checkpoint, args.config, device)
    print("✅ Modelo cargado")
    
    # Cargar anotaciones COCO
    print(f"\nCargando anotaciones desde: {args.ann_file}")
    coco = COCO(args.ann_file)
    
    category_names = {cat['id']: cat['name'] for cat in coco.dataset['categories']}
    img_ids = sorted(coco.getImgIds())
    
    print(f"Total de imágenes: {len(img_ids)}")
    
    # Seleccionar imágenes
    if args.num_images == -1:
        selected_ids = img_ids
    else:
        if args.random:
            selected_ids = random.sample(img_ids, min(args.num_images, len(img_ids)))
        else:
            selected_ids = img_ids[:min(args.num_images, len(img_ids))]
    
    print(f"Visualizando {len(selected_ids)} imágenes...\n")
    
    # Crear directorio de salida
    output_dir = os.path.join(os.path.dirname(args.checkpoint), 'visualizations_test')
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualizar cada imagen
    for i, img_id in enumerate(selected_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(args.img_folder, img_info['file_name'])
        
        # Obtener ground truth
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        gt_boxes = [ann['bbox'] for ann in anns]
        gt_labels = [ann['category_id'] for ann in anns]
        
        # Visualizar
        save_path = os.path.join(output_dir, f"{os.path.splitext(img_info['file_name'])[0]}_prediction.png")
        
        visualize_image(
            model=model,
            image_path=img_path,
            ground_truth_boxes=gt_boxes,
            ground_truth_labels=gt_labels,
            category_names=category_names,
            device=device,
            cfg=cfg,
            score_threshold=args.score_threshold,
            save_path=save_path
        )
        
        if (i + 1) % 10 == 0:
            print(f"  Procesadas {i + 1}/{len(selected_ids)} imágenes")
    
    print(f"\n✅ Visualizaciones guardadas en: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualizar predicciones de DEIMv2')
    
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--img-folder', type=str, required=True)
    parser.add_argument('--ann-file', type=str, required=True)
    parser.add_argument('--num-images', type=int, default=20)
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--score-threshold', type=float, default=0.85)
    
    args = parser.parse_args()
    main(args)