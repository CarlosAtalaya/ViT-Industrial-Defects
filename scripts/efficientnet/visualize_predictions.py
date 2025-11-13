"""
Script para visualizar predicciones del modelo de detección.
Genera imágenes con bounding boxes de predicciones y ground truth.
"""

import os
import argparse
import random

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from dataset_industrial_defects_loader import (
    IndustrialDefectsDataset,
    collate_fn,
    get_transform
)
from train_efficientnet_fasterrcnn import get_model_efficientnet_fasterrcnn


# Colores para cada categoría (distintos para visualización)
COLORS = [
    '#FF6B6B',  # Rojo
    '#4ECDC4',  # Turquesa
    '#45B7D1',  # Azul
    '#FFA07A',  # Salmón
    '#98D8C8',  # Verde menta
    '#F7DC6F',  # Amarillo
    '#BB8FCE',  # Púrpura
]


def denormalize_image(image_tensor):
    """
    Desnormaliza una imagen normalizada con estadísticas de ImageNet.
    
    Args:
        image_tensor: Tensor [C, H, W] normalizado
    
    Returns:
        Array numpy [H, W, C] en rango [0, 1]
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Convertir a numpy y transponer
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    
    # Desnormalizar
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    return image


def visualize_predictions(
    image,
    predictions,
    ground_truth,
    category_id_to_name,
    score_threshold=0.5,
    save_path=None,
    show=False
):
    """
    Visualiza predicciones y ground truth en una imagen.
    
    Args:
        image: Tensor de imagen [C, H, W]
        predictions: Dict con 'boxes', 'labels', 'scores'
        ground_truth: Dict con 'boxes', 'labels'
        category_id_to_name: Mapeo de IDs a nombres
        score_threshold: Umbral de confianza
        save_path: Ruta para guardar la imagen
        show: Si mostrar la imagen con plt.show()
    """
    # Desnormalizar imagen
    img_array = denormalize_image(image)
    
    # Crear figura con dos subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Ground Truth
    ax1 = axes[0]
    ax1.imshow(img_array)
    ax1.set_title('Ground Truth', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    for box, label in zip(ground_truth['boxes'], ground_truth['labels']):
        x_min, y_min, x_max, y_max = box.cpu().numpy()
        width = x_max - x_min
        height = y_max - y_min
        
        class_id = label.item()
        class_name = category_id_to_name.get(class_id, f"Class_{class_id}")
        color = COLORS[class_id % len(COLORS)]
        
        # Dibujar bounding box
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax1.add_patch(rect)
        
        # Agregar etiqueta
        ax1.text(
            x_min, y_min - 5,
            class_name,
            color='white',
            fontsize=10,
            fontweight='bold',
            bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=2)
        )
    
    # Predictions
    ax2 = axes[1]
    ax2.imshow(img_array)
    ax2.set_title('Predictions', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Filtrar por score threshold
    keep_indices = predictions['scores'] >= score_threshold
    pred_boxes = predictions['boxes'][keep_indices]
    pred_labels = predictions['labels'][keep_indices]
    pred_scores = predictions['scores'][keep_indices]
    
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        x_min, y_min, x_max, y_max = box.cpu().numpy()
        width = x_max - x_min
        height = y_max - y_min
        
        class_id = label.item()
        class_name = category_id_to_name.get(class_id, f"Class_{class_id}")
        color = COLORS[class_id % len(COLORS)]
        
        # Dibujar bounding box
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax2.add_patch(rect)
        
        # Agregar etiqueta con score
        label_text = f"{class_name} {score:.2f}"
        ax2.text(
            x_min, y_min - 5,
            label_text,
            color='white',
            fontsize=10,
            fontweight='bold',
            bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=2)
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Imagen guardada en: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


@torch.no_grad()
def main(args):
    """Función principal de visualización."""
    
    print("=" * 80)
    print("VISUALIZACIÓN DE PREDICCIONES")
    print("=" * 80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Cargar dataset
    print(f"\nCargando dataset de {args.split}...")
    dataset = IndustrialDefectsDataset(
        root_dir=args.dataset_path,
        split=args.split,
        transforms=get_transform(train=False)
    )
    
    print(f"Número de imágenes: {len(dataset)}")
    
    # Cargar modelo
    print(f"\nCargando modelo desde: {args.checkpoint}")
    
    num_classes = len(dataset.categories) + 1
    model = get_model_efficientnet_fasterrcnn(
        num_classes=num_classes,
        pretrained_backbone=False
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Modelo cargado (época {checkpoint['epoch']})")
    
    # Crear directorio de salida
    output_dir = os.path.join(
        os.path.dirname(args.checkpoint),
        f'visualizations_{args.split}'
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGuardando visualizaciones en: {output_dir}")
    
    # Seleccionar imágenes a visualizar
    if args.num_images == -1:
        indices = list(range(len(dataset)))
    else:
        if args.random:
            indices = random.sample(range(len(dataset)), min(args.num_images, len(dataset)))
        else:
            indices = list(range(min(args.num_images, len(dataset))))
    
    print(f"\nVisualizando {len(indices)} imágenes...")
    
    # Visualizar cada imagen
    for i, idx in enumerate(indices):
        # Cargar imagen y ground truth
        image, target = dataset[idx]
        
        # Hacer predicción
        image_batch = image.unsqueeze(0).to(device)
        predictions = model(image_batch)[0]
        
        # Obtener nombre de archivo
        img_info = dataset.images[idx]
        img_filename = os.path.splitext(img_info['file_name'])[0]
        
        # Guardar visualización
        save_path = os.path.join(output_dir, f'{img_filename}_prediction.png')
        
        visualize_predictions(
            image=image,
            predictions=predictions,
            ground_truth=target,
            category_id_to_name=dataset.category_id_to_name,
            score_threshold=args.score_threshold,
            save_path=save_path,
            show=args.show
        )
        
        if (i + 1) % 10 == 0:
            print(f"  Procesadas {i + 1}/{len(indices)} imágenes")
    
    print(f"\n✓ Visualizaciones completadas y guardadas en: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualizar predicciones del modelo de detección'
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
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Split del dataset a visualizar'
    )
    parser.add_argument(
        '--num-images',
        type=int,
        default=20,
        help='Número de imágenes a visualizar (-1 para todas)'
    )
    parser.add_argument(
        '--random',
        action='store_true',
        help='Seleccionar imágenes aleatoriamente'
    )
    parser.add_argument(
        '--score-threshold',
        type=float,
        default=0.5,
        help='Umbral de confianza para filtrar predicciones'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Mostrar imágenes con plt.show()'
    )
    
    args = parser.parse_args()
    main(args)