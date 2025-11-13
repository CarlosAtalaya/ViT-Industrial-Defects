"""
Dataset loader para detección de defectos industriales en formato COCO.
Soporta 6 categorías de defectos + NORMAL.
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from typing import Dict, List, Tuple, Optional
import numpy as np


class IndustrialDefectsDataset(Dataset):
    """
    Dataset para detección de defectos en componentes industriales.
    Formato COCO con anotaciones en bounding boxes.
    """
    
    def __init__(
        self, 
        root_dir: str, 
        split: str = 'train',
        transforms: Optional[object] = None
    ):
        """
        Args:
            root_dir: Ruta al directorio raíz del dataset
            split: 'train', 'val' o 'test'
            transforms: Transformaciones de albumentations (opcional)
        """
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        
        # Rutas a los archivos
        self.split_dir = os.path.join(root_dir, split)
        self.images_dir = os.path.join(self.split_dir, 'images')
        self.json_path = os.path.join(self.split_dir, f'{split}.json')
        
        # Cargar anotaciones COCO
        print(f"Cargando anotaciones desde: {self.json_path}")
        with open(self.json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        self.categories = self.coco_data['categories']
        
        # Crear mapeo de categorías
        self.category_id_to_name = {
            cat['id']: cat.get('unified_category_name', cat.get('name'))
            for cat in self.categories
        }
        self.category_name_to_id = {
            cat.get('unified_category_name', cat.get('name')): cat['id']
            for cat in self.categories
        }
        
        # Crear índice de anotaciones por imagen
        self.image_id_to_annotations = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append(ann)
        
        print(f"Dataset cargado: {len(self.images)} imágenes, "
              f"{len(self.annotations)} anotaciones, "
              f"{len(self.categories)} categorías")
        print(f"Categorías: {list(self.category_name_to_id.keys())}")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Retorna imagen y target en formato esperado por Faster R-CNN.
        
        Returns:
            image: tensor [C, H, W] normalizado
            target: dict con 'boxes', 'labels', 'image_id', 'area', 'iscrowd'
        """
        # Obtener información de la imagen
        img_info = self.images[idx]
        img_id = img_info['id']
        img_filename = img_info['file_name']
        img_path = os.path.join(self.images_dir, img_filename)
        
        # Cargar imagen
        image = Image.open(img_path).convert('RGB')
        
        # Obtener anotaciones para esta imagen
        annotations = self.image_id_to_annotations.get(img_id, [])
        
        # Preparar boxes y labels
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in annotations:
            # Obtener bbox en formato [x, y, width, height]
            bbox = ann['bbox']
            x, y, w, h = bbox
            
            # Convertir a formato [x_min, y_min, x_max, y_max]
            x_min = x
            y_min = y
            x_max = x + w
            y_max = y + h
            
            # Validar que el bbox sea válido
            if w > 0 and h > 0:
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(ann['category_id'])
                areas.append(ann.get('area', w * h))
                iscrowd.append(ann.get('iscrowd', 0))
        
        # Convertir a tensors
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            # Imagen sin anotaciones (posiblemente solo NORMAL)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        
        # Preparar target en formato Faster R-CNN
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': iscrowd
        }
        
        # Aplicar transformaciones si existen
        if self.transforms is not None:
            image = self.transforms(image)
        else:
            # Transformación por defecto: convertir a tensor
            image = T.ToTensor()(image)
        
        return image, target
    
    def get_category_distribution(self) -> Dict[str, int]:
        """Retorna la distribución de categorías en el dataset."""
        distribution = {}
        for ann in self.annotations:
            cat_name = self.category_id_to_name[ann['category_id']]
            distribution[cat_name] = distribution.get(cat_name, 0) + 1
        return distribution


def collate_fn(batch):
    """
    Función de colación personalizada para Faster R-CNN.
    Necesaria porque las imágenes pueden tener diferente número de objetos.
    """
    return tuple(zip(*batch))


def get_transform(train: bool = True):
    """
    Retorna transformaciones para imágenes.
    
    Args:
        train: Si True, incluye data augmentation
    """
    transforms = []
    transforms.append(T.ToTensor())
    
    # Normalización estándar de ImageNet (ResNet preentrenado en ImageNet)
    transforms.append(T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ))
    
    # TODO: Agregar data augmentation si train=True
    # Se puede usar albumentations para augmentations más avanzadas
    
    return T.Compose(transforms)


if __name__ == "__main__":
    # Test del dataset
    dataset_path = "curated_dataset_splitted_20251101_provisional_1st_version"
    
    print("=== Testing Train Dataset ===")
    train_dataset = IndustrialDefectsDataset(
        root_dir=dataset_path,
        split='train',
        transforms=get_transform(train=True)
    )
    
    print(f"\nNúmero de imágenes: {len(train_dataset)}")
    print(f"\nDistribución de categorías:")
    dist = train_dataset.get_category_distribution()
    for cat, count in sorted(dist.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")
    
    # Probar carga de una imagen
    print("\n=== Testing Sample Loading ===")
    img, target = train_dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Number of objects: {len(target['boxes'])}")
    print(f"Boxes: {target['boxes']}")
    print(f"Labels: {target['labels']}")
    print(f"Category names: {[train_dataset.category_id_to_name[l.item()] for l in target['labels']]}")