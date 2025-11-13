"""
Script de entrenamiento para detección de defectos industriales.
Usa ResNet-18 como backbone en Faster R-CNN.
"""

import os
import sys
import json
import time
from datetime import datetime
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import resnet18, ResNet18_Weights

from dataset_industrial_defects_loader import (
    IndustrialDefectsDataset, 
    collate_fn, 
    get_transform
)


def get_model_resnet18_fasterrcnn(num_classes: int, pretrained_backbone: bool = True):
    """
    Crea un modelo Faster R-CNN con backbone ResNet-18.
    
    Args:
        num_classes: Número de clases (incluyendo background)
        pretrained_backbone: Si usar pesos preentrenados de ImageNet
    
    Returns:
        Modelo Faster R-CNN
    """
    # Cargar ResNet-18 preentrenado
    if pretrained_backbone:
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        backbone = resnet18(weights=None)
    
    # ResNet-18 tiene las siguientes capas:
    # conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
    # Queremos usar hasta layer4 (antes del avgpool y fc)
    
    # Remover las capas de clasificación (avgpool y fc)
    backbone = torch.nn.Sequential(
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,
        backbone.layer1,  # stride 4
        backbone.layer2,  # stride 8
        backbone.layer3,  # stride 16
        backbone.layer4,  # stride 32
    )
    
    # ResNet-18 layer4 tiene 512 canales de salida
    backbone.out_channels = 512
    
    # Definir anchor generator
    # Los anchors son las "cajas base" que la RPN ajustará
    anchor_generator = AnchorGenerator(
        sizes=((8, 16, 32, 64, 128, 256),),  # ← Añadir 8px y 16px
        aspect_ratios=((0.33, 0.5, 1.0, 2.0, 3.0),)  # ← Más ratios
    )
    
    # ROI Pooling (convierte features de diferentes tamaños a tamaño fijo)
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],  # Solo usamos una feature map
        output_size=7,        # Tamaño de salida 7x7
        sampling_ratio=2
    )
    
    # Crear el modelo Faster R-CNN
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """
    Entrena el modelo por una época.
    
    Returns:
        dict con métricas promedio de la época
    """
    model.train()
    
    total_loss = 0
    total_loss_classifier = 0
    total_loss_box_reg = 0
    total_loss_objectness = 0
    total_loss_rpn_box_reg = 0
    
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Acumular pérdidas
        total_loss += losses.item()
        total_loss_classifier += loss_dict['loss_classifier'].item()
        total_loss_box_reg += loss_dict['loss_box_reg'].item()
        total_loss_objectness += loss_dict['loss_objectness'].item()
        total_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()
        
        if (i + 1) % print_freq == 0:
            print(f"  Epoch [{epoch}] Iter [{i+1}/{len(data_loader)}] "
                  f"Loss: {losses.item():.4f} "
                  f"(cls: {loss_dict['loss_classifier'].item():.4f}, "
                  f"box: {loss_dict['loss_box_reg'].item():.4f}, "
                  f"obj: {loss_dict['loss_objectness'].item():.4f}, "
                  f"rpn: {loss_dict['loss_rpn_box_reg'].item():.4f})")
    
    # Calcular promedios
    n_batches = len(data_loader)
    metrics = {
        'loss': total_loss / n_batches,
        'loss_classifier': total_loss_classifier / n_batches,
        'loss_box_reg': total_loss_box_reg / n_batches,
        'loss_objectness': total_loss_objectness / n_batches,
        'loss_rpn_box_reg': total_loss_rpn_box_reg / n_batches,
    }
    
    return metrics


@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    Evalúa el modelo en el conjunto de validación.
    Calcula pérdidas promedio.
    """
    model.train()  # Necesario para obtener pérdidas
    
    total_loss = 0
    total_loss_classifier = 0
    total_loss_box_reg = 0
    total_loss_objectness = 0
    total_loss_rpn_box_reg = 0
    
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        total_loss += losses.item()
        total_loss_classifier += loss_dict['loss_classifier'].item()
        total_loss_box_reg += loss_dict['loss_box_reg'].item()
        total_loss_objectness += loss_dict['loss_objectness'].item()
        total_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()
    
    n_batches = len(data_loader)
    metrics = {
        'val_loss': total_loss / n_batches,
        'val_loss_classifier': total_loss_classifier / n_batches,
        'val_loss_box_reg': total_loss_box_reg / n_batches,
        'val_loss_objectness': total_loss_objectness / n_batches,
        'val_loss_rpn_box_reg': total_loss_rpn_box_reg / n_batches,
    }
    
    return metrics


def main(args):
    """Función principal de entrenamiento."""
    
    print("=" * 80)
    print("ENTRENAMIENTO DE DETECCIÓN DE DEFECTOS INDUSTRIALES")
    print("Arquitectura: ResNet-18 + Faster R-CNN")
    print("=" * 80)
    
    # Configurar device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Crear directorios de resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"resnet18_fasterrcnn_{timestamp}"
    output_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    print(f"\nDirectorio de salida: {output_dir}")
    
    # Guardar configuración
    config = vars(args)
    config['device'] = str(device)
    config['experiment_name'] = experiment_name
    config['timestamp'] = timestamp
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Cargar datasets
    print("\n" + "=" * 80)
    print("CARGANDO DATASETS")
    print("=" * 80)
    
    train_dataset = IndustrialDefectsDataset(
        root_dir=args.dataset_path,
        split='train',
        transforms=get_transform(train=True)
    )
    
    val_dataset = IndustrialDefectsDataset(
        root_dir=args.dataset_path,
        split='val',
        transforms=get_transform(train=False)
    )
    
    print(f"\nDistribución de categorías (train):")
    train_dist = train_dataset.get_category_distribution()
    for cat, count in sorted(train_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")
    
    # Crear data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Crear modelo
    print("\n" + "=" * 80)
    print("CREANDO MODELO")
    print("=" * 80)
    
    # Número de clases = categorías + background
    num_classes = len(train_dataset.categories) + 1
    print(f"\nNúmero de clases (con background): {num_classes}")
    
    model = get_model_resnet18_fasterrcnn(
        num_classes=num_classes,
        pretrained_backbone=args.pretrained_backbone
    )
    model.to(device)
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros totales: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}")
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma
    )
    
    print(f"\nOptimizer: SGD")
    print(f"Learning rate: {args.lr}")
    print(f"Momentum: {args.momentum}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"LR scheduler: StepLR (step={args.lr_step_size}, gamma={args.lr_gamma})")
    
    # Entrenamiento
    print("\n" + "=" * 80)
    print("INICIANDO ENTRENAMIENTO")
    print("=" * 80)
    
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch + 1}/{args.epochs}")
        print(f"{'='*80}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_metrics = train_one_epoch(
            model, optimizer, train_loader, device, epoch + 1, 
            print_freq=args.print_freq
        )
        
        # Validation
        print("\n  Evaluando en conjunto de validación...")
        val_metrics = evaluate(model, val_loader, device)
        
        # Update learning rate
        lr_scheduler.step()
        
        # Mostrar resultados
        epoch_time = time.time() - epoch_start_time
        print(f"\n  Epoch {epoch + 1} completada en {epoch_time:.2f}s")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        
        # Guardar métricas
        epoch_metrics = {
            'epoch': epoch + 1,
            'time': epoch_time,
            'lr': optimizer.param_groups[0]['lr'],
            **train_metrics,
            **val_metrics
        }
        training_history.append(epoch_metrics)
        
        # Guardar historial
        with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Guardar checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': config
        }
        
        # Guardar último checkpoint
        torch.save(
            checkpoint,
            os.path.join(checkpoints_dir, 'last_checkpoint.pth')
        )
        
        # Guardar mejor modelo
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            torch.save(
                checkpoint,
                os.path.join(checkpoints_dir, 'best_checkpoint.pth')
            )
            print(f"  ✓ Mejor modelo guardado (val_loss: {best_val_loss:.4f})")
        
        # Guardar checkpoint periódicamente
        if (epoch + 1) % args.save_freq == 0:
            torch.save(
                checkpoint,
                os.path.join(checkpoints_dir, f'checkpoint_epoch_{epoch+1}.pth')
            )
    
    print("\n" + "=" * 80)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 80)
    print(f"Mejor val_loss: {best_val_loss:.4f}")
    print(f"Resultados guardados en: {output_dir}")
    
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Entrenar Faster R-CNN con ResNet-18 para detección de defectos'
    )
    
    # Dataset
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='curated_dataset_splitted_20251101_provisional_1st_version',
        help='Ruta al dataset'
    )
    
    # Modelo
    parser.add_argument(
        '--pretrained-backbone',
        action='store_true',
        default=True,
        help='Usar backbone preentrenado en ImageNet'
    )
    
    # Training
    parser.add_argument('--epochs', type=int, default=20, help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--lr-step-size', type=int, default=5, help='LR scheduler step size')
    parser.add_argument('--lr-gamma', type=float, default=0.1, help='LR scheduler gamma')
    
    # Misc
    parser.add_argument('--num-workers', type=int, default=4, help='Num workers para DataLoader')
    parser.add_argument('--print-freq', type=int, default=10, help='Frecuencia de print')
    parser.add_argument('--save-freq', type=int, default=5, help='Frecuencia de guardado')
    parser.add_argument('--output-dir', type=str, default='results/training', help='Directorio de salida')
    
    args = parser.parse_args()
    
    main(args)