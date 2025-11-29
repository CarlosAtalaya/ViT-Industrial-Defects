#!/usr/bin/env python3
"""
Script de Evaluación Fase 2 (Multimodal)
Genera métricas comparables (JSON) usando el wrapper de fusión.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torchvision.transforms as T
from PIL import Image

# Setup Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEIMV2_PATH = PROJECT_ROOT / "DEIMv2"
sys.path.insert(0, str(DEIMV2_PATH))
sys.path.insert(0, str(Path(__file__).parent))

from engine.core import YAMLConfig
from models_utils import TextEncoder
from data import get_text_prompts
from deimv2_fusion_wrapper import build_deimv2_with_fusion

def evaluate_fase2(checkpoint_path, output_dir, device='cuda'):
    # Configuración hardcodeada para reproducibilidad
    config_path = "scripts/deimv2_multimodal/configs/deimv2_industrial_defects.yml"
    # IMPORTANTE: Apuntar a tu dataset de TEST real
    test_ann_file = "dataset/annotations/instances_test.json" 
    test_img_folder = "dataset/test"
    
    # 1. Preparar Modelo Multimodal
    print("📝 Generando embeddings y cargando modelo...")
    text_encoder = TextEncoder(freeze=True).to(device)
    text_embeddings = text_encoder.encode_texts(get_text_prompts(), device)
    
    model = build_deimv2_with_fusion(
        checkpoint_path="scripts/deimv2_multimodal/outputs/deimv2_1024_300epochs/best_stg1.pth", # Dummy path para init
        config_path=config_path,
        text_embeddings=text_embeddings,
        device=device
    )
    
    # Cargar pesos entrenados de Fase 2
    print(f"📂 Cargando checkpoint Fase 2: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt) # Casos de state_dict directo
    model.eval()

    # 2. Inferencia
    print(f"🚀 Evaluando sobre: {test_ann_file}")
    coco_gt = COCO(test_ann_file)
    img_ids = sorted(coco_gt.getImgIds())
    results = []
    
    transform = T.Compose([
        T.Resize([1024, 1024]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("   Procesando imágenes...")
    for img_id in img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        path = os.path.join(test_img_folder, img_info['file_name'])
        img = Image.open(path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = model(input_tensor)
            
        # Procesar salidas (logica simplificada para DEIMv2 output)
        logits = out['pred_logits'][0]
        boxes = out['pred_boxes'][0]
        
        scores = logits.softmax(-1).max(-1)[0]
        labels = logits.softmax(-1).argmax(-1)
        
        # Rescale boxes
        h, w = img_info['height'], img_info['width']
        boxes[:, 0] *= w; boxes[:, 2] *= w
        boxes[:, 1] *= h; boxes[:, 3] *= h
        
        # Convert cxcywh -> xywh
        boxes[:, 0] -= boxes[:, 2]/2
        boxes[:, 1] -= boxes[:, 3]/2
        
        # Filtrado (Thresholds fase 1)
        score_thr = 0.15
        for s, l, b in zip(scores, labels, boxes):
            if s >= score_thr and l < 6:
                results.append({
                    "image_id": img_id,
                    "category_id": int(l),
                    "bbox": b.tolist(),
                    "score": float(s)
                })

    # 3. Calcular Métricas
    print("📊 Calculando métricas COCO...")
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 4. Generar JSON Comparable
    metrics = {
        "mAP": float(coco_eval.stats[0]),
        "iou_threshold": 0.5,
        "score_threshold": 0.15,
        "num_test_images": len(img_ids),
        "AP_per_class": {}, # Rellenar con lógica si se necesita detalle por clase
    }
    
    # Cálculo AP por clase (simplificado para ejemplo)
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    for i, cat in enumerate(cats):
        # Nota: Esto es una aproximación, COCOeval tiene la matriz completa
        # Se requiere lógica compleja para extraer exacto como en script Fase 1
        pass 

    os.makedirs(output_dir, exist_ok=True)
    res_path = os.path.join(output_dir, "test_evaluation_results_comparable.json")
    with open(res_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    det_path = os.path.join(output_dir, "test_detections_filtered.json")
    with open(det_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Resultados guardados en: {output_dir}")

if __name__ == "__main__":
    # Ajusta esto a tu último checkpoint
    CKPT = "demo-Multimodal/finetuning-simple-incremental/outputs/fase2/final.pth"
    OUT = "demo-Multimodal/finetuning-simple-incremental/outputs/fase2"
    evaluate_fase2(CKPT, OUT)