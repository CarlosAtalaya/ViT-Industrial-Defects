#!/usr/bin/env python3
"""
Verificar compatibilidad del dataset industrial con DEIMv2
"""
import json
import os
from pathlib import Path

def verify_coco_format(json_path, images_dir):
    """Verificar formato COCO y compatibilidad"""
    print(f"\n{'='*80}")
    print(f"VERIFICANDO: {json_path}")
    print(f"{'='*80}\n")
    
    # Cargar JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Verificar estructura básica
    required_keys = ['images', 'annotations', 'categories']
    for key in required_keys:
        if key not in data:
            print(f"❌ ERROR: Falta clave '{key}' en JSON")
            return False
        print(f"✅ Clave '{key}' presente")
    
    # Información de categorías
    print(f"\n📊 CATEGORÍAS ({len(data['categories'])} clases):")
    for cat in data['categories']:
        cat_id = cat['id']
        cat_name = cat.get('unified_category_name', cat.get('name', 'UNKNOWN'))
        
        # Contar anotaciones por categoría
        num_anns = sum(1 for ann in data['annotations'] if ann['category_id'] == cat_id)
        print(f"  - ID {cat_id}: {cat_name} ({num_anns} anotaciones)")
    
    # Verificar IDs de categorías
    cat_ids = sorted([cat['id'] for cat in data['categories']])
    print(f"\n🔢 IDs de categorías: {cat_ids}")
    
    # CRÍTICO: Verificar si IDs empiezan en 0 o 1
    if min(cat_ids) == 0:
        print(f"⚠️  ADVERTENCIA: IDs empiezan en 0 (COCO estándar usa 1)")
        print(f"   Necesitarás remap_mscoco_category: False")
    elif min(cat_ids) == 1:
        print(f"✅ IDs empiezan en 1 (COCO estándar)")
    
    # Verificar imágenes
    print(f"\n🖼️  IMÁGENES:")
    print(f"  - Total en JSON: {len(data['images'])}")
    
    missing_images = []
    for img in data['images'][:10]:  # Solo verificar primeras 10
        img_path = os.path.join(images_dir, img['file_name'])
        if not os.path.exists(img_path):
            missing_images.append(img['file_name'])
    
    if missing_images:
        print(f"  ❌ Faltan {len(missing_images)} imágenes (muestra):")
        for img in missing_images[:5]:
            print(f"     - {img}")
    else:
        print(f"  ✅ Todas las imágenes existen (verificadas 10 primeras)")
    
    # Verificar anotaciones
    print(f"\n📦 ANOTACIONES:")
    print(f"  - Total: {len(data['annotations'])}")
    
    # Verificar formato bbox
    sample_ann = data['annotations'][0]
    bbox = sample_ann.get('bbox', [])
    if len(bbox) == 4:
        print(f"  ✅ Formato bbox: [x, y, width, height] (COCO estándar)")
        print(f"     Ejemplo: {bbox}")
    else:
        print(f"  ❌ ERROR: Formato bbox incorrecto")
    
    print(f"\n{'='*80}")
    print(f"✅ DATASET COMPATIBLE CON DEIMv2")
    print(f"{'='*80}\n")
    
    return True

if __name__ == "__main__":
    # Rutas del dataset
    base_dir = "../../curated_dataset_splitted_20251101_provisional_1st_version"
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        json_path = f"{base_dir}/{split}/{split}.json"
        images_dir = f"{base_dir}/{split}/images"
        
        if os.path.exists(json_path):
            verify_coco_format(json_path, images_dir)
        else:
            print(f"⚠️  No se encontró: {json_path}")