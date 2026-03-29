#!/usr/bin/env python3
"""
04_balance_dataset_FINAL_FIX.py
CORRECCIÓN DEFINITIVA: Copia TODAS las imágenes necesarias

CAMBIOS CLAVE:
1. Copia exhaustiva de imágenes originales ANTES de cualquier operación
2. Validación post-copia de existencia
3. Logging detallado de cada operación de copia
"""

import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import Counter
import random
import numpy as np
from PIL import Image
import albumentations as A


class DatasetBalancerFinalFix:
    """
    Balanceador con estrategia FAIL-SAFE: Copiar TODO primero
    """
    
    def __init__(self, source_path: str, output_path: str, seed: int = 42):
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.target_distribution = {
            "NORMAL": 300,
            "DEFORMACIONES": 133,
            "ROTURA_FRACTURA": 169,
            "RAYONES_ARANAZOS": 150,
            "PERFORACIONES": 187,
            "CONTAMINACION": 120
        }
        
        self._setup_augmentation()
        
        self.stats = {
            "original_distribution": {},
            "target_distribution": self.target_distribution,
            "final_distribution": {},
            "undersampled": {},
            "oversampled": {},
            "augmented_images": [],
            "copy_operations": {
                "attempted": 0,
                "successful": 0,
                "failed": 0,
                "already_exists": 0
            }
        }
    
    def _setup_augmentation(self):
        """Augmentación conservadora"""
        self.augmentation_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=5, p=0.5, border_mode=0),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.5
            ),
            A.GaussNoise(var_limit=(5.0, 15.0), p=0.3),
        ], bbox_params=A.BboxParams(
            format='coco',
            label_fields=['category_ids'],
            min_visibility=0.3
        ))
    
    def load_unified_dataset(self) -> Dict:
        """Carga dataset unificado"""
        print(f"\n📂 Cargando dataset unificado desde: {self.source_path}")
        
        anno_file = self.source_path / "annotations_unified.coco.json"
        with open(anno_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        category_counts = Counter()
        for ann in data["annotations"]:
            category_counts[ann["unified_category_name"]] += 1
        
        self.stats["original_distribution"] = dict(category_counts)
        
        print(f"✅ Dataset cargado:")
        print(f"   📷 Imágenes: {len(data['images'])}")
        print(f"   📋 Anotaciones: {len(data['annotations'])}")
        
        return data
    
    def copy_image_safe(self, filename: str, source_dir: Path, dest_dir: Path) -> bool:
        """
        ✅ NUEVA FUNCIÓN: Copia una imagen con verificación exhaustiva
        
        Returns:
            True si la copia fue exitosa o el archivo ya existe
            False si falló
        """
        src = source_dir / filename
        dst = dest_dir / filename
        
        self.stats["copy_operations"]["attempted"] += 1
        
        # Verificar si ya existe
        if dst.exists():
            self.stats["copy_operations"]["already_exists"] += 1
            return True
        
        # Verificar si existe en source
        if not src.exists():
            print(f"      ⚠️ Source no existe: {filename}")
            self.stats["copy_operations"]["failed"] += 1
            return False
        
        try:
            # Copiar
            shutil.copy2(src, dst)
            
            # Verificar que se copió
            if dst.exists() and dst.stat().st_size > 0:
                self.stats["copy_operations"]["successful"] += 1
                return True
            else:
                print(f"      ❌ Copia corrupta: {filename}")
                self.stats["copy_operations"]["failed"] += 1
                return False
                
        except Exception as e:
            print(f"      ❌ Error copiando {filename}: {e}")
            self.stats["copy_operations"]["failed"] += 1
            return False
    
    def copy_all_original_images_first(self, original_data: Dict):
        """
        ✅ NUEVA FUNCIÓN CLAVE: Copia TODAS las imágenes originales ANTES de hacer nada
        
        Esto garantiza que todas las imágenes necesarias estén disponibles
        para las operaciones posteriores.
        """
        print("\n📋 PASO 0: Copiando TODAS las imágenes originales...")
        
        source_images_dir = self.source_path / "images"
        output_images_dir = self.output_path / "images"
        output_images_dir.mkdir(parents=True, exist_ok=True)
        
        total = len(original_data["images"])
        successful = 0
        failed = 0
        
        print(f"   Total a copiar: {total}")
        
        for i, img in enumerate(original_data["images"], 1):
            if self.copy_image_safe(img["file_name"], source_images_dir, output_images_dir):
                successful += 1
            else:
                failed += 1
            
            # Progress cada 100 imágenes
            if i % 100 == 0:
                print(f"   ⏳ Progreso: {i}/{total} ({i/total*100:.0f}%)")
        
        print(f"\n   ✅ Copiadas exitosamente: {successful}/{total}")
        if failed > 0:
            print(f"   ❌ Fallos: {failed}")
        
        # CRÍTICO: Si faltan más del 5%, abortar
        if failed / total > 0.05:
            raise RuntimeError(f"Demasiadas imágenes faltantes ({failed}/{total}). Abortando.")
    
    def undersample_category(self, 
                            images: List[Dict], 
                            annotations: List[Dict],
                            category_name: str,
                            target_count: int) -> Tuple[List[Dict], List[Dict]]:
        """Under-sampling aleatorio"""
        print(f"\n🔽 Under-sampling {category_name}...")
        
        cat_image_ids = set()
        for ann in annotations:
            if ann["unified_category_name"] == category_name:
                cat_image_ids.add(ann["image_id"])
        
        cat_images_dict = {img["id"]: img for img in images if img["id"] in cat_image_ids}
        cat_images = list(cat_images_dict.values())
        
        original_count = len(cat_images)
        
        if original_count <= target_count:
            print(f"   ℹ️ No es necesario ({original_count} ≤ {target_count})")
            selected_annotations = [a for a in annotations if a["image_id"] in cat_image_ids]
            return cat_images, selected_annotations
        
        # Seleccionar subset
        selected_images = random.sample(cat_images, target_count)
        selected_image_ids = set(img["id"] for img in selected_images)
        
        selected_annotations = [
            ann for ann in annotations 
            if ann["image_id"] in selected_image_ids
        ]
        
        removed = original_count - target_count
        self.stats["undersampled"][category_name] = removed
        
        print(f"   ✅ {original_count} → {target_count} (-{removed})")
        
        return selected_images, selected_annotations
    
    def oversample_category(self,
                           images: List[Dict],
                           annotations: List[Dict],
                           category_name: str,
                           target_count: int,
                           next_img_id: int,
                           next_ann_id: int) -> Tuple[List[Dict], List[Dict], int, int]:
        """Over-sampling mediante augmentación"""
        print(f"\n🔼 Over-sampling {category_name}...")
        
        cat_image_ids = set()
        for ann in annotations:
            if ann["unified_category_name"] == category_name:
                cat_image_ids.add(ann["image_id"])
        
        cat_images_dict = {img["id"]: img for img in images if img["id"] in cat_image_ids}
        cat_images = list(cat_images_dict.values())
        cat_annotations = [ann for ann in annotations if ann["image_id"] in cat_image_ids]
        
        original_count = len(cat_images)
        
        if original_count >= target_count:
            print(f"   ℹ️ No es necesario ({original_count} ≥ {target_count})")
            return cat_images, cat_annotations, next_img_id, next_ann_id
        
        needed = target_count - original_count
        print(f"   📊 Generando {needed} imágenes augmentadas...")
        
        # Augmentar
        augmented_images = []
        augmented_annotations = []
        
        source_images = cat_images.copy()
        random.shuffle(source_images)
        
        successful_augs = 0
        
        # LÍMITE: Máximo 3 augmentaciones por imagen source
        source_aug_counts = Counter()
        max_augs_per_source = 3
        
        attempts = 0
        max_attempts = needed * 5  # Permitir múltiples intentos
        
        while successful_augs < needed and attempts < max_attempts:
            attempts += 1
            
            # Seleccionar source
            source_img = source_images[attempts % len(source_images)]
            
            # Verificar límite
            if source_aug_counts[source_img["id"]] >= max_augs_per_source:
                continue
            
            source_anns = [a for a in cat_annotations if a["image_id"] == source_img["id"]]
            
            # Augmentar
            aug_img_info, aug_anns = self._augment_image(
                source_img,
                source_anns,
                next_img_id,
                next_ann_id
            )
            
            if aug_img_info:
                augmented_images.append(aug_img_info)
                augmented_annotations.extend(aug_anns)
                
                next_img_id += 1
                next_ann_id += len(aug_anns)
                successful_augs += 1
                source_aug_counts[source_img["id"]] += 1
                
                self.stats["augmented_images"].append({
                    "source": source_img["file_name"],
                    "augmented": aug_img_info["file_name"],
                    "category": category_name
                })
                
                if successful_augs % 10 == 0:
                    print(f"   ⏳ Progreso: {successful_augs}/{needed}")
        
        self.stats["oversampled"][category_name] = successful_augs
        
        if successful_augs < needed:
            print(f"   ⚠️ Solo se generaron {successful_augs}/{needed} augmentaciones")
        
        print(f"   ✅ {original_count} → {original_count + successful_augs}")
        
        combined_images = cat_images + augmented_images
        combined_annotations = cat_annotations + augmented_annotations
        
        return combined_images, combined_annotations, next_img_id, next_ann_id
    
    def _augment_image(self,
                      source_img: Dict,
                      source_anns: List[Dict],
                      new_img_id: int,
                      new_ann_id: int) -> Tuple[Dict, List[Dict]]:
        """Aplica augmentación a una imagen"""
        
        # Buscar imagen (ahora debe existir porque ya la copiamos)
        img_path = self.output_path / "images" / source_img["file_name"]
        
        if not img_path.exists():
            print(f"      ⚠️ Imagen no encontrada para augmentar: {source_img['file_name']}")
            return None, []
        
        try:
            image = np.array(Image.open(img_path).convert('RGB'))
            
            bboxes = [ann["bbox"] for ann in source_anns]
            category_ids = [ann["category_id"] for ann in source_anns]
            
            transformed = self.augmentation_pipeline(
                image=image,
                bboxes=bboxes,
                category_ids=category_ids
            )
            
            # Guardar
            source_name = Path(source_img["file_name"]).stem
            new_name = f"{source_name}_aug{new_img_id}.png"
            output_img_path = self.output_path / "images" / new_name
            
            aug_image_pil = Image.fromarray(transformed["image"])
            aug_image_pil.save(output_img_path)
            
            # Verificar guardado
            if not output_img_path.exists():
                return None, []
            
            # Crear metadata
            new_img_info = {
                **source_img,
                "id": new_img_id,
                "file_name": new_name,
                "is_augmented": True,
                "augmented_from": source_img["file_name"]
            }
            
            new_annotations = []
            for i, bbox in enumerate(transformed["bboxes"]):
                source_ann = source_anns[i]
                new_ann = {
                    **source_ann,
                    "id": new_ann_id + i,
                    "image_id": new_img_id,
                    "bbox": list(bbox),
                    "is_augmented": True
                }
                new_annotations.append(new_ann)
            
            return new_img_info, new_annotations
            
        except Exception as e:
            print(f"      ⚠️ Error: {e}")
            return None, []
    
    def balance_dataset(self, original_data: Dict) -> Dict:
        """Aplica balanceo completo"""
        print("\n🔄 Aplicando estrategia de balanceo...")
        
        balanced_images = []
        balanced_annotations = []
        
        next_img_id = max(img["id"] for img in original_data["images"]) + 1
        next_ann_id = max(ann["id"] for ann in original_data["annotations"]) + 1
        
        for category_name, target_count in self.target_distribution.items():
            current_count = self.stats["original_distribution"].get(category_name, 0)
            
            print(f"\n📊 {category_name}: {current_count} → {target_count}")
            
            if current_count > target_count:
                cat_images, cat_annotations = self.undersample_category(
                    original_data["images"],
                    original_data["annotations"],
                    category_name,
                    target_count
                )
            elif current_count < target_count:
                cat_images, cat_annotations, next_img_id, next_ann_id = self.oversample_category(
                    original_data["images"],
                    original_data["annotations"],
                    category_name,
                    target_count,
                    next_img_id,
                    next_ann_id
                )
            else:
                # Sin cambios - solo seleccionar
                cat_image_ids = set()
                for ann in original_data["annotations"]:
                    if ann["unified_category_name"] == category_name:
                        cat_image_ids.add(ann["image_id"])
                
                cat_images_dict = {img["id"]: img for img in original_data["images"] 
                                   if img["id"] in cat_image_ids}
                cat_images = list(cat_images_dict.values())
                cat_annotations = [ann for ann in original_data["annotations"] 
                                   if ann["image_id"] in cat_image_ids]
            
            balanced_images.extend(cat_images)
            balanced_annotations.extend(cat_annotations)
        
        # Verificar unicidad
        unique_img_ids = set(img["id"] for img in balanced_images)
        if len(balanced_images) != len(unique_img_ids):
            balanced_images_dict = {img["id"]: img for img in balanced_images}
            balanced_images = list(balanced_images_dict.values())
        
        # Calcular distribución final
        final_counts = Counter()
        for ann in balanced_annotations:
            final_counts[ann["unified_category_name"]] += 1
        
        self.stats["final_distribution"] = dict(final_counts)
        
        balanced_data = {
            "info": {
                **original_data.get("info", {}),
                "version": "4.1",
                "description": "Dataset Balanceado (FIXED - todas las imágenes copiadas)",
                "date_balanced": datetime.now().isoformat()
            },
            "licenses": original_data.get("licenses", []),
            "images": balanced_images,
            "annotations": balanced_annotations,
            "categories": original_data["categories"]
        }
        
        return balanced_data
    
    def save_balanced_dataset(self, balanced_data: Dict):
        """Guarda dataset"""
        print("\n💾 Guardando dataset balanceado...")
        
        anno_file = self.output_path / "annotations_balanced.coco.json"
        with open(anno_file, 'w', encoding='utf-8') as f:
            json.dump(balanced_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Dataset guardado: {anno_file}")
    
    def save_metadata(self):
        """Guarda metadata"""
        metadata_dir = self.output_path / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "phase": "PHASE_4_BALANCING_FINAL_FIX",
            "seed": self.seed,
            "strategy": "hybrid_with_exhaustive_copy",
            "statistics": self.stats
        }
        
        metadata_file = metadata_dir / "phase4_balancing_log.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Metadata guardada: {metadata_file}")
    
    def print_summary(self):
        """Resumen"""
        print("\n" + "="*70)
        print("RESUMEN DE BALANCEO - FASE 4 (FINAL FIX)")
        print("="*70)
        
        print(f"\n📊 OPERACIONES DE COPIA:")
        print(f"   Intentadas: {self.stats['copy_operations']['attempted']}")
        print(f"   ✅ Exitosas: {self.stats['copy_operations']['successful']}")
        print(f"   ℹ️ Ya existían: {self.stats['copy_operations']['already_exists']}")
        print(f"   ❌ Fallidas: {self.stats['copy_operations']['failed']}")
        
        print(f"\n📊 DISTRIBUCIÓN:")
        for cat in sorted(self.stats["original_distribution"].keys()):
            orig = self.stats["original_distribution"].get(cat, 0)
            target = self.stats["target_distribution"].get(cat, 0)
            final = self.stats["final_distribution"].get(cat, 0)
            delta = final - orig
            sign = "+" if delta > 0 else ""
            print(f"   {cat:<20} {orig:<10} → {target:<10} = {final:<10} ({sign}{delta})")
        
        counts = list(self.stats["final_distribution"].values())
        ratio = max(counts) / min(counts) if min(counts) > 0 else 0
        
        print(f"\n📈 RATIO: {ratio:.2f}:1 {'✅ ÓPTIMO' if ratio < 3.0 else '⚠️ REVISAR'}")
        
        print("\n" + "="*70)
    
    def balance(self):
        """Pipeline completo"""
        print("\n" + "="*70)
        print("FASE 4: BALANCEO (FINAL FIX)")
        print("="*70)
        
        try:
            # Paso 1: Cargar
            original_data = self.load_unified_dataset()
            
            # ✅ PASO 0 CRÍTICO: Copiar TODAS las imágenes originales primero
            self.copy_all_original_images_first(original_data)
            
            # Paso 2: Balancear
            balanced_data = self.balance_dataset(original_data)
            
            # Paso 3: Guardar
            self.save_balanced_dataset(balanced_data)
            self.save_metadata()
            
            # Paso 4: Resumen
            self.print_summary()
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description="Balanceo FINAL FIX - Fase 4")
    parser.add_argument("--source", required=True, help="Dataset unificado (Fase 3)")
    parser.add_argument("--output", required=True, help="Dataset balanceado")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    
    args = parser.parse_args()
    
    balancer = DatasetBalancerFinalFix(args.source, args.output, args.seed)
    success = balancer.balance()
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()