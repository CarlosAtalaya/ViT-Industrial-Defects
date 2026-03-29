#!/usr/bin/env python3
"""
05_create_splits_with_folders.py
Creación de Splits Estratificados CON Estructura de Carpetas

TFG: Vision Transformers para Detección de Anomalías Industriales
Fase 5 del Pipeline de Curación - CON ORGANIZACIÓN DE CARPETAS

ESTRUCTURA DE SALIDA:
output_dir/
├── train/
│   ├── images/          # Imágenes de entrenamiento
│   └── train.json       # Anotaciones COCO
├── val/
│   ├── images/          # Imágenes de validación
│   └── val.json         # Anotaciones COCO
├── test/
│   ├── images/          # Imágenes de test
│   └── test.json        # Anotaciones COCO
└── metadata/
    └── phase5_splits_log.json

Autor: [Tu nombre]
Fecha: Noviembre 2025
"""

import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency


class StratifiedSplitterWithFolders:
    """
    Creador de splits estratificados con estructura de carpetas organizada
    """
    
    def __init__(self, source_path: str, output_path: str, seed: int = 42):
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.seed = seed
        
        # Configuración de splits
        self.split_ratios = {
            "train": 0.70,
            "val": 0.10,
            "test": 0.20
        }
        
        self.stats = {
            "total_images": 0,
            "splits": {},
            "distribution_per_split": {},
            "chi2_test_results": {},
            "leakage_check": "passed"
        }
    
    def load_balanced_dataset(self) -> Dict:
        """Carga dataset balanceado (Fase 4)"""
        print(f"\n📂 Cargando dataset balanceado desde: {self.source_path}")
        
        anno_file = self.source_path / "annotations_balanced.coco.json"
        with open(anno_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.stats["total_images"] = len(data["images"])
        
        print(f"✅ Dataset cargado:")
        print(f"   📷 Imágenes: {len(data['images'])}")
        print(f"   📋 Anotaciones: {len(data['annotations'])}")
        
        return data
    
    def create_stratify_keys(self, data: Dict) -> Tuple[List, List]:
        """Crea claves de estratificación combinadas"""
        print("\n🔑 Creando claves de estratificación...")
        
        # Mapear image_id -> categoría principal
        image_categories = {}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in image_categories:
                image_categories[img_id] = ann["unified_category_name"]
        
        # Crear claves estratificadas
        stratify_keys = []
        image_list = []
        
        for img in data["images"]:
            img_id = img["id"]
            category = image_categories.get(img_id, "UNKNOWN")
            source = img.get("source_dataset", "unknown")
            
            stratify_key = f"{category}_{source}"
            
            stratify_keys.append(stratify_key)
            image_list.append(img)
        
        # Mostrar distribución
        key_counts = Counter(stratify_keys)
        print(f"✅ Claves únicas: {len(key_counts)}")
        print(f"\n📊 Top 10 combinaciones:")
        for key, count in key_counts.most_common(10):
            print(f"   • {key}: {count}")
        
        return image_list, stratify_keys
    
    def create_stratified_splits(self, 
                                images: List[Dict],
                                stratify_keys: List[str]) -> Dict[str, List[Dict]]:
        """Crea splits estratificados usando sklearn"""
        print("\n✂️ Creando splits estratificados...")
        
        # Convertir a numpy arrays
        images_array = np.array(images)
        stratify_array = np.array(stratify_keys)
        
        # Primera división: Train vs Temp
        train_images, temp_images, train_keys, temp_keys = train_test_split(
            images_array,
            stratify_array,
            test_size=(1 - self.split_ratios["train"]),
            stratify=stratify_array,
            random_state=self.seed
        )
        
        print(f"✅ Split 1: Train vs (Val+Test)")
        print(f"   Train: {len(train_images)} ({len(train_images)/len(images)*100:.1f}%)")
        print(f"   Temp:  {len(temp_images)} ({len(temp_images)/len(images)*100:.1f}%)")
        
        # Segunda división: Val vs Test
        val_ratio_in_temp = self.split_ratios["val"] / (1 - self.split_ratios["train"])
        
        val_images, test_images, val_keys, test_keys = train_test_split(
            temp_images,
            temp_keys,
            test_size=(1 - val_ratio_in_temp),
            stratify=temp_keys,
            random_state=self.seed
        )
        
        print(f"\n✅ Split 2: Val vs Test")
        print(f"   Val:  {len(val_images)} ({len(val_images)/len(images)*100:.1f}%)")
        print(f"   Test: {len(test_images)} ({len(test_images)/len(images)*100:.1f}%)")
        
        # Convertir a listas
        splits = {
            "train": train_images.tolist(),
            "val": val_images.tolist(),
            "test": test_images.tolist()
        }
        
        self.stats["splits"] = {
            name: len(imgs) for name, imgs in splits.items()
        }
        
        return splits
    
    def validate_split_distributions(self, 
                                    splits: Dict[str, List[Dict]],
                                    original_data: Dict):
        """Valida distribución de categorías por split"""
        print("\n📊 Validando distribuciones por split...")
        
        # Mapear image_id -> categoría
        image_categories = {}
        for ann in original_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in image_categories:
                image_categories[img_id] = ann["unified_category_name"]
        
        # Calcular distribución por split
        distributions = {}
        for split_name, images in splits.items():
            dist = Counter()
            for img in images:
                category = image_categories.get(img["id"], "UNKNOWN")
                dist[category] += 1
            distributions[split_name] = dist
        
        # Distribución global
        global_dist = Counter()
        for dist in distributions.values():
            global_dist.update(dist)
        
        self.stats["distribution_per_split"] = {
            split_name: dict(dist) for split_name, dist in distributions.items()
        }
        
        # Test Chi-cuadrado
        print(f"\n🔬 Test Chi-cuadrado (similitud con distribución global):")
        
        all_categories = sorted(global_dist.keys())
        
        for split_name, split_dist in distributions.items():
            observed = [split_dist.get(cat, 0) for cat in all_categories]
            expected_ratio = [global_dist[cat] / sum(global_dist.values()) for cat in all_categories]
            expected = [ratio * sum(observed) for ratio in expected_ratio]
            
            chi2, p_value = chi2_contingency([observed, expected])[:2]
            
            significant = p_value < 0.05
            status = "⚠️ DIFERENTE" if significant else "✅ SIMILAR"
            
            print(f"   {split_name}: χ²={chi2:.2f}, p-value={p_value:.4f} {status}")
            
            self.stats["chi2_test_results"][split_name] = {
                "chi2": float(chi2),
                "p_value": float(p_value),
                "similar_to_global": not significant
            }
            
            print(f"   Distribución {split_name}:")
            for cat in all_categories:
                count = split_dist.get(cat, 0)
                pct = count / sum(split_dist.values()) * 100
                print(f"      • {cat}: {count} ({pct:.1f}%)")
    
    def validate_no_leakage(self, splits: Dict[str, List[Dict]]):
        """Verifica que no hay imágenes compartidas entre splits"""
        print("\n🔍 Validando no-leakage entre splits...")
        
        train_ids = set(img["id"] for img in splits["train"])
        val_ids = set(img["id"] for img in splits["val"])
        test_ids = set(img["id"] for img in splits["test"])
        
        train_val_overlap = train_ids & val_ids
        train_test_overlap = train_ids & test_ids
        val_test_overlap = val_ids & test_ids
        
        has_leakage = bool(train_val_overlap or train_test_overlap or val_test_overlap)
        
        if has_leakage:
            print("❌ LEAKAGE DETECTADO:")
            if train_val_overlap:
                print(f"   Train-Val: {len(train_val_overlap)} imágenes")
            if train_test_overlap:
                print(f"   Train-Test: {len(train_test_overlap)} imágenes")
            if val_test_overlap:
                print(f"   Val-Test: {len(val_test_overlap)} imágenes")
            
            self.stats["leakage_check"] = "failed"
        else:
            print("✅ No leakage: Splits completamente disjuntos")
            self.stats["leakage_check"] = "passed"
        
        return not has_leakage
    
    def copy_images_to_split_folders(self, splits: Dict[str, List[Dict]]):
        """
        NUEVO: Copia imágenes a carpetas organizadas por split
        
        Estructura:
        output/
        ├── train/images/
        ├── val/images/
        └── test/images/
        """
        print("\n📁 Copiando imágenes a carpetas de splits...")
        
        source_images_dir = self.source_path / "images"
        
        if not source_images_dir.exists():
            print(f"⚠️ Directorio de imágenes no encontrado: {source_images_dir}")
            return
        
        for split_name, images in splits.items():
            # Crear directorio de imágenes para este split
            split_images_dir = self.output_path / split_name / "images"
            split_images_dir.mkdir(parents=True, exist_ok=True)
            
            copied = 0
            errors = 0
            
            for img in images:
                filename = img["file_name"]
                src = source_images_dir / filename
                dst = split_images_dir / filename
                
                try:
                    if src.exists():
                        shutil.copy2(src, dst)
                        copied += 1
                    else:
                        print(f"   ⚠️ Imagen no encontrada: {filename}")
                        errors += 1
                except Exception as e:
                    print(f"   ❌ Error copiando {filename}: {e}")
                    errors += 1
            
            print(f"   ✅ {split_name}/images/: {copied} imágenes copiadas")
            if errors > 0:
                print(f"   ⚠️ {split_name}: {errors} errores")
    
    def save_splits(self, splits: Dict[str, List[Dict]], original_data: Dict):
        """
        Guarda splits en estructura organizada con carpetas
        
        Estructura final:
        output/
        ├── train/
        │   ├── images/       # Imágenes de train
        │   └── train.json    # Anotaciones de train
        ├── val/
        │   ├── images/       # Imágenes de val
        │   └── val.json      # Anotaciones de val
        ├── test/
        │   ├── images/       # Imágenes de test
        │   └── test.json     # Anotaciones de test
        └── metadata/
            └── phase5_splits_log.json
        """
        print("\n💾 Guardando splits con estructura organizada...")
        
        # Crear estructura de carpetas
        for split_name in splits.keys():
            (self.output_path / split_name).mkdir(parents=True, exist_ok=True)
        
        # Copiar imágenes a carpetas separadas
        self.copy_images_to_split_folders(splits)
        
        # Guardar anotaciones COCO en cada carpeta de split
        for split_name, images in splits.items():
            split_image_ids = set(img["id"] for img in images)
            
            # Filtrar anotaciones
            split_annotations = [
                ann for ann in original_data["annotations"]
                if ann["image_id"] in split_image_ids
            ]
            
            # Crear COCO JSON
            split_data = {
                "info": {
                    **original_data.get("info", {}),
                    "split": split_name,
                    "split_created": datetime.now().isoformat(),
                    "description": f"Split {split_name} del dataset curado"
                },
                "licenses": original_data.get("licenses", []),
                "images": images,
                "annotations": split_annotations,
                "categories": original_data["categories"]
            }
            
            # Guardar en carpeta del split
            split_file = self.output_path / split_name / f"{split_name}.json"
            with open(split_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ {split_name}/{split_name}.json: {len(images)} imgs, {len(split_annotations)} anns")
        
        # Guardar también listas de filenames (opcional, para referencia)
        for split_name, images in splits.items():
            filenames = [img["file_name"] for img in images]
            txt_file = self.output_path / split_name / f"{split_name}_files.txt"
            with open(txt_file, 'w') as f:
                f.write('\n'.join(filenames))
            print(f"   ✅ {split_name}/{split_name}_files.txt: {len(filenames)} filenames")
    
    def save_metadata(self):
        """Guarda metadata de splits"""
        print("\n📊 Guardando metadata...")
        
        metadata_dir = self.output_path / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "phase": "PHASE_5_SPLIT_CREATION_WITH_FOLDERS",
            "seed": self.seed,
            "split_ratios": self.split_ratios,
            "stratification": "dual_category_and_source",
            "statistics": self.stats,
            "folder_structure": {
                "description": "Splits organizados en carpetas separadas",
                "layout": {
                    "train": "train/images/ + train/train.json",
                    "val": "val/images/ + val/val.json",
                    "test": "test/images/ + test/test.json"
                }
            }
        }
        
        metadata_file = metadata_dir / "phase5_splits_log.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Metadata guardada: {metadata_file}")
    
    def print_summary(self):
        """Resumen de splits"""
        print("\n" + "="*70)
        print("RESUMEN DE SPLITS - FASE 5 (CON CARPETAS)")
        print("="*70)
        
        print(f"\n📊 SPLITS CREADOS:")
        total = self.stats["total_images"]
        for split_name, count in self.stats["splits"].items():
            pct = count / total * 100
            target = self.split_ratios[split_name] * 100
            print(f"   {split_name.upper():<6}: {count:>4} imágenes ({pct:>5.1f}% - target: {target:.0f}%)")
        
        print(f"\n✅ VALIDACIONES:")
        print(f"   Leakage check: {self.stats['leakage_check'].upper()}")
        
        all_similar = all(
            result["similar_to_global"] 
            for result in self.stats["chi2_test_results"].values()
        )
        print(f"   Distribución: {'✅ PRESERVADA' if all_similar else '⚠️ REVISAR'}")
        
        print(f"\n📁 ESTRUCTURA DE CARPETAS:")
        print(f"   {self.output_path}/")
        print(f"   ├── train/")
        print(f"   │   ├── images/ ({self.stats['splits']['train']} archivos)")
        print(f"   │   └── train.json")
        print(f"   ├── val/")
        print(f"   │   ├── images/ ({self.stats['splits']['val']} archivos)")
        print(f"   │   └── val.json")
        print(f"   ├── test/")
        print(f"   │   ├── images/ ({self.stats['splits']['test']} archivos)")
        print(f"   │   └── test.json")
        print(f"   └── metadata/")
        print(f"       └── phase5_splits_log.json")
        
        print("\n" + "="*70)
        print("✅ FASE 5 COMPLETADA (CON ESTRUCTURA ORGANIZADA)")
        print(f"📁 Splits guardados en: {self.output_path}")
        print("="*70)
    
    def create_splits(self):
        """Pipeline completo de creación de splits con carpetas"""
        print("\n" + "="*70)
        print("FASE 5: CREACIÓN DE SPLITS CON ESTRUCTURA DE CARPETAS")
        print("="*70)
        
        try:
            # Paso 1: Cargar dataset balanceado
            original_data = self.load_balanced_dataset()
            
            # Paso 2: Crear claves de estratificación
            images, stratify_keys = self.create_stratify_keys(original_data)
            
            # Paso 3: Crear splits
            splits = self.create_stratified_splits(images, stratify_keys)
            
            # Paso 4: Validar distribuciones
            self.validate_split_distributions(splits, original_data)
            
            # Paso 5: Validar no-leakage
            valid = self.validate_no_leakage(splits)
            if not valid:
                print("\n⚠️ ADVERTENCIA: Leakage detectado!")
            
            # Paso 6: Guardar con estructura de carpetas
            self.save_splits(splits, original_data)
            self.save_metadata()
            
            # Paso 7: Resumen
            self.print_summary()
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Creación de Splits con Carpetas - Fase 5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Estructura de salida:
  output_dir/
  ├── train/
  │   ├── images/          # Imágenes de entrenamiento
  │   └── train.json       # Anotaciones COCO
  ├── val/
  │   ├── images/          # Imágenes de validación
  │   └── val.json         # Anotaciones COCO
  ├── test/
  │   ├── images/          # Imágenes de test
  │   └── test.json        # Anotaciones COCO
  └── metadata/
      └── phase5_splits_log.json

Ejemplo de uso:
  python3 05_create_splits_with_folders.py \\
    --source curated_dataset_v3_balanced_20251101 \\
    --output curated_dataset_v5_splits_FINAL \\
    --seed 42
        """
    )
    
    parser.add_argument("--source", required=True, help="Dataset balanceado (Fase 4)")
    parser.add_argument("--output", required=True, help="Directorio de splits con carpetas")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reproducibilidad")
    
    args = parser.parse_args()
    
    splitter = StratifiedSplitterWithFolders(args.source, args.output, args.seed)
    success = splitter.create_splits()
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()