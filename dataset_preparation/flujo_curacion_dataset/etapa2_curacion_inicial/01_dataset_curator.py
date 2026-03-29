#!/usr/bin/env python3
"""
Dataset Curator para TFG: Vision Transformers para Detección de Anomalías
Combina VISION-Datasets y MVTec AD según categorías específicas
Versión: Curación paso a paso - Dataset unificado sin splits
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Set
import argparse
from datetime import datetime
from PIL import Image
import uuid

class DatasetCurator:
    def __init__(self, vision_path: str, mvtec_path: str, output_path: str):
        """
        Inicializa el curador de datasets
        
        Args:
            vision_path: Ruta a VISION-Datasets
            mvtec_path: Ruta a mvtec-ad  
            output_path: Ruta de salida para el dataset curado
        """
        self.vision_path = Path(vision_path)
        self.mvtec_path = Path(mvtec_path)
        self.output_path = Path(output_path)
        
        # Configuración de categorías y defectos objetivo
        self.defectos_objetivo = {
            "ROTURA_FRACTURA": {
                "vision_labels": ["break", "defect"],
                "mvtec_labels": ["broken", "broken_large", "crack"],
                "importancia": "CRITICA"
            },
            "CONTAMINACION": {
                "vision_labels": ["Dirty", "impurities"],
                "mvtec_labels": ["contamination", "metal_contamination"],
                "importancia": "ALTA"
            },
            "RAYONES_ARANAZOS": {
                "vision_labels": ["Scratch", "s_scratch", "t_scratch"],
                "mvtec_labels": ["scratch", "scratch_head"],
                "importancia": "MEDIA"
            },
            "PERFORACIONES": {
                "vision_labels": ["Hole", "missing_hole"],
                "mvtec_labels": ["hole", "cut"],
                "importancia": "CRITICA"
            },
            "DEFORMACIONES": {
                "vision_labels": ["short", "spur"],
                "mvtec_labels": ["bent", "bent_lead", "bent_wire"],
                "importancia": "ALTA"
            }
        }
        
        # Categorías MVTec prioritarias
        self.categorias_mvtec = ["transistor", "metal_nut", "cable", "capsule", "hazelnut"]
        
        # Componentes VISION prioritarios
        self.componentes_vision = ["PCB_1", "PCB_2", "Console", "Electronics", "Cable", "Lens"]
        
        # Contadores para COCO JSON
        self.next_image_id = 1
        self.next_annotation_id = 1
        self.next_category_id = 1
        
        # Almacenar datos COCO
        self.coco_data = {
            "info": {
                "description": "TFG Dataset: VISION-Datasets + MVTec AD Curado",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "TFG Vision Transformers",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Mapeo de categorías para mantener consistencia
        self.category_name_to_id = {}
        
        self.stats = {
            "vision_images": 0,
            "mvtec_images": 0,
            "total_images": 0,
            "defect_distribution": {},
            "conversion_errors": [],
            "processed_categories": set()
        }
    
    def setup_output_structure(self):
        """Crea la estructura de directorios de salida"""
        print("🏗️ Creando estructura de directorios...")
        
        # Estructura simplificada
        dirs = [
            "images",
            "metadata"
        ]
        
        for dir_path in dirs:
            (self.output_path / dir_path).mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Estructura creada en: {self.output_path}")
    
    def load_mvtec_annotations(self) -> List[Dict]:
        """Carga las anotaciones de MVTec AD con manejo robusto de formatos"""
        samples_file = self.mvtec_path / "samples.json"
        
        if not samples_file.exists():
            raise FileNotFoundError(f"No se encuentra {samples_file}")
        
        with open(samples_file, 'r') as f:
            raw_data = json.load(f)
        
        print(f"📋 Archivo samples.json cargado: {type(raw_data)}")
        
        # Manejar diferentes formatos posibles
        annotations = []
        
        if isinstance(raw_data, dict):
            # Si es un diccionario, buscar las anotaciones dentro
            if 'samples' in raw_data:
                annotations = raw_data['samples']
                print(f"📋 Usando clave 'samples' con {len(annotations)} elementos")
            elif 'images' in raw_data:
                annotations = raw_data['images']
                print(f"📋 Usando clave 'images' con {len(annotations)} elementos")
            else:
                print(f"⚠️ Diccionario con claves: {list(raw_data.keys())}")
                # Intentar usar el primer valor que sea una lista
                for key, value in raw_data.items():
                    if isinstance(value, list):
                        annotations = value
                        print(f"📋 Usando clave '{key}' como anotaciones con {len(annotations)} elementos")
                        break
        
        elif isinstance(raw_data, list):
            # Si es una lista directamente
            annotations = raw_data
            print(f"📋 Lista directa con {len(annotations)} elementos")
        
        if not annotations:
            print("❌ No se pudieron extraer anotaciones del archivo samples.json")
            raise ValueError("Formato de samples.json no compatible")
        
        # Procesar anotaciones para formato estándar
        processed_annotations = []
        
        for i, sample in enumerate(annotations):
            if not isinstance(sample, dict):
                print(f"⚠️ Elemento {i} no es diccionario: {type(sample)}")
                continue
            
            # Mostrar estructura del primer elemento para debug
            if i == 0:
                print(f"🔍 Estructura del primer elemento: {list(sample.keys())}")
                print(f"🔍 Ejemplo: {sample}")
            
            processed_sample = self._process_mvtec_sample(sample)
            if processed_sample:
                processed_annotations.append(processed_sample)
        
        print(f"📋 Anotaciones MVTec procesadas: {len(processed_annotations)} muestras")
        return processed_annotations
    
    def _process_mvtec_sample(self, sample: Dict) -> Dict:
        """Procesa una muestra individual de MVTec para formato estándar"""
        try:
            # Extraer filepath
            filepath = sample.get('filepath', '')
            if not filepath:
                print(f"⚠️ Sample sin filepath: {sample}")
                return None
            
            # Parsear información del filepath
            # Formato: "data/data_X/filename.png"
            path_parts = Path(filepath).parts
            
            if len(path_parts) < 3:
                print(f"⚠️ Filepath con formato inesperado: {filepath}")
                return None
            
            # Extraer componentes
            data_folder = path_parts[1]  # data_X
            filename = path_parts[2]     # filename.png
            
            # Extraer categoría de la estructura MongoDB
            category = 'unknown'
            category_obj = sample.get('category', {})
            if isinstance(category_obj, dict) and 'label' in category_obj:
                category = category_obj['label']
            
            # Extraer defect_type de la estructura MongoDB
            defect_type = 'unknown'
            defect_obj = sample.get('defect', {})
            if isinstance(defect_obj, dict) and 'label' in defect_obj:
                defect_type = defect_obj['label']
            
            # Extraer split
            split = sample.get('split', 'unknown')
            
            # Verificar si el archivo realmente existe
            full_path = self.mvtec_path / filepath
            if not full_path.exists():
                print(f"⚠️ Archivo no existe: {full_path}")
                return None
            
            processed_sample = {
                'image_path': filepath,
                'image_name': filename,
                'category': category,
                'split': split,
                'defect_type': defect_type,
                'data_folder': data_folder,
                'original_sample': sample  # Preservar datos originales
            }
            
            return processed_sample
            
        except Exception as e:
            print(f"⚠️ Error procesando sample: {e}")
            print(f"    Sample keys: {sample.keys() if isinstance(sample, dict) else 'Not a dict'}")
            return None
    
    def load_vision_annotations(self, component: str) -> Dict:
        """Carga las anotaciones COCO de un componente VISION"""
        annotations = {}
        
        for split in ['train', 'val']:
            anno_file = self.vision_path / component / split / "_annotations.coco.json"
            
            if anno_file.exists():
                with open(anno_file, 'r') as f:
                    data = json.load(f)
                    annotations[split] = data
                    print(f"📋 {component}/{split}: {len(data.get('images', []))} imágenes")
        
        return annotations
    
    def get_or_create_category(self, category_name: str, source_dataset: str = "unknown") -> int:
        """Obtiene o crea una categoría en el COCO JSON"""
        if category_name in self.category_name_to_id:
            return self.category_name_to_id[category_name]
        
        category_id = self.next_category_id
        self.next_category_id += 1
        
        self.coco_data["categories"].append({
            "id": category_id,
            "name": category_name,
            "supercategory": f"{source_dataset}_defect"
        })
        
        self.category_name_to_id[category_name] = category_id
        return category_id
    
    def convert_image_to_png(self, src_path: Path, dst_path: Path) -> bool:
        """Convierte imagen a PNG manteniendo calidad"""
        try:
            with Image.open(src_path) as img:
                # Convertir a RGB si es necesario (para evitar problemas con RGBA, etc.)
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                # Guardar como PNG
                img.save(dst_path, 'PNG', optimize=True)
                return True
                
        except Exception as e:
            print(f"⚠️ Error convirtiendo {src_path}: {e}")
            self.stats["conversion_errors"].append(str(src_path))
            return False
    
    def filter_mvtec_samples(self, annotations: List[Dict]) -> List[Dict]:
        """Filtra muestras MVTec según categorías y defectos objetivo"""
        filtered_samples = []
        defectos_target = set()
        
        # Recopilar todos los defectos objetivo de MVTec
        for defecto_info in self.defectos_objetivo.values():
            defectos_target.update(defecto_info["mvtec_labels"])
        
        print(f"🎯 Defectos MVTec objetivo: {defectos_target}")
        
        for sample in annotations:
            # Verificar que sample es un diccionario
            if not isinstance(sample, dict):
                print(f"⚠️ Muestra no es diccionario: {type(sample)} - {sample}")
                continue
                
            category = sample.get('category', '')
            defect_type = sample.get('defect_type', '')
            
            # Filtrar por categoría
            if category not in self.categorias_mvtec:
                continue
            
            # Incluir normales siempre
            if defect_type == 'good':
                filtered_samples.append(sample)
                continue
                
            # Incluir anomalías que nos interesan
            if defect_type in defectos_target:
                filtered_samples.append(sample)
                # Actualizar estadísticas
                defecto_normalizado = self._normalize_defect_name(defect_type, 'mvtec')
                self.stats["defect_distribution"][defecto_normalizado] = \
                    self.stats["defect_distribution"].get(defecto_normalizado, 0) + 1
        
        print(f"🔍 MVTec filtrado: {len(filtered_samples)} muestras de {len(annotations)}")
        return filtered_samples
    
    def filter_vision_annotations(self, component: str, annotations: Dict) -> Dict:
        """Filtra anotaciones VISION según defectos objetivo"""
        filtered_annotations = {}
        defectos_target = set()
        
        # Recopilar defectos objetivo de VISION
        for defecto_info in self.defectos_objetivo.values():
            defectos_target.update(defecto_info["vision_labels"])
        
        print(f"🎯 Defectos VISION objetivo: {defectos_target}")
        
        for split in annotations:
            anno_data = annotations[split]
            filtered_images = []
            filtered_annotations_list = []
            
            # Crear mapeo imagen_id -> anotaciones
            image_annotations = {}
            for ann in anno_data.get('annotations', []):
                img_id = ann['image_id']
                if img_id not in image_annotations:
                    image_annotations[img_id] = []
                image_annotations[img_id].append(ann)
            
            # Filtrar imágenes por categorías de defectos
            for img_info in anno_data.get('images', []):
                img_id = img_info['id']
                img_annotations = image_annotations.get(img_id, [])
                
                # Verificar si alguna anotación contiene defectos objetivo
                has_target_defect = False
                valid_annotations = []
                
                for ann in img_annotations:
                    category_id = ann.get('category_id')
                    if category_id is not None:
                        # Buscar nombre de categoría
                        for cat in anno_data.get('categories', []):
                            if cat['id'] == category_id and cat['name'] in defectos_target:
                                has_target_defect = True
                                valid_annotations.append(ann)
                                defecto_normalizado = self._normalize_defect_name(cat['name'], 'vision')
                                self.stats["defect_distribution"][defecto_normalizado] = \
                                    self.stats["defect_distribution"].get(defecto_normalizado, 0) + 1
                                break
                
                if has_target_defect:
                    filtered_images.append(img_info)
                    filtered_annotations_list.extend(valid_annotations)
            
            filtered_annotations[split] = {
                'images': filtered_images,
                'annotations': filtered_annotations_list,
                'categories': anno_data.get('categories', [])
            }
            
            print(f"🔍 {component}/{split} filtrado: {len(filtered_images)} imágenes")
        
        return filtered_annotations
    
    def _normalize_defect_name(self, defect_name: str, dataset_type: str) -> str:
        """Normaliza nombres de defectos a categorías unificadas"""
        for categoria, info in self.defectos_objetivo.items():
            labels_key = f"{dataset_type}_labels"
            if defect_name in info[labels_key]:
                return categoria
        return f"OTROS_{dataset_type.upper()}"
    
    def process_mvtec_samples(self, filtered_samples: List[Dict]):
        """Procesa imágenes filtradas de MVTec y las añade al COCO unificado"""
        print("📂 Procesando muestras MVTec...")
        
        for sample in filtered_samples:
            filepath = sample['image_path']  # Ej: "data/data_36/007-69.png"
            
            # CORREGIR: filepath ya incluye "data/", usar directamente
            src_path = self.mvtec_path / filepath
            
            if not src_path.exists():
                print(f"⚠️ Imagen no encontrada: {src_path}")
                continue
            
            # Mantener nombre original pero cambiar extensión a .png
            original_name = src_path.stem  # nombre sin extensión
            new_name = f"{original_name}.png"
            dst_path = self.output_path / "images" / new_name
            
            # Evitar sobrescribir si ya existe (puede pasar con nombres duplicados)
            counter = 1
            while dst_path.exists():
                new_name = f"{original_name}_{counter}.png"
                dst_path = self.output_path / "images" / new_name
                counter += 1
            
            # Convertir y copiar imagen
            if not self.convert_image_to_png(src_path, dst_path):
                continue
            
            # Obtener dimensiones de la imagen
            try:
                with Image.open(dst_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"⚠️ Error leyendo dimensiones de {dst_path}: {e}")
                continue
            
            # Crear entrada de imagen en COCO
            image_info = {
                "id": self.next_image_id,
                "width": width,
                "height": height,
                "file_name": new_name,
                "license": 0,
                "date_captured": datetime.now().isoformat(),
                "source_dataset": "mvtec",
                "original_category": sample.get('category', 'unknown'),
                "original_split": sample.get('split', 'unknown'),
                "defect_type": sample.get('defect_type', 'good')
            }
            
            self.coco_data["images"].append(image_info)
            
            # Crear anotación COCO para MVTec
            defect_type = sample.get('defect_type', 'good')
            category_name = defect_type if defect_type != 'good' else 'normal'
            category_id = self.get_or_create_category(category_name, "mvtec")
            
            # Para MVTec, crear anotación de imagen completa (sin bounding box específico)
            annotation = {
                "id": self.next_annotation_id,
                "image_id": self.next_image_id,
                "category_id": category_id,
                "segmentation": [],
                "area": width * height,  # Área completa de la imagen
                "bbox": [0, 0, width, height],  # Bounding box de toda la imagen
                "iscrowd": 0,
                "source_dataset": "mvtec"
            }
            
            self.coco_data["annotations"].append(annotation)
            
            # Actualizar contadores
            self.next_image_id += 1
            self.next_annotation_id += 1
            self.stats["mvtec_images"] += 1
            self.stats["total_images"] += 1
            self.stats["processed_categories"].add(sample.get('category', 'unknown'))
    
    def process_vision_samples(self, component: str, filtered_annotations: Dict):
        """Procesa imágenes filtradas de VISION y las añade al COCO unificado"""
        print(f"📂 Procesando muestras VISION/{component}...")
        
        for split in filtered_annotations:
            anno_data = filtered_annotations[split]
            
            for img_info in anno_data['images']:
                img_name = img_info['file_name']
                src_path = self.vision_path / component / split / img_name
                
                if not src_path.exists():
                    print(f"⚠️ Imagen no encontrada: {src_path}")
                    continue
                
                # Mantener nombre original pero cambiar extensión a .png
                original_name = src_path.stem
                new_name = f"{original_name}.png"
                dst_path = self.output_path / "images" / new_name
                
                # Evitar sobrescribir si ya existe (puede pasar con nombres duplicados)
                counter = 1
                while dst_path.exists():
                    new_name = f"{original_name}_{counter}.png"
                    dst_path = self.output_path / "images" / new_name
                    counter += 1
                
                # Convertir y copiar imagen
                if not self.convert_image_to_png(src_path, dst_path):
                    continue
                
                # Obtener dimensiones reales de la imagen convertida
                try:
                    with Image.open(dst_path) as img:
                        width, height = img.size
                except Exception as e:
                    print(f"⚠️ Error leyendo dimensiones de {dst_path}: {e}")
                    continue
                
                # Crear entrada de imagen en COCO unificado
                image_info = {
                    "id": self.next_image_id,
                    "width": width,
                    "height": height,
                    "file_name": new_name,
                    "license": 0,
                    "date_captured": datetime.now().isoformat(),
                    "source_dataset": "vision",
                    "original_component": component,
                    "original_split": split,
                    "original_id": img_info.get('id')
                }
                
                self.coco_data["images"].append(image_info)
                
                # Procesar anotaciones de esta imagen
                img_id_original = img_info['id']
                
                for ann in anno_data['annotations']:
                    if ann['image_id'] == img_id_original:
                        # Encontrar nombre de categoría
                        category_name = None
                        for cat in anno_data['categories']:
                            if cat['id'] == ann['category_id']:
                                category_name = cat['name']
                                break
                        
                        if category_name:
                            category_id = self.get_or_create_category(category_name, "vision")
                            
                            # Crear anotación en COCO unificado
                            new_annotation = {
                                "id": self.next_annotation_id,
                                "image_id": self.next_image_id,
                                "category_id": category_id,
                                "segmentation": ann.get('segmentation', []),
                                "area": ann.get('area', 0),
                                "bbox": ann.get('bbox', []),
                                "iscrowd": ann.get('iscrowd', 0),
                                "source_dataset": "vision",
                                "original_annotation_id": ann.get('id')
                            }
                            
                            self.coco_data["annotations"].append(new_annotation)
                            self.next_annotation_id += 1
                
                # Actualizar contadores
                self.next_image_id += 1
                self.stats["vision_images"] += 1
                self.stats["total_images"] += 1
                self.stats["processed_categories"].add(component)
    
    def save_coco_json(self):
        """Guarda el archivo COCO JSON unificado"""
        coco_file = self.output_path / "annotations.coco.json"
        
        with open(coco_file, 'w', encoding='utf-8') as f:
            json.dump(self.coco_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 COCO JSON guardado: {coco_file}")
        print(f"   📊 {len(self.coco_data['images'])} imágenes")
        print(f"   📋 {len(self.coco_data['annotations'])} anotaciones")
        print(f"   🏷️ {len(self.coco_data['categories'])} categorías")
    
    def save_metadata(self):
        """Guarda metadatos del dataset curado"""
        metadata = {
            "created_at": datetime.now().isoformat(),
            "source_datasets": ["VISION-Datasets", "MVTec AD"],
            "target_defects": self.defectos_objetivo,
            "mvtec_categories": self.categorias_mvtec,
            "vision_components": self.componentes_vision,
            "statistics": {
                **self.stats,
                "processed_categories": list(self.stats["processed_categories"])
            },
            "coco_info": {
                "total_images": len(self.coco_data['images']),
                "total_annotations": len(self.coco_data['annotations']),
                "total_categories": len(self.coco_data['categories']),
                "categories": [cat['name'] for cat in self.coco_data['categories']]
            },
            "output_structure": {
                "images/": "Todas las imágenes filtradas en formato PNG",
                "annotations.coco.json": "Anotaciones COCO unificadas",
                "metadata/": "Información del proceso de curación"
            }
        }
        
        metadata_file = self.output_path / "metadata" / "dataset_info.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Metadatos guardados en: {metadata_file}")
    
    def curate_dataset(self):
        """Ejecuta el proceso completo de curación"""
        print("🚀 Iniciando curación de dataset unificado...")
        print(f"📁 VISION-Datasets: {self.vision_path}")
        print(f"📁 MVTec AD: {self.mvtec_path}")
        print(f"📁 Salida: {self.output_path}")
        
        # 1. Configurar estructura
        self.setup_output_structure()
        
        # 2. Procesar MVTec AD
        print("\n" + "="*50)
        print("🔄 Procesando MVTec AD...")
        mvtec_annotations = self.load_mvtec_annotations()
        filtered_mvtec = self.filter_mvtec_samples(mvtec_annotations)
        self.process_mvtec_samples(filtered_mvtec)
        
        # 3. Procesar VISION-Datasets
        print("\n" + "="*50)
        print("🔄 Procesando VISION-Datasets...")
        for component in self.componentes_vision:
            if (self.vision_path / component).exists():
                print(f"\n📦 Procesando {component}...")
                vision_annotations = self.load_vision_annotations(component)
                if vision_annotations:  # Solo procesar si hay anotaciones
                    filtered_vision = self.filter_vision_annotations(component, vision_annotations)
                    self.process_vision_samples(component, filtered_vision)
            else:
                print(f"⚠️ Componente no encontrado: {component}")
        
        # 4. Guardar COCO JSON unificado
        print("\n" + "="*50)
        self.save_coco_json()
        
        # 5. Guardar metadatos
        self.save_metadata()
        
        # 6. Resumen final
        self.print_summary()
    
    def print_summary(self):
        """Imprime resumen del proceso de curación"""
        print("\n" + "="*50)
        print("📊 RESUMEN DE CURACIÓN")
        print("="*50)
        print(f"🖼️ Total imágenes procesadas: {self.stats['total_images']}")
        print(f"   • VISION-Datasets: {self.stats['vision_images']}")
        print(f"   • MVTec AD: {self.stats['mvtec_images']}")
        
        print(f"\n📈 Distribución por tipo de defecto:")
        for defecto, count in self.stats['defect_distribution'].items():
            print(f"   • {defecto}: {count}")
        
        print(f"\n📂 Componentes/categorías procesadas:")
        for categoria in sorted(self.stats["processed_categories"]):
            print(f"   • {categoria}")
        
        if self.stats["conversion_errors"]:
            print(f"\n⚠️ Errores de conversión: {len(self.stats['conversion_errors'])}")
            for error in self.stats["conversion_errors"][:5]:  # Mostrar solo los primeros 5
                print(f"   • {error}")
            if len(self.stats["conversion_errors"]) > 5:
                print(f"   • ... y {len(self.stats['conversion_errors']) - 5} más")
        
        print(f"\n🎯 COCO JSON creado:")
        print(f"   • {len(self.coco_data['images'])} imágenes")
        print(f"   • {len(self.coco_data['annotations'])} anotaciones")
        print(f"   • {len(self.coco_data['categories'])} categorías")
        
        print(f"\n📁 Dataset curado guardado en: {self.output_path}")
        print("✅ Curación completada exitosamente!")
        print("\n💡 Siguiente paso: Analizar distribución para crear splits train/val/test")


def main():
    parser = argparse.ArgumentParser(description="Curador de Dataset TFG - Versión Unificada")
    parser.add_argument("--vision-path", required=True, help="Ruta a VISION-Datasets")
    parser.add_argument("--mvtec-path", required=True, help="Ruta a mvtec-ad")
    parser.add_argument("--output-path", required=True, help="Ruta de salida")
    
    args = parser.parse_args()
    
    curator = DatasetCurator(args.vision_path, args.mvtec_path, args.output_path)
    curator.curate_dataset()


if __name__ == "__main__":
    main()