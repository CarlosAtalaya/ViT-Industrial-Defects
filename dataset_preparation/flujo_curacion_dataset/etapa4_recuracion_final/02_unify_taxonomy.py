#!/usr/bin/env python3
"""
03_unify_taxonomy.py
Unificación de Taxonomía y Esquema de Anotación

TFG: Vision Transformers para Detección de Anomalías Industriales
Fase 3 del Pipeline de Curación

Objetivos:
1. Mapear 15 categorías originales → 6 categorías unificadas
2. Convertir a esquema híbrido COCO-compatible
3. Añadir metadata de trazabilidad completa
4. Validar consistencia post-unificación

Taxonomía Unificada:
- 0: NORMAL
- 1: DEFORMACIONES  
- 2: ROTURA_FRACTURA
- 3: RAYONES_ARANAZOS
- 4: PERFORACIONES
- 5: CONTAMINACION

Autor: [Tu nombre]
Fecha: Noviembre 2025
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import pandas as pd


class TaxonomyUnifier:
    """
    Unificador científico de taxonomía con esquema híbrido
    """
    
    def __init__(self, source_path: str, output_path: str):
        """
        Inicializa el unificador
        
        Args:
            source_path: Dataset re-curado (salida de Fase 2)
            output_path: Dataset con taxonomía unificada
        """
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        
        # Configurar taxonomía unificada
        self._setup_unified_taxonomy()
        
        # Estadísticas
        self.stats = {
            "original_categories": 0,
            "unified_categories": 6,
            "images_processed": 0,
            "annotations_processed": 0,
            "category_mapping_counts": {},
            "unmapped_categories": []
        }
    
    def _setup_unified_taxonomy(self):
        """
        Define la taxonomía unificada científicamente
        
        Basado en:
        - Normativas industriales de calidad
        - Análisis de clustering semántico
        - Criticidad de defectos
        """
        # TAXONOMÍA UNIFICADA (6 categorías)
        self.unified_taxonomy = {
            0: {
                "name": "NORMAL",
                "description": "Componente funcional sin defectos detectables",
                "criticality": "NONE",
                "color": "#2ecc71",  # Verde
                "supercategory": "none"
            },
            1: {
                "name": "DEFORMACIONES",
                "description": "Alteración geométrica o estructural del componente",
                "criticality": "ALTA",
                "color": "#f39c12",  # Naranja
                "supercategory": "defect"
            },
            2: {
                "name": "ROTURA_FRACTURA",
                "description": "Discontinuidad estructural, grietas o roturas completas",
                "criticality": "CRITICA",
                "color": "#e74c3c",  # Rojo
                "supercategory": "defect"
            },
            3: {
                "name": "RAYONES_ARANAZOS",
                "description": "Daño superficial por abrasión o contacto",
                "criticality": "MEDIA",
                "color": "#9b59b6",  # Púrpura
                "supercategory": "defect"
            },
            4: {
                "name": "PERFORACIONES",
                "description": "Agujeros, cortes o ausencia de material",
                "criticality": "CRITICA",
                "color": "#c0392b",  # Rojo oscuro
                "supercategory": "defect"
            },
            5: {
                "name": "CONTAMINACION",
                "description": "Presencia de material extraño o impurezas",
                "criticality": "ALTA",
                "color": "#95a5a6",  # Gris
                "supercategory": "defect"
            }
        }
        
        # MAPEO CIENTÍFICO: Labels originales → Categoría unificada
        self.category_mapping = {
            # MVTec labels
            "good": 0,
            "normal": 0,
            
            "bent": 1,
            "bent_lead": 1,
            "bent_wire": 1,
            "short": 1,
            "spur": 1,
            
            "crack": 2,
            "break": 2,
            "broken": 2,
            "broken_large": 2,
            "broken_small": 2,
            "defect": 2,  # VISION generic defect → ROTURA (revisar)
            
            "scratch": 3,
            "Scratch": 3,  # Case-sensitive
            "s_scratch": 3,
            "t_scratch": 3,
            "scratch_head": 3,
            "scratch_neck": 3,
            
            "hole": 4,
            "Hole": 4,  # Case-sensitive
            "missing_hole": 4,
            "cut": 4,
            "cut_inner_insulation": 4,
            "cut_outer_insulation": 4,
            "cut_lead": 4,
            
            "contamination": 5,
            "metal_contamination": 5,
            "Dirty": 5,
            "impurities": 5
        }
        
        # Invertir mapeo para trazabilidad
        self.reverse_mapping = defaultdict(list)
        for original, unified in self.category_mapping.items():
            self.reverse_mapping[unified].append(original)
    
    def load_recurated_dataset(self) -> Dict:
        """
        Carga el dataset re-curado (Fase 2)
        """
        print(f"\n📂 Cargando dataset re-curado desde: {self.source_path}")
        
        anno_file = self.source_path / "annotations.coco.json"
        if not anno_file.exists():
            raise FileNotFoundError(f"No se encuentra: {anno_file}")
        
        with open(anno_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.stats["images_processed"] = len(data["images"])
        self.stats["annotations_processed"] = len(data["annotations"])
        self.stats["original_categories"] = len(data["categories"])
        
        print(f"✅ Dataset cargado:")
        print(f"   📷 Imágenes: {len(data['images'])}")
        print(f"   📋 Anotaciones: {len(data['annotations'])}")
        print(f"   🏷️ Categorías originales: {len(data['categories'])}")
        
        return data
    
    def create_category_id_mapping(self, original_categories: List[Dict]) -> Dict[int, int]:
        """
        Crea mapeo de category_id original → unified_category_id
        
        Args:
            original_categories: Lista de categorías del COCO original
        
        Returns:
            {original_cat_id: unified_cat_id}
        """
        print("\n🔄 Creando mapeo de categorías...")
        
        cat_id_mapping = {}
        unmapped_count = 0
        
        for cat in original_categories:
            original_id = cat["id"]
            original_name = cat["name"]
            
            # Buscar en mapeo
            if original_name in self.category_mapping:
                unified_id = self.category_mapping[original_name]
                cat_id_mapping[original_id] = unified_id
                
                # Estadística
                unified_name = self.unified_taxonomy[unified_id]["name"]
                key = f"{original_name} → {unified_name}"
                self.stats["category_mapping_counts"][key] = 0  # Se incrementa al procesar anotaciones
            else:
                print(f"⚠️ Categoría sin mapeo: '{original_name}'")
                self.stats["unmapped_categories"].append(original_name)
                unmapped_count += 1
        
        print(f"✅ Mapeo creado:")
        print(f"   Categorías mapeadas: {len(cat_id_mapping)}")
        print(f"   Categorías sin mapeo: {unmapped_count}")
        
        if unmapped_count > 0:
            print(f"\n⚠️ ADVERTENCIA: {unmapped_count} categorías sin mapeo!")
            print(f"   Revisa: {self.stats['unmapped_categories']}")
        
        return cat_id_mapping
    
    def unify_annotations(self, 
                         original_data: Dict, 
                         cat_id_mapping: Dict[int, int]) -> List[Dict]:
        """
        Convierte anotaciones a esquema híbrido con taxonomía unificada
        
        Esquema híbrido:
        - Todas las anotaciones tienen bbox (requerido por COCO)
        - Segmentación opcional (MVTec no tiene, VISION sí)
        - Metadata de trazabilidad completa
        """
        print("\n🔧 Unificando anotaciones a esquema híbrido...")
        
        unified_annotations = []
        category_counts = Counter()
        
        for ann in original_data["annotations"]:
            original_cat_id = ann["category_id"]
            
            # Mapear a categoría unificada
            if original_cat_id not in cat_id_mapping:
                print(f"⚠️ Anotación con category_id sin mapeo: {original_cat_id}")
                continue
            
            unified_cat_id = cat_id_mapping[original_cat_id]
            unified_cat_name = self.unified_taxonomy[unified_cat_id]["name"]
            
            # Incrementar conteo
            category_counts[unified_cat_name] += 1
            
            # Obtener información de la imagen
            image_info = self._get_image_info(original_data["images"], ann["image_id"])
            source = image_info.get("source_dataset", "unknown")
            
            # Crear anotación unificada (esquema híbrido)
            unified_ann = {
                # IDs (mantenemos originales por ahora)
                "id": ann["id"],
                "image_id": ann["image_id"],
                
                # CATEGORÍA UNIFICADA
                "category_id": unified_cat_id,
                "unified_category_name": unified_cat_name,
                
                # LOCALIZACIÓN (híbrido)
                "bbox": ann.get("bbox", []),
                "segmentation": ann.get("segmentation", []),
                "area": ann.get("area", 0),
                "iscrowd": ann.get("iscrowd", 0),
                
                # METADATA DEL ESQUEMA HÍBRIDO
                "has_segmentation": bool(ann.get("segmentation")),
                "localization_type": self._determine_localization_type(ann, source),
                "confidence": 1.0,  # Anotaciones manuales
                
                # TRAZABILIDAD
                "source_dataset": source,
                "original_category_id": original_cat_id,
                "original_label": self._get_original_label(
                    original_data["categories"], 
                    original_cat_id
                ),
                
                # METADATA ADICIONAL
                "unified_at": datetime.now().isoformat(),
                "curation_phase": "PHASE_3_UNIFICATION"
            }
            
            unified_annotations.append(unified_ann)
        
        # Actualizar estadísticas
        for cat_name, count in category_counts.items():
            if cat_name in [self.unified_taxonomy[i]["name"] for i in range(6)]:
                # Buscar mapeo original
                for key in self.stats["category_mapping_counts"].keys():
                    if key.endswith(f"→ {cat_name}"):
                        self.stats["category_mapping_counts"][key] = count
        
        print(f"✅ Anotaciones unificadas: {len(unified_annotations)}")
        print(f"\n📊 Distribución por categoría unificada:")
        for cat_id in sorted(category_counts.keys()):
            count = category_counts[cat_id]
            percentage = count / len(unified_annotations) * 100
            print(f"   • {cat_id}: {count} ({percentage:.1f}%)")
        
        return unified_annotations
    
    def _get_image_info(self, images: List[Dict], image_id: int) -> Dict:
        """Helper para obtener info de imagen por ID"""
        for img in images:
            if img["id"] == image_id:
                return img
        return {}
    
    def _get_original_label(self, categories: List[Dict], cat_id: int) -> str:
        """Helper para obtener label original de categoría"""
        for cat in categories:
            if cat["id"] == cat_id:
                return cat["name"]
        return "unknown"
    
    def _determine_localization_type(self, annotation: Dict, source: str) -> str:
        """
        Determina el tipo de localización de la anotación
        """
        has_seg = bool(annotation.get("segmentation"))
        bbox = annotation.get("bbox", [])
        
        # VISION con segmentación → pixel-level
        if has_seg and source == "vision":
            return "pixel_level"
        
        # VISION con solo bbox → bbox-level
        if bbox and not has_seg and source == "vision":
            return "bbox_level"
        
        # MVTec (bbox de imagen completa) → image-level
        if source == "mvtec":
            return "image_level"
        
        return "unknown"
    
    def create_unified_categories(self) -> List[Dict]:
        """
        Crea lista de categorías unificadas en formato COCO
        """
        unified_categories = []
        
        for cat_id, cat_info in self.unified_taxonomy.items():
            unified_categories.append({
                "id": cat_id,
                "name": cat_info["name"],
                "supercategory": cat_info["supercategory"],
                "description": cat_info["description"],
                "criticality": cat_info["criticality"],
                "color": cat_info["color"],
                "original_labels": self.reverse_mapping[cat_id]
            })
        
        return unified_categories
    
    def validate_unification(self, unified_data: Dict):
        """
        Valida que la unificación es correcta
        """
        print("\n✅ Validando unificación...")
        
        issues = []
        
        # Validación 1: Todas las anotaciones tienen categoría válida
        for ann in unified_data["annotations"]:
            cat_id = ann["category_id"]
            if cat_id not in range(6):
                issues.append(f"Annotation {ann['id']} tiene category_id inválido: {cat_id}")
        
        # Validación 2: Todas las categorías están presentes
        present_categories = set(ann["category_id"] for ann in unified_data["annotations"])
        if len(present_categories) < 6:
            missing = set(range(6)) - present_categories
            missing_names = [self.unified_taxonomy[i]["name"] for i in missing]
            issues.append(f"Categorías faltantes: {missing_names}")
        
        # Validación 3: Metadata de trazabilidad presente
        for ann in unified_data["annotations"]:
            required_fields = ["unified_category_name", "source_dataset", "original_label"]
            for field in required_fields:
                if field not in ann:
                    issues.append(f"Annotation {ann['id']} falta campo: {field}")
                    break  # Solo reportar una vez por anotación
        
        if issues:
            print(f"⚠️ Se encontraron {len(issues)} issues:")
            for issue in issues[:10]:  # Mostrar primeros 10
                print(f"   • {issue}")
            if len(issues) > 10:
                print(f"   ... y {len(issues) - 10} más")
        else:
            print("✅ Validación exitosa: No se encontraron issues")
        
        return len(issues) == 0
    
    def save_unified_dataset(self, unified_data: Dict):
        """
        Guarda el dataset unificado
        """
        print("\n💾 Guardando dataset unificado...")
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Guardar COCO JSON
        anno_file = self.output_path / "annotations_unified.coco.json"
        with open(anno_file, 'w', encoding='utf-8') as f:
            json.dump(unified_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Dataset unificado guardado: {anno_file}")
    
    def save_metadata(self):
        """
        Guarda metadata de la unificación
        """
        print("\n📊 Guardando metadata...")
        
        metadata_dir = self.output_path / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata de unificación
        unification_metadata = {
            "timestamp": datetime.now().isoformat(),
            "phase": "PHASE_3_TAXONOMY_UNIFICATION",
            "source_dataset": str(self.source_path),
            "output_dataset": str(self.output_path),
            "statistics": self.stats,
            "unified_taxonomy": {
                cat_id: {
                    "name": info["name"],
                    "description": info["description"],
                    "criticality": info["criticality"],
                    "original_labels": self.reverse_mapping[cat_id]
                }
                for cat_id, info in self.unified_taxonomy.items()
            },
            "category_mapping": self.category_mapping
        }
        
        metadata_file = metadata_dir / "phase3_unification_log.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(unification_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Metadata guardada: {metadata_file}")
        
        # CSV de distribución
        distribution_data = []
        for ann in self.stats.get("annotations_by_category", {}):
            pass  # Se construye durante el proceso
        
        # Guardar mapeo como CSV
        mapping_df = pd.DataFrame([
            {
                "Original_Label": orig,
                "Unified_Category_ID": unified_id,
                "Unified_Category_Name": self.unified_taxonomy[unified_id]["name"],
                "Criticality": self.unified_taxonomy[unified_id]["criticality"]
            }
            for orig, unified_id in self.category_mapping.items()
        ])
        
        mapping_file = metadata_dir / "category_mapping.csv"
        mapping_df.to_csv(mapping_file, index=False)
        print(f"✅ Mapeo guardado: {mapping_file}")
    
    def print_summary(self):
        """
        Imprime resumen final
        """
        print("\n" + "="*70)
        print("RESUMEN DE UNIFICACIÓN - FASE 3")
        print("="*70)
        
        print(f"\n📂 DATASET ORIGINAL:")
        print(f"   Categorías originales: {self.stats['original_categories']}")
        
        print(f"\n📂 DATASET UNIFICADO:")
        print(f"   Categorías unificadas: {self.stats['unified_categories']}")
        print(f"   Reducción: {self.stats['original_categories']} → {self.stats['unified_categories']}")
        print(f"   Factor: {self.stats['original_categories']/self.stats['unified_categories']:.1f}x")
        
        print(f"\n🔄 MAPEO DE CATEGORÍAS:")
        for key, count in sorted(self.stats["category_mapping_counts"].items()):
            if count > 0:
                print(f"   {key}: {count} anotaciones")
        
        print(f"\n📋 PROCESAMIENTO:")
        print(f"   Imágenes: {self.stats['images_processed']}")
        print(f"   Anotaciones: {self.stats['annotations_processed']}")
        
        if self.stats["unmapped_categories"]:
            print(f"\n⚠️ Categorías sin mapeo: {len(self.stats['unmapped_categories'])}")
            for cat in self.stats['unmapped_categories']:
                print(f"   • {cat}")
        
        print("\n" + "="*70)
        print(f"✅ FASE 3 COMPLETADA")
        print(f"📁 Dataset unificado en: {self.output_path}")
        print("="*70)
    
    def unify(self):
        """
        Ejecuta el pipeline completo de unificación
        """
        print("\n" + "="*70)
        print("FASE 3: UNIFICACIÓN DE TAXONOMÍA")
        print("="*70)
        
        try:
            # Paso 1: Cargar dataset re-curado
            original_data = self.load_recurated_dataset()
            
            # Paso 2: Crear mapeo de categorías
            cat_id_mapping = self.create_category_id_mapping(original_data["categories"])
            
            # Paso 3: Unificar anotaciones
            unified_annotations = self.unify_annotations(original_data, cat_id_mapping)
            
            # Paso 4: Crear categorías unificadas
            unified_categories = self.create_unified_categories()
            
            # Paso 5: Construir dataset unificado
            unified_data = {
                "info": {
                    **original_data.get("info", {}),
                    "version": "3.0",
                    "description": "Dataset con Taxonomía Unificada (6 categorías)",
                    "date_unified": datetime.now().isoformat(),
                    "curation_phase": "PHASE_3_TAXONOMY_UNIFICATION"
                },
                "licenses": original_data.get("licenses", []),
                "images": original_data["images"],  # Imágenes sin cambios
                "annotations": unified_annotations,
                "categories": unified_categories
            }
            
            # Paso 6: Validar
            valid = self.validate_unification(unified_data)
            if not valid:
                print("\n⚠️ ADVERTENCIA: Se encontraron issues en validación")
            
            # Paso 7: Guardar
            self.save_unified_dataset(unified_data)
            
            # Paso 8: Guardar metadata
            self.save_metadata()
            
            # Paso 9: Resumen
            self.print_summary()
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR en unificación: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Unificación de Taxonomía - Fase 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  python 03_unify_taxonomy.py \\
    --source curated_dataset_v2_20251101 \\
    --output curated_dataset_v3_unified

Taxonomía resultante:
  0: NORMAL
  1: DEFORMACIONES
  2: ROTURA_FRACTURA
  3: RAYONES_ARANAZOS
  4: PERFORACIONES
  5: CONTAMINACION

Fase: 3/8 del Pipeline de Curación
        """
    )
    
    parser.add_argument(
        "--source",
        required=True,
        help="Dataset re-curado (salida Fase 2)"
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Dataset con taxonomía unificada"
    )
    
    args = parser.parse_args()
    
    # Ejecutar unificación
    unifier = TaxonomyUnifier(
        source_path=args.source,
        output_path=args.output
    )
    
    success = unifier.unify()
    
    if success:
        print("\n🎉 Unificación completada exitosamente!")
        exit(0)
    else:
        print("\n❌ Unificación falló")
        exit(1)


if __name__ == "__main__":
    main()