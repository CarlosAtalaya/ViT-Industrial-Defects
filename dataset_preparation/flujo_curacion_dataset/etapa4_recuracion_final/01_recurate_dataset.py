#!/usr/bin/env python3
"""
02_recurate_dataset_v2.py
Re-curación del Dataset con Filtrado Mejorado

TFG: Vision Transformers para Detección de Anomalías Industriales
Fase 2 del Pipeline de Curación

Objetivos:
1. Eliminar categoría 'hazelnut' de MVTec AD (fuera de scope electrónico)
2. Remover duplicados detectados en análisis exploratorio
3. Filtrar solo componentes relevantes para electrónica
4. Generar dataset base limpio para unificación

Autor: [Tu nombre]
Fecha: Noviembre 2025
"""

import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple
import pandas as pd
from collections import defaultdict


class DatasetRecurator:
    """
    Re-curador científico de dataset con eliminación de componentes irrelevantes
    """
    
    def __init__(self, source_path: str, output_path: str, duplicates_csv: str = None):
        """
        Inicializa el re-curador
        
        Args:
            source_path: Ruta al dataset original
            output_path: Ruta de salida para dataset re-curado
            duplicates_csv: CSV con duplicados detectados (opcional)
        """
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.duplicates_csv = duplicates_csv
        
        # Configuración basada en análisis científico
        self._setup_filters()
        
        # Estadísticas
        self.stats = {
            "original_images": 0,
            "filtered_images": 0,
            "removed_hazelnut": 0,
            "removed_duplicates": 0,
            "removed_other": 0,
            "mvtec_images": 0,
            "vision_images": 0,
            "categories_removed": [],
            "components_kept": []
        }
        
        # Log de operaciones
        self.operation_log = []
    
    def _setup_filters(self):
        """
        Configura filtros basados en decisiones científicas documentadas
        """
        # DECISIÓN #1: Componentes MVTec permitidos
        # Justificación: Solo componentes electrónicos/industriales
        self.mvtec_categories_allowed = [
            "transistor",   # ✅ Componente electrónico
            "metal_nut",    # ✅ Componente mecánico industrial
            "cable",        # ✅ Conectividad electrónica
            "capsule"       # ✅ Componente industrial (píldoras en packaging)
        ]
        
        # DECISIÓN #2: Componentes VISION permitidos
        # Justificación: Todos relevantes para componentes electrónicos
        self.vision_components_allowed = [
            "PCB_1",        # ✅ Circuito impreso
            "PCB_2",        # ✅ Circuito impreso
            "Electronics",  # ✅ Componentes electrónicos generales
            "Console",      # ✅ Dispositivos electrónicos
            "Cable",        # ✅ Cables de conexión
            "Lens"          # ✅ Óptica para sensores
        ]
        
        # DECISIÓN #3: Defectos objetivo (para referencia)
        self.target_defects_mvtec = [
            "bent", "bent_lead", "bent_wire",           # DEFORMACIONES
            "crack", "broken", "broken_large",          # ROTURA_FRACTURA
            "scratch", "scratch_head",                  # RAYONES_ARANAZOS
            "hole", "cut",                              # PERFORACIONES
            "contamination", "metal_contamination",     # CONTAMINACION
            "good"                                      # NORMAL
        ]
        
        self.target_defects_vision = [
            "short", "spur",                            # DEFORMACIONES
            "break", "defect",                          # ROTURA_FRACTURA
            "Scratch", "s_scratch", "t_scratch",        # RAYONES_ARANAZOS
            "Hole", "missing_hole",                     # PERFORACIONES
            "Dirty", "impurities"                       # CONTAMINACION
        ]
        
        # Lista de duplicados a eliminar (se carga desde CSV)
        self.duplicates_to_remove = set()
    
    def load_duplicates(self):
        """
        Carga lista de duplicados desde el CSV de análisis
        
        Estrategia: Mantener file1, eliminar file2 de cada par
        """
        if not self.duplicates_csv or not Path(self.duplicates_csv).exists():
            print("⚠️ No se proporcionó CSV de duplicados o no existe")
            return
        
        print(f"📂 Cargando duplicados desde: {self.duplicates_csv}")
        
        df = pd.read_csv(self.duplicates_csv)
        
        # DECISIÓN: Eliminar file2 de cada par (mantener file1)
        # Justificación: file1 aparece primero en el dataset (menor ID)
        for _, row in df.iterrows():
            self.duplicates_to_remove.add(row['file2'])
        
        print(f"✅ {len(self.duplicates_to_remove)} duplicados marcados para eliminación")
        
        # Log de decisión
        self.operation_log.append({
            "operation": "load_duplicates",
            "timestamp": datetime.now().isoformat(),
            "files_to_remove": len(self.duplicates_to_remove),
            "strategy": "keep_file1_remove_file2"
        })
    
    def should_keep_image(self, img_info: Dict) -> Tuple[bool, str]:
        """
        Determina si una imagen debe mantenerse según reglas científicas
        
        Args:
            img_info: Información de la imagen del COCO JSON
        
        Returns:
            (keep: bool, reason: str)
        """
        source = img_info.get("source_dataset", "unknown")
        filename = img_info["file_name"]
        
        # REGLA 1: Eliminar duplicados
        if filename in self.duplicates_to_remove:
            self.stats["removed_duplicates"] += 1
            return False, f"duplicate_of_{filename}"
        
        # REGLA 2: Filtrar por dataset origen
        if source == "mvtec":
            return self._should_keep_mvtec(img_info)
        elif source == "vision":
            return self._should_keep_vision(img_info)
        else:
            self.stats["removed_other"] += 1
            return False, "unknown_source"
    
    def _should_keep_mvtec(self, img_info: Dict) -> Tuple[bool, str]:
        """
        Reglas específicas para imágenes MVTec
        """
        category = img_info.get("original_category", "")
        
        # REGLA CRÍTICA: Eliminar hazelnut
        if category == "hazelnut":
            self.stats["removed_hazelnut"] += 1
            if category not in self.stats["categories_removed"]:
                self.stats["categories_removed"].append(category)
            return False, "hazelnut_out_of_scope"
        
        # Verificar si está en lista permitida
        if category not in self.mvtec_categories_allowed:
            self.stats["removed_other"] += 1
            if category not in self.stats["categories_removed"]:
                self.stats["categories_removed"].append(category)
            return False, f"mvtec_category_not_allowed_{category}"
        
        return True, "mvtec_approved"
    
    def _should_keep_vision(self, img_info: Dict) -> Tuple[bool, str]:
        """
        Reglas específicas para imágenes VISION
        """
        component = img_info.get("original_component", "")
        
        # Verificar si está en lista permitida
        if component not in self.vision_components_allowed:
            self.stats["removed_other"] += 1
            if component not in self.stats["categories_removed"]:
                self.stats["categories_removed"].append(component)
            return False, f"vision_component_not_allowed_{component}"
        
        return True, "vision_approved"
    
    def load_original_dataset(self) -> Dict:
        """
        Carga el dataset original
        """
        print(f"\n📂 Cargando dataset original desde: {self.source_path}")
        
        anno_file = self.source_path / "annotations.coco.json"
        if not anno_file.exists():
            raise FileNotFoundError(f"No se encuentra: {anno_file}")
        
        with open(anno_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.stats["original_images"] = len(data["images"])
        
        print(f"✅ Dataset cargado:")
        print(f"   📷 Imágenes: {len(data['images'])}")
        print(f"   📋 Anotaciones: {len(data['annotations'])}")
        print(f"   🏷️ Categorías: {len(data['categories'])}")
        
        return data
    
    def filter_dataset(self, original_data: Dict) -> Dict:
        """
        Aplica filtros al dataset original
        
        Returns:
            Dataset filtrado en formato COCO
        """
        print("\n🔍 Aplicando filtros al dataset...")
        
        # Estructuras para nuevo dataset
        new_images = []
        kept_image_ids = set()
        removal_reasons = defaultdict(int)
        
        # Filtrar imágenes
        for img_info in original_data["images"]:
            keep, reason = self.should_keep_image(img_info)
            
            if keep:
                new_images.append(img_info)
                kept_image_ids.add(img_info["id"])
                
                # Estadísticas por origen
                source = img_info.get("source_dataset", "unknown")
                if source == "mvtec":
                    self.stats["mvtec_images"] += 1
                elif source == "vision":
                    self.stats["vision_images"] += 1
                
                # Registrar componente
                component = img_info.get("original_category") or img_info.get("original_component")
                if component and component not in self.stats["components_kept"]:
                    self.stats["components_kept"].append(component)
            else:
                removal_reasons[reason] += 1
        
        self.stats["filtered_images"] = len(new_images)
        
        print(f"\n📊 Resultados del filtrado:")
        print(f"   Imágenes originales:  {self.stats['original_images']}")
        print(f"   Imágenes mantenidas:  {self.stats['filtered_images']}")
        print(f"   Tasa retención:       {self.stats['filtered_images']/self.stats['original_images']*100:.1f}%")
        print(f"\n📉 Razones de eliminación:")
        for reason, count in sorted(removal_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {reason}: {count}")
        
        # Filtrar anotaciones (solo las de imágenes mantenidas)
        new_annotations = []
        for ann in original_data["annotations"]:
            if ann["image_id"] in kept_image_ids:
                new_annotations.append(ann)
        
        print(f"\n📋 Anotaciones:")
        print(f"   Originales:  {len(original_data['annotations'])}")
        print(f"   Mantenidas:  {len(new_annotations)}")
        
        # Crear nuevo dataset COCO
        new_data = {
            "info": {
                **original_data.get("info", {}),
                "version": "2.0",
                "description": "Dataset Re-Curado: Sin hazelnut, sin duplicados, componentes electrónicos",
                "date_created": datetime.now().isoformat(),
                "curation_phase": "PHASE_2_RECURATION"
            },
            "licenses": original_data.get("licenses", []),
            "images": new_images,
            "annotations": new_annotations,
            "categories": original_data["categories"]  # Mantenemos todas por ahora (se unifican en FASE 3)
        }
        
        return new_data
    
    def copy_images(self, filtered_data: Dict):
        """
        Copia imágenes filtradas al directorio de salida
        """
        print("\n📁 Copiando imágenes filtradas...")
        
        images_dir = self.output_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        source_images_dir = self.source_path / "images"
        
        copied = 0
        errors = 0
        
        for img_info in filtered_data["images"]:
            filename = img_info["file_name"]
            src = source_images_dir / filename
            dst = images_dir / filename
            
            try:
                if src.exists():
                    shutil.copy2(src, dst)
                    copied += 1
                else:
                    print(f"⚠️ Imagen no encontrada: {src}")
                    errors += 1
            except Exception as e:
                print(f"❌ Error copiando {filename}: {e}")
                errors += 1
        
        print(f"✅ Imágenes copiadas: {copied}")
        if errors > 0:
            print(f"⚠️ Errores: {errors}")
    
    def save_filtered_dataset(self, filtered_data: Dict):
        """
        Guarda el dataset filtrado
        """
        print("\n💾 Guardando dataset re-curado...")
        
        # Guardar COCO JSON
        anno_file = self.output_path / "annotations.coco.json"
        with open(anno_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Anotaciones guardadas: {anno_file}")
    
    def save_metadata(self):
        """
        Guarda metadata del proceso de re-curación
        """
        print("\n📊 Guardando metadata de curación...")
        
        metadata_dir = self.output_path / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata de re-curación
        recuration_metadata = {
            "timestamp": datetime.now().isoformat(),
            "phase": "PHASE_2_RECURATION",
            "source_dataset": str(self.source_path),
            "output_dataset": str(self.output_path),
            "statistics": self.stats,
            "filters_applied": {
                "mvtec_categories_allowed": self.mvtec_categories_allowed,
                "vision_components_allowed": self.vision_components_allowed,
                "duplicates_removed": len(self.duplicates_to_remove)
            },
            "decisions": {
                "hazelnut_removed": {
                    "reason": "Out of scope for electronic components research",
                    "count": self.stats["removed_hazelnut"]
                },
                "duplicates_strategy": "keep_first_occurrence",
                "category_focus": "electronic_and_industrial_components"
            },
            "operation_log": self.operation_log
        }
        
        # Guardar
        metadata_file = metadata_dir / "phase2_recuration_log.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(recuration_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Metadata guardada: {metadata_file}")
        
        # Guardar CSV resumen
        summary_df = pd.DataFrame([{
            "Métrica": k,
            "Valor": v
        } for k, v in self.stats.items() if isinstance(v, (int, float, str))])
        
        summary_file = metadata_dir / "phase2_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"✅ Resumen CSV guardado: {summary_file}")
    
    def print_summary(self):
        """
        Imprime resumen final detallado
        """
        print("\n" + "="*70)
        print("RESUMEN DE RE-CURACIÓN - FASE 2")
        print("="*70)
        
        print(f"\n📂 DATASET ORIGINAL:")
        print(f"   Ubicación: {self.source_path}")
        print(f"   Imágenes:  {self.stats['original_images']}")
        
        print(f"\n📂 DATASET RE-CURADO:")
        print(f"   Ubicación: {self.output_path}")
        print(f"   Imágenes:  {self.stats['filtered_images']}")
        print(f"   └── MVTec:   {self.stats['mvtec_images']} ({self.stats['mvtec_images']/self.stats['filtered_images']*100:.1f}%)")
        print(f"   └── VISION:  {self.stats['vision_images']} ({self.stats['vision_images']/self.stats['filtered_images']*100:.1f}%)")
        
        print(f"\n❌ IMÁGENES ELIMINADAS:")
        total_removed = (self.stats['original_images'] - self.stats['filtered_images'])
        print(f"   Total:       {total_removed} ({total_removed/self.stats['original_images']*100:.1f}%)")
        print(f"   └── Hazelnut:   {self.stats['removed_hazelnut']}")
        print(f"   └── Duplicados: {self.stats['removed_duplicates']}")
        print(f"   └── Otros:      {self.stats['removed_other']}")
        
        print(f"\n🏷️ COMPONENTES MANTENIDOS ({len(self.stats['components_kept'])}):")
        for comp in sorted(self.stats['components_kept']):
            print(f"   • {comp}")
        
        if self.stats['categories_removed']:
            print(f"\n🚫 CATEGORÍAS ELIMINADAS ({len(self.stats['categories_removed'])}):")
            for cat in sorted(self.stats['categories_removed']):
                print(f"   • {cat}")
        
        print(f"\n✅ CALIDAD DEL DATASET:")
        retention_rate = self.stats['filtered_images'] / self.stats['original_images']
        if retention_rate > 0.90:
            quality = "EXCELENTE (>90% retenido)"
        elif retention_rate > 0.80:
            quality = "BUENA (80-90% retenido)"
        elif retention_rate > 0.70:
            quality = "ACEPTABLE (70-80% retenido)"
        else:
            quality = "REVISAR (<70% retenido)"
        
        print(f"   Tasa de retención: {retention_rate*100:.1f}% - {quality}")
        
        print("\n" + "="*70)
        print(f"✅ FASE 2 COMPLETADA")
        print(f"📁 Dataset re-curado disponible en: {self.output_path}")
        print("="*70)
    
    def recurate(self):
        """
        Ejecuta el pipeline completo de re-curación
        """
        print("\n" + "="*70)
        print("FASE 2: RE-CURACIÓN DEL DATASET")
        print("="*70)
        
        try:
            # Paso 1: Cargar duplicados
            self.load_duplicates()
            
            # Paso 2: Cargar dataset original
            original_data = self.load_original_dataset()
            
            # Paso 3: Aplicar filtros
            filtered_data = self.filter_dataset(original_data)
            
            # Paso 4: Copiar imágenes
            self.copy_images(filtered_data)
            
            # Paso 5: Guardar dataset filtrado
            self.save_filtered_dataset(filtered_data)
            
            # Paso 6: Guardar metadata
            self.save_metadata()
            
            # Paso 7: Resumen
            self.print_summary()
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR en re-curación: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Re-curación del Dataset - Fase 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Re-curar con eliminación de duplicados
  python 02_recurate_dataset_v2.py \\
    --source curated_dataset_20250921_115859 \\
    --output curated_dataset_v2_20251101 \\
    --duplicates analysis_reports/phase1/duplicates.csv

  # Re-curar sin CSV de duplicados (solo elimina hazelnut)
  python 02_recurate_dataset_v2.py \\
    --source curated_dataset_20250921_115859 \\
    --output curated_dataset_v2_20251101

Fase: 2/8 del Pipeline de Curación
        """
    )
    
    parser.add_argument(
        "--source",
        required=True,
        help="Ruta al dataset original"
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Ruta de salida para dataset re-curado"
    )
    
    parser.add_argument(
        "--duplicates",
        default=None,
        help="CSV con duplicados detectados (opcional)"
    )
    
    args = parser.parse_args()
    
    # Ejecutar re-curación
    recurator = DatasetRecurator(
        source_path=args.source,
        output_path=args.output,
        duplicates_csv=args.duplicates
    )
    
    success = recurator.recurate()
    
    if success:
        print("\n🎉 Re-curación completada exitosamente!")
        exit(0)
    else:
        print("\n❌ Re-curación falló")
        exit(1)


if __name__ == "__main__":
    main()