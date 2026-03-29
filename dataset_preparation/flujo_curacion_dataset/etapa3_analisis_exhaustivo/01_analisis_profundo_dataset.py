#!/usr/bin/env python3
"""
01_deep_analysis.py
Análisis Exploratorio Profundo del Dataset Actual
TFG: Vision Transformers para Detección de Anomalías Industriales
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd
import imagehash
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class DatasetDeepAnalyzer:
    """
    Análisis científico riguroso del dataset actual
    """
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.annotations = self._load_annotations()
        self.metadata = self._load_metadata()
        
    def _load_annotations(self):
        """Carga el archivo de anotaciones COCO"""
        anno_path = self.dataset_path / "annotations.coco.json"
        if not anno_path.exists():
            raise FileNotFoundError(f"No se encuentra: {anno_path}")
        
        with open(anno_path) as f:
            return json.load(f)
    
    def _load_metadata(self):
        """Carga el metadata del dataset"""
        meta_path = self.dataset_path / "metadata" / "dataset_info.json"
        if not meta_path.exists():
            print(f"⚠️ No se encuentra metadata en: {meta_path}")
            return {}
        
        with open(meta_path) as f:
            return json.load(f)
    
    def _get_category_name(self, cat_id):
        """Obtiene el nombre de categoría por ID"""
        for cat in self.annotations["categories"]:
            if cat["id"] == cat_id:
                return cat["name"]
        return "unknown"
    
    def analyze_category_distribution(self):
        """
        Análisis de distribución de categorías originales
        """
        print("\n📊 Analizando distribución de categorías...")
        
        categories = {}
        for ann in self.annotations["annotations"]:
            cat_id = ann["category_id"]
            cat_name = self._get_category_name(cat_id)
            categories[cat_name] = categories.get(cat_name, 0) + 1
        
        df = pd.DataFrame.from_dict(
            categories, 
            orient='index', 
            columns=['count']
        ).sort_values('count', ascending=False)
        
        df['percentage'] = (df['count'] / df['count'].sum() * 100).round(2)
        
        print(df)
        return df
    
    def analyze_source_distribution(self):
        """
        Analiza distribución por dataset de origen
        """
        print("\n📊 Analizando distribución por origen...")
        
        sources = defaultdict(lambda: {"images": 0, "annotations": 0})
        
        for img_info in self.annotations["images"]:
            source = img_info.get("source_dataset", "unknown")
            sources[source]["images"] += 1
        
        for ann in self.annotations["annotations"]:
            source = ann.get("source_dataset", "unknown")
            sources[source]["annotations"] += 1
        
        df = pd.DataFrame.from_dict(sources, orient='index')
        print(df)
        return df
    
    def detect_duplicates(self, hash_size=16, threshold=5):
        """
        Detecta imágenes duplicadas usando perceptual hashing
        """
        print("\n🔍 Detectando imágenes duplicadas...")
        
        duplicates = []
        hashes = {}
        
        images_dir = self.dataset_path / "images"
        
        for img_info in self.annotations["images"]:
            img_path = images_dir / img_info["file_name"]
            
            if not img_path.exists():
                print(f"⚠️ Imagen no encontrada: {img_path}")
                continue
            
            try:
                # Calcular hash perceptual
                img = Image.open(img_path)
                img_hash = imagehash.phash(img, hash_size=hash_size)
                
                # Buscar duplicados
                for existing_hash, existing_file in hashes.items():
                    distance = abs(img_hash - existing_hash)
                    if distance < threshold:
                        duplicates.append({
                            'file1': existing_file,
                            'file2': img_info["file_name"],
                            'distance': distance,
                            'source1': self._get_image_source(existing_file),
                            'source2': img_info.get("source_dataset")
                        })
                
                hashes[img_hash] = img_info["file_name"]
                
            except Exception as e:
                print(f"⚠️ Error procesando {img_info['file_name']}: {e}")
        
        df = pd.DataFrame(duplicates)
        print(f"✅ Duplicados encontrados: {len(df)}")
        return df
    
    def _get_image_source(self, filename):
        """Helper para obtener source de una imagen por filename"""
        for img in self.annotations["images"]:
            if img["file_name"] == filename:
                return img.get("source_dataset", "unknown")
        return "unknown"
    
    def analyze_image_quality(self):
        """
        Analiza resolución, formato, y calidad de imágenes
        """
        print("\n📐 Analizando calidad de imágenes...")
        
        resolutions = []
        corrupted = []
        
        images_dir = self.dataset_path / "images"
        
        for img_info in self.annotations["images"]:
            img_path = images_dir / img_info["file_name"]
            
            if not img_path.exists():
                corrupted.append({
                    'file': img_info["file_name"],
                    'error': 'File not found'
                })
                continue
            
            try:
                with Image.open(img_path) as img:
                    resolutions.append({
                        'file': img_info["file_name"],
                        'width': img.size[0],
                        'height': img.size[1],
                        'format': img.format,
                        'mode': img.mode,
                        'source': img_info.get("source_dataset", "unknown"),
                        'aspect_ratio': round(img.size[0] / img.size[1], 2)
                    })
            except Exception as e:
                corrupted.append({
                    'file': img_info["file_name"],
                    'error': str(e)
                })
        
        df_res = pd.DataFrame(resolutions)
        df_corr = pd.DataFrame(corrupted)
        
        if len(df_res) > 0:
            print(f"✅ Imágenes válidas: {len(df_res)}")
            print(f"\n📊 Estadísticas de resolución:")
            print(df_res[['width', 'height']].describe())
            
            # Agrupar por resolución
            print(f"\n📊 Resoluciones más comunes:")
            resolution_counts = df_res.groupby(['width', 'height']).size().sort_values(ascending=False).head(10)
            print(resolution_counts)
        
        if len(df_corr) > 0:
            print(f"❌ Imágenes corruptas: {len(df_corr)}")
        
        return df_res, df_corr
    
    def analyze_annotation_consistency(self):
        """
        Verifica consistencia entre imágenes y anotaciones
        """
        print("\n🔍 Verificando consistencia de anotaciones...")
        
        issues = []
        
        # Mapear image_id -> anotaciones
        img_annotations = defaultdict(list)
        for ann in self.annotations["annotations"]:
            img_id = ann["image_id"]
            img_annotations[img_id].append(ann)
        
        # Verificar cada imagen
        for img_info in self.annotations["images"]:
            img_id = img_info["id"]
            anns = img_annotations.get(img_id, [])
            
            # Regla 1: Todas las imágenes deben tener al menos una anotación
            if len(anns) == 0:
                issues.append({
                    'image_id': img_id,
                    'file': img_info["file_name"],
                    'issue': 'No annotations for this image',
                    'severity': 'MEDIUM',
                    'source': img_info.get("source_dataset")
                })
            
            # Regla 2: Imágenes MVTec "good" deberían tener categoría "normal"
            if img_info.get("defect_type") == "good":
                defect_anns = [a for a in anns if self._get_category_name(a["category_id"]) != "normal"]
                if defect_anns:
                    issues.append({
                        'image_id': img_id,
                        'file': img_info["file_name"],
                        'issue': f'Image marked as "good" has {len(defect_anns)} defect annotation(s)',
                        'severity': 'HIGH',
                        'source': img_info.get("source_dataset")
                    })
            
            # Regla 3: Bboxes deben estar dentro de límites
            for ann in anns:
                bbox = ann.get("bbox", [])
                if bbox and len(bbox) == 4:
                    x, y, w, h = bbox
                    if (x < 0 or y < 0 or 
                        x + w > img_info["width"] or 
                        y + h > img_info["height"]):
                        issues.append({
                            'image_id': img_id,
                            'annotation_id': ann["id"],
                            'file': img_info["file_name"],
                            'issue': f'Bbox outside boundaries: [{x}, {y}, {w}, {h}] vs image [{img_info["width"]}, {img_info["height"]}]',
                            'severity': 'CRITICAL',
                            'source': img_info.get("source_dataset")
                        })
            
            # Regla 4: Áreas de bbox sospechosamente pequeñas
            for ann in anns:
                area = ann.get("area", 0)
                if area > 0 and area < 100:  # Menos de 10x10 pixels
                    issues.append({
                        'image_id': img_id,
                        'annotation_id': ann["id"],
                        'file': img_info["file_name"],
                        'issue': f'Suspiciously small bbox area: {area} pixels',
                        'severity': 'LOW',
                        'source': img_info.get("source_dataset")
                    })
        
        df = pd.DataFrame(issues)
        
        if len(df) > 0:
            print(f"⚠️ Issues encontrados: {len(df)}")
            print(f"\n📊 Por severidad:")
            print(df['severity'].value_counts())
        else:
            print(f"✅ No se encontraron issues de consistencia")
        
        return df
    
    def analyze_defect_mapping(self):
        """
        Analiza cómo se mapearían las categorías actuales a la taxonomía unificada
        """
        print("\n🗂️ Analizando mapeo a taxonomía unificada...")
        
        # Definir mapeo (del metadata)
        mapping = {}
        if "target_defects" in self.metadata:
            for unified_name, defect_info in self.metadata["target_defects"].items():
                for label in defect_info.get("mvtec_labels", []):
                    mapping[label] = unified_name
                for label in defect_info.get("vision_labels", []):
                    mapping[label] = unified_name
        
        # Añadir "normal"
        mapping["normal"] = "NORMAL"
        mapping["good"] = "NORMAL"
        
        # Analizar cobertura
        current_categories = {cat["name"] for cat in self.annotations["categories"]}
        mapped = set()
        unmapped = set()
        
        for cat_name in current_categories:
            if cat_name in mapping:
                mapped.add(cat_name)
            else:
                unmapped.add(cat_name)
        
        print(f"✅ Categorías con mapeo: {len(mapped)}/{len(current_categories)}")
        if unmapped:
            print(f"⚠️ Categorías SIN mapeo: {unmapped}")
        
        # Simular distribución post-unificación
        unified_dist = defaultdict(int)
        for ann in self.annotations["annotations"]:
            cat_name = self._get_category_name(ann["category_id"])
            unified_name = mapping.get(cat_name, "UNMAPPED")
            unified_dist[unified_name] += 1
        
        df = pd.DataFrame.from_dict(
            unified_dist, 
            orient='index', 
            columns=['count']
        ).sort_values('count', ascending=False)
        
        df['percentage'] = (df['count'] / df['count'].sum() * 100).round(2)
        
        print(f"\n📊 Distribución esperada post-unificación:")
        print(df)
        
        return df, mapping
    
    def generate_visualizations(self, output_dir):
        """
        Genera gráficos de análisis
        """
        print("\n📊 Generando visualizaciones...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Distribución de categorías
        cat_dist = self.analyze_category_distribution()
        
        plt.figure(figsize=(12, 6))
        cat_dist['count'].plot(kind='bar')
        plt.title('Distribución de Categorías Actuales')
        plt.xlabel('Categoría')
        plt.ylabel('Número de Anotaciones')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'category_distribution.png', dpi=300)
        plt.close()
        
        # 2. Distribución por origen
        source_dist = self.analyze_source_distribution()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        source_dist['images'].plot(kind='bar', ax=ax1, color='steelblue')
        ax1.set_title('Imágenes por Dataset')
        ax1.set_ylabel('Cantidad')
        
        source_dist['annotations'].plot(kind='bar', ax=ax2, color='coral')
        ax2.set_title('Anotaciones por Dataset')
        ax2.set_ylabel('Cantidad')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'source_distribution.png', dpi=300)
        plt.close()
        
        # 3. Distribución de resoluciones
        resolutions, _ = self.analyze_image_quality()
        
        if len(resolutions) > 0:
            plt.figure(figsize=(10, 6))
            plt.scatter(resolutions['width'], resolutions['height'], 
                       c=resolutions['source'].astype('category').cat.codes, 
                       alpha=0.6)
            plt.xlabel('Ancho (pixels)')
            plt.ylabel('Alto (pixels)')
            plt.title('Distribución de Resoluciones')
            plt.colorbar(label='Dataset')
            plt.tight_layout()
            plt.savefig(output_dir / 'resolution_scatter.png', dpi=300)
            plt.close()
        
        print(f"✅ Visualizaciones guardadas en: {output_dir}")
    
    def generate_report(self, output_dir):
        """
        Genera reporte completo con todos los análisis
        """
        print("\n" + "="*60)
        print("GENERANDO REPORTE COMPLETO DE ANÁLISIS")
        print("="*60)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ejecutar todos los análisis
        cat_dist = self.analyze_category_distribution()
        source_dist = self.analyze_source_distribution()
        dupes = self.detect_duplicates()
        resolutions, corrupted = self.analyze_image_quality()
        issues = self.analyze_annotation_consistency()
        unified_dist, mapping = self.analyze_defect_mapping()
        
        # Crear reporte resumen
        report = {
            "timestamp": datetime.now().isoformat(),
            "dataset_path": str(self.dataset_path),
            "summary": {
                "total_images": len(self.annotations["images"]),
                "total_annotations": len(self.annotations["annotations"]),
                "total_categories": len(self.annotations["categories"]),
                "duplicates_found": len(dupes),
                "corrupted_images": len(corrupted),
                "consistency_issues": len(issues)
            },
            "categories": cat_dist.to_dict(),
            "source_distribution": source_dist.to_dict(),
            "unified_preview": unified_dist.to_dict(),
            "quality_metrics": {
                "resolution_stats": resolutions[['width', 'height']].describe().to_dict() if len(resolutions) > 0 else {},
                "format_distribution": resolutions['format'].value_counts().to_dict() if len(resolutions) > 0 else {}
            }
        }
        
        # Guardar JSONs y CSVs
        with open(output_dir / "analysis_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        cat_dist.to_csv(output_dir / "category_distribution.csv")
        source_dist.to_csv(output_dir / "source_distribution.csv")
        unified_dist.to_csv(output_dir / "unified_distribution_preview.csv")
        
        if len(dupes) > 0:
            dupes.to_csv(output_dir / "duplicates.csv", index=False)
        
        if len(corrupted) > 0:
            corrupted.to_csv(output_dir / "corrupted_images.csv", index=False)
        
        if len(issues) > 0:
            issues.to_csv(output_dir / "consistency_issues.csv", index=False)
        
        if len(resolutions) > 0:
            resolutions.to_csv(output_dir / "image_resolutions.csv", index=False)
        
        # Generar visualizaciones
        self.generate_visualizations(output_dir)
        
        # Resumen en consola
        print("\n" + "="*60)
        print("RESUMEN DEL ANÁLISIS")
        print("="*60)
        print(f"📁 Dataset: {self.dataset_path}")
        print(f"🖼️ Total imágenes: {report['summary']['total_images']}")
        print(f"📋 Total anotaciones: {report['summary']['total_annotations']}")
        print(f"🏷️ Total categorías: {report['summary']['total_categories']}")
        print(f"👥 Duplicados: {report['summary']['duplicates_found']}")
        print(f"❌ Corruptas: {report['summary']['corrupted_images']}")
        print(f"⚠️ Issues: {report['summary']['consistency_issues']}")
        print("\n✅ Reporte completo guardado en:", output_dir)
        print("="*60)
        
        return report


def main():
    parser = argparse.ArgumentParser(
        description="Análisis Exploratorio Profundo del Dataset"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Ruta al dataset (carpeta con annotations.coco.json)"
    )
    parser.add_argument(
        "--output",
        default="analysis_reports/phase1",
        help="Directorio de salida para reportes"
    )
    
    args = parser.parse_args()
    
    # Ejecutar análisis
    analyzer = DatasetDeepAnalyzer(args.dataset)
    analyzer.generate_report(args.output)


if __name__ == "__main__":
    main()