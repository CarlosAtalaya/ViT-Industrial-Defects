#!/usr/bin/env python3
"""
dataset_balance_analysis.py
Análisis exhaustivo del balance del dataset en train/val/test.

Analiza:
- Distribución de categorías
- Tamaños de imágenes (width, height, area, short_edge)
- Tamaños de bboxes (width, height, area, aspect_ratio)
- Balance entre splits
- Procedencia (source_dataset)
- Imágenes augmentadas vs originales
- Correlaciones entre variables

Uso:
    python dataset_balance_analysis.py --dataset-root path/to/dataset --output-dir analysis_results
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class DatasetBalanceAnalyzer:
    """Analizador de balance del dataset."""
    
    def __init__(self, dataset_root, output_dir):
        self.dataset_root = Path(dataset_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.splits = ['inspect_train', 'inspect_val', 'inspect_test']
        self.data = {}
        
    def load_data(self):
        """Carga todos los CSVs de los tres splits."""
        print("=" * 80)
        print("CARGANDO DATOS")
        print("=" * 80)
        
        for split in self.splits:
            split_dir = self.dataset_root / split
            
            # Verificar que exista el directorio
            if not split_dir.exists():
                print(f"⚠️  WARNING: Split '{split}' no encontrado en {split_dir}")
                continue
            
            print(f"\nCargando split: {split}")
            
            self.data[split] = {}
            
            # Cargar CSVs
            csv_files = {
                'annotations': 'annotations_table.csv',
                'bboxes': 'bboxes_stats.csv',
                'images': 'images_table.csv',
                'images_area': 'images_with_area.csv'
            }
            
            for key, filename in csv_files.items():
                csv_path = split_dir / filename
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    self.data[split][key] = df
                    print(f"  ✓ {filename}: {len(df)} filas")
                else:
                    print(f"  ✗ {filename}: NO ENCONTRADO")
                    self.data[split][key] = None
        
        print("\n" + "=" * 80)
    
    def analyze_category_distribution(self):
        """Analiza la distribución de categorías entre splits."""
        print("\n" + "=" * 80)
        print("ANÁLISIS DE DISTRIBUCIÓN DE CATEGORÍAS")
        print("=" * 80)
        
        # Recopilar categorías
        category_counts = defaultdict(lambda: defaultdict(int))
        
        for split in self.splits:
            if split not in self.data or self.data[split]['annotations'] is None:
                continue
            
            df = self.data[split]['annotations']
            
            if 'unified_category_name' in df.columns:
                counts = df['unified_category_name'].value_counts()
                for cat, count in counts.items():
                    category_counts[cat][split] = count
        
        # Crear DataFrame
        df_cats = pd.DataFrame(category_counts).T.fillna(0).astype(int)
        df_cats['total'] = df_cats.sum(axis=1)
        df_cats = df_cats.sort_values('total', ascending=False)
        
        # Calcular porcentajes
        df_cats_pct = df_cats.copy()
        for split in self.splits:
            if split in df_cats_pct.columns:
                total = df_cats_pct[split].sum()
                df_cats_pct[f'{split}_pct'] = (df_cats_pct[split] / total * 100).round(2)
        
        # Guardar CSV
        output_path = self.output_dir / 'category_distribution.csv'
        df_cats.to_csv(output_path)
        print(f"\n✓ Guardado: {output_path}")
        
        # Imprimir tabla
        print("\nDistribución de categorías (anotaciones):")
        print(df_cats.to_string())
        
        # Gráfica de barras comparativa
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(df_cats))
        width = 0.25
        
        for i, split in enumerate(self.splits):
            if split in df_cats.columns:
                offset = (i - 1) * width
                ax.bar(x + offset, df_cats[split], width, label=split.capitalize())
        
        ax.set_xlabel('Categoría', fontsize=12, fontweight='bold')
        ax.set_ylabel('Número de Anotaciones', fontsize=12, fontweight='bold')
        ax.set_title('Distribución de Categorías por Split', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df_cats.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        fig_path = self.output_dir / 'category_distribution.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"✓ Gráfica guardada: {fig_path}")
        
        # Gráfica de proporciones (pie charts)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Proporción de Categorías por Split', fontsize=14, fontweight='bold')
        
        for i, split in enumerate(self.splits):
            if split in df_cats.columns and df_cats[split].sum() > 0:
                ax = axes[i]
                data = df_cats[split][df_cats[split] > 0]
                ax.pie(data, labels=data.index, autopct='%1.1f%%', startangle=90)
                ax.set_title(f'{split.capitalize()} ({data.sum()} anotaciones)')
        
        plt.tight_layout()
        fig_path = self.output_dir / 'category_proportions.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"✓ Gráfica guardada: {fig_path}")
        
        return df_cats
    
    def analyze_image_sizes(self):
        """Analiza los tamaños de imágenes."""
        print("\n" + "=" * 80)
        print("ANÁLISIS DE TAMAÑOS DE IMÁGENES")
        print("=" * 80)
        
        stats_summary = []
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Distribución de Tamaños de Imágenes', fontsize=14, fontweight='bold')
        
        for idx, split in enumerate(self.splits):
            if split not in self.data or self.data[split]['images_area'] is None:
                continue
            
            df = self.data[split]['images_area']
            
            # Estadísticas
            stats = {
                'split': split,
                'n_images': len(df),
                'width_mean': df['width'].mean(),
                'width_std': df['width'].std(),
                'width_min': df['width'].min(),
                'width_max': df['width'].max(),
                'height_mean': df['height'].mean(),
                'height_std': df['height'].std(),
                'height_min': df['height'].min(),
                'height_max': df['height'].max(),
                'area_mean': df['area_px'].mean(),
                'area_std': df['area_px'].std(),
                'short_edge_mean': df['short_edge'].mean(),
                'short_edge_std': df['short_edge'].std(),
                'short_edge_min': df['short_edge'].min(),
                'short_edge_max': df['short_edge'].max(),
            }
            stats_summary.append(stats)
            
            # Gráfica: Distribución de short_edge
            ax = axes[0, idx]
            ax.hist(df['short_edge'], bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(df['short_edge'].mean(), color='red', linestyle='--', 
                      label=f'Media: {df["short_edge"].mean():.1f}')
            ax.axvline(df['short_edge'].median(), color='green', linestyle='--',
                      label=f'Mediana: {df["short_edge"].median():.1f}')
            ax.set_xlabel('Short Edge (px)', fontsize=10)
            ax.set_ylabel('Frecuencia', fontsize=10)
            ax.set_title(f'{split.capitalize()} - Short Edge', fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            
            # Gráfica: Distribución de area
            ax = axes[1, idx]
            ax.hist(df['area_px'], bins=30, alpha=0.7, edgecolor='black', color='orange')
            ax.axvline(df['area_px'].mean(), color='red', linestyle='--',
                      label=f'Media: {df["area_px"].mean():.0f}')
            ax.axvline(df['area_px'].median(), color='green', linestyle='--',
                      label=f'Mediana: {df["area_px"].median():.0f}')
            ax.set_xlabel('Área (px²)', fontsize=10)
            ax.set_ylabel('Frecuencia', fontsize=10)
            ax.set_title(f'{split.capitalize()} - Área Imagen', fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        fig_path = self.output_dir / 'image_sizes_distribution.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"✓ Gráfica guardada: {fig_path}")
        
        # Guardar estadísticas
        df_stats = pd.DataFrame(stats_summary)
        output_path = self.output_dir / 'image_sizes_stats.csv'
        df_stats.to_csv(output_path, index=False)
        print(f"✓ Estadísticas guardadas: {output_path}")
        
        print("\nEstadísticas de tamaños de imágenes:")
        print(df_stats.to_string(index=False))
        
        return df_stats
    
    def analyze_bbox_sizes(self):
        """Analiza los tamaños de bounding boxes."""
        print("\n" + "=" * 80)
        print("ANÁLISIS DE TAMAÑOS DE BOUNDING BOXES")
        print("=" * 80)
        
        stats_summary = []
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        fig.suptitle('Distribución de Bounding Boxes', fontsize=14, fontweight='bold')
        
        for idx, split in enumerate(self.splits):
            if split not in self.data or self.data[split]['bboxes'] is None:
                continue
            
            df = self.data[split]['bboxes']
            
            if len(df) == 0:
                print(f"⚠️  WARNING: No hay bboxes en split '{split}'")
                continue
            
            # Estadísticas
            stats = {
                'split': split,
                'n_bboxes': len(df),
                'width_mean': df['width'].mean(),
                'width_std': df['width'].std(),
                'width_min': df['width'].min(),
                'width_max': df['width'].max(),
                'width_median': df['width'].median(),
                'height_mean': df['height'].mean(),
                'height_std': df['height'].std(),
                'height_min': df['height'].min(),
                'height_max': df['height'].max(),
                'height_median': df['height'].median(),
                'area_mean': df['area'].mean(),
                'area_std': df['area'].std(),
                'area_min': df['area'].min(),
                'area_max': df['area'].max(),
                'area_median': df['area'].median(),
                'aspect_ratio_mean': df['aspect_ratio'].mean(),
                'aspect_ratio_std': df['aspect_ratio'].std(),
                'aspect_ratio_min': df['aspect_ratio'].min(),
                'aspect_ratio_max': df['aspect_ratio'].max(),
                'aspect_ratio_median': df['aspect_ratio'].median(),
            }
            
            # Análisis crítico: bboxes pequeños (< 32px)
            small_width = (df['width'] < 32).sum()
            small_height = (df['height'] < 32).sum()
            stats['small_width_count'] = small_width
            stats['small_height_count'] = small_height
            stats['small_width_pct'] = (small_width / len(df) * 100)
            stats['small_height_pct'] = (small_height / len(df) * 100)
            
            # Análisis crítico: aspect ratios extremos
            extreme_ar = ((df['aspect_ratio'] < 0.25) | (df['aspect_ratio'] > 4.0)).sum()
            stats['extreme_aspect_ratio_count'] = extreme_ar
            stats['extreme_aspect_ratio_pct'] = (extreme_ar / len(df) * 100)
            
            stats_summary.append(stats)
            
            # Gráfica 1: Distribución de width
            ax = axes[0, idx]
            ax.hist(df['width'], bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(32, color='red', linestyle='--', label='32px (RPN min)')
            ax.axvline(df['width'].median(), color='green', linestyle='--',
                      label=f'Mediana: {df["width"].median():.1f}')
            ax.set_xlabel('Width (px)', fontsize=10)
            ax.set_ylabel('Frecuencia', fontsize=10)
            ax.set_title(f'{split.capitalize()} - BBox Width', fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            
            # Gráfica 2: Distribución de area
            ax = axes[1, idx]
            ax.hist(df['area'], bins=50, alpha=0.7, edgecolor='black', color='orange')
            ax.axvline(df['area'].median(), color='green', linestyle='--',
                      label=f'Mediana: {df["area"].median():.1f}')
            ax.set_xlabel('Área (px²)', fontsize=10)
            ax.set_ylabel('Frecuencia', fontsize=10)
            ax.set_title(f'{split.capitalize()} - BBox Área', fontweight='bold')
            ax.set_yscale('log')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            
            # Gráfica 3: Distribución de aspect_ratio
            ax = axes[2, idx]
            # Filtrar outliers para mejor visualización
            ar_filtered = df['aspect_ratio'][(df['aspect_ratio'] > 0) & (df['aspect_ratio'] < 10)]
            ax.hist(ar_filtered, bins=50, alpha=0.7, edgecolor='black', color='purple')
            ax.axvline(0.5, color='blue', linestyle='--', alpha=0.5, label='Faster R-CNN: 0.5')
            ax.axvline(1.0, color='blue', linestyle='--', alpha=0.5, label='Faster R-CNN: 1.0')
            ax.axvline(2.0, color='blue', linestyle='--', alpha=0.5, label='Faster R-CNN: 2.0')
            ax.axvline(ar_filtered.median(), color='green', linestyle='--',
                      label=f'Mediana: {ar_filtered.median():.2f}')
            ax.set_xlabel('Aspect Ratio (W/H)', fontsize=10)
            ax.set_ylabel('Frecuencia', fontsize=10)
            ax.set_title(f'{split.capitalize()} - Aspect Ratio', fontweight='bold')
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        fig_path = self.output_dir / 'bbox_distribution.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"✓ Gráfica guardada: {fig_path}")
        
        # Verificar si hay datos antes de crear DataFrame
        if not stats_summary:
            print("⚠️  WARNING: No se encontraron estadísticas de bboxes en ningún split")
            return None
        
        # Guardar estadísticas
        df_stats = pd.DataFrame(stats_summary)
        output_path = self.output_dir / 'bbox_stats.csv'
        df_stats.to_csv(output_path, index=False)
        print(f"✓ Estadísticas guardadas: {output_path}")
        
        # Imprimir tabla con las columnas que existen
        print("\nEstadísticas de bounding boxes:")
        cols_to_show = ['split', 'n_bboxes', 'width_median', 'height_median', 
                        'area_median', 'aspect_ratio_median']
        # Verificar que las columnas existen
        cols_available = [col for col in cols_to_show if col in df_stats.columns]
        if cols_available:
            print(df_stats[cols_available].to_string(index=False))
        else:
            print(df_stats.to_string(index=False))
        
        print("\n⚠️  PROBLEMAS CRÍTICOS DETECTADOS:")
        for stats in stats_summary:
            print(f"\n{stats['split'].upper()}:")
            if stats['small_width_pct'] > 10:
                print(f"  🚨 {stats['small_width_pct']:.1f}% de bboxes tienen width < 32px")
            if stats['small_height_pct'] > 10:
                print(f"  🚨 {stats['small_height_pct']:.1f}% de bboxes tienen height < 32px")
            if stats['extreme_aspect_ratio_pct'] > 20:
                print(f"  ⚠️  {stats['extreme_aspect_ratio_pct']:.1f}% de bboxes tienen aspect ratio extremo (<0.25 o >4.0)")
        
        return df_stats
    
    def analyze_source_dataset(self):
        """Analiza la distribución por dataset de origen."""
        print("\n" + "=" * 80)
        print("ANÁLISIS DE DATASET DE ORIGEN")
        print("=" * 80)
        
        source_counts = defaultdict(lambda: defaultdict(int))
        
        for split in self.splits:
            if split not in self.data or self.data[split]['images'] is None:
                continue
            
            df = self.data[split]['images']
            
            if 'source_dataset' in df.columns:
                counts = df['source_dataset'].value_counts()
                for source, count in counts.items():
                    source_counts[source][split] = count
        
        # Crear DataFrame
        df_sources = pd.DataFrame(source_counts).T.fillna(0).astype(int)
        df_sources['total'] = df_sources.sum(axis=1)
        df_sources = df_sources.sort_values('total', ascending=False)
        
        # Guardar CSV
        output_path = self.output_dir / 'source_dataset_distribution.csv'
        df_sources.to_csv(output_path)
        print(f"\n✓ Guardado: {output_path}")
        
        print("\nDistribución por dataset de origen:")
        print(df_sources.to_string())
        
        # Gráfica
        fig, ax = plt.subplots(figsize=(10, 6))
        df_sources[self.splits].plot(kind='bar', ax=ax, alpha=0.8)
        ax.set_xlabel('Dataset de Origen', fontsize=12, fontweight='bold')
        ax.set_ylabel('Número de Imágenes', fontsize=12, fontweight='bold')
        ax.set_title('Distribución de Imágenes por Dataset de Origen', fontsize=14, fontweight='bold')
        ax.legend(title='Split')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        fig_path = self.output_dir / 'source_dataset_distribution.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"✓ Gráfica guardada: {fig_path}")
        
        return df_sources
    
    def analyze_augmentation(self):
        """Analiza la proporción de imágenes augmentadas."""
        print("\n" + "=" * 80)
        print("ANÁLISIS DE AUGMENTATION")
        print("=" * 80)
        
        aug_stats = []
        
        for split in self.splits:
            if split not in self.data or self.data[split]['images'] is None:
                continue
            
            df = self.data[split]['images']
            
            if 'is_augmented' in df.columns:
                total = len(df)
                augmented = df['is_augmented'].sum()
                original = total - augmented
                
                aug_stats.append({
                    'split': split,
                    'total': total,
                    'augmented': augmented,
                    'original': original,
                    'augmented_pct': (augmented / total * 100),
                    'original_pct': (original / total * 100)
                })
        
        df_aug = pd.DataFrame(aug_stats)
        
        # Guardar CSV
        output_path = self.output_dir / 'augmentation_stats.csv'
        df_aug.to_csv(output_path, index=False)
        print(f"\n✓ Guardado: {output_path}")
        
        print("\nEstadísticas de augmentation:")
        print(df_aug.to_string(index=False))
        
        # Gráfica
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(df_aug))
        width = 0.35
        
        ax.bar(x - width/2, df_aug['original'], width, label='Original', alpha=0.8)
        ax.bar(x + width/2, df_aug['augmented'], width, label='Augmented', alpha=0.8)
        
        ax.set_xlabel('Split', fontsize=12, fontweight='bold')
        ax.set_ylabel('Número de Imágenes', fontsize=12, fontweight='bold')
        ax.set_title('Imágenes Originales vs Augmentadas', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df_aug['split'].str.capitalize())
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Añadir porcentajes
        for i, row in df_aug.iterrows():
            ax.text(i - width/2, row['original'], f"{row['original_pct']:.1f}%", 
                   ha='center', va='bottom', fontsize=9)
            ax.text(i + width/2, row['augmented'], f"{row['augmented_pct']:.1f}%",
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        fig_path = self.output_dir / 'augmentation_distribution.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"✓ Gráfica guardada: {fig_path}")
        
        return df_aug
    
    def generate_summary_report(self):
        """Genera un reporte resumen en texto."""
        print("\n" + "=" * 80)
        print("GENERANDO REPORTE RESUMEN")
        print("=" * 80)
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("REPORTE DE ANÁLISIS DE BALANCE DEL DATASET")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Resumen general
        report_lines.append("## RESUMEN GENERAL")
        report_lines.append("")
        
        total_images = 0
        total_annotations = 0
        
        for split in self.splits:
            if split in self.data:
                if self.data[split]['images'] is not None:
                    n_images = len(self.data[split]['images'])
                    total_images += n_images
                    report_lines.append(f"- {split.capitalize()}: {n_images} imágenes")
                
                if self.data[split]['annotations'] is not None:
                    n_anns = len(self.data[split]['annotations'])
                    total_annotations += n_anns
                    report_lines.append(f"  - {n_anns} anotaciones")
        
        report_lines.append("")
        report_lines.append(f"**Total**: {total_images} imágenes, {total_annotations} anotaciones")
        report_lines.append("")
        
        # Problemas críticos detectados
        report_lines.append("## PROBLEMAS CRÍTICOS DETECTADOS")
        report_lines.append("")
        
        # Leer estadísticas de bbox
        bbox_stats_path = self.output_dir / 'bbox_stats.csv'
        if bbox_stats_path.exists():
            df_bbox = pd.read_csv(bbox_stats_path)
            
            for _, row in df_bbox.iterrows():
                report_lines.append(f"### {row['split'].upper()}")
                
                if row['small_width_pct'] > 10 or row['small_height_pct'] > 10:
                    report_lines.append(f"🚨 CRÍTICO: Muchos bboxes pequeños")
                    report_lines.append(f"   - {row['small_width_pct']:.1f}% tienen width < 32px")
                    report_lines.append(f"   - {row['small_height_pct']:.1f}% tienen height < 32px")
                    report_lines.append(f"   ⚠️  Faster R-CNN tiene RPN con anchors mínimos de 32px")
                    report_lines.append("")
                
                if row['extreme_aspect_ratio_pct'] > 20:
                    report_lines.append(f"⚠️  ATENCIÓN: {row['extreme_aspect_ratio_pct']:.1f}% de bboxes con aspect ratio extremo")
                    report_lines.append(f"   Faster R-CNN usa ratios: 0.5, 1.0, 2.0")
                    report_lines.append("")
                
                # Estadísticas medianas
                report_lines.append(f"Estadísticas medianas:")
                report_lines.append(f"   - Width: {row['width_median']:.1f} px")
                report_lines.append(f"   - Height: {row['height_median']:.1f} px")
                report_lines.append(f"   - Área: {row['area_median']:.1f} px²")
                report_lines.append(f"   - Aspect Ratio: {row['aspect_ratio_median']:.2f}")
                report_lines.append("")
        
        # Recomendaciones
        report_lines.append("## RECOMENDACIONES")
        report_lines.append("")
        
        # Leer estadísticas para hacer recomendaciones
        if bbox_stats_path.exists():
            df_bbox = pd.read_csv(bbox_stats_path)
            
            # Verificar si hay problema de tamaños pequeños
            has_small_boxes = any(df_bbox['small_width_pct'] > 10)
            has_extreme_ar = any(df_bbox['extreme_aspect_ratio_pct'] > 20)
            
            if has_small_boxes:
                report_lines.append("### 1. Modificar Anchors de Faster R-CNN")
                report_lines.append("```python")
                report_lines.append("anchor_generator = AnchorGenerator(")
                report_lines.append("    sizes=((16, 32, 64, 128, 256),),  # Añadir 16px")
                report_lines.append("    aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),)  # Más ratios")
                report_lines.append(")")
                report_lines.append("```")
                report_lines.append("")
            
            if has_extreme_ar:
                report_lines.append("### 2. Considerar Multi-Scale Training")
                report_lines.append("- Augmentation con diferentes escalas")
                report_lines.append("- Feature Pyramid Networks (FPN)")
                report_lines.append("")
            
            report_lines.append("### 3. Considerar Vision Transformers")
            report_lines.append("- DINOv2 maneja mejor objetos pequeños y aspect ratios variados")
            report_lines.append("- Attention global captura mejor defectos pequeños")
            report_lines.append("")
        
        # Guardar reporte
        report_text = "\n".join(report_lines)
        output_path = self.output_dir / 'BALANCE_REPORT.txt'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n✓ Reporte guardado: {output_path}")
        print("\n" + report_text)
    
    def run_full_analysis(self):
        """Ejecuta el análisis completo."""
        print("\n")
        print("=" * 80)
        print("ANÁLISIS EXHAUSTIVO DE BALANCE DEL DATASET")
        print("=" * 80)
        print(f"Dataset: {self.dataset_root}")
        print(f"Output: {self.output_dir}")
        print("=" * 80)
        
        # Cargar datos
        self.load_data()
        
        # Ejecutar análisis
        self.analyze_category_distribution()
        self.analyze_image_sizes()
        self.analyze_bbox_sizes()
        self.analyze_source_dataset()
        self.analyze_augmentation()
        self.generate_summary_report()
        
        print("\n" + "=" * 80)
        print("✓ ANÁLISIS COMPLETADO")
        print("=" * 80)
        print(f"\nTodos los resultados guardados en: {self.output_dir}")
        print("\nArchivos generados:")
        for file in sorted(self.output_dir.glob('*')):
            print(f"  - {file.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Análisis exhaustivo de balance del dataset'
    )
    
    parser.add_argument(
        '--dataset-root',
        type=str,
        required=True,
        help='Ruta al directorio raíz del dataset (contiene train/val/test)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='dataset_balance_analysis',
        help='Directorio donde guardar los resultados'
    )
    
    args = parser.parse_args()
    
    # Ejecutar análisis
    analyzer = DatasetBalanceAnalyzer(args.dataset_root, args.output_dir)
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()