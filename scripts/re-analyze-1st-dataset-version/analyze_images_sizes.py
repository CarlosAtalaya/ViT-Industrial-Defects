#!/usr/bin/env python3
"""
Análisis de distribución de tamaños de imágenes para decidir resolución de entrenamiento
"""

import json
import os
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Configuración
DATASET_BASE = Path("../../curated_dataset_splitted_20251101_provisional_1st_version")
SPLITS = ['train', 'val', 'test']

def analyze_split(split_name):
    """Analiza tamaños de imágenes en un split del dataset."""
    
    ann_file = DATASET_BASE / split_name / f"{split_name}.json"
    img_folder = DATASET_BASE / split_name / "images"
    
    print(f"\n{'='*80}")
    print(f"Analizando: {split_name.upper()}")
    print(f"{'='*80}")
    
    # Cargar annotations
    with open(ann_file) as f:
        data = json.load(f)
    
    sizes = []
    widths = []
    heights = []
    aspect_ratios = []
    areas = []
    
    for img_info in data['images']:
        img_path = img_folder / img_info['file_name']
        
        if not img_path.exists():
            print(f"⚠️  Imagen no encontrada: {img_path}")
            continue
        
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                sizes.append(min(w, h))  # Lado más corto
                aspect_ratios.append(max(w, h) / min(w, h))
                areas.append(w * h)
        except Exception as e:
            print(f"⚠️  Error leyendo {img_path}: {e}")
    
    # Convertir a arrays numpy
    widths = np.array(widths)
    heights = np.array(heights)
    sizes = np.array(sizes)
    aspect_ratios = np.array(aspect_ratios)
    areas = np.array(areas)
    
    # Estadísticas
    print(f"\n📊 ESTADÍSTICAS DE TAMAÑOS ({len(sizes)} imágenes)")
    print(f"{'-'*80}")
    
    print(f"\n🔹 Ancho (width):")
    print(f"   Min:     {widths.min():.0f} px")
    print(f"   Q1:      {np.percentile(widths, 25):.0f} px")
    print(f"   Median:  {np.median(widths):.0f} px")
    print(f"   Mean:    {widths.mean():.0f} px")
    print(f"   Q3:      {np.percentile(widths, 75):.0f} px")
    print(f"   Max:     {widths.max():.0f} px")
    print(f"   Std:     {widths.std():.0f} px")
    
    print(f"\n🔹 Alto (height):")
    print(f"   Min:     {heights.min():.0f} px")
    print(f"   Q1:      {np.percentile(heights, 25):.0f} px")
    print(f"   Median:  {np.median(heights):.0f} px")
    print(f"   Mean:    {heights.mean():.0f} px")
    print(f"   Q3:      {np.percentile(heights, 75):.0f} px")
    print(f"   Max:     {heights.max():.0f} px")
    print(f"   Std:     {heights.std():.0f} px")
    
    print(f"\n🔹 Lado más corto (min side):")
    print(f"   Min:     {sizes.min():.0f} px")
    print(f"   Q1:      {np.percentile(sizes, 25):.0f} px")
    print(f"   Median:  {np.median(sizes):.0f} px")
    print(f"   Mean:    {sizes.mean():.0f} px")
    print(f"   Q3:      {np.percentile(sizes, 75):.0f} px")
    print(f"   Max:     {sizes.max():.0f} px")
    
    print(f"\n🔹 Aspect Ratio (max/min):")
    print(f"   Min:     {aspect_ratios.min():.2f}")
    print(f"   Median:  {np.median(aspect_ratios):.2f}")
    print(f"   Mean:    {aspect_ratios.mean():.2f}")
    print(f"   Max:     {aspect_ratios.max():.2f}")
    
    print(f"\n🔹 Área (píxeles):")
    print(f"   Min:     {areas.min():.0f} px² ({areas.min()/1e6:.2f} MP)")
    print(f"   Median:  {np.median(areas):.0f} px² ({np.median(areas)/1e6:.2f} MP)")
    print(f"   Mean:    {areas.mean():.0f} px² ({areas.mean()/1e6:.2f} MP)")
    print(f"   Max:     {areas.max():.0f} px² ({areas.max()/1e6:.2f} MP)")
    
    # Distribución por rangos
    print(f"\n📦 DISTRIBUCIÓN POR RANGOS (lado más corto)")
    print(f"{'-'*80}")
    
    ranges = [
        (0, 500, "Muy pequeñas (<500px)"),
        (500, 800, "Pequeñas (500-800px)"),
        (800, 1200, "Medianas (800-1200px)"),
        (1200, 1600, "Grandes (1200-1600px)"),
        (1600, 2000, "Muy grandes (1600-2000px)"),
        (2000, 10000, "Extra grandes (>2000px)")
    ]
    
    for min_val, max_val, label in ranges:
        count = np.sum((sizes >= min_val) & (sizes < max_val))
        pct = 100 * count / len(sizes)
        bar = '█' * int(pct / 2)
        print(f"   {label:30s} {count:4d} ({pct:5.1f}%) {bar}")
    
    # Identificar extremos
    print(f"\n🔍 IMÁGENES EXTREMAS")
    print(f"{'-'*80}")
    
    # 5 más pequeñas
    smallest_idx = np.argsort(sizes)[:5]
    print(f"\n🔻 5 más pequeñas:")
    for idx in smallest_idx:
        img_info = data['images'][idx]
        print(f"   {img_info['file_name']:50s} {widths[idx]:.0f}×{heights[idx]:.0f} px")
    
    # 5 más grandes
    largest_idx = np.argsort(sizes)[-5:][::-1]
    print(f"\n🔺 5 más grandes:")
    for idx in largest_idx:
        img_info = data['images'][idx]
        print(f"   {img_info['file_name']:50s} {widths[idx]:.0f}×{heights[idx]:.0f} px")
    
    return {
        'split': split_name,
        'n_images': len(sizes),
        'widths': widths,
        'heights': heights,
        'sizes': sizes,
        'aspect_ratios': aspect_ratios,
        'areas': areas
    }


def recommend_resolution(all_stats):
    """Recomienda resolución de entrenamiento basada en análisis."""
    
    # Combinar todos los splits
    all_sizes = np.concatenate([s['sizes'] for s in all_stats])
    all_widths = np.concatenate([s['widths'] for s in all_stats])
    all_heights = np.concatenate([s['heights'] for s in all_stats])
    
    print(f"\n{'='*80}")
    print(f"💡 RECOMENDACIONES DE RESOLUCIÓN")
    print(f"{'='*80}")
    
    # Percentiles clave
    p10 = np.percentile(all_sizes, 10)
    p25 = np.percentile(all_sizes, 25)
    p50 = np.percentile(all_sizes, 50)
    p75 = np.percentile(all_sizes, 75)
    p90 = np.percentile(all_sizes, 90)
    
    print(f"\n📐 Percentiles del lado más corto:")
    print(f"   P10: {p10:.0f} px")
    print(f"   P25: {p25:.0f} px")
    print(f"   P50 (mediana): {p50:.0f} px")
    print(f"   P75: {p75:.0f} px")
    print(f"   P90: {p90:.0f} px")
    
    # Encontrar múltiplos de 14 más cercanos
    def nearest_multiple_14(val):
        return int(round(val / 14) * 14)
    
    # Opciones de resolución
    print(f"\n🎯 OPCIONES DE RESOLUCIÓN (múltiplos de 14 para ViT):")
    print(f"{'-'*80}")
    
    options = [
        (p25, "Conservadora", "Minimiza upscaling de imágenes pequeñas"),
        (p50, "Balanceada", "Compromiso entre pequeñas y grandes"),
        (p75, "Agresiva", "Preserva más detalles de imágenes grandes"),
        (1120, "Estándar ViT", "Tamaño común en papers (1120×1120)"),
        (1400, "Alta resolución", "Máxima calidad (requiere más memoria)"),
    ]
    
    print(f"\n{'Opción':<20} {'Resolución':<15} {'Justificación':<40} {'Impacto':<30}")
    print(f"{'-'*110}")
    
    for base_val, name, reason in options:
        res = nearest_multiple_14(base_val)
        
        # Calcular impacto
        upscale = np.sum(all_sizes < res) / len(all_sizes) * 100
        downscale = np.sum(all_sizes > res) / len(all_sizes) * 100
        
        print(f"{name:<20} {res}×{res:<10} {reason:<40} ↑{upscale:.0f}% ↓{downscale:.0f}%")
    
    # Recomendación final
    print(f"\n✅ RECOMENDACIÓN FINAL:")
    print(f"{'-'*80}")
    
    recommended = nearest_multiple_14(p50)
    
    print(f"""
Para tu dataset con:
- Rango: {all_sizes.min():.0f}px - {all_sizes.max():.0f}px
- Mediana: {p50:.0f}px

Recomiendo: {recommended}×{recommended} px

RAZONES:
1. ✓ Múltiplo de 14 (compatible con DINOv3 patch size)
2. ✓ Cerca de la mediana del dataset
3. ✓ Balance entre upscaling ({np.sum(all_sizes < recommended) / len(all_sizes) * 100:.0f}% imágenes) 
   y downscaling ({np.sum(all_sizes > recommended) / len(all_sizes) * 100:.0f}% imágenes)
4. ✓ Manejable en RTX 4070 12GB con batch_size=1

ALTERNATIVAS:
- Si OOM → {nearest_multiple_14(p25)}×{nearest_multiple_14(p25)} (más conservador)
- Si quieres máxima calidad → 1400×1400 (requiere más memoria)

CONFIG YAML:
  - {{type: Resize, size: [{recommended}, {recommended}]}}
  collate_fn:
    base_size: {recommended}
  eval_spatial_size: [{recommended}, {recommended}]
""")
    
    return recommended


def plot_distributions(all_stats, output_dir='analysis_plots'):
    """Genera gráficas de distribución."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Combinar todos los splits
    all_widths = np.concatenate([s['widths'] for s in all_stats])
    all_heights = np.concatenate([s['heights'] for s in all_stats])
    all_sizes = np.concatenate([s['sizes'] for s in all_stats])
    all_aspect_ratios = np.concatenate([s['aspect_ratios'] for s in all_stats])
    
    # Plot 1: Histograma de tamaños
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].hist(all_widths, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(np.median(all_widths), color='red', linestyle='--', label=f'Mediana: {np.median(all_widths):.0f}')
    axes[0, 0].set_xlabel('Ancho (píxeles)', fontsize=12)
    axes[0, 0].set_ylabel('Frecuencia', fontsize=12)
    axes[0, 0].set_title('Distribución de Anchos', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].hist(all_heights, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(np.median(all_heights), color='red', linestyle='--', label=f'Mediana: {np.median(all_heights):.0f}')
    axes[0, 1].set_xlabel('Alto (píxeles)', fontsize=12)
    axes[0, 1].set_ylabel('Frecuencia', fontsize=12)
    axes[0, 1].set_title('Distribución de Altos', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].hist(all_sizes, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].axvline(np.median(all_sizes), color='red', linestyle='--', label=f'Mediana: {np.median(all_sizes):.0f}')
    axes[1, 0].set_xlabel('Lado más corto (píxeles)', fontsize=12)
    axes[1, 0].set_ylabel('Frecuencia', fontsize=12)
    axes[1, 0].set_title('Distribución de Lado Más Corto', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].hist(all_aspect_ratios, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].axvline(np.median(all_aspect_ratios), color='red', linestyle='--', label=f'Mediana: {np.median(all_aspect_ratios):.2f}')
    axes[1, 1].set_xlabel('Aspect Ratio', fontsize=12)
    axes[1, 1].set_ylabel('Frecuencia', fontsize=12)
    axes[1, 1].set_title('Distribución de Aspect Ratio', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'image_size_distributions.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Gráfica guardada: {output_path}")
    plt.close()
    
    # Plot 2: Scatter ancho vs alto
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for stats in all_stats:
        ax.scatter(stats['widths'], stats['heights'], 
                  alpha=0.6, s=30, label=stats['split'].capitalize())
    
    ax.set_xlabel('Ancho (píxeles)', fontsize=12)
    ax.set_ylabel('Alto (píxeles)', fontsize=12)
    ax.set_title('Distribución Ancho vs Alto por Split', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Añadir líneas de referencia para resoluciones comunes
    for res in [640, 896, 1120, 1400]:
        ax.axhline(res, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.axvline(res, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.text(res + 50, ax.get_ylim()[1] * 0.95, f'{res}px', 
               fontsize=9, color='gray', alpha=0.7)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'width_vs_height_scatter.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Gráfica guardada: {output_path}")
    plt.close()


def main():
    """Ejecuta análisis completo."""
    
    print("="*80)
    print("🔬 ANÁLISIS DE DISTRIBUCIÓN DE TAMAÑOS - DATASET INDUSTRIAL")
    print("="*80)
    
    # Verificar que existe el dataset
    if not DATASET_BASE.exists():
        print(f"\n❌ Error: No se encuentra el dataset en {DATASET_BASE}")
        print("   Por favor verifica la ruta en el script.")
        return
    
    # Analizar cada split
    all_stats = []
    for split in SPLITS:
        stats = analyze_split(split)
        all_stats.append(stats)
    
    # Generar recomendación
    recommended_res = recommend_resolution(all_stats)
    
    # Generar gráficas
    print(f"\n{'='*80}")
    print("📊 GENERANDO GRÁFICAS...")
    print(f"{'='*80}")
    plot_distributions(all_stats)
    
    print(f"\n{'='*80}")
    print("✅ ANÁLISIS COMPLETADO")
    print(f"{'='*80}")
    print(f"\nResolución recomendada: {recommended_res}×{recommended_res} px")
    print("\nConsulta las gráficas en: ./analysis_plots/")


if __name__ == '__main__':
    main()