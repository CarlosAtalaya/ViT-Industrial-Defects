import pandas as pd

# Cargar CSVs
bbox = pd.read_csv("bbox_stats.csv")
cats = pd.read_csv("category_distribution.csv")
sizes = pd.read_csv("image_sizes_stats.csv")
aug = pd.read_csv("augmentation_stats.csv")

# Preparar datos
total_images = sizes["n_images"].sum()
category_totals = cats.set_index(cats.columns[0])["total"]

# Crear resumen
contenido = []

contenido.append("RESUMEN EJECUTIVO – ANÁLISIS DEL DATASET PARA ENTRENAMIENTO ViT + DINOv3\n")
contenido.append(f"Total de imágenes: {total_images}\n")

contenido.append("\n=== DISTRIBUCIÓN DE CATEGORÍAS ===\n")
for cat in category_totals.index:
    if cat != "":
        contenido.append(f"{cat}: {category_totals[cat]}\n")

contenido.append("\n=== ESTADÍSTICAS DE BOUNDING BOXES ===\n")
for _, row in bbox.iterrows():
    contenido.append(
        f"{row['split']} -> "
        f"bboxes={row['n_bboxes']}, "
        f"width_mean={row['width_mean']:.1f}, "
        f"height_mean={row['height_mean']:.1f}, "
        f"area_mean={row['area_mean']:.1f}, "
        f"AR_mean={row['aspect_ratio_mean']:.2f}, "
        f"small_pct={row['small_width_pct']:.2f}%\n"
    )

contenido.append("\n=== TAMAÑOS DE IMÁGENES ===\n")
for _, row in sizes.iterrows():
    contenido.append(
        f"{row['split']} -> "
        f"n={row['n_images']}, "
        f"width_mean={row['width_mean']:.1f}, "
        f"height_mean={row['height_mean']:.1f}, "
        f"area_mean={row['area_mean']:.1f}\n"
    )

contenido.append("\n=== AUGMENTATION ===\n")
for _, row in aug.iterrows():
    contenido.append(
        f"{row['split']} -> augmented={row['augmented_pct']:.2f}%, "
        f"original={row['original_pct']:.2f}%\n"
    )

contenido.append("\n=== OBSERVACIONES ===\n")
contenido.append("Dataset balanceado, variabilidad alta en tamaños, presencia de objetos pequeños, AR extremos y buen nivel de augmentación.\n")

# Guardar archivo
with open("resumen_ejecutivo.txt", "w", encoding="utf-8") as f:
    f.writelines(contenido)

print("Archivo resumen_ejecutivo.txt generado correctamente.")
