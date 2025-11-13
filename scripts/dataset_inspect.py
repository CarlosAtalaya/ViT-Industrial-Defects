#!/usr/bin/env python3
"""
dataset_inspect.py
Inspección y análisis del dataset COCO-style antes de entrenar.

Uso:
    python dataset_inspect.py --coco-json path/to/train.json --images-dir path/to/images --out-dir ./inspect_output

Genera:
 - resumen por split y por categoría (CSV + texto)
 - verificación existencia de archivos
 - histograma de tamaños de imagen
 - estadísticas de bboxes (área, aspect ratio)
 - análisis de segmentaciones (presencia, area estimada)
 - archivos CSV con detalles por imagen y por anotación
"""
import os
import json
import argparse
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# No fijamos estilos ni colores (a petición del entorno)
plt.rcParams["figure.figsize"] = (8,5)

def load_coco(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def images_dict(coco):
    return {img['id']: img for img in coco.get('images', [])}

def categories_dict(coco):
    # usa id y nombre del objeto categories en COCO (ahora tienes "id" y "name")
    cat_map = {}
    for c in coco.get('categories', []):
        cat_map[c['id']] = c
    return cat_map

def annotations_per_image(coco):
    ann_map = defaultdict(list)
    for ann in coco.get('annotations', []):
        # COCO usa 'image_id', tu JSON mezcla 'id' vs 'image_id' en las muestras; asumimos COCO estándar en 'annotations'
        image_id = ann.get('image_id') or ann.get('id')  # fallback
        ann_map[image_id].append(ann)
    return ann_map

def verify_images_exist(images, images_dir):
    missing = []
    sizes = []
    for img in images.values():
        path = os.path.join(images_dir, img['file_name'])
        exist = os.path.exists(path)
        if not exist:
            missing.append(img['file_name'])
        sizes.append((img['width'], img['height']))
    return missing, sizes

def bbox_stats(anns):
    areas = []
    aspect_ratios = []
    widths = []
    heights = []
    for a in anns:
        if 'bbox' in a and a['bbox']:
            # bbox [x,y,w,h] as floats
            x,y,w,h = a['bbox']
            widths.append(w); heights.append(h)
            aspect_ratios.append(w / (h+1e-9))
            # some COCOs include 'area'
            areas.append(a.get('area', w*h))
    return np.array(widths), np.array(heights), np.array(aspect_ratios), np.array(areas)

def segmentation_presence(anns):
    with_seg = 0
    total = 0
    seg_areas = []
    for a in anns:
        total += 1
        if a.get('has_segmentation') or a.get('segmentation'):
            with_seg += 1
            if 'area' in a:
                seg_areas.append(a['area'])
    return with_seg, total, seg_areas

def main(args):
    coco = load_coco(args.coco_json)
    imgs = images_dict(coco)
    cats = categories_dict(coco)
    anns = coco.get('annotations', [])
    ann_map = annotations_per_image(coco)

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Verificar existencia de archivos
    missing, sizes = verify_images_exist(imgs, args.images_dir)
    print(f"Total images in JSON: {len(imgs)}")
    print(f"Missing image files: {len(missing)}")
    if len(missing) > 0:
        with open(os.path.join(args.out_dir, 'missing_images.txt'), 'w', encoding='utf-8') as f:
            for m in missing:
                f.write(m + "\n")
    # Image size distribution
    widths = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]
    df_images = pd.DataFrame([
        {'image_id': iid, 'file_name': img['file_name'], 'width': img['width'], 'height': img['height'],
         'source_dataset': img.get('source_dataset', None),
         'original_category': img.get('original_category', None),
         'defect_type': img.get('defect_type', None),
         'is_augmented': img.get('is_augmented', False)}
        for iid, img in imgs.items()
    ])
    df_images.to_csv(os.path.join(args.out_dir, 'images_table.csv'), index=False)

    # 2) Estadísticas por categoría (labels)
    # Map annotations to category names: in tu JSON categories tienen 'id' y 'name'
    category_counts = Counter()
    catid_to_unified = {}
    for c in coco.get('categories', []):
        catid_to_unified[c['id']] = c['name']

    for a in anns:
        cid = a.get('category_id')
        category_counts[catid_to_unified.get(cid, str(cid))] += 1

    df_cat = pd.DataFrame([
        {'category_id': k, 'n_annotations': v} if isinstance(k, int) else {'category_name': k, 'n_annotations': v}
        for k, v in category_counts.items()
    ])
    # But more useful: counts of images per unified_category_name (from annotations 'unified_category_name' if present)
    image_category_counter = Counter()
    for img_id, img_anns in ann_map.items():
        # Each image may have multiple anns; get unique unified_category per image
        unified_set = set()
        for a in img_anns:
            unified = a.get('unified_category_name') or catid_to_unified.get(a.get('category_id'))
            if unified:
                unified_set.add(unified)
        if len(unified_set)==0:
            unified_set.add('UNKNOWN')
        for u in unified_set:
            image_category_counter[u] += 1

    df_image_cat = pd.DataFrame([
        {'unified_category_name': k, 'n_images': v} for k, v in image_category_counter.items()
    ]).sort_values('n_images', ascending=False)
    df_image_cat.to_csv(os.path.join(args.out_dir, 'images_per_unified_category.csv'), index=False)
    print("\nImages per (unified) category (top):")
    print(df_image_cat.head(20).to_string(index=False))

    # 3) Stats de bboxes
    widths_arr, heights_arr, ar_arr, areas_arr = bbox_stats(anns)
    if areas_arr.size>0:
        print("\nBBox stats (count={}):".format(len(areas_arr)))
        print(" - area: mean {:.1f}, median {:.1f}, min {:.1f}, max {:.1f}".format(
            areas_arr.mean(), np.median(areas_arr), areas_arr.min(), areas_arr.max()))
        print(" - width: mean {:.1f}, height mean {:.1f}".format(widths_arr.mean(), heights_arr.mean()))
        print(" - aspect ratio (w/h) mean {:.2f}".format(ar_arr.mean()))
        # save dataframe
        df_bbox = pd.DataFrame({
            'width': widths_arr, 'height': heights_arr, 'aspect_ratio': ar_arr, 'area': areas_arr
        })
        df_bbox.to_csv(os.path.join(args.out_dir, 'bboxes_stats.csv'), index=False)
        # plot histograms
        plt.hist(areas_arr, bins=60)
        plt.title("Distribución área bboxes")
        plt.xlabel("area (px^2)")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'hist_bbox_area.png'))
        plt.clf()

        plt.hist(ar_arr, bins=50)
        plt.title("Aspect ratio (w/h) de bboxes")
        plt.xlabel("w/h")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'hist_bbox_aspect_ratio.png'))
        plt.clf()
    else:
        print("No hay bboxes con área/ bbox info detectada en annotations.")

    # 4) Segmentations
    with_seg, total_ann, seg_areas = segmentation_presence(anns)
    print(f"\nSegmentations: {with_seg} / {total_ann} annotations tienen 'segmentation' o 'has_segmentation' flag.")
    if len(seg_areas)>0:
        seg_areas = np.array(seg_areas)
        print(" - segmentation areas: mean {:.1f}, median {:.1f}".format(seg_areas.mean(), np.median(seg_areas)))

    # 5) Split inspect — si JSON contiene 'info.split' o si tienes ficheros train_files.txt
    split_info = coco.get('info', {}).get('split', None)
    print("\nInfo.split en JSON:", split_info)
    # además calculamos distribución por 'is_augmented' y por source_dataset
    src_counter = Counter([img.get('source_dataset', 'UNKNOWN') for img in imgs.values()])
    print("\nSource datasets in images:")
    for k,v in src_counter.items():
        print(f"  {k}: {v}")

    # augmented
    aug_count = sum(1 for img in imgs.values() if img.get('is_augmented', False))
    print(f"\nAugmented images flagged: {aug_count}")

    # 6) Export annotations table (each ann row)
    df_anns = pd.DataFrame([
        {
            'annotation_id': a.get('id'),
            'image_id': a.get('image_id') or a.get('id'),
            'category_id': a.get('category_id'),
            'unified_category_name': a.get('unified_category_name'),
            'bbox_w': (a.get('bbox') or [None,None,None,None])[2] if a.get('bbox') else None,
            'bbox_h': (a.get('bbox') or [None,None,None,None])[3] if a.get('bbox') else None,
            'area': a.get('area'),
            'iscrowd': a.get('iscrowd', 0),
            'has_segmentation': bool(a.get('segmentation') or a.get('has_segmentation', False))
        }
        for a in anns
    ])
    df_anns.to_csv(os.path.join(args.out_dir, 'annotations_table.csv'), index=False)

    # 7) Basic plots for image sizes
    df_images['area_px'] = df_images['width'] * df_images['height']
    df_images['short_edge'] = df_images[['width','height']].min(axis=1)
    df_images.to_csv(os.path.join(args.out_dir, 'images_with_area.csv'), index=False)

    plt.hist(df_images['short_edge'], bins=40)
    plt.title("Distribución de short_edge de imágenes")
    plt.xlabel("short edge (px)")
    plt.ylabel("n imágenes")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'hist_short_edge.png'))
    plt.clf()

    # 8) Report final resumido
    report_lines = []
    report_lines.append(f"Total images in JSON: {len(imgs)}")
    report_lines.append(f"Total annotations: {len(anns)}")
    report_lines.append(f"Missing image files: {len(missing)}")
    report_lines.append(f"Augmented images flagged: {aug_count}")
    report_lines.append("Top unified categories by image count:")
    for k,v in image_category_counter.most_common():
        report_lines.append(f"  {k}: {v}")
    report_txt = "\n".join(report_lines)
    with open(os.path.join(args.out_dir, 'dataset_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_txt)
    print("\nReport saved to:", os.path.join(args.out_dir, 'dataset_report.txt'))
    print(report_txt)
    print("\nTodos los CSV/plots guardados en:", args.out_dir)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--coco-json', required=True, help='path to train.json')
    p.add_argument('--images-dir', required=True, help='path to images directory for that split')
    p.add_argument('--out-dir', default='./inspect_output', help='output directory')
    args = p.parse_args()
    main(args)
