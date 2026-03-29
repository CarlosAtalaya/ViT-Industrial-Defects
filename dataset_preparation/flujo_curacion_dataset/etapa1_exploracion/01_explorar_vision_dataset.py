# Dataset VISION - Análisis Exploratorio
import argparse
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
from collections import defaultdict, Counter
import tarfile

class VisionDatasetExplorer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.components = []
        self.stats = defaultdict(dict)
        
    def extract_all_components(self):
        """Extrae todos los .tar.gz si no están extraídos"""
        for item in os.listdir(self.dataset_path):
            if item.endswith('.tar.gz'):
                component_name = item.replace('.tar.gz', '')
                component_path = os.path.join(self.dataset_path, component_name)
                
                if not os.path.exists(component_path):
                    print(f"Extrayendo {item}...")
                    with tarfile.open(os.path.join(self.dataset_path, item), 'r:gz') as tar:
                        tar.extractall(self.dataset_path)
                
                self.components.append(component_name)
    
    def analyze_component_structure(self, component_name):
        """Analiza la estructura de un componente específico"""
        component_path = os.path.join(self.dataset_path, component_name)
        
        if not os.path.exists(component_path):
            print(f"Componente {component_name} no encontrado")
            return
            
        splits = ['train', 'val', 'inference']
        component_stats = {}
        
        for split in splits:
            split_path = os.path.join(component_path, split)
            if os.path.exists(split_path):
                # Contar imágenes
                images = [f for f in os.listdir(split_path) if f.endswith('.jpg')]
                
                # Buscar archivo de anotaciones COCO
                coco_files = [f for f in os.listdir(split_path) if f.endswith('.json')]
                
                component_stats[split] = {
                    'num_images': len(images),
                    'coco_files': coco_files,
                    'sample_images': images[:5]  # Primeras 5 para inspección
                }
                
                # Analizar anotaciones COCO si existen
                if coco_files:
                    coco_path = os.path.join(split_path, coco_files[0])
                    try:
                        with open(coco_path, 'r') as f:
                            coco_data = json.load(f)
                        
                        component_stats[split]['categories'] = coco_data.get('categories', [])
                        component_stats[split]['num_annotations'] = len(coco_data.get('annotations', []))
                        
                    except Exception as e:
                        print(f"Error leyendo COCO {coco_path}: {e}")
        
        self.stats[component_name] = component_stats
        return component_stats
    
    def get_image_stats(self, component_name, split='train', max_images=50):
        """Obtiene estadísticas de las imágenes"""
        split_path = os.path.join(self.dataset_path, component_name, split)
        if not os.path.exists(split_path):
            return None
            
        images = [f for f in os.listdir(split_path) if f.endswith('.jpg')][:max_images]
        
        heights, widths, channels = [], [], []
        
        for img_name in images:
            img_path = os.path.join(split_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                h, w, c = img.shape
                heights.append(h)
                widths.append(w)
                channels.append(c)
        
        return {
            'heights': {'mean': np.mean(heights), 'std': np.std(heights), 'min': min(heights), 'max': max(heights)},
            'widths': {'mean': np.mean(widths), 'std': np.std(widths), 'min': min(widths), 'max': max(widths)},
            'channels': Counter(channels)
        }
    
    def analyze_defect_distribution(self, component_name, split='train'):
        """Analiza la distribución de tipos de defectos"""
        split_path = os.path.join(self.dataset_path, component_name, split)
        coco_files = [f for f in os.listdir(split_path) if f.endswith('.json')]
        
        if not coco_files:
            return None
            
        coco_path = os.path.join(split_path, coco_files[0])
        
        try:
            with open(coco_path, 'r') as f:
                coco_data = json.load(f)
            
            # Mapear categorías
            categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
            
            # Contar anotaciones por categoría
            category_counts = Counter()
            for ann in coco_data.get('annotations', []):
                cat_id = ann['category_id']
                cat_name = categories.get(cat_id, f'category_{cat_id}')
                category_counts[cat_name] += 1
            
            return {
                'categories': categories,
                'distribution': dict(category_counts),
                'total_annotations': len(coco_data.get('annotations', []))
            }
            
        except Exception as e:
            print(f"Error analizando defectos en {component_name}: {e}")
            return None
    
    def create_summary_report(self):
        """Genera un reporte resumen del dataset"""
        report = {
            'total_components': len(self.components),
            'components': list(self.components),
            'component_details': {}
        }
        
        total_train_images = 0
        total_val_images = 0
        all_categories = set()
        
        for component in self.components:
            print(f"\nAnalizando {component}...")
            
            # Estructura básica
            comp_stats = self.analyze_component_structure(component)
            
            # Estadísticas de imágenes
            img_stats = self.get_image_stats(component, 'train')
            
            # Distribución de defectos
            defect_dist = self.analyze_defect_distribution(component, 'train')
            
            report['component_details'][component] = {
                'structure': comp_stats,
                'image_stats': img_stats,
                'defect_distribution': defect_dist
            }
            
            # Acumular estadísticas globales
            if 'train' in comp_stats:
                total_train_images += comp_stats['train']['num_images']
            if 'val' in comp_stats:
                total_val_images += comp_stats['val']['num_images']
            
            if defect_dist and 'categories' in defect_dist:
                all_categories.update(defect_dist['categories'].values())
        
        report['global_stats'] = {
            'total_train_images': total_train_images,
            'total_val_images': total_val_images,
            'unique_defect_categories': len(all_categories),
            'all_categories': list(all_categories)
        }
        
        return report
    
    def visualize_component_samples(self, component_name, split='train', n_samples=6):
        """Visualiza muestras de un componente"""
        split_path = os.path.join(self.dataset_path, component_name, split)
        images = [f for f in os.listdir(split_path) if f.endswith('.jpg')][:n_samples]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, img_name in enumerate(images):
            if i >= len(axes):
                break
                
            img_path = os.path.join(split_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i].imshow(img_rgb)
                axes[i].set_title(f"{component_name}: {img_name}")
                axes[i].axis('off')
        
        # Ocultar axes vacíos
        for i in range(len(images), len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.savefig(f'samples_{component_name}_{split}.png', dpi=300, bbox_inches='tight')
        plt.show()

# Uso del explorador
if __name__ == "__main__":
    _repo_root = Path(__file__).resolve().parents[3]
    _parser = argparse.ArgumentParser(
        description="Exploración del dataset VISION-Datasets (Hugging Face / tar.gz por componente)."
    )
    _parser.add_argument(
        "--vision-path",
        type=str,
        default=os.environ.get("VISION_DATASETS_PATH", str(_repo_root / "VISION-Datasets")),
        help="Directorio raíz con los .tar.gz o carpetas extraídas (véase DESCARGA_DATASETS_ORIGEN.md).",
    )
    _args = _parser.parse_args()

    explorer = VisionDatasetExplorer(_args.vision_path)
    
    # Extraer todos los componentes
    explorer.extract_all_components()
    
    # Generar reporte completo
    print("Generando reporte del dataset VISION...")
    report = explorer.create_summary_report()
    
    # Guardar reporte
    with open('vision_dataset_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Mostrar estadísticas principales
    print(f"\n=== RESUMEN DATASET VISION ===")
    print(f"Total componentes: {report['total_components']}")
    print(f"Componentes: {', '.join(report['components'])}")
    print(f"Total imágenes train: {report['global_stats']['total_train_images']}")
    print(f"Total imágenes val: {report['global_stats']['total_val_images']}")
    print(f"Categorías de defectos únicas: {report['global_stats']['unique_defect_categories']}")
    
    # Visualizar algunas muestras
    sample_components = ['PCB_1', 'Capacitor', 'Electronics']  # Ajustar según disponibilidad
    for comp in sample_components:
        if comp in explorer.components:
            print(f"\nVisualizando muestras de {comp}...")
            explorer.visualize_component_samples(comp, 'train')
            break