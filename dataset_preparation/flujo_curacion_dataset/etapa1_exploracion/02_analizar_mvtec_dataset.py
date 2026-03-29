# Analizador específico para MVTec AD dataset
import argparse
import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

class MVTecAnalyzer:
    def __init__(self, mvtec_path="mvtec-ad"):
        self.mvtec_path = mvtec_path
        self.samples_data = None
        self.report = {}
        
    def load_samples_json(self):
        """Carga el archivo samples.json de MVTec"""
        samples_path = os.path.join(self.mvtec_path, "samples.json")
        
        if not os.path.exists(samples_path):
            print(f"ERROR: No se encuentra samples.json en {samples_path}")
            return False
            
        try:
            with open(samples_path, 'r') as f:
                data = json.load(f)
                
            # El formato puede ser {"samples": [...]} o directamente [...]
            if isinstance(data, dict) and 'samples' in data:
                self.samples_data = data['samples']
            elif isinstance(data, list):
                self.samples_data = data
            else:
                print(f"ERROR: Formato inesperado en samples.json")
                return False
                
            print(f"✅ Cargadas {len(self.samples_data)} muestras de MVTec AD")
            return True
            
        except Exception as e:
            print(f"ERROR cargando samples.json: {e}")
            return False
    
    def analyze_structure(self):
        """Analiza la estructura del dataset MVTec"""
        if not self.samples_data:
            return None
            
        categories = set()
        defects = set()
        splits = set()
        
        category_counts = Counter()
        defect_counts = Counter()
        split_counts = Counter()
        
        # Analizar cada muestra
        for sample in self.samples_data:
            # Extraer información
            category = sample.get('category', {}).get('label', 'unknown')
            defect = sample.get('defect', {}).get('label', 'unknown') 
            split = sample.get('split', 'unknown')
            
            categories.add(category)
            defects.add(defect)
            splits.add(split)
            
            category_counts[category] += 1
            defect_counts[defect] += 1
            split_counts[split] += 1
        
        structure_analysis = {
            'total_samples': len(self.samples_data),
            'unique_categories': len(categories),
            'unique_defects': len(defects),
            'splits': list(splits),
            'categories_list': sorted(list(categories)),
            'defects_list': sorted(list(defects)),
            'category_distribution': dict(category_counts),
            'defect_distribution': dict(defect_counts),
            'split_distribution': dict(split_counts)
        }
        
        return structure_analysis
    
    def analyze_category_details(self):
        """Analiza detalles por categoría"""
        if not self.samples_data:
            return None
            
        category_details = defaultdict(lambda: {
            'total_samples': 0,
            'train_samples': 0,
            'test_samples': 0,
            'defect_types': set(),
            'normal_samples': 0,
            'anomaly_samples': 0
        })
        
        for sample in self.samples_data:
            category = sample.get('category', {}).get('label', 'unknown')
            defect = sample.get('defect', {}).get('label', 'unknown')
            split = sample.get('split', 'unknown')
            
            details = category_details[category]
            details['total_samples'] += 1
            details['defect_types'].add(defect)
            
            # Contar por split
            if split == 'train':
                details['train_samples'] += 1
            elif split == 'test':
                details['test_samples'] += 1
            
            # Contar normal vs anomaly
            if defect == 'good':
                details['normal_samples'] += 1
            else:
                details['anomaly_samples'] += 1
        
        # Convertir sets a listas para JSON
        for category, details in category_details.items():
            details['defect_types'] = sorted(list(details['defect_types']))
            details['unique_defects'] = len(details['defect_types'])
        
        return dict(category_details)
    
    def analyze_defect_patterns(self):
        """Analiza patrones de defectos por categoría"""
        if not self.samples_data:
            return None
            
        defect_patterns = defaultdict(lambda: defaultdict(int))
        
        for sample in self.samples_data:
            category = sample.get('category', {}).get('label', 'unknown')
            defect = sample.get('defect', {}).get('label', 'unknown')
            
            defect_patterns[category][defect] += 1
        
        return dict(defect_patterns)
    
    def generate_mvtec_report(self):
        """Genera reporte completo de MVTec AD"""
        print("🔍 Analizando MVTec AD dataset...")
        
        # Cargar datos
        if not self.load_samples_json():
            return None
        
        # Análisis estructural
        structure = self.analyze_structure()
        category_details = self.analyze_category_details()
        defect_patterns = self.analyze_defect_patterns()
        
        # Crear reporte
        self.report = {
            'dataset_name': 'MVTec AD',
            'dataset_path': self.mvtec_path,
            'structure_analysis': structure,
            'category_details': category_details,
            'defect_patterns': defect_patterns,
            'characteristics': [
                'Benchmark estándar para detección de anomalías industriales',
                'Enfoque unsupervised: train solo normal, test normal+anomaly',
                'Máscaras pixel-level para localización precisa',
                'Amplia variedad de objetos y texturas industriales',
                '15 categorías diferentes con defectos reales'
            ]
        }
        
        return self.report
    
    def print_summary(self):
        """Imprime resumen del análisis"""
        if not self.report:
            print("No hay reporte generado")
            return
            
        structure = self.report['structure_analysis']
        
        print(f"\n{'='*60}")
        print(f"📊 RESUMEN MVTec AD DATASET")
        print(f"{'='*60}")
        
        print(f"\n🔢 ESTADÍSTICAS GENERALES:")
        print(f"   Total muestras: {structure['total_samples']:,}")
        print(f"   Categorías: {structure['unique_categories']}")
        print(f"   Tipos de defectos: {structure['unique_defects']}")
        print(f"   Splits: {', '.join(structure['splits'])}")
        
        print(f"\n📈 DISTRIBUCIÓN POR SPLIT:")
        for split, count in structure['split_distribution'].items():
            percentage = (count/structure['total_samples'])*100
            print(f"   {split}: {count:,} ({percentage:.1f}%)")
        
        print(f"\n🏭 CATEGORÍAS:")
        categories = structure['categories_list']
        print(f"   {', '.join(categories)}")
        
        print(f"\n🔍 TIPOS DE DEFECTOS:")
        defects = [d for d in structure['defects_list'] if d != 'good']
        print(f"   Normal: good")
        print(f"   Anomalías: {', '.join(defects[:10])}")
        if len(defects) > 10:
            print(f"   ... y {len(defects)-10} más")
        
        # Top 5 categorías por volumen
        top_categories = sorted(structure['category_distribution'].items(), 
                              key=lambda x: x[1], reverse=True)[:5]
        print(f"\n📊 TOP 5 CATEGORÍAS (por volumen):")
        for category, count in top_categories:
            print(f"   {category}: {count:,} muestras")
    
    def save_report(self, filename="mvtec_ad_report.json"):
        """Guarda el reporte en JSON"""
        if not self.report:
            print("No hay reporte para guardar")
            return False
            
        try:
            with open(filename, 'w') as f:
                json.dump(self.report, f, indent=2)
            print(f"📄 Reporte MVTec AD guardado en: {filename}")
            return True
        except Exception as e:
            print(f"ERROR guardando reporte: {e}")
            return False
    
    def create_visualizations(self):
        """Crea visualizaciones del dataset"""
        if not self.report:
            return
            
        structure = self.report['structure_analysis']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Distribución por split
        splits = list(structure['split_distribution'].keys())
        split_counts = list(structure['split_distribution'].values())
        
        axes[0,0].pie(split_counts, labels=splits, autopct='%1.1f%%', startangle=90)
        axes[0,0].set_title('Distribución por Split')
        
        # 2. Top 10 categorías
        top_cats = sorted(structure['category_distribution'].items(), 
                         key=lambda x: x[1], reverse=True)[:10]
        cat_names = [item[0] for item in top_cats]
        cat_counts = [item[1] for item in top_cats]
        
        axes[0,1].barh(cat_names, cat_counts, color='orange')
        axes[0,1].set_title('Top 10 Categorías')
        axes[0,1].set_xlabel('Número de Muestras')
        
        # 3. Distribución Normal vs Anomaly
        normal_count = structure['defect_distribution'].get('good', 0)
        anomaly_count = structure['total_samples'] - normal_count
        
        axes[1,0].bar(['Normal (good)', 'Anomalías'], [normal_count, anomaly_count], 
                     color=['green', 'red'], alpha=0.7)
        axes[1,0].set_title('Normal vs Anomalías')
        axes[1,0].set_ylabel('Número de Muestras')
        
        # 4. Top defectos (excluyendo 'good')
        defect_dist = {k: v for k, v in structure['defect_distribution'].items() if k != 'good'}
        top_defects = sorted(defect_dist.items(), key=lambda x: x[1], reverse=True)[:8]
        
        if top_defects:
            defect_names = [item[0][:10] for item in top_defects]  # Truncar nombres
            defect_counts = [item[1] for item in top_defects]
            
            axes[1,1].bar(defect_names, defect_counts, color='salmon')
            axes[1,1].set_title('Top 8 Tipos de Defectos')
            axes[1,1].set_ylabel('Número de Muestras')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('mvtec_analysis_charts.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Gráficos guardados en: mvtec_analysis_charts.png")

# Script principal
if __name__ == "__main__":
    _repo_root = Path(__file__).resolve().parents[3]
    _parser = argparse.ArgumentParser(
        description="Análisis exploratorio de MVTec AD (requiere samples.json en la raíz del path; véase DESCARGA_DATASETS_ORIGEN.md)."
    )
    _parser.add_argument(
        "--mvtec-path",
        type=str,
        default=os.environ.get("MVTEC_AD_PATH", str(_repo_root / "mvtec-ad")),
        help="Directorio con samples.json y datos MVTec preparados para este pipeline.",
    )
    _args = _parser.parse_args()
    MVTEC_PATH = _args.mvtec_path

    print(f"🚀 Iniciando análisis MVTec AD en: {MVTEC_PATH}")
    
    # Crear analizador
    analyzer = MVTecAnalyzer(MVTEC_PATH)
    
    # Generar reporte completo
    report = analyzer.generate_mvtec_report()
    
    if report:
        # Mostrar resumen
        analyzer.print_summary()
        
        # Guardar reporte
        analyzer.save_report("mvtec_ad_report.json")
        
        # Crear visualizaciones
        analyzer.create_visualizations()
        
        print(f"\n✅ Análisis MVTec AD completado!")
        print(f"📁 Archivos generados:")
        print(f"   - mvtec_ad_report.json")
        print(f"   - mvtec_analysis_charts.png")
        
    else:
        print(f"\n❌ Error en el análisis. Verifica:")
        print(f"   - Path del dataset: {MVTEC_PATH}")
        print(f"   - Existencia de samples.json")
        print(f"   - Formato del archivo JSON")