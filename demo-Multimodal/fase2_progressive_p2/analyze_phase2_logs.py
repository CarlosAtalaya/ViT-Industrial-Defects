import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def parse_log_file(log_path):
    """
    Lee el fichero de log y extrae las métricas por época usando Regex.
    """
    if not os.path.exists(log_path):
        print(f"❌ Error: No se encuentra el archivo {log_path}")
        return None

    # Estructuras para almacenar datos
    epochs_data = []
    
    # Patrones Regex basados en tu script de logging
    # Patrón: 🌀 EPOCH 1/100
    epoch_pattern = re.compile(r"🌀 EPOCH (\d+)/(\d+)")
    
    # Patrón: 📉 Training Loss: 42.0251
    loss_pattern = re.compile(r"📉 Training Loss: ([\d\.]+)")
    
    # Patrón: 📊 mAP: 0.6719 | Recall (Críticas): 0.6410
    metrics_pattern = re.compile(r"📊 mAP: ([\d\.]+) \| Recall \(Críticas\): ([\d\.]+)")
    
    # Patrón: 🏆 ¡MEJORA DETECTADA! (Score: 0.6626)
    score_pattern = re.compile(r"Score: ([\d\.]+)\)")

    current_epoch = -1
    current_data = {}

    print(f"📖 Leyendo {log_path}...")
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # 1. Detectar Nueva Época
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                # Si ya teníamos datos de una época anterior, guardarlos
                if current_epoch != -1 and 'loss' in current_data:
                    epochs_data.append(current_data)
                
                # Iniciar nueva época
                current_epoch = int(epoch_match.group(1))
                current_data = {'epoch': current_epoch}
                continue

            # 2. Extraer Loss
            loss_match = loss_pattern.search(line)
            if loss_match:
                current_data['loss'] = float(loss_match.group(1))
                continue

            # 3. Extraer Métricas (mAP y Recall)
            metrics_match = metrics_pattern.search(line)
            if metrics_match:
                current_data['map'] = float(metrics_match.group(1))
                current_data['recall'] = float(metrics_match.group(2))
                
                # Calcular el Score compuesto manualmente por si no sale en el log
                # Score = 0.7 * mAP + 0.3 * Recall
                score_calc = (current_data['map'] * 0.7) + (current_data['recall'] * 0.3)
                current_data['score_calc'] = score_calc
                continue

            # 4. Extraer Score guardado (opcional, para verificar)
            score_match = score_pattern.search(line)
            if score_match:
                current_data['score_saved'] = float(score_match.group(1))

        # Añadir la última época si quedó pendiente
        if current_epoch != -1 and 'loss' in current_data:
            epochs_data.append(current_data)

    df = pd.DataFrame(epochs_data)
    return df

def plot_metrics(df, output_dir):
    """
    Genera gráficas de estilo académico.
    """
    # Estilo general
    plt.style.use('ggplot')
    
    # 1. Gráfica de LOSS (Entrenamiento)
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['loss'], label='Training Loss', color='#E24A33', linewidth=2, marker='o', markersize=4)
    plt.title('Evolución de la Función de Pérdida (Fase 2 Multimodal)', fontsize=14)
    plt.xlabel('Épocas', fontsize=12)
    plt.ylabel('Focal Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'fase2_training_loss.png'), dpi=300)
    print("   📈 Gráfica guardada: fase2_training_loss.png")
    
    # 2. Gráfica Comparativa: mAP vs Recall Crítico
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['map'], label='Global mAP (COCO)', color='#348ABD', linewidth=2)
    plt.plot(df['epoch'], df['recall'], label='Recall (Clases Críticas)', color='#988ED5', linewidth=2, linestyle='--')
    
    # Línea de Baseline Fase 1 (Referencia)
    plt.axhline(y=0.785, color='gray', linestyle=':', label='Baseline Fase 1 (0.785)')
    
    plt.title('Dinámica de Métricas: Precisión Global vs Sensibilidad Crítica', fontsize=14)
    plt.xlabel('Épocas', fontsize=12)
    plt.ylabel('Score (0-1)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'fase2_metrics_comparison.png'), dpi=300)
    print("   📈 Gráfica guardada: fase2_metrics_comparison.png")

    # 3. Gráfica de Score Híbrido (Criterio de Parada)
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['score_calc'], label='Score Híbrido (0.7·mAP + 0.3·Recall)', color='#8EBA42', linewidth=2, marker='s', markersize=4)
    plt.title('Evolución del Score Híbrido de Decisión', fontsize=14)
    plt.xlabel('Épocas', fontsize=12)
    plt.ylabel('Score Compuesto', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'fase2_composite_score.png'), dpi=300)
    print("   📈 Gráfica guardada: fase2_composite_score.png")
    
    plt.close('all')

def main():
    parser = argparse.ArgumentParser(description="Analizador de Logs Fase 2")
    parser.add_argument('--log', type=str, default="outputs/fase2_progressive_auto/experiment_log.txt", help="Ruta al fichero de log")
    parser.add_argument('--out', type=str, default="./analysis_output", help="Carpeta de salida")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1. Parsear
    df = parse_log_file(args.log)
    
    if df is not None and not df.empty:
        print(f"✅ Se han extraído datos de {len(df)} épocas.")
        
        # 2. Guardar CSV
        csv_path = os.path.join(args.out, 'fase2_metrics.csv')
        df.to_csv(csv_path, index=False)
        print(f"💾 Datos exportados a: {csv_path}")
        
        # 3. Generar Gráficas
        plot_metrics(df, args.out)
        
        # 4. Resumen Estadístico en consola
        best_epoch = df.loc[df['score_calc'].idxmax()]
        print("\n" + "="*40)
        print("📊 RESUMEN EJECUTIVO FASE 2")
        print("="*40)
        print(f"• Total Épocas: {len(df)}")
        print(f"• Loss Inicial: {df.iloc[0]['loss']:.4f} -> Final: {df.iloc[-1]['loss']:.4f}")
        print(f"• Mejor Score Híbrido: {best_epoch['score_calc']:.4f} (Epoch {int(best_epoch['epoch'])})")
        print(f"  - mAP asociado: {best_epoch['map']:.4f}")
        print(f"  - Recall asociado: {best_epoch['recall']:.4f}")
        print("="*40)
    else:
        print("⚠️ No se encontraron datos válidos en el log.")

if __name__ == "__main__":
    main()