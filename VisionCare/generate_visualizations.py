# =====================================================================
# 📊 GENERATE ALL VISUALIZATIONS (No Training Required!)
# =====================================================================
# This script generates all charts using your HARDCODED training results.
# Just run it in Colab - no training, no checkpoints needed!
# =====================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Try to mount Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    OUTPUT_DIR = "/content/drive/MyDrive/symile-mimic/MultiLabel_Results"
except:
    OUTPUT_DIR = "./MultiLabel_Results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== YOUR HARDCODED RESULTS =====================

LABELS = ['Cardiomegaly', 'Edema', 'Atelectasis', 'Pleural Effusion', 'Lung Opacity', 'No Finding']

# Results per disease (from your training output)
RESULTS = {
    'Cardiomegaly':     {'auc': 0.7003, 'precision': 0.5861, 'recall': 0.4767, 'f1': 0.5257, 'specificity': 0.7756, 'accuracy': 0.6560},
    'Edema':            {'auc': 0.8334, 'precision': 0.5803, 'recall': 0.5600, 'f1': 0.5700, 'specificity': 0.8527, 'accuracy': 0.7747},
    'Atelectasis':      {'auc': 0.7511, 'precision': 0.5758, 'recall': 0.5672, 'f1': 0.5714, 'specificity': 0.7676, 'accuracy': 0.6960},
    'Pleural Effusion': {'auc': 0.8484, 'precision': 0.7366, 'recall': 0.6893, 'f1': 0.7122, 'specificity': 0.8532, 'accuracy': 0.7920},
    'Lung Opacity':     {'auc': 0.6824, 'precision': 0.4941, 'recall': 0.3621, 'f1': 0.4179, 'specificity': 0.8340, 'accuracy': 0.6880},
    'No Finding':       {'auc': 0.8057, 'precision': 0.4907, 'recall': 0.4690, 'f1': 0.4796, 'specificity': 0.9137, 'accuracy': 0.8467}
}

MACRO = {'auc': 0.7702, 'precision': 0.5773, 'recall': 0.5207, 'f1': 0.5461, 'specificity': 0.8328, 'accuracy': 0.7422}

# Individual modality performance
VISION_AUC = 0.7694       # ConvNeXt-Tiny
SIGNAL_AUC = 0.6023       # 1D-CNN
CLINICAL_AUC = 0.6118     # MLP
FUSION_AUC = 0.7702       # Multi-Modal Fusion

# Training history (7 epochs)
TRAIN_LOSS = [0.5656, 0.5159, 0.5062, 0.5006, 0.4971, 0.4934, 0.4911]
VAL_AUC = [0.7667, 0.7702, 0.7669, 0.7651, 0.7646, 0.7669, 0.7643]


# ===================== 1. ARCHITECTURE DIAGRAM =====================

def plot_architecture_diagram():
    """Create a detailed architecture diagram for teacher presentation."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    ax.set_xlim(0, 18)
    ax.set_ylim(-3, 14)
    ax.axis('off')
    ax.set_facecolor('#f8f9fa')
    
    # Title
    ax.text(9, 13.5, '🔀 Multi-Modal Fusion Architecture for Disease Detection', 
            fontsize=20, fontweight='bold', ha='center', color='#2c3e50')
    ax.text(9, 12.9, 'Intermediate Feature-Level Fusion | SYMILE-MIMIC Dataset', 
            fontsize=12, ha='center', color='#7f8c8d', style='italic')
    
    # ===== INPUT MODALITIES =====
    ax.add_patch(plt.Rectangle((0.5, 9), 3.5, 3, facecolor='#3498db', edgecolor='#2980b9', linewidth=3, alpha=0.9))
    ax.text(2.25, 11.5, '🩻 CHEST X-RAY', fontsize=12, fontweight='bold', ha='center', color='white')
    ax.text(2.25, 10.8, 'Input: 224×224×3', fontsize=10, ha='center', color='white')
    ax.text(2.25, 10.2, 'RGB Image', fontsize=9, ha='center', color='#bdc3c7')
    ax.text(2.25, 9.5, '(Radiograph)', fontsize=9, ha='center', color='#bdc3c7')
    
    ax.add_patch(plt.Rectangle((7.25, 9), 3.5, 3, facecolor='#e74c3c', edgecolor='#c0392b', linewidth=3, alpha=0.9))
    ax.text(9, 11.5, '❤️ 12-LEAD ECG', fontsize=12, fontweight='bold', ha='center', color='white')
    ax.text(9, 10.8, 'Input: 12×5000', fontsize=10, ha='center', color='white')
    ax.text(9, 10.2, 'Time Series', fontsize=9, ha='center', color='#bdc3c7')
    ax.text(9, 9.5, '(10 seconds)', fontsize=9, ha='center', color='#bdc3c7')
    
    ax.add_patch(plt.Rectangle((14, 9), 3.5, 3, facecolor='#9b59b6', edgecolor='#8e44ad', linewidth=3, alpha=0.9))
    ax.text(15.75, 11.5, '🧪 LAB VALUES', fontsize=12, fontweight='bold', ha='center', color='white')
    ax.text(15.75, 10.8, 'Input: 100D', fontsize=10, ha='center', color='white')
    ax.text(15.75, 10.2, '50 Values + 50 Flags', fontsize=9, ha='center', color='#bdc3c7')
    ax.text(15.75, 9.5, '(Blood Tests)', fontsize=9, ha='center', color='#bdc3c7')
    
    # Arrows to encoders
    ax.annotate('', xy=(2.25, 7.6), xytext=(2.25, 9), arrowprops=dict(arrowstyle='->', color='#3498db', lw=3))
    ax.annotate('', xy=(9, 7.6), xytext=(9, 9), arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=3))
    ax.annotate('', xy=(15.75, 7.6), xytext=(15.75, 9), arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=3))
    
    # ===== ENCODERS =====
    ax.add_patch(plt.Rectangle((0.5, 5.5), 3.5, 2.1, facecolor='#2980b9', edgecolor='#1a5276', linewidth=2))
    ax.text(2.25, 7.1, 'ConvNeXt-Tiny', fontsize=11, fontweight='bold', ha='center', color='white')
    ax.text(2.25, 6.5, '(Pre-trained)', fontsize=9, ha='center', color='#bdc3c7')
    ax.text(2.25, 5.9, '❄️ FROZEN', fontsize=8, ha='center', color='#f39c12', fontweight='bold')
    
    ax.add_patch(plt.Rectangle((7.25, 5.5), 3.5, 2.1, facecolor='#c0392b', edgecolor='#922b21', linewidth=2))
    ax.text(9, 7.1, '1D-CNN', fontsize=11, fontweight='bold', ha='center', color='white')
    ax.text(9, 6.5, '(Pre-trained)', fontsize=9, ha='center', color='#bdc3c7')
    ax.text(9, 5.9, '❄️ FROZEN', fontsize=8, ha='center', color='#f39c12', fontweight='bold')
    
    ax.add_patch(plt.Rectangle((14, 5.5), 3.5, 2.1, facecolor='#8e44ad', edgecolor='#6c3483', linewidth=2))
    ax.text(15.75, 7.1, 'MLP', fontsize=11, fontweight='bold', ha='center', color='white')
    ax.text(15.75, 6.5, '(Pre-trained)', fontsize=9, ha='center', color='#bdc3c7')
    ax.text(15.75, 5.9, '❄️ FROZEN', fontsize=8, ha='center', color='#f39c12', fontweight='bold')
    
    # Feature vectors
    ax.annotate('', xy=(2.25, 4.6), xytext=(2.25, 5.5), arrowprops=dict(arrowstyle='->', color='#2980b9', lw=2))
    ax.annotate('', xy=(9, 4.6), xytext=(9, 5.5), arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2))
    ax.annotate('', xy=(15.75, 4.6), xytext=(15.75, 5.5), arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=2))
    
    ax.add_patch(plt.Rectangle((1, 4.1), 2.5, 0.5, facecolor='#5dade2', edgecolor='#2980b9', linewidth=1))
    ax.text(2.25, 4.35, '768D', fontsize=10, fontweight='bold', ha='center', color='white')
    ax.add_patch(plt.Rectangle((7.75, 4.1), 2.5, 0.5, facecolor='#ec7063', edgecolor='#c0392b', linewidth=1))
    ax.text(9, 4.35, '256D', fontsize=10, fontweight='bold', ha='center', color='white')
    ax.add_patch(plt.Rectangle((14.5, 4.1), 2.5, 0.5, facecolor='#bb8fce', edgecolor='#8e44ad', linewidth=1))
    ax.text(15.75, 4.35, '64D', fontsize=10, fontweight='bold', ha='center', color='white')
    
    # Arrows to concatenation
    ax.annotate('', xy=(7, 3.35), xytext=(2.25, 4.1), arrowprops=dict(arrowstyle='->', color='#34495e', lw=2))
    ax.annotate('', xy=(9, 3.35), xytext=(9, 4.1), arrowprops=dict(arrowstyle='->', color='#34495e', lw=2))
    ax.annotate('', xy=(11, 3.35), xytext=(15.75, 4.1), arrowprops=dict(arrowstyle='->', color='#34495e', lw=2))
    
    # Concatenation
    ax.add_patch(plt.Rectangle((6, 2.6), 6, 0.75, facecolor='#1abc9c', edgecolor='#16a085', linewidth=2))
    ax.text(9, 2.95, '🔗 CONCATENATE: 768 + 256 + 64 = 1088D', fontsize=11, fontweight='bold', ha='center', color='white')
    
    # Fusion MLP
    ax.annotate('', xy=(9, 1.85), xytext=(9, 2.6), arrowprops=dict(arrowstyle='->', color='#1abc9c', lw=3))
    ax.add_patch(plt.Rectangle((5.5, 0.85), 7, 1, facecolor='#f39c12', edgecolor='#d68910', linewidth=3))
    ax.text(9, 1.55, '🔥 FUSION MLP (Trainable)', fontsize=12, fontweight='bold', ha='center', color='white')
    ax.text(9, 1.1, '1088 → 512 → 256 → 128 → 6', fontsize=10, ha='center', color='white')
    
    # Output arrow
    ax.annotate('', xy=(9, 0.05), xytext=(9, 0.85), arrowprops=dict(arrowstyle='->', color='#27ae60', lw=3))
    
    # Output boxes
    diseases = ['Cardiomegaly', 'Edema', 'Atelectasis', 'Pleural\nEffusion', 'Lung\nOpacity', 'No\nFinding']
    aucs = [0.7003, 0.8334, 0.7511, 0.8484, 0.6824, 0.8057]
    x_positions = [3, 5.4, 7.8, 10.2, 12.6, 15]
    
    for x, disease, auc in zip(x_positions, diseases, aucs):
        ax.add_patch(plt.Rectangle((x-1.1, -1.3), 2.2, 1.1, facecolor='#27ae60', edgecolor='#1e8449', linewidth=1.5, alpha=0.9))
        ax.text(x, -0.55, disease, fontsize=8, fontweight='bold', ha='center', va='center', color='white')
        ax.text(x, -1.0, f'AUC:{auc:.2f}', fontsize=7, ha='center', va='center', color='white')
    
    ax.text(9, -1.8, '📊 6 Disease Predictions (Multi-Label Sigmoid Output)', fontsize=11, ha='center', color='#2c3e50', style='italic')
    
    # Legend
    ax.add_patch(plt.Rectangle((0.3, -2.7), 4.5, 1.8, facecolor='white', edgecolor='#bdc3c7', linewidth=1))
    ax.text(2.55, -1.1, 'LEGEND', fontsize=9, fontweight='bold', ha='center', color='#2c3e50')
    ax.text(0.6, -1.5, '❄️ = Frozen (Pre-trained, No Gradient)', fontsize=8, ha='left', color='#7f8c8d')
    ax.text(0.6, -1.9, '🔥 = Trainable (724K params)', fontsize=8, ha='left', color='#7f8c8d')
    ax.text(0.6, -2.3, '→ = Data Flow', fontsize=8, ha='left', color='#7f8c8d')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fusion_architecture_diagram.png", dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    print("✅ Saved: fusion_architecture_diagram.png")
    plt.show()


# ===================== 2. MODALITY COMPARISON =====================

def plot_modality_comparison():
    """Bar chart comparing individual modalities vs fusion."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    modalities = ['Vision\n(ConvNeXt-Tiny)', 'Signal\n(1D-CNN)', 'Clinical\n(MLP)', '🔀 FUSION\n(Multi-Modal)']
    aucs = [VISION_AUC, SIGNAL_AUC, CLINICAL_AUC, FUSION_AUC]
    colors = ['#3498db', '#e74c3c', '#9b59b6', '#27ae60']
    
    bars = ax.bar(modalities, aucs, color=colors, edgecolor='black', linewidth=2)
    
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{auc:.4f}', 
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    best_single = max(VISION_AUC, SIGNAL_AUC, CLINICAL_AUC)
    ax.axhline(y=best_single, color='gray', linestyle='--', lw=2, label=f'Best Single: {best_single:.4f}')
    
    improvement = FUSION_AUC - best_single
    ax.annotate(f'+{improvement:.4f}', xy=(3, FUSION_AUC + 0.02), fontsize=12, color='#27ae60', fontweight='bold', ha='center')
    
    ax.set_ylabel('Macro-AUC', fontsize=14)
    ax.set_ylim(0.5, 0.85)
    ax.set_title('📊 Modality Performance Comparison', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/modality_comparison.png", dpi=300, bbox_inches='tight')
    print("✅ Saved: modality_comparison.png")
    plt.show()


# ===================== 3. PER-CLASS METRICS =====================

def plot_per_class_metrics():
    """Bar chart comparing metrics across diseases."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    x = np.arange(len(LABELS))
    width = 0.65
    
    metrics_to_plot = [
        ('auc', 'AUC-ROC', '#3498db'),
        ('f1', 'F1-Score', '#e74c3c'),
        ('precision', 'Precision', '#2ecc71'),
        ('recall', 'Recall/Sensitivity', '#9b59b6')
    ]
    
    for ax, (metric_key, metric_name, color) in zip(axes.flatten(), metrics_to_plot):
        values = [RESULTS[l][metric_key] for l in LABELS]
        macro_avg = MACRO[metric_key]
        
        bars = ax.bar(x, values, width, color=color, edgecolor='black', alpha=0.85)
        ax.axhline(y=macro_avg, color='red', linestyle='--', lw=2, label=f'Macro Avg: {macro_avg:.3f}')
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace(' ', '\n') for l in LABELS], fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_title(f'{metric_name} per Disease', fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle('📊 Per-Disease Performance Metrics - Multi-Modal Fusion', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/per_class_metrics.png", dpi=300, bbox_inches='tight')
    print("✅ Saved: per_class_metrics.png")
    plt.show()


# ===================== 4. TRAINING HISTORY =====================

def plot_training_history():
    """Plot training loss and validation AUC over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(TRAIN_LOSS) + 1)
    
    # Loss
    axes[0].plot(epochs, TRAIN_LOSS, 'b-o', lw=2.5, markersize=8, label='Training Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('📉 Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    
    # AUC
    axes[1].plot(epochs, VAL_AUC, 'g-o', lw=2.5, markersize=8, label='Validation Macro-AUC')
    best_epoch = np.argmax(VAL_AUC) + 1
    best_auc = max(VAL_AUC)
    axes[1].axvline(x=best_epoch, color='red', linestyle='--', lw=1.5, label=f'Best Epoch: {best_epoch}')
    axes[1].scatter([best_epoch], [best_auc], color='red', s=150, zorder=5, marker='*')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Macro-AUC', fontsize=12)
    axes[1].set_title(f'📈 Validation AUC (Best: {best_auc:.4f})', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)
    
    plt.suptitle('🔀 Multi-Modal Fusion Training History', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/training_history.png", dpi=300, bbox_inches='tight')
    print("✅ Saved: training_history.png")
    plt.show()


# ===================== 5. RADAR CHART =====================

def plot_radar_chart():
    """Radar chart comparing all metrics across diseases."""
    categories = ['AUC', 'Precision', 'Recall', 'F1', 'Specificity', 'Accuracy']
    N = len(categories)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(polar=True))
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(LABELS)))
    
    for label, color in zip(LABELS, colors):
        values = [RESULTS[label]['auc'], RESULTS[label]['precision'], RESULTS[label]['recall'],
                  RESULTS[label]['f1'], RESULTS[label]['specificity'], RESULTS[label]['accuracy']]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('📊 Per-Disease Metrics Radar Chart', fontsize=14, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/radar_chart.png", dpi=300, bbox_inches='tight')
    print("✅ Saved: radar_chart.png")
    plt.show()


# ===================== 6. COMPREHENSIVE TABLE =====================

def plot_metrics_table():
    """Create a nice table image of all metrics."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    # Header
    columns = ['Disease', 'AUC', 'Precision', 'Recall', 'F1', 'Specificity', 'Accuracy']
    
    # Data
    data = []
    for label in LABELS:
        r = RESULTS[label]
        data.append([label, f"{r['auc']:.4f}", f"{r['precision']:.4f}", f"{r['recall']:.4f}", 
                     f"{r['f1']:.4f}", f"{r['specificity']:.4f}", f"{r['accuracy']:.4f}"])
    
    data.append(['MACRO AVERAGE', f"{MACRO['auc']:.4f}", f"{MACRO['precision']:.4f}", f"{MACRO['recall']:.4f}",
                 f"{MACRO['f1']:.4f}", f"{MACRO['specificity']:.4f}", f"{MACRO['accuracy']:.4f}"])
    
    colors_cells = [['#ecf0f1'] * 7 for _ in range(6)] + [['#27ae60'] * 7]
    
    table = ax.table(cellText=data, colLabels=columns, cellLoc='center', loc='center',
                     cellColours=colors_cells, colColours=['#3498db'] * 7)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Header styling
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(fontweight='bold', color='white')
        if i == 7:  # Macro average row
            cell.set_text_props(fontweight='bold', color='white')
    
    ax.set_title('📊 Multi-Modal Fusion: Comprehensive Metrics', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/metrics_table.png", dpi=300, bbox_inches='tight')
    print("✅ Saved: metrics_table.png")
    plt.show()


# ===================== 7. HEATMAP =====================

def plot_metrics_heatmap():
    """Heatmap of all metrics across diseases."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    metrics = ['auc', 'precision', 'recall', 'f1', 'specificity', 'accuracy']
    data = [[RESULTS[label][m] for m in metrics] for label in LABELS]
    
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
                xticklabels=['AUC', 'Precision', 'Recall', 'F1', 'Specificity', 'Accuracy'],
                yticklabels=LABELS, vmin=0.3, vmax=1.0, annot_kws={'size': 12})
    
    ax.set_title('📊 Performance Heatmap - Multi-Modal Fusion', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Disease', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/metrics_heatmap.png", dpi=300, bbox_inches='tight')
    print("✅ Saved: metrics_heatmap.png")
    plt.show()


# ===================== RUN ALL =====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("📊 GENERATING ALL VISUALIZATIONS (Hardcoded Results)")
    print("="*70)
    print(f"Output Directory: {OUTPUT_DIR}\n")
    
    plot_architecture_diagram()
    plot_modality_comparison()
    plot_per_class_metrics()
    plot_training_history()
    plot_radar_chart()
    plot_metrics_table()
    plot_metrics_heatmap()
    
    print("\n" + "="*70)
    print("✅ ALL VISUALIZATIONS GENERATED!")
    print("="*70)
    print(f"\n📁 Files saved to: {OUTPUT_DIR}/")
    print("   • fusion_architecture_diagram.png   ← For your teacher! 📐")
    print("   • modality_comparison.png")
    print("   • per_class_metrics.png")
    print("   • training_history.png")
    print("   • radar_chart.png")
    print("   • metrics_table.png")
    print("   • metrics_heatmap.png")
    print("\n🎉 DONE!")
