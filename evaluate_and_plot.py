import os
import pretty_midi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MIDI_DIR = os.path.join(BASE_DIR, "eval_midi_samples")
OUTPUT_PLOT_DIR = os.path.join(BASE_DIR, "paper_figures_final")

os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

def calculate_metrics(file_path):
    try:
        pm = pretty_midi.PrettyMIDI(file_path)
        notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
        if not notes: return None
        
        nd = len(notes) 
        pitches = [n.pitch for n in notes]
        counts = np.bincount(pitches)
        probs = counts[counts > 0] / len(pitches)
        entropy = -np.sum(probs * np.log2(probs))
        return nd, entropy
    except:
        return None

def run_evaluation():
    if not os.path.exists(MIDI_DIR): return print(f"❌ 找不到 {MIDI_DIR}")

    files = [f for f in os.listdir(MIDI_DIR) if f.endswith('.mid')]
    results = []
    
    for f in tqdm(files, desc="计算指标"):
        metrics = calculate_metrics(os.path.join(MIDI_DIR, f))
        if metrics:
            if f.startswith("Main_L"):
                level = int(f.split("_")[1].replace("L", ""))
                results.append({"Model": "REMI+ (Ours)", "Target_Level": level, "Display_Category": f"L{level}", "Actual_ND": metrics[0], "Pitch_Entropy": metrics[1]})
            elif f.startswith("Ablation_"):
                results.append({"Model": "Unconditional Baseline", "Target_Level": 0, "Display_Category": "Baseline", "Actual_ND": metrics[0], "Pitch_Entropy": metrics[1]})

    if not results: return print("❌ 数据为空！")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_PLOT_DIR, "final_experiment_metrics.csv"), index=False)
    df_main = df[df["Model"] == "REMI+ (Ours)"].copy()
    
    # ---------------------------------------------------------
    # 图 1：Pearson 回归散点图
    plt.figure(figsize=(8, 6))
    if len(df_main["Actual_ND"].unique()) > 1:
        r, p = pearsonr(df_main["Target_Level"].astype(int), df_main["Actual_ND"])
        sns.regplot(data=df_main, x="Target_Level", y="Actual_ND", scatter_kws={'alpha':0.4, 'color':'#2c3e50'}, line_kws={'color':'#e74c3c', 'label':f'Pearson $r={r:.3f}$'})
        plt.legend()
        print(f"\n✅ Pearson r 最终突破！值域为: {r:.3f}")
    else:
        sns.scatterplot(data=df_main, x="Target_Level", y="Actual_ND")
        
    plt.title("Correlation: Target Level vs. Note Density")
    plt.xlabel("Target Complexity Level")
    plt.ylabel("Generated Note Count (Density Proxy)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "Fig1_Pearson_Scatter.pdf"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 图 2：微观阶梯箱线图
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df_main, x="Target_Level", y="Actual_ND", palette="Blues")
    plt.title("Step-wise Distribution of Generated Note Density")
    plt.xlabel("REMI+ Target Level")
    plt.ylabel("Generated Note Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "Fig2_Stepwise_Boxplot.pdf"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 图 3：消融对比全景图 (Ablation)
    plt.figure(figsize=(11, 6))
    order = ["Baseline"] + [f"L{i}" for i in range(1, 11)]
    custom_palette = {"Baseline": "#e74c3c"}
    for i in range(1, 11): custom_palette[f"L{i}"] = "#3498db"
        
    sns.boxplot(data=df, x="Display_Category", y="Actual_ND", order=order, palette=custom_palette)
    baseline_mean = df[df["Model"] == "Unconditional Baseline"]["Actual_ND"].mean()
    plt.axhline(baseline_mean, color='#c0392b', linestyle='--', alpha=0.6, label="Baseline Mean")
    
    plt.title("Ablation Study: Unconditional Baseline vs. REMI+ Steerable Levels")
    plt.xlabel("Model Configuration")
    plt.ylabel("Generated Note Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "Fig3_Ablation_Comparison.pdf"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 图 4：特征解耦 (音高熵)
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_main, x="Target_Level", y="Pitch_Entropy", color="#27ae60", errorbar="sd", capsize=0.1)
    plt.title("Verification of Feature Decoupling (Pitch Shannon Entropy)")
    plt.ylim(0, max(df_main["Pitch_Entropy"].max() + 1, 5.5))
    plt.xlabel("Target Complexity Level")
    plt.ylabel("Pitch Entropy (Bits)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "Fig4_Feature_Decoupling.pdf"), dpi=300)
    plt.close()

    print(f"\n🎉 完美收官！全套矢量图表已在 '{OUTPUT_PLOT_DIR}' 生成！")

if __name__ == "__main__":
    run_evaluation()
