import os
import pretty_midi
import numpy as np
from tqdm import tqdm

RAW_MIDI_DIR = "raw_midi_500"
OUTPUT_DATA_FILE = "remi_dataset.txt"
WINDOW_SIZE = 4.0  # 4秒为一个切片(近似一小节)

def process_midi_by_windows(file_path):
    try:
        pm = pretty_midi.PrettyMIDI(file_path)
        notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
        notes.sort(key=lambda x: x.start)
        if not notes: return []
        
        end_time = pm.get_end_time()
        chunks = []
        for start_t in np.arange(0, end_time, WINDOW_SIZE):
            end_t = start_t + WINDOW_SIZE
            window_notes = [n for n in notes if start_t <= n.start < end_t]
            if len(window_notes) >= 4:
                nd = len(window_notes) / WINDOW_SIZE
                tokens = [f"Pitch_{n.pitch}" for n in window_notes]
                chunks.append((nd, tokens))
        return chunks
    except:
        return []

def build_dataset():
    if not os.path.exists(RAW_MIDI_DIR):
        print(f"❌ 找不到 {RAW_MIDI_DIR}")
        return

    files = [f for f in os.listdir(RAW_MIDI_DIR) if f.endswith(('.mid', '.midi'))]
    print("🔍 阶段一：按小节 (4秒窗口) 对 MIDI 进行高维切片...")

    all_chunks = []
    densities = []

    for f in tqdm(files):
        chunks = process_midi_by_windows(os.path.join(RAW_MIDI_DIR, f))
        for nd, tokens in chunks:
            if len(tokens) < 120: 
                all_chunks.append({"nd": nd, "tokens": tokens})
                densities.append(nd)

    percentiles = np.percentile(densities, np.arange(10, 101, 10))
    print(f"📊 严格切片后的密度边界 (Deciles): {np.round(percentiles, 2)}")

    def get_level(nd):
        for i, threshold in enumerate(percentiles):
            if nd <= threshold: return i + 1
        return 10

    print("✍️ 阶段二：注入标签与 [EOS] 刹车符...")
    with open(OUTPUT_DATA_FILE, "w", encoding="utf-8") as out_f:
        for chunk in all_chunks:
            level = get_level(chunk["nd"])
            # 关键修复：加入 [EOS] 
            seq_str = f"[Level_{level}] " + " ".join(chunk["tokens"]) + " [EOS]"
            out_f.write(seq_str + "\n")

    print(f"🎉 主数据集构建完成！有效训练片段数: {len(all_chunks)}")

if __name__ == "__main__":
    build_dataset()
