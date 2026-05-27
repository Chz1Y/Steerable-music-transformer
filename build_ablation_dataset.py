import os
import pretty_midi
import numpy as np
from tqdm import tqdm

RAW_MIDI_DIR = "raw_midi_500"
OUTPUT_DATA_FILE = "ablation_dataset.txt"
WINDOW_SIZE = 4.0

def build_ablation_dataset():
    files = [f for f in os.listdir(RAW_MIDI_DIR) if f.endswith(('.mid', '.midi'))]
    print("🔍 正在构建切片化、但无标签的消融数据集...")

    with open(OUTPUT_DATA_FILE, "w", encoding="utf-8") as out_f:
        for f in tqdm(files):
            try:
                pm = pretty_midi.PrettyMIDI(os.path.join(RAW_MIDI_DIR, f))
                notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
                notes.sort(key=lambda x: x.start)
                end_time = pm.get_end_time()
                
                for start_t in np.arange(0, end_time, WINDOW_SIZE):
                    end_t = start_t + WINDOW_SIZE
                    window_notes = [n for n in notes if start_t <= n.start < end_t]
                    
                    if 4 <= len(window_notes) < 120:
                        tokens = [f"Pitch_{n.pitch}" for n in window_notes]
                        # 核心区别：没有 Level 标签，但有 [EOS] 刹车！
                        out_f.write(" ".join(tokens) + " [EOS]\n")
            except:
                pass

    print(f"🎉 消融数据集构建完成！")

if __name__ == "__main__":
    build_ablation_dataset()
