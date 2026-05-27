import os
import torch
import pretty_midi
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from tqdm import tqdm

# --- 终极动态路径配置区 ---
# 自动获取当前 python 脚本所在的绝对目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MAIN_MODEL_DIR = os.path.join(BASE_DIR, "saved_micro_model_500")
ABLATION_MODEL_DIR = os.path.join(BASE_DIR, "saved_ablation_model")
OUTPUT_MIDI_DIR = os.path.join(BASE_DIR, "eval_midi_samples")

NUM_SAMPLES_PER_LEVEL = 30  # 建议先用 30 跑通出图，后续写论文前可以改成 100

os.makedirs(OUTPUT_MIDI_DIR, exist_ok=True)

def generate_midi_sequence(model, tokenizer, prompt_text, filename):
    """基于核采样生成 MIDI 序列"""
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    
    output_ids = model.generate(
        input_ids, max_length=256, do_sample=True, top_p=0.9, temperature=1.0,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
    )
    
    generated_tokens = tokenizer.decode(output_ids[0], skip_special_tokens=True).split()
    
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0) # 0为钢琴
    current_time = 0.0
    
    for token in generated_tokens:
        if token.startswith("Pitch_"):
            try:
                pitch = int(token.split("_")[1])
                note = pretty_midi.Note(velocity=100, pitch=pitch, start=current_time, end=current_time + 0.25)
                inst.notes.append(note)
                current_time += 0.25
            except:
                pass
                
    if len(inst.notes) > 0:
        pm.instruments.append(inst)
        pm.write(filename)
        return True
    return False

def run_generation():
    print(f"📂 当前工作目录已被锁定为: {BASE_DIR}")
    
    print("\n🚀 阶段 1/2：加载主模型 (REMI+) 并开始生成...")
    if not os.path.exists(MAIN_MODEL_DIR):
        print(f"❌ 找不到主模型: {MAIN_MODEL_DIR}")
        return
        
    main_tokenizer = PreTrainedTokenizerFast.from_pretrained(MAIN_MODEL_DIR)
    main_model = GPT2LMHeadModel.from_pretrained(MAIN_MODEL_DIR)
    
    for level in range(1, 11):
        prompt = f"[Level_{level}]"
        for i in tqdm(range(NUM_SAMPLES_PER_LEVEL), desc=f"生成主模型 Level {level}"):
            filename = os.path.join(OUTPUT_MIDI_DIR, f"Main_L{level}_{i}.mid")
            generate_midi_sequence(main_model, main_tokenizer, prompt, filename)

    print("\n🚀 阶段 2/2：加载消融模型 (Baseline) 并开始盲猜生成...")
    if not os.path.exists(ABLATION_MODEL_DIR):
        print(f"❌ 找不到消融模型: {ABLATION_MODEL_DIR}")
        return
        
    ablation_tokenizer = PreTrainedTokenizerFast.from_pretrained(ABLATION_MODEL_DIR)
    ablation_model = GPT2LMHeadModel.from_pretrained(ABLATION_MODEL_DIR)
    
    # 消融模型没有 Level 概念，因此生成 10 倍数量的样本作为对比基线
    for i in tqdm(range(NUM_SAMPLES_PER_LEVEL * 10), desc="生成消融 Baseline"):
        filename = os.path.join(OUTPUT_MIDI_DIR, f"Ablation_None_{i}.mid")
        generate_midi_sequence(ablation_model, ablation_tokenizer, "Pitch_60", filename)

    print(f"\n🎉 所有的 MIDI 生成完毕，已保存在 '{OUTPUT_MIDI_DIR}' 文件夹中！")
    print(f"👉 下一步：请运行 evaluate_and_plot.py 来出图！")

if __name__ == "__main__":
    run_generation()
