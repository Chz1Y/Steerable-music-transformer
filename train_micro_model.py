import os
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast, GPT2Config, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

DATA_FILE = "remi_dataset.txt"
MODEL_OUTPUT_DIR = "saved_micro_model_500"

def train_model():
    print("🚀 阶段 1：构建 REMI+ 分词器 (识别 EOS)...")
    if not os.path.exists(DATA_FILE): return print(f"❌ 找不到数据文件 {DATA_FILE}")
        
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"])
    tokenizer.train(files=[DATA_FILE], trainer=trainer)
    
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    fast_tokenizer.add_special_tokens({'pad_token': '[PAD]', 'bos_token': '[BOS]', 'eos_token': '[EOS]'})
    
    print("🧠 阶段 2：初始化微缩架构 (4 Layers, 8 Heads)...")
    config = GPT2Config(
        vocab_size=len(fast_tokenizer) + 100,  # 🔪 核心修复 1：扩大词表矩阵，防止越界崩溃
        n_positions=512, n_embd=256, n_layer=4, n_head=8,
        bos_token_id=fast_tokenizer.bos_token_id, 
        eos_token_id=fast_tokenizer.eos_token_id, 
        pad_token_id=fast_tokenizer.pad_token_id
    )
    model = GPT2LMHeadModel(config)
    
    dataset = load_dataset("text", data_files={"train": DATA_FILE})
    def tokenize_function(examples):
        return fast_tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
        
    # 🔪 核心修复 2：强制清空历史数据缓存，读取带有 [EOS] 的全新数据
    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"],
        load_from_cache_file=False 
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=fast_tokenizer, mlm=False)

    print("🔥 阶段 3：启动 50 轮训练...")
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR, num_train_epochs=50, per_device_train_batch_size=16,
        learning_rate=5e-4, lr_scheduler_type="cosine", warmup_steps=500,
        weight_decay=0.01, save_strategy="epoch", save_total_limit=1, report_to="none"
    )

    trainer = Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=tokenized_datasets["train"])
    trainer.train()
    
    trainer.save_model(MODEL_OUTPUT_DIR)
    fast_tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"🎉 主模型训练彻底完成！")

if __name__ == "__main__":
    train_model()
