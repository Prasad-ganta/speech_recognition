import os
import pandas as pd
import torch
from datasets import Dataset, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer

# 📂 Path to your Common Voice English folder
DATASET_DIR = "path/to/cv-corpus-21.0-delta-2025-03-14/en"
CLIPS_DIR = os.path.join(DATASET_DIR, "clips")
TSV_PATH = os.path.join(DATASET_DIR, "validated.tsv")

# ✅ Load and filter data
df = pd.read_csv(TSV_PATH, sep="\t")
df = df[["path", "sentence"]].dropna()
df["path"] = df["path"].apply(lambda x: os.path.join(CLIPS_DIR, x))
df = df.rename(columns={"path": "audio", "sentence": "text"})

# 🔄 Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# 🔌 Load processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# 🎛️ Preprocessing
def prepare(batch):
    audio = batch["audio"]
    inputs = processor(audio["array"], sampling_rate=16000, return_tensors="pt", padding="longest")
    with processor.as_target_processor():
        labels = processor(batch["text"], return_tensors="pt", padding="longest").input_ids
    batch["input_values"] = inputs.input_values[0]
    batch["labels"] = labels[0]
    return batch

dataset = dataset.map(prepare, remove_columns=dataset.column_names, num_proc=2)

# 🧾 Training setup
training_args = TrainingArguments(
    output_dir="./asr_model",
    per_device_train_batch_size=4,
    evaluation_strategy="no",
    num_train_epochs=1,
    save_steps=100,
    logging_steps=10,
    save_total_limit=2,
    fp16=torch.cuda.is_available()
)

# 🏋️ Train
def data_collator(features):
    input_values = torch.stack([f["input_values"] for f in features])
    labels = torch.stack([f["labels"] for f in features])
    return {"input_values": input_values, "labels": labels}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained("./asr_model")
processor.save_pretrained("./asr_model")
