import os
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

# Paths
dataset_path = os.path.join("data", "finetune_dataset.txt")
model_output_path = os.path.join("models", "gpt2-finetuned")

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Ensure padding token is defined (GPT2 doesn’t have one by default)
tokenizer.pad_token = tokenizer.eos_token

# Prepare dataset
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=dataset_path,
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # we are not using masked language modeling
)

training_args = TrainingArguments(
    output_dir=model_output_path,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_dir=os.path.join(model_output_path, "logs"),
    logging_steps=10,
    fp16=False,  # keep this False on CPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

print("🚀 Starting fine-tuning...")
trainer.train()

# Save the model and tokenizer
trainer.save_model(model_output_path)
tokenizer.save_pretrained(model_output_path)

print("✅ Model fine-tuned and saved at:", model_output_path)
