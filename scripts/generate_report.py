from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os

# Load fine-tuned model and tokenizer
model_dir = os.path.join("models", "gpt2-finetuned")
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)

# Ensure tokenizer has padding token set
tokenizer.pad_token = tokenizer.eos_token
model.eval()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_investment_report(stock_trend, news, max_length=100):
    """
    Generate a GPT-2 based investment report given stock trend and news.
    Returns only the generated continuation (cleaned, without prompt).
    """
    prompt = f"Stock Trend: {stock_trend}\nNews: {news}\nGenerate Report:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=len(inputs["input_ids"][0]) + max_length,
    do_sample=True,
    temperature=0.9,
    top_k=50,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id
)

    generated = tokenizer.decode(output[0], skip_special_tokens=True)

    # Smart trimming: get only the actual generated report
    if "Generate Report:" in generated:
        summary = generated.split("Generate Report:")[-1].strip()
    else:
        summary = generated.strip()

    return summary

