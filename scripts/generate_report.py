from transformers import pipeline

def generate_investment_report(prompt, model_dir="models/investai_gpt2"):
    generator = pipeline("text-generation", model=model_dir)
    return generator(prompt, max_length=512, do_sample=True)[0]['generated_text']