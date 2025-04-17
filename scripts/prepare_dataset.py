import os

input_path = os.path.join("data", "finetune_corpus.txt")
output_path = os.path.join("data", "finetune_dataset.txt")

with open(input_path, "r", encoding="utf-8") as f:
    blocks = f.read().strip().split("Generate Report:")

cleaned = []
for block in blocks:
    lines = block.strip().split("\n")
    if len(lines) < 2:
        continue

    has_trend = any("Stock Trend:" in line for line in lines)
    has_news = any("News:" in line for line in lines)

    if has_trend and has_news:
        cleaned_block = "\n".join([line for line in lines if line.strip()]) + "\nGenerate Report:"
        cleaned.append(cleaned_block)

with open(output_path, "w", encoding="utf-8") as f:
    for item in cleaned:
        f.write(item.strip() + "\n\n")

print(f"✅ Cleaned dataset saved to: {output_path}")
print(f"Total examples: {len(cleaned)}")
