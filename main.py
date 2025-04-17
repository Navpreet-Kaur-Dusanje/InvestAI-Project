import os
import pandas as pd
import matplotlib.pyplot as plt
from scripts.data_fetcher import get_data
from scripts.data_preprocessor import (
    preprocess_stock_data,
    preprocess_news_articles,
    create_finetune_corpus,
)
from scripts.generate_report import generate_investment_report
from utils.config import TICKERS, START_DATE, END_DATE

print("📦 Fetching stock & news data...")
stock_data_dict, news_data_dict = get_data(TICKERS, START_DATE, END_DATE)

print("🔧 Preprocessing...")
for ticker in TICKERS:
    stock_data_dict[ticker] = preprocess_stock_data(stock_data_dict[ticker])
    news_data_dict[ticker] = preprocess_news_articles(news_data_dict[ticker])

print("📝 Creating fine-tuning corpus...")
create_finetune_corpus(stock_data_dict, news_data_dict)
print("✅ Done! Corpus saved to data/finetune_corpus.txt")

print("🧠 Generating investment reports from latest data...\n")
report_samples = []

for ticker in TICKERS:
    stock_df = stock_data_dict[ticker]
    news_df = news_data_dict[ticker]

    if stock_df.empty or news_df.empty:
        continue

    try:
        latest_row = stock_df.tail(1).copy().reset_index(drop=True)
        stock_price = float(latest_row["Close"].iloc[0])
        stock_date = pd.to_datetime(latest_row["Date"].iloc[0]).strftime("%Y-%m-%d")

        latest_news = news_df.sort_values(by="publishedAt", ascending=False).head(1)
        news_sentence = latest_news["content"].values[0][:300]

        summary = generate_investment_report(str(stock_price), news_sentence)

        # Enhanced Structured Report
        enhanced_report = f"""
🧠 [AI Investment Summary]
────────────────────────────
📈 Ticker: {ticker}
📅 Date: {stock_date}
💵 Closing Price: ${stock_price}

📰 News Snapshot:
{news_sentence}

📝 AI Insight:
{summary}

────────────────────────────
"""
        print(enhanced_report)

        report_samples.append({
            "ticker": ticker,
            "date": stock_date,
            "price": stock_price,
            "summary": summary,
            "full_report": enhanced_report
        })

    except Exception as e:
        print(f"⚠️ Skipped {ticker} due to: {e}")

# 📝 Export report to text file
if report_samples:
    with open("report_summary.txt", "w", encoding="utf-8") as f:
        for r in report_samples:
            f.write(r["full_report"])
            f.write("\n")

# 📊 Horizontal Bar Chart
if report_samples:
    report_samples = sorted(report_samples, key=lambda x: x["price"], reverse=True)
    labels = [f"{r['ticker']} ({r['date']})" for r in report_samples]
    prices = [r["price"] for r in report_samples]
    summaries = [r["summary"][:80] + "..." if len(r["summary"]) > 80 else r["summary"] for r in report_samples]

    plt.figure(figsize=(12, 7))
    bars = plt.barh(labels, prices, color="skyblue", edgecolor="black")
    plt.xlabel("Stock Price ($)")
    plt.title("📈 Latest Stock Prices with GPT-2 Generated Investment Reports")

    for bar, summary in zip(bars, summaries):
        plt.text(
            bar.get_width() + 5,
            bar.get_y() + bar.get_height() / 2,
            summary,
            va="center",
            fontsize=9,
            color="green"
        )

    plt.tight_layout()
    plt.savefig("report_summary_plot.png", bbox_inches="tight")
    plt.show()
