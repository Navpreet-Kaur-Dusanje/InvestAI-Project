# ───────────────────────────────────────────────
# InvestAI – Dynamic Configuration (Big Tech)
# Automatically sets date range to last 90 days
# ───────────────────────────────────────────────

from datetime import datetime, timedelta

# List of Big Tech stock tickers
TICKERS = [
    "AAPL",   # Apple Inc.
    "MSFT",   # Microsoft Corp
    "GOOGL",  # Alphabet Inc. (Class A)
    "AMZN",   # Amazon.com Inc.
    "META",   # Meta Platforms Inc.
    "NVDA",   # NVIDIA Corp
    "TSLA"    # Tesla Inc.
]

# Get today's date and 90 days ago
END_DATE = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")
