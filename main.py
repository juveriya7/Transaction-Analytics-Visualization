import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

print("=" * 55)
print("   CREDIT CARD TRANSACTION DATA ANALYSIS PROJECT")
print("=" * 55)

# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────
print("\n[1/5] Loading datasets...")

transactions = pd.read_csv("transactions_data.csv", nrows=50000)
users        = pd.read_csv("users_data.csv")
cards        = pd.read_csv("cards_data.csv")

print(f"  Transactions loaded : {len(transactions):,} rows")
print(f"  Users loaded        : {len(users):,} rows")
print(f"  Cards loaded        : {len(cards):,} rows")

# ─────────────────────────────────────────────
# STEP 2: DATA CLEANING
# ─────────────────────────────────────────────
print("\n[2/5] Cleaning data...")

before = len(transactions)

# Remove duplicate rows
transactions = transactions.drop_duplicates()
dupes_removed = before - len(transactions)

# Clean amount column — remove $ signs and convert to float
transactions["amount"] = (
    transactions["amount"]
    .astype(str)
    .str.replace(r"[$,]", "", regex=True)
    .str.strip()
    .astype(float)
)

# Parse dates
transactions["date"] = pd.to_datetime(transactions["date"], errors="coerce")

# Fill missing merchant state
missing_state_before = transactions["merchant_state"].isna().sum()
transactions["merchant_state"] = transactions["merchant_state"].fillna("Unknown")

# Standardize text columns
for col in ["use_chip", "merchant_state", "merchant_city"]:
    if col in transactions.columns:
        transactions[col] = transactions[col].astype(str).str.strip().str.title()

# Flag negative amounts (potential refunds or errors)
transactions["is_negative"] = transactions["amount"] < 0

print(f"  Duplicate rows removed       : {dupes_removed:,}")
print(f"  Missing merchant states fixed: {missing_state_before:,}")
print(f"  Negative amount entries      : {transactions['is_negative'].sum():,}")
print(f"  Clean dataset size           : {len(transactions):,} rows")

# ─────────────────────────────────────────────
# STEP 3: EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
print("\n[3/5] Running analysis...")

# Transaction amount stats
avg_amount   = transactions["amount"].mean()
median_amt   = transactions["amount"].median()
total_volume = transactions["amount"].sum()
max_txn      = transactions["amount"].max()

print(f"\n  Transaction Summary:")
print(f"    Total volume   : ${total_volume:,.2f}")
print(f"    Average amount : ${avg_amount:.2f}")
print(f"    Median amount  : ${median_amt:.2f}")
print(f"    Largest txn    : ${max_txn:,.2f}")

# Monthly trend
transactions["month"] = transactions["date"].dt.to_period("M")
monthly = transactions.groupby("month")["amount"].sum().reset_index()
monthly["month"] = monthly["month"].astype(str)

# Top merchants
top_states   = transactions["merchant_state"].value_counts().head(10)
top_cities   = transactions["merchant_city"].value_counts().head(10)  if "merchant_city"  in transactions.columns else None
txn_types    = transactions["use_chip"].value_counts()                if "use_chip"       in transactions.columns else None

# High value transactions (top 1%)
threshold    = transactions["amount"].quantile(0.99)
high_value   = transactions[transactions["amount"] >= threshold]
print(f"\n  High-value transactions (top 1%): {len(high_value):,} (threshold: ${threshold:,.2f})")

# ─────────────────────────────────────────────
# STEP 4: VISUALIZATIONS (DASHBOARD)
# ─────────────────────────────────────────────
print("\n[4/5] Building dashboard...")

fig = plt.figure(figsize=(16, 11), facecolor="#F8F9FA")
fig.suptitle(
    "Credit Card Transaction Analysis Dashboard",
    fontsize=18, fontweight="bold", color="#1A1A2E", y=0.98
)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Color palette
BLUE   = "#2196F3"
GREEN  = "#4CAF50"
ORANGE = "#FF9800"
RED    = "#F44336"
PURPLE = "#9C27B0"

# ── Chart 1: Monthly Transaction Volume ──
ax1 = fig.add_subplot(gs[0, :2])
if len(monthly) > 1:
    ax1.fill_between(range(len(monthly)), monthly["amount"], alpha=0.15, color=BLUE)
    ax1.plot(range(len(monthly)), monthly["amount"], color=BLUE, linewidth=2.5, marker="o", markersize=4)
    step = max(1, len(monthly) // 8)
    ax1.set_xticks(range(0, len(monthly), step))
    ax1.set_xticklabels(monthly["month"].iloc[::step], rotation=35, ha="right", fontsize=8)
ax1.set_title("Monthly Transaction Volume ($)", fontweight="bold", fontsize=11, color="#1A1A2E")
ax1.set_ylabel("Total Amount ($)", fontsize=9)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M" if x >= 1e6 else f"${x/1e3:.0f}K"))
ax1.grid(axis="y", alpha=0.3, linestyle="--")
ax1.set_facecolor("#FFFFFF")

# ── Chart 2: Transaction Type Breakdown ──
ax2 = fig.add_subplot(gs[0, 2])
if txn_types is not None and len(txn_types) > 0:
    colors_pie = [BLUE, GREEN, ORANGE, RED, PURPLE][:len(txn_types)]
    wedges, texts, autotexts = ax2.pie(
        txn_types.values,
        labels=None,
        autopct="%1.1f%%",
        colors=colors_pie,
        startangle=90,
        pctdistance=0.75
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax2.legend(
        wedges, txn_types.index,
        loc="lower center", fontsize=7,
        bbox_to_anchor=(0.5, -0.15), ncol=1
    )
ax2.set_title("Transaction Types", fontweight="bold", fontsize=11, color="#1A1A2E")

# ── Chart 3: Top 10 Merchant States ──
ax3 = fig.add_subplot(gs[1, :2])
bars = ax3.barh(
    top_states.index[::-1],
    top_states.values[::-1],
    color=BLUE, alpha=0.85, edgecolor="white", height=0.6
)
for bar, val in zip(bars, top_states.values[::-1]):
    ax3.text(bar.get_width() + top_states.max() * 0.01, bar.get_y() + bar.get_height() / 2,
             f"{val:,}", va="center", fontsize=8, color="#444")
ax3.set_title("Top 10 Merchant States by Transaction Count", fontweight="bold", fontsize=11, color="#1A1A2E")
ax3.set_xlabel("Number of Transactions", fontsize=9)
ax3.grid(axis="x", alpha=0.3, linestyle="--")
ax3.set_facecolor("#FFFFFF")

# ── Chart 4: Amount Distribution ──
ax4 = fig.add_subplot(gs[1, 2])
clean_amounts = transactions["amount"][(transactions["amount"] > 0) & (transactions["amount"] < threshold)]
ax4.hist(clean_amounts, bins=40, color=PURPLE, alpha=0.75, edgecolor="white")
ax4.axvline(avg_amount,  color=RED,   linestyle="--", linewidth=1.5, label=f"Mean: ${avg_amount:.0f}")
ax4.axvline(median_amt,  color=GREEN, linestyle="--", linewidth=1.5, label=f"Median: ${median_amt:.0f}")
ax4.set_title("Transaction Amount Distribution", fontweight="bold", fontsize=11, color="#1A1A2E")
ax4.set_xlabel("Amount ($)", fontsize=9)
ax4.set_ylabel("Frequency", fontsize=9)
ax4.legend(fontsize=8)
ax4.grid(axis="y", alpha=0.3, linestyle="--")
ax4.set_facecolor("#FFFFFF")

plt.savefig("transaction_dashboard.png", dpi=150, bbox_inches="tight", facecolor="#F8F9FA")
print("  Dashboard saved as transaction_dashboard.png")
plt.show()

# ─────────────────────────────────────────────
# STEP 5: SUMMARY REPORT
# ─────────────────────────────────────────────
print("\n[5/5] Generating summary report...")

report = f"""
╔══════════════════════════════════════════════════════╗
║          DATA ANALYSIS SUMMARY REPORT                ║
╚══════════════════════════════════════════════════════╝

DATASET OVERVIEW
────────────────
  Total transactions analysed : {len(transactions):,}
  Duplicate rows removed      : {dupes_removed:,}
  Missing values handled      : {missing_state_before:,} merchant states
  Date range                  : {transactions['date'].min().date()} → {transactions['date'].max().date()}

TRANSACTION METRICS
───────────────────
  Total transaction volume    : ${total_volume:,.2f}
  Average transaction amount  : ${avg_amount:.2f}
  Median transaction amount   : ${median_amt:.2f}
  Largest single transaction  : ${max_txn:,.2f}
  High-value transactions     : {len(high_value):,} (top 1%, ≥ ${threshold:,.2f})
  Negative/refund entries     : {transactions['is_negative'].sum():,}

TOP MERCHANT STATE
──────────────────
  #{1}: {top_states.index[0]} ({top_states.values[0]:,} transactions)
  #{2}: {top_states.index[1]} ({top_states.values[1]:,} transactions)
  #{3}: {top_states.index[2]} ({top_states.values[2]:,} transactions)

DATA QUALITY SCORE
──────────────────
  Completeness  : {(1 - transactions.isnull().sum().sum() / (len(transactions) * len(transactions.columns))) * 100:.1f}%
  Duplicate-free: {"Yes" if dupes_removed == 0 else f"Fixed ({dupes_removed} removed)"}
  Format valid  : Yes (amounts cleaned, dates parsed)

OUTPUT
──────
  Dashboard saved : transaction_dashboard.png
"""

print(report)

with open("analysis_report.txt", "w") as f:
    f.write(report)

print("Report saved as analysis_report.txt")
print("\n" + "=" * 55)
print("   ANALYSIS COMPLETE")
print("=" * 55)