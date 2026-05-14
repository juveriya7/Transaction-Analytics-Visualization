import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

print("=" * 60)
print(" CREDIT CARD TRANSACTION ANALYSIS DASHBOARD ")
print("=" * 60)

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------
print("\nLoading datasets...\n")

transactions = pd.read_csv(
    "transactions_data.csv",
    nrows=50000,
    low_memory=False
)

users = pd.read_csv("users_data.csv")
cards = pd.read_csv("cards_data.csv")

print(f"Transactions : {len(transactions):,}")
print(f"Users        : {len(users):,}")
print(f"Cards        : {len(cards):,}")

# ------------------------------------------------
# CLEAN DATA
# ------------------------------------------------
print("\nCleaning dataset...\n")

# Remove duplicates
transactions = transactions.drop_duplicates()

# Clean amount column
transactions["amount"] = (
    transactions["amount"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
)

transactions["amount"] = pd.to_numeric(
    transactions["amount"],
    errors="coerce"
)

# Convert date column
transactions["date"] = pd.to_datetime(
    transactions["date"],
    errors="coerce"
)

# Remove bad rows
transactions = transactions.dropna(subset=["amount", "date"])

# Fix missing states
transactions["merchant_state"] = (
    transactions["merchant_state"]
    .fillna("Unknown")
)

print("Dataset cleaned successfully!")

# ------------------------------------------------
# ANALYSIS
# ------------------------------------------------
print("\nRunning analysis...\n")

# Monthly trend
transactions["month"] = (
    transactions["date"]
    .dt.strftime("%Y-%m")
)

monthly = (
    transactions.groupby("month")["amount"]
    .sum()
    .reset_index()
)

# Statistics
avg_amount = transactions["amount"].mean()
median_amount = transactions["amount"].median()
max_amount = transactions["amount"].max()
total_volume = transactions["amount"].sum()

# Transaction types
txn_types = (
    transactions["use_chip"]
    .value_counts()
)

# Top states
top_states = (
    transactions["merchant_state"]
    .value_counts()
    .head(10)
)

print(f"Total transaction volume : ${total_volume:,.2f}")
print(f"Average transaction      : ${avg_amount:.2f}")
print(f"Median transaction       : ${median_amount:.2f}")
print(f"Largest transaction      : ${max_amount:,.2f}")

# ------------------------------------------------
# VISUALIZATION
# ------------------------------------------------
print("\nGenerating dashboard...\n")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

fig.suptitle(
    "Credit Card Transaction Analysis Dashboard",
    fontsize=20,
    fontweight="bold"
)

# -----------------------------
# Chart 1: Monthly Trend
# -----------------------------
axes[0, 0].plot(
    monthly["month"],
    monthly["amount"],
    marker="o"
)

axes[0, 0].set_title("Monthly Transaction Volume")
axes[0, 0].set_xlabel("Month")
axes[0, 0].set_ylabel("Amount ($)")
axes[0, 0].tick_params(axis="x", rotation=45)

# -----------------------------
# Chart 2: Transaction Types
# -----------------------------
axes[0, 1].pie(
    txn_types.values,
    labels=txn_types.index,
    autopct="%1.1f%%"
)

axes[0, 1].set_title("Transaction Types")

# -----------------------------
# Chart 3: Top States
# -----------------------------
top_states.sort_values().plot(
    kind="barh",
    ax=axes[1, 0]
)

axes[1, 0].set_title("Top Merchant States")
axes[1, 0].set_xlabel("Transactions")

# -----------------------------
# Chart 4: Amount Distribution
# -----------------------------
axes[1, 1].hist(
    transactions["amount"],
    bins=40
)

axes[1, 1].axvline(
    avg_amount,
    linestyle="--",
    label=f"Mean: ${avg_amount:.0f}"
)

axes[1, 1].axvline(
    median_amount,
    linestyle="--",
    label=f"Median: ${median_amount:.0f}"
)

axes[1, 1].set_title("Transaction Amount Distribution")
axes[1, 1].set_xlabel("Amount ($)")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].legend()

# Layout
plt.tight_layout()

# Save dashboard
plt.savefig(
    "transaction_dashboard.png",
    dpi=300
)

print("Dashboard saved as transaction_dashboard.png")

# ------------------------------------------------
# REPORT
# ------------------------------------------------
report = f"""
CREDIT CARD ANALYSIS REPORT
===========================

Total Transactions : {len(transactions):,}

Total Volume        : ${total_volume:,.2f}
Average Amount      : ${avg_amount:.2f}
Median Amount       : ${median_amount:.2f}
Largest Transaction : ${max_amount:,.2f}

Top Merchant States:
{top_states.to_string()}
"""

with open("analysis_report.txt", "w") as f:
    f.write(report)

print("\nReport saved as analysis_report.txt")

print("\nPROJECT EXECUTED SUCCESSFULLY!")
print("=" * 60)