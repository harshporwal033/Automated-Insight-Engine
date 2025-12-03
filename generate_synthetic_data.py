#!/usr/bin/env python3
"""
generate_synthetic_user_data.py

Create a 'user_data/' folder and populate it with synthetic but realistic datasets:
 - CSV (traffic)
 - TSV (clicks)
 - Excel with multiple sheets (finance)
 - Parquet (users)
 - JSONL (events)
 - SQLite DB with multiple tables (impressions, conversions)
 - Small campaign CSV with overlapping columns

Run:
    python generate_synthetic_user_data.py
"""
import os
import random
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import uuid

USER_DATA_DIR = Path("user_data")


def ensure_dir():
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)


def random_dates(start: datetime, n: int, spread_days: int = 30):
    return [start + timedelta(days=random.randint(0, spread_days), seconds=random.randint(0, 86400)) for _ in range(n)]


def generate_traffic_csv(n=500):
    """session-level data (CSV)"""
    sessions = []
    base = datetime.now() - timedelta(days=60)
    countries = ["US", "IN", "GB", "DE", "FR", "JP"]
    devices = ["mobile", "desktop", "tablet"]
    campaigns = [f"CMP_{i}" for i in range(1, 8)]

    for i in range(n):
        session_id = str(uuid.uuid4())
        user_id = random.randint(1000, 2000)
        ts = random_dates(base, 1)[0].isoformat()
        country = random.choice(countries)
        device = random.choice(devices)
        page_views = max(1, int(random.expovariate(1/3)))
        session_duration = max(5, int(random.gauss(180, 60)))
        campaign_id = random.choice(campaigns) if random.random() < 0.6 else None
        # small chance of a typo in column values (simulate noise)
        if random.random() < 0.02:
            country = country.lower()
        sessions.append({
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": ts,
            "country": country,
            "device": device,
            "page_views": page_views,
            "session_duration": session_duration,
            "campaign_id": campaign_id
        })

    df = pd.DataFrame(sessions)
    # introduce some missing values and duplicated rows
    for _ in range(int(n*0.03)):
        ix = random.choice(df.index)
        df.loc[ix, random.choice(df.columns)] = None
    df = pd.concat([df, df.sample(frac=0.01)])  # tiny duplication
    path = USER_DATA_DIR / "traffic.csv"
    df.to_csv(path, index=False)
    print(f"[gen] wrote {path}")
    return path


def generate_clicks_tsv(n=400):
    """click-level data (TSV)"""
    base = datetime.now() - timedelta(days=60)
    campaigns = [f"CMP_{i}" for i in range(1, 9)]
    ads = [f"AD_{i}" for i in range(1, 50)]
    rows = []
    for i in range(n):
        click_id = str(uuid.uuid4())
        user_id = random.randint(900, 2100)
        ts = random_dates(base, 1)[0].isoformat()
        ad_id = random.choice(ads)
        campaign_id = random.choice(campaigns)
        clicked = int(random.random() < 0.4)
        revenue = round(random.random() * 5 * clicked, 4)  # revenue only when clicked
        rows.append({
            "click_id": click_id,
            "user_id": user_id,
            "timestamp": ts,
            "ad_id": ad_id,
            "campaign_id": campaign_id,
            "clicked": clicked,
            "revenue": revenue
        })
    df = pd.DataFrame(rows)
    # introduce a slight column-name typo occasionally to validate robustness
    if random.random() < 0.5:
        df.rename(columns={"campaign_id": "campaign id"}, inplace=True)
    path = USER_DATA_DIR / "clicks.tsv"
    df.to_csv(path, sep="\t", index=False)
    print(f"[gen] wrote {path}")
    return path


def generate_finance_excel():
    """Excel with multiple sheets (daily + monthly aggregates)"""
    base = datetime.now().date() - timedelta(days=60)
    campaigns = [f"CMP_{i}" for i in range(1, 8)]
    days = [base + timedelta(days=i) for i in range(60)]
    daily = []
    for d in days:
        for c in campaigns:
            impressions = random.randint(100, 20000)
            clicks = int(impressions * random.uniform(0.001, 0.05))
            spend = round(random.uniform(10, 500) * (impressions / 1000), 2)
            revenue = round(spend * random.uniform(0.8, 2.0), 2)
            daily.append({
                "date": d.isoformat(),
                "campaign_id": c,
                "impressions": impressions,
                "clicks": clicks,
                "spend": spend,
                "revenue": revenue
            })
    df_daily = pd.DataFrame(daily)
    df_monthly = df_daily.copy()
    df_monthly["month"] = pd.to_datetime(df_monthly["date"]).dt.to_period("M").astype(str)
    df_monthly = df_monthly.groupby(["month", "campaign_id"]).agg({
        "impressions": "sum",
        "clicks": "sum",
        "spend": "sum",
        "revenue": "sum"
    }).reset_index()

    path = USER_DATA_DIR / "finance.xlsx"
    with pd.ExcelWriter(path) as writer:
        df_daily.to_excel(writer, sheet_name="daily", index=False)
        df_monthly.to_excel(writer, sheet_name="monthly", index=False)
    print(f"[gen] wrote {path}")
    return path


def generate_users_parquet(n=300):
    """User profile table saved as Parquet"""
    start = datetime.now() - timedelta(days=365*2)
    countries = ["US", "IN", "GB", "DE", "FR", "JP"]
    ages = list(range(18, 70))
    records = []
    for i in range(n):
        user_id = 1000 + i
        signup_date = (start + timedelta(days=random.randint(0, 700))).date().isoformat()
        country = random.choice(countries)
        age = random.choice(ages)
        gender = random.choice(["M", "F", None])
        records.append({
            "user_id": user_id,
            "signup_date": signup_date,
            "country": country,
            "age": age,
            "gender": gender
        })
    df = pd.DataFrame(records)
    # occasionally add a 'userId' type column to simulate schema drift
    if random.random() < 0.4:
        df["userId"] = df["user_id"]
    path = USER_DATA_DIR / "users.parquet"
    df.to_parquet(path, index=False)
    print(f"[gen] wrote {path}")
    return path


def generate_events_jsonl(n=800):
    """Write JSONL events (one JSON object per line), fully valid."""
    base = datetime.now() - timedelta(days=60)
    event_types = ["page_view", "click", "purchase", "signup"]

    path = USER_DATA_DIR / "events.jsonl"          # FIXED: JSONL extension

    with open(path, "w", encoding="utf-8") as fh:
        for _ in range(n):
            ev = {
                "event_id": str(uuid.uuid4()),
                "user_id": random.randint(900, 2100),
                "event_type": random.choice(event_types),
                "timestamp": random_dates(base, 1)[0].isoformat(),
                "session_id": str(uuid.uuid4()) if random.random() < 0.9 else None,
                "value": round(random.random() * 100, 2)
            }

            # small noise key
            if random.random() < 0.03:
                ev["co_untry"] = random.choice(["US", "IN"])

            fh.write(json.dumps(ev) + "\n")        # JSONL line — CORRECT

    print(f"[gen] wrote {path}")
    return path

def generate_sqlite_db():
    """Create a SQLite DB with two tables: impressions, conversions (overlapping columns)"""
    path = USER_DATA_DIR / "analytics.db"
    conn = sqlite3.connect(path)
    try:
        # impressions
        imps = []
        for i in range(500):
            imps.append({
                "impression_id": str(uuid.uuid4()),
                "timestamp": (datetime.now() - timedelta(days=random.randint(0, 90))).isoformat(),
                "campaign_id": random.choice([f"CMP_{i}" for i in range(1, 8)]),
                "ad_id": f"AD_{random.randint(1, 120)}",
                "impressions": random.randint(1, 10)
            })
        df_imps = pd.DataFrame(imps)
        df_imps.to_sql("impressions", conn, if_exists="replace", index=False)

        # conversions
        conv = []
        for i in range(200):
            conv.append({
                "conversion_id": str(uuid.uuid4()),
                "timestamp": (datetime.now() - timedelta(days=random.randint(0, 90))).isoformat(),
                "user_id": random.randint(900, 2100),
                "campaign_id": random.choice([f"CMP_{i}" for i in range(1, 8)]),
                "revenue": round(random.random() * 50, 2)
            })
        df_conv = pd.DataFrame(conv)
        df_conv.to_sql("conversions", conn, if_exists="replace", index=False)
    finally:
        conn.close()
    print(f"[gen] wrote {path}")
    return path


def generate_campaigns_csv():
    """Campaign meta info (small CSV) — overlapping campaign_id column"""
    campaigns = []
    for i in range(1, 9):
        campaigns.append({
            "campaign_id": f"CMP_{i}",
            "campaign_name": f"Campaign {i}",
            "start_date": (datetime.now() - timedelta(days=random.randint(10, 200))).date().isoformat(),
            "end_date": (datetime.now() + timedelta(days=random.randint(5, 90))).date().isoformat(),
            "budget": round(random.uniform(1000, 50000), 2)
        })
    df = pd.DataFrame(campaigns)
    path = USER_DATA_DIR / "ad_campaigns.csv"
    df.to_csv(path, index=False)
    print(f"[gen] wrote {path}")
    return path


def main():
    ensure_dir()
    generate_traffic_csv()
    generate_clicks_tsv()
    generate_finance_excel()
    generate_users_parquet()
    generate_events_jsonl()
    generate_sqlite_db()
    generate_campaigns_csv()
    print("[gen] All synthetic files created in 'user_data/'.")


if __name__ == "__main__":
    main()
