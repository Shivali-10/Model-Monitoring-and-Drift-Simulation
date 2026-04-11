import os
import time
import json
import random
import requests
import pandas as pd
from dotenv import load_dotenv

# Load env variables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Critical Error: SUPABASE_URL or SUPABASE_KEY is missing in .env")
    exit(1)

# Pick an offline dataset to stream
DATA_FILE = os.path.join(BASE_DIR, "week4_drift.csv")

if not os.path.exists(DATA_FILE):
    # Fallback to week1 if week4 is missing
    DATA_FILE = os.path.join(BASE_DIR, "week1_baseline.csv")

if not os.path.exists(DATA_FILE):
    print(f"❌ Error: Could not find any CSV dataset to stream at {BASE_DIR}")
    exit(1)

def get_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

print(f"🚀 Starting Live Streamer...")
print(f"📁 Reading source data: {os.path.basename(DATA_FILE)} (This may take a moment)")

# Load the file once to save memory on 8GB laptops. We only keep 5000 rows in memory to be lightweight.
df = pd.read_csv(DATA_FILE, nrows=5000)
df = df.fillna(0)
records = df.to_dict(orient="records")

print("✅ Data loaded successfully. Commencing live streaming to Supabase!")
print("-" * 50)

batch_size = 1000  # Pushing a larger batch since it only runs weekly

# --- CLOUD AUTOMATION MODE (GITHUB ACTIONS) ---
if os.getenv("GITHUB_ACTIONS") == "true":
    print("☁️ Running in Cloud (GitHub Actions) - Single Shot Mode")
    # Take a random sample so it inserts different data every week
    batch = random.sample(records, min(batch_size, len(records)))
    try:
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/transactions",
            headers=get_headers(),
            json=batch,
            timeout=10
        )
        if response.status_code == 201:
            print(f"[LIVE] 📡 Synced {len(batch)} new transactions to Supabase!")
        else:
            print(f"[ERR] ❌ Failed to insert! HTTP Status: {response.status_code}")
    except Exception as e:
        print(f"[ERR] ❌ Network/Connection error: {e}")
    exit(0) # Exit immediately after pushing

# --- LOCAL DEMO MODE ---
print("💻 Running Locally - Continuous Stream Mode")
index = 0

try:
    while index < len(records):
        batch = records[index:index+batch_size]
        
        # Sleeps for exactly 1 week (7 days * 24 hrs * 60 mins * 60 secs)
        # Note: If you want to show this LIVE in an interview, change this back to 5!
        time.sleep(604800)
        
        try:
            response = requests.post(
                f"{SUPABASE_URL}/rest/v1/transactions",
                headers=get_headers(),
                json=batch,
                timeout=10
            )
            
            if response.status_code == 201:
                print(f"[LIVE] 📡 Synced {len(batch)} new transactions to Supabase... (Total session: {index+len(batch)})")
            else:
                print(f"[ERR] ❌ Failed to insert! HTTP Status: {response.status_code}")
                
        except Exception as e:
            print(f"[ERR] ❌ Network/Connection error: {e}")
            
        index += batch_size
        
        # Loop back to begining if we drain the 5000 rows
        if index >= len(records):
            index = 0
            random.shuffle(records) # Shuffle to keep the stream looking organic

except KeyboardInterrupt:
    print("\n🛑 Live Streamer stopped by user.")
