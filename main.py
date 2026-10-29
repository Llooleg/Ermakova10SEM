#!/usr/bin/env python3
"""
Pulls the current AbuseIPDB blacklist (community-reported malicious IPs)
and dumps it to CSV. Needs an API key: https://www.abuseipdb.com/account/api
"""

import requests
import csv
import sys
import os

API_KEY = os.environ.get("ABUSEIPDB_KEY", "dcb9162b931613587d2997b0f359bd4c7baddeea0a2f10b91d8fbd1994af82bc2594e9f76220a7b0")
CONFIDENCE_MIN = 90     # how sure AbuseIPDB has to be before you trust it
LIMIT = 50000           # free tier will silently clamp this lower, don't @ me

def fetch_blacklist():
    url = "https://api.abuseipdb.com/api/v2/blacklist"
    headers = {"Key": API_KEY, "Accept": "application/json"}
    params = {"confidenceMinimum": CONFIDENCE_MIN, "limit": LIMIT}

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()["data"]

def save_csv(data, filename="abuseipdb_blacklist.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ipAddress", "abuseConfidenceScore", "lastReportedAt", 'countryCode'])
        writer.writeheader()
        writer.writerows(data)
    print(f"Saved {len(data)} IPs to {filename}. Congrats, you now own a list of the internet's worst neighbors.")

if __name__ == "__main__":
    if API_KEY == "YOUR_API_KEY_HERE":
        sys.exit("No API key set. Export ABUSEIPDB_KEY or hardcode it in like it's 2004. Either way, get one first.")
    try:
        data = fetch_blacklist()
        save_csv(data)
    except requests.HTTPError as e:
        sys.exit(f"AbuseIPDB said no: {e}")