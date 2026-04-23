import sqlite3
import os
import requests
import argparse
import random
import json
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

# --- Configuration ---
BASE_S3_URL = "https://pmc-oa-opendata.s3.amazonaws.com"
PUBMED_URL_TEMPLATE = "https://pubmed.ncbi.nlm.nih.gov/{}/"
SAFE_LICENSES = ['CC BY', 'CC BY-SA', 'CC0']
TARGET_YEARS = [2026]

def download_file(url, folder):
    """Helper to download a single file into a folder."""
    filename = os.path.basename(url.split("?")[0])
    filepath = os.path.join(folder, filename)
    
    if os.path.exists(filepath):
        return True

    try:
        response = requests.get(url, timeout=30, stream=True)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        return f"HTTP {response.status_code}"
    except Exception as e:
        return str(e)

def process_article(row_data, output_dir):
    """Worker to create JSON metadata file and download S3 assets."""
    data = {k.lower(): v for k, v in dict(row_data).items()}
    pmid = data.get('pmid')
    pmc_id = str(data.get('pmc_id', ''))
    
    article_id = str(pmid) if pmid else pmc_id
    if not article_id:
        return ("Unknown", "No valid ID found in database row")
        
    article_folder = os.path.join(output_dir, article_id)
    os.makedirs(article_folder, exist_ok=True)

    try:
        # 1. Write metadata
        metadata_path = os.path.join(article_folder, "metadata.json")
        if 'raw_response' in data and isinstance(data['raw_response'], str):
            try:
                data['raw_response'] = json.loads(data['raw_response'])
            except json.JSONDecodeError:
                pass 
        
        data['external_links'] = {
            "pubmed_url": PUBMED_URL_TEMPLATE.format(pmid) if pmid else None,
            "aws_s3_prefix": f"{pmc_id}.1" if pmc_id else None
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        # 2. Fetch and Download S3 Files
        pmc_str = f"PMC{pmc_id}" if not pmc_id.startswith("PMC") else pmc_id
        metadata_url = f"{BASE_S3_URL}/metadata/{pmc_str}.1.json"
        
        res = requests.get(metadata_url, timeout=15)
        if res.status_code != 200:
            return (article_id, f"S3 Metadata missing (HTTP {res.status_code})")

        meta_json = res.json()
        s3_paths = []
        if meta_json.get('pdf_url'): s3_paths.append(meta_json.get('pdf_url'))
        if meta_json.get('xml_url'): s3_paths.append(meta_json.get('xml_url'))
        if meta_json.get('media_urls'): s3_paths.extend(meta_json.get('media_urls'))

        failures = []
        for s3_path in s3_paths:
            clean_path = s3_path.replace("s3://pmc-oa-opendata/", "").split("?")[0]
            download_url = f"{BASE_S3_URL}/{clean_path}"
            result = download_file(download_url, article_folder)
            if result is not True:
                failures.append(f"{os.path.basename(clean_path)} failed: {result}")

        if failures:
            return (article_id, " | ".join(failures))
        
        return None # Success

    except Exception as e:
        return (article_id, f"Critical Error: {str(e)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default="data/llm_filtered.db")
    parser.add_argument("--out", type=str, default="data/2026")
    parser.add_argument("--log_dir", type=str, default="log/preprocessing", help="Directory for failure logs")
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=8) 
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    
    # 🛠 Ensure directories exist
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    if not os.path.exists(args.db):
        print(f"❌ Error: DB not found at {args.db}")
        return

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row  
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = cursor.fetchall()
    if not tables:
        print("❌ Error: No tables found.")
        return
    table_name = tables[0]['name']
    
    license_placeholders = ", ".join([f"'{lic}'" for lic in SAFE_LICENSES])
    year_placeholders = ", ".join([str(y) for y in TARGET_YEARS])

    query = f"""
    SELECT * FROM {table_name} 
    WHERE is_case_report = 1 
    AND YEAR IN ({year_placeholders})
    AND license IN ({license_placeholders})
    """
    
    print(f"🔍 Searching database for years: {TARGET_YEARS}")
    cursor.execute(query)
    all_rows = cursor.fetchall()
    conn.close()

    if not all_rows:
        print("⚠️ No matching records found.")
        return

    rows_by_year = defaultdict(list)
    for row in all_rows:
        rows_by_year[row['year']].append(row)

    # Balanced Stratified Sampling
    samples_per_year = max(1, args.limit // len(TARGET_YEARS))
    selected_rows = []

    print(f"📊 Balancing target (~{samples_per_year} per year)...")
    for year in TARGET_YEARS:
        year_pool = rows_by_year.get(year, [])
        count_to_take = min(len(year_pool), samples_per_year)
        if year_pool:
            selected_rows.extend(random.sample(year_pool, count_to_take))
            print(f"   - {year}: Sampled {count_to_take}")

    random.shuffle(selected_rows)
    print(f"✅ Total articles to process: {len(selected_rows)}")

    # --- Execution ---
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(
            executor.map(lambda row: process_article(row, args.out), selected_rows),
            total=len(selected_rows),
            desc="Downloading"
        ))
        
        # Filter for failures
        failed_logs = [r for r in results if r is not None]

    # --- Write Failure Log ---
    if failed_logs:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(args.log_dir, f"failures_{timestamp}.txt")
        
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"DOWNLOAD FAILURE LOG\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Total Attempted: {len(selected_rows)}\n")
            f.write(f"Total Failures: {len(failed_logs)}\n")
            f.write("="*60 + "\n\n")
            
            for article_id, reason in failed_logs:
                f.write(f"ID: {article_id:<15} | REASON: {reason}\n")
        
        print(f"\n⚠️ {len(failed_logs)} articles failed. Details saved to: {log_path}")
    else:
        print("\n🚀 All downloads completed with 100% success!")

if __name__ == "__main__":
    main()