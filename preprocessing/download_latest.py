import sqlite3
import os
import requests
import argparse
import random
import json
import time
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Configuration ---
NCBI_API_KEY = "42d8712f11ddac302f4df11fe70ae9bb1709"  # <--- PASTE YOUR API KEY HERE
BASE_S3_URL = "https://pmc-oa-opendata.s3.amazonaws.com"
PUBMED_URL_TEMPLATE = "https://pubmed.ncbi.nlm.nih.gov/{}/"
SAFE_LICENSES = ['CC BY', 'CC BY-SA', 'CC0']
TARGET_YEARS = [2026] 
TARGET_DATE = datetime(2026, 2, 1) 

def create_resilient_session():
    """Creates a requests Session that automatically retries on connection drops."""
    session = requests.Session()
    # Retry on specific errors: 429 (Too Many Requests), 500, 502, 503, 504
    # backoff_factor=1 means it will sleep for [0.5, 1, 2, 4...] seconds between retries
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def parse_ncbi_date(date_str):
    """
    Parses NCBI date strings like '2026 Feb 2', '2026/02/01', or '2026'.
    Returns None if date_str is empty or unparseable.
    """
    if not date_str or not isinstance(date_str, str):
        return None
    
    # Standardize separators
    clean_date = date_str.replace('/', ' ').replace('-', ' ').strip()
    parts = clean_date.split()
    
    if not parts:
        return None

    try:
        # 1. Handle Year
        year = int(parts[0])
        month = 1
        day = 1
        
        # 2. Handle Month (Supports "Feb", "02", or "2")
        if len(parts) > 1:
            month_part = parts[1]
            if month_part.isdigit():
                month = int(month_part)
            else:
                month_str = month_part[:3].title()
                months = {
                    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
                }
                month = months.get(month_str, 1)
            
        # 3. Handle Day
        if len(parts) > 2:
            day_part = parts[2]
            if day_part.isdigit():
                day = int(day_part)
            
        return datetime(year, month, day)
    except Exception:
        return None

def fetch_valid_dates_from_api(pmc_ids):
    """
    Strictly verifies that the 'epubdate' exists and is >= Feb 1, 2026.
    Skips any article missing 'epubdate' or using only 'pubdate'.
    """
    valid_ids = set()
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    session = create_resilient_session()
    
    chunk_size = 30 
    for i in tqdm(range(0, len(pmc_ids), chunk_size), desc="Strict EpubDate Verification"):
        chunk = pmc_ids[i:i + chunk_size]
        params = {
            "db": "pmc",
            "id": ",".join(chunk),
            "retmode": "json",
            "api_key": NCBI_API_KEY
        }
        
        try:
            res = session.get(base_url, params=params, timeout=20)
            if res.status_code == 200:
                data = res.json()
                result_dict = data.get("result", {})
                
                for pmcid in chunk:
                    article_data = result_dict.get(str(pmcid), {})
                    if not isinstance(article_data, dict):
                        continue
                    
                    # --- THE STRICT FILTER ---
                    # We ONLY look at epubdate. 
                    # If this field is empty, the article is rejected.
                    epub_date_raw = article_data.get("epubdate")
                    
                    if epub_date_raw:
                        parsed_date = parse_ncbi_date(epub_date_raw)
                        # Ensure the electronic timestamp is Feb 1, 2026 or later
                        if parsed_date and parsed_date >= TARGET_DATE:
                            valid_ids.add(str(pmcid))
                    # else:
                    #     print(f"Skipping PMC{pmcid}: Missing epubdate")
            
            # API Key allows 10 req/s. 0.2s is a safe balance for throughput.
            time.sleep(0.2) 
            
        except requests.exceptions.RequestException as e:
            print(f"❌ API Error on chunk {i}: {e}")
            
    return valid_ids

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
        
        return None 

    except Exception as e:
        return (article_id, f"Critical Error: {str(e)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default="data/llm_filtered.db")
    parser.add_argument("--out", type=str, default="data/download_files")
    parser.add_argument("--log_dir", type=str, default="log/preprocessing")
    parser.add_argument("--workers", type=int, default=8) 
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
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

    query = f"""
    SELECT * FROM {table_name} 
    WHERE is_case_report = 1 
    AND YEAR >= 2026 
    AND license IN ({license_placeholders})
    """
    
    print(f"🔍 Phase 1: Querying local DB for baseline year >= 2026...")
    cursor.execute(query)
    all_rows = cursor.fetchall()
    conn.close()

    if not all_rows:
        print("⚠️ No matching records found in database.")
        return

    raw_candidates = []
    pmcid_list_for_api = []
    
    for row in all_rows:
        row_dict = dict(row)
        pmc_raw = str(row_dict.get('pmc_id', ''))
        pmc_num = pmc_raw.replace('PMC', '')
        
        pmid = row_dict.get('pmid')
        article_id = str(pmid) if pmid else pmc_raw
        if os.path.exists(os.path.join(args.out, article_id, "metadata.json")):
            continue 
            
        if pmc_num.isdigit():
            raw_candidates.append(row)
            pmcid_list_for_api.append(pmc_num)

    print(f"📊 Found {len(pmcid_list_for_api)} new candidate articles. Checking dates via NCBI API...")

    verified_pmc_ids = fetch_valid_dates_from_api(pmcid_list_for_api)
    
    final_rows_to_download = [
        row for row in raw_candidates 
        if str(dict(row).get('pmc_id', '')).replace('PMC', '') in verified_pmc_ids
    ]

    print(f"✅ Phase 2 complete. {len(final_rows_to_download)} articles published strictly after Feb 1, 2026.")

    if not final_rows_to_download:
        print("🚀 No new articles to download. Exiting.")
        return

    print("\n⬇️ Phase 3: Downloading S3 Metadata and Assets...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(
            executor.map(lambda row: process_article(row, args.out), final_rows_to_download),
            total=len(final_rows_to_download),
            desc="Downloading"
        ))
        
        failed_logs = [r for r in results if r is not None]

    if failed_logs:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(args.log_dir, f"failures_{timestamp}.txt")
        
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"DOWNLOAD FAILURE LOG\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Total Attempted: {len(final_rows_to_download)}\n")
            f.write(f"Total Failures: {len(failed_logs)}\n")
            f.write("="*60 + "\n\n")
            
            for article_id, reason in failed_logs:
                f.write(f"ID: {article_id:<15} | REASON: {reason}\n")
        
        print(f"\n⚠️ {len(failed_logs)} articles failed. Details saved to: {log_path}")
    else:
        print("\n🚀 All downloads completed with 100% success!")

if __name__ == "__main__":
    main()