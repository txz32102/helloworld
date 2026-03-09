import sqlite3
import os
import requests
import argparse
import random
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# --- Configuration ---
BASE_S3_URL = "https://pmc-oa-opendata.s3.amazonaws.com"
PUBMED_URL_TEMPLATE = "https://pubmed.ncbi.nlm.nih.gov/{}/"

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
    except Exception:
        pass
    return False

def process_article(row_data, output_dir):
    """Worker to create JSON metadata file and download S3 assets."""
    data = {k.lower(): v for k, v in dict(row_data).items()}
    
    pmid = data.get('pmid')
    pmc_id = str(data.get('pmc_id', ''))
    
    # Use PMID as folder name, fallback to PMC_ID if PMID is missing
    folder_name = str(pmid) if pmid else pmc_id
    if not folder_name:
        return # Skip if we have no identifiers
        
    article_folder = os.path.join(output_dir, folder_name)
    os.makedirs(article_folder, exist_ok=True)

    # 1. Write metadata
    metadata_path = os.path.join(article_folder, "metadata.json")
    
    # Try to parse the raw string response into actual JSON if it exists
    if 'raw_response' in data and isinstance(data['raw_response'], str):
        try:
            data['raw_response'] = json.loads(data['raw_response'])
        except json.JSONDecodeError:
            pass # Keep as string if it fails
            
    # Add external links cleanly into the JSON structure
    data['external_links'] = {
        "pubmed_url": PUBMED_URL_TEMPLATE.format(pmid) if pmid else None,
        "aws_s3_prefix": f"{pmc_id}.1" if pmc_id else None
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    # 2. Fetch and Download S3 Files
    pmc_str = f"PMC{pmc_id}" if not pmc_id.startswith("PMC") else pmc_id
    metadata_url = f"{BASE_S3_URL}/metadata/{pmc_str}.1.json"
    
    try:
        res = requests.get(metadata_url, timeout=15)
        if res.status_code == 200:
            meta_json = res.json()
            s3_paths = []
            if meta_json.get('pdf_url'): s3_paths.append(meta_json.get('pdf_url'))
            if meta_json.get('xml_url'): s3_paths.append(meta_json.get('xml_url'))
            if meta_json.get('media_urls'): s3_paths.extend(meta_json.get('media_urls'))

            for s3_path in s3_paths:
                # Convert s3:// paths to https:// public URLs
                clean_path = s3_path.replace("s3://pmc-oa-opendata/", "").split("?")[0]
                download_url = f"{BASE_S3_URL}/{clean_path}"
                download_file(download_url, article_folder)
        else:
            os.makedirs("log", exist_ok=True)
            with open("log/download_errors.log", "a") as log:
                log.write(f"Metadata 404 for PMID {pmid} (PMC {pmc_str})\n")
    except Exception as e:
        with open("download_errors.log", "a") as log:
            log.write(f"Error processing PMID {pmid}: {str(e)}\n")

def main():
    parser = argparse.ArgumentParser()
    # Updated default DB path
    parser.add_argument("--db", type=str, default="data/llm_filtered.db")
    parser.add_argument("--out", type=str, default="data/download_files")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8) 
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if not os.path.exists(args.db):
        print(f"Error: DB not found at {args.db}")
        return

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row  
    cursor = conn.cursor()
    
    # Dynamically find the table name (in case it's not 'case_reports')
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = cursor.fetchall()
    if not tables:
        print("Error: No tables found in the database.")
        return
    table_name = tables[0]['name']
    
    # Filter using the new YEAR column instead of LIKE '2020%'
    query = f"""
    SELECT * FROM {table_name} 
    WHERE is_case_report = 1 
    AND YEAR >= 2020
    """
    
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
    except sqlite3.OperationalError as e:
        print(f"Database error: {e}")
        print("Please ensure your table has 'is_case_report' and 'YEAR' columns.")
        return
    finally:
        conn.close()

    if not rows:
        print("No records found matching the criteria (is_case_report=1, Year >= 2020).")
        return

    selected_rows = random.sample(rows, min(len(rows), args.limit))
    print(f"Found {len(rows)} matching articles. Sampling {len(selected_rows)} (Seed: {args.seed})...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(tqdm(
            executor.map(lambda row: process_article(row, args.out), selected_rows),
            total=len(selected_rows),
            desc="Downloading"
        ))

    print(f"\nSuccess! Files saved to: {args.out}")

if __name__ == "__main__":
    main()