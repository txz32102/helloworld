import os
import sqlite3
import requests
import time
import re
import logging
from datetime import datetime
import xml.etree.ElementTree as ET

# --- Configuration ---
SOURCE_DB = "data/source_metadata.db"
OUTPUT_DIR = "data/title_abstract_db"

# https://www.ncbi.nlm.nih.gov/myncbi/
# https://account.ncbi.nlm.nih.gov/settings/
# API_KEY = "92a4e59ca5b7525f720d6568e363b2a5b308" 
API_KEY = None
YEAR_RANGE = [2020, 2027]
BATCH_SIZE = 200
SLEEP_TIME = 0.1 if API_KEY else 0.4 
MAX_RETRIES = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/preprocessing", exist_ok=True)
# --- Logging Setup ---
log_file = os.path.join(f"{OUTPUT_DIR}/preprocessing", f"crawler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def fetch_batch_details(pmids, attempt=1):
    clean_ids = [re.sub(r'\D', '', str(pid)) for pid in pmids]
    ids_str = ",".join(clean_ids)

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": ids_str, "retmode": "xml"}
    if API_KEY: params["api_key"] = API_KEY

    results = {}
    try:
        response = requests.get(url, params=params, timeout=1)
        
        if response.status_code == 429:
            wait = 10 * attempt
            logger.warning(f"Rate limited (429). Waiting {wait}s...")
            time.sleep(wait)
            return fetch_batch_details(pmids, attempt + 1)

        response.raise_for_status()
        root = ET.fromstring(response.content)

        for article in root.findall(".//PubmedArticle"):
            pmid_node = article.find(".//PMID")
            if pmid_node is None: continue
            
            clean_pmid = pmid_node.text.strip()
            t_node = article.find(".//ArticleTitle")
            title = "".join(t_node.itertext()).strip() if t_node is not None else "N/A"

            abstract_parts = []
            for abstract_node in article.findall(".//AbstractText"):
                label = abstract_node.get("Label")
                text = "".join(abstract_node.itertext()).strip()
                abstract_parts.append(f"{label}: {text}" if label else text)
            
            abstract = "\n".join(abstract_parts) if abstract_parts else "N/A"
            results[clean_pmid] = (title, abstract)

    except (requests.exceptions.RequestException, ET.ParseError) as e:
        if attempt <= MAX_RETRIES:
            logger.warning(f"Retry {attempt}/{MAX_RETRIES} for batch. Error: {type(e).__name__}")
            time.sleep(5 * attempt)
            return fetch_batch_details(pmids, attempt + 1)
        else:
            logger.error(f"BATCH FAILED PERMANENTLY: {type(e).__name__}. Skipping.")
            return {} 

    return results

def setup_dest_db(year):
    db_path = os.path.join(OUTPUT_DIR, f"pub_abstracts_{year}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute('''
        CREATE TABLE IF NOT EXISTS publications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT, journal TEXT, pub_date TEXT, year INTEGER,
            volume TEXT, pages TEXT, pmc_id INTEGER, pmid INTEGER,
            license TEXT, title TEXT, abstract TEXT
        )
    ''')
    conn.commit()
    return conn

def get_already_processed_count(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM publications")
        return cursor.fetchone()[0]
    except:
        return 0

def run_crawler():
    if not os.path.exists(SOURCE_DB):
        logger.error(f"Source database {SOURCE_DB} not found!")
        return

    src_conn = sqlite3.connect(SOURCE_DB)
    src_cursor = src_conn.cursor()
    src_cursor.execute("SELECT DISTINCT year FROM publications WHERE year BETWEEN ? AND ? ORDER BY year DESC", 
                       (YEAR_RANGE[0], YEAR_RANGE[1]))
    years = [row[0] for row in src_cursor.fetchall()]

    for year in years:
        logger.info(f"Starting Year: {year}")
        src_cursor.execute("SELECT * FROM publications WHERE year = ? AND pmid IS NOT NULL", (year,))
        columns = [description[0] for description in src_cursor.description]
        records = [dict(zip(columns, row)) for row in src_cursor.fetchall()]
        
        total_in_year = len(records)
        if total_in_year == 0: continue

        dest_conn = setup_dest_db(year)
        
        # RESUME LOGIC: Check how many records already exist in the target DB
        processed_count = get_already_processed_count(dest_conn)
        if processed_count >= total_in_year:
            logger.info(f"Year {year} already fully processed ({processed_count} records). Skipping.")
            dest_conn.close()
            continue
        elif processed_count > 0:
            logger.info(f"Resuming Year {year} from record {processed_count}...")

        year_start_time = time.time()

        for i in range(processed_count, total_in_year, BATCH_SIZE):
            batch = records[i : i + BATCH_SIZE]
            batch_pmids = [str(r['pmid']) for r in batch]
            pubmed_data = fetch_batch_details(batch_pmids)

            insert_payload = []
            for r in batch:
                pmid_str = str(r['pmid'])
                title, abstract = pubmed_data.get(pmid_str, ("N/A", "N/A"))
                insert_payload.append((
                    r['file_path'], r['journal'], r['pub_date'], r['year'],
                    r['volume'], r['pages'], r['pmc_id'], r['pmid'], 
                    r['license'], title, abstract
                ))

            dest_conn.executemany('''
                INSERT INTO publications 
                (file_path, journal, pub_date, year, volume, pages, pmc_id, pmid, license, title, abstract)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', insert_payload)
            dest_conn.commit()

            processed_count += len(batch)
            elapsed = time.time() - year_start_time
            speed = processed_count / elapsed if elapsed > 0 else 0
            
            if processed_count % (BATCH_SIZE * 5) == 0 or processed_count >= total_in_year:
                logger.info(f"Year {year} Progress: {processed_count}/{total_in_year} ({speed:.1f} rec/s)")
            
            time.sleep(SLEEP_TIME)

        dest_conn.close()
        logger.info(f"✅ Finished Year {year}")

    src_conn.close()
    logger.info("🏁 Finished processing range.")

if __name__ == "__main__":
    run_crawler()