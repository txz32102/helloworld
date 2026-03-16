import sqlite3
import re
import os
import requests
import time
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm

# LLM Imports (Loaded conditionally to save memory if you only want to run early stages)
import torch
from transformers import pipeline

# --- Configuration ---
DATA_DIR = "data"
INPUT_FILE = os.path.join(DATA_DIR, "oa_file_list.txt")
META_DB = os.path.join(DATA_DIR, "source_metadata.db")
ABSTRACT_DB_2026 = os.path.join(DATA_DIR, "title_abstract_db", "pub_abstracts_2026.db")
KEYWORD_DB = os.path.join(DATA_DIR, "keyword_filtered.db")
LLM_DB = os.path.join(DATA_DIR, "llm_filtered.db")

CURRENT_YEAR = 2026 # Focus updates on the active year

# Reusing your regex patterns
LINE_PATTERN = re.compile(
    r'^(?P<path>\S+)\s+(?P<journal>.+?)\.\s+(?P<date>\d{4}[^;]*);\s+'
    r'(?P<vol>[^:]+):(?P<pages>\S+)\s+PMC(?P<pmc>\d+)\s*(?:PMID:(?P<pmid>\d+))?\s*(?P<license>[\w\s-]+)$'
)
KEYWORDS = [r"case report", r"case reports", r"case study", r"case studies", r"clinical study"]
KW_PATTERN = re.compile(f"({'|'.join(KEYWORDS)})", re.IGNORECASE)

# --- Utilities ---
def get_existing_pmcs(db_path, table="publications"):
    """Quickly loads all PMC IDs from a database into a set for fast O(1) lookups."""
    if not os.path.exists(db_path):
        return set()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(f"SELECT pmc_id FROM {table}")
            return {row[0] for row in cursor.fetchall()}
        except sqlite3.OperationalError:
            return set() # Table doesn't exist yet

# --- Stage 0: Check and Download OA File List ---
def check_and_download_oa_list(url="https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.txt", dest_path=INPUT_FILE):
    print("\n--- 0. Checking for updates to OA File List ---")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    etag_file = dest_path + ".etag"
    local_etag = ""
    if os.path.exists(etag_file):
        with open(etag_file, "r") as f:
            local_etag = f.read().strip()

    try:
        # Use a HEAD request to fetch headers without downloading the file body
        response = requests.head(url, allow_redirects=True, timeout=10)
        response.raise_for_status()
        
        # Servers often send the MD5 hash in the 'ETag' header
        remote_etag = response.headers.get("ETag", "").strip('"')
        remote_modified = response.headers.get("Last-Modified", "")
        
        # Fallback to Last-Modified if ETag is stripped by proxies
        remote_version = remote_etag if remote_etag else remote_modified

        if os.path.exists(dest_path) and local_etag == remote_version and remote_version != "":
            print("✅ Local file is already up-to-date. Skipping download.")
            return

        print("🔄 File changed (or missing). Starting download...")
        
        # Proceed with streaming download
        with requests.get(url, stream=True, allow_redirects=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f, tqdm(
                desc="Downloading oa_file_list.txt", 
                total=total_size, 
                unit='iB', 
                unit_scale=True,
                unit_divisor=1024
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        size = f.write(chunk)
                        bar.update(size)
                        
        # Save the new ETag/Version so we don't download it again next time
        if remote_version:
            with open(etag_file, "w") as f:
                f.write(remote_version)
        print("✅ Download complete.")
        
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Failed to check or download the file: {e}")
        if not os.path.exists(dest_path):
            raise SystemExit("❌ No local file exists and download failed. Exiting.")
        print("⚠️ Proceeding with the existing local file.")

# --- Stage 1: Update Metadata ---
def update_metadata():
    print("\n--- 1. Updating Source Metadata ---")
    existing_pmcs = get_existing_pmcs(META_DB)
    
    conn = sqlite3.connect(META_DB)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS publications (
                        file_path TEXT, journal TEXT, pub_date TEXT, year INTEGER, 
                        volume TEXT, pages TEXT, pmc_id INTEGER PRIMARY KEY, 
                        pmid INTEGER, license TEXT)''')
    conn.execute("BEGIN")
    
    records_inserted = 0
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        next(f, None) # Skip header
        for line in tqdm(f, desc="Scanning for new lines"):
            line = line.strip()
            if not line or not line.startswith('oa_package'): continue
            
            match = LINE_PATTERN.search(line)
            if match:
                data = match.groupdict()
                try:
                    pmc_val = int(data['pmc'])
                    if pmc_val in existing_pmcs:
                        continue # Skip existing!
                    
                    year_val = int(data['date'][:4])
                    pmid_val = int(data['pmid']) if data['pmid'] else None
                except (ValueError, IndexError):
                    continue

                cursor.execute(
                    '''INSERT INTO publications 
                       (file_path, journal, pub_date, year, volume, pages, pmc_id, pmid, license) 
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (data['path'], data['journal'].strip(), data['date'].strip(), year_val, 
                     data['vol'].strip(), data['pages'].strip(), pmc_val, pmid_val, data['license'].strip())
                )
                records_inserted += 1
                if records_inserted % 50000 == 0:
                    conn.commit()
                    conn.execute("BEGIN")

    conn.commit()
    conn.close()
    print(f"✅ Inserted {records_inserted} new metadata records.")

# --- Stage 2: Update Abstracts ---
def fetch_batch_details(pmids, api_key="42d8712f11ddac302f4df11fe70ae9bb1709"):
    """Fetches PubMed details, utilizing an NCBI API key for higher rate limits."""
    clean_ids = [re.sub(r'\D', '', str(pid)) for pid in pmids]
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    params = {
        "db": "pubmed", 
        "id": ",".join(clean_ids), 
        "retmode": "xml"
    }
    
    if api_key:
        params["api_key"] = api_key
        
    results = {}
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        for article in root.findall(".//PubmedArticle"):
            pmid_node = article.find(".//PMID")
            if pmid_node is None: continue
            clean_pmid = pmid_node.text.strip()
            t_node = article.find(".//ArticleTitle")
            title = "".join(t_node.itertext()).strip() if t_node is not None else "N/A"
            abstract_parts = [f"{n.get('Label')}: {''.join(n.itertext()).strip()}" if n.get("Label") else "".join(n.itertext()).strip() for n in article.findall(".//AbstractText")]
            results[clean_pmid] = ("\n".join(abstract_parts) if abstract_parts else "N/A", title)
    except Exception as e:
        print(f"API Error: {e}")
    return results

def update_abstracts():
    print(f"\n--- 2. Updating Abstracts for {CURRENT_YEAR} ---")
    meta_conn = sqlite3.connect(META_DB)
    meta_conn.row_factory = sqlite3.Row
    
    abstract_pmcs = get_existing_pmcs(ABSTRACT_DB_2026)
    
    cursor = meta_conn.cursor()
    cursor.execute("SELECT * FROM publications WHERE year = ? AND pmid IS NOT NULL", (CURRENT_YEAR,))
    
    pending_records = [dict(row) for row in cursor.fetchall() if row["pmc_id"] not in abstract_pmcs]
    meta_conn.close()
    
    if not pending_records:
        print("✅ Abstracts are up to date.")
        return

    print(f"🔍 Found {len(pending_records)} new records to fetch.")
    
    dest_conn = sqlite3.connect(ABSTRACT_DB_2026)
    dest_conn.execute('''CREATE TABLE IF NOT EXISTS publications (
                        file_path TEXT, journal TEXT, pub_date TEXT, year INTEGER, 
                        volume TEXT, pages TEXT, pmc_id INTEGER PRIMARY KEY, 
                        pmid INTEGER, license TEXT, title TEXT, abstract TEXT)''')
                        
    BATCH_SIZE = 200
    
    for i in tqdm(range(0, len(pending_records), BATCH_SIZE), desc="Fetching from NCBI"):
        batch = pending_records[i:i+BATCH_SIZE]
        batch_pmids = [str(r['pmid']) for r in batch]
        pubmed_data = fetch_batch_details(batch_pmids)
        
        insert_payload = []
        for r in batch:
            abstract, title = pubmed_data.get(str(r['pmid']), ("N/A", "N/A"))
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
        time.sleep(0.4) # Be nice to NCBI

    dest_conn.close()

# --- Stage 3: Update Keyword Filter ---
def update_keywords():
    print("\n--- 3. Updating Keyword Filter ---")
    kw_pmcs = get_existing_pmcs(KEYWORD_DB)
    
    if not os.path.exists(ABSTRACT_DB_2026):
        print("⚠️ Abstract DB not found. Skipping keyword update.")
        return
        
    abs_conn = sqlite3.connect(ABSTRACT_DB_2026)
    abs_conn.row_factory = sqlite3.Row
    abs_cursor = abs_conn.cursor()
    
    abs_cursor.execute("SELECT * FROM publications")
    pending_records = [dict(r) for r in abs_cursor.fetchall() if r["pmc_id"] not in kw_pmcs]
    abs_conn.close()
    
    if not pending_records:
        print("✅ Keyword DB is up to date.")
        return
        
    print(f"🔍 Found {len(pending_records)} new abstracts to regex filter.")
    
    out_conn = sqlite3.connect(KEYWORD_DB)
    out_cursor = out_conn.cursor()
    out_cursor.execute('''CREATE TABLE IF NOT EXISTS publications (
                        file_path TEXT, journal TEXT, pub_date TEXT, year INTEGER, 
                        volume TEXT, pages TEXT, pmc_id INTEGER PRIMARY KEY, 
                        pmid INTEGER, license TEXT, title TEXT, abstract TEXT)''')
    
    batch_data = []
    for r in pending_records:
        if KW_PATTERN.search(str(r['title'] or "")) or KW_PATTERN.search(str(r['abstract'] or "")):
            batch_data.append((
                r['file_path'], r['journal'], r['pub_date'], r['year'], r['volume'], 
                r['pages'], r['pmc_id'], r['pmid'], r['license'], r['title'], r['abstract']
            ))
            
    if batch_data:
        out_cursor.executemany("""
            INSERT INTO publications (
                file_path, journal, pub_date, year, volume, 
                pages, pmc_id, pmid, license, title, abstract
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, batch_data)
        out_conn.commit()
        
    out_conn.close()
    print(f"✅ Added {len(batch_data)} new keyword-matched records.")

# --- Stage 4: Update LLM Filter ---
def update_llm():
    print("\n--- 4. Updating LLM Predictions ---")
    llm_pmcs = get_existing_pmcs(LLM_DB)
    
    if not os.path.exists(KEYWORD_DB):
        print("⚠️ Keyword DB not found. Skipping LLM update.")
        return
        
    kw_conn = sqlite3.connect(KEYWORD_DB)
    kw_conn.row_factory = sqlite3.Row
    kw_cursor = kw_conn.cursor()
    kw_cursor.execute("SELECT * FROM publications")
    
    pending_records = [dict(r) for r in kw_cursor.fetchall() if r["pmc_id"] not in llm_pmcs]
    kw_conn.close()
    
    if not pending_records:
        print("✅ LLM DB is up to date.")
        return
        
    print(f"🧠 Found {len(pending_records)} new records for LLM inference. Loading model...")
    
    pipe = pipeline("text-generation", model="/home/data1/musong/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1", device=0, torch_dtype=torch.bfloat16, trust_remote_code=True)
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    
    out_conn = sqlite3.connect(LLM_DB)
    out_cursor = out_conn.cursor()
    out_cursor.execute('''CREATE TABLE IF NOT EXISTS publications (
                        file_path TEXT, journal TEXT, pub_date TEXT, year INTEGER, 
                        volume TEXT, pages TEXT, pmc_id INTEGER PRIMARY KEY, 
                        pmid INTEGER, license TEXT, title TEXT, abstract TEXT,
                        category TEXT, is_case_report INTEGER, rarity_level TEXT, 
                        reasoning TEXT, raw_response TEXT)''')

    def data_generator():
        for rec in pending_records:
            prompt = (
                "You are an expert clinical researcher. First, classify this medical abstract EXACTLY into ONE of the following string categories (DO NOT deviate from these exact strings):\n"
                "1. 'Case Report': Detailed study of a single patient or a very small group (<3 patients).\n"
                "2. 'Group Study': Clinical trials, cohorts, or case-control studies involving many participants.\n"
                "3. 'Review': Systematic reviews, meta-analyses, or general literature summaries.\n"
                "4. 'Non-Clinical': Animal/in-vitro studies or basic laboratory experiments.\n"
                "5. 'Other': Editorials, study protocols, conference announcements, or ambiguous texts.\n\n"
                "SPECIAL INSTRUCTION FOR CASE REPORTS: If the abstract is a 'Case Report', determine its level of rarity. "
                "Classify 'rarity_level' STRICTLY into ONE of these exact string values (No other values are allowed):\n"
                "- 'rare_disease': The underlying disease/condition itself is epidemiologically rare (e.g., orphan diseases, rare genetic anomalies).\n"
                "- 'unprecedented_presentation': Extremely rare. A 'first-of-its-kind' report, a completely novel manifestation of a known disease, or a highly bizarre anatomical variant.\n"
                "- 'atypical_presentation': Uncommon but not unprecedented. A known disease presenting with uncommon features, in an unusual demographic, or in a less typical location.\n"
                "- 'standard_case': A relatively standard teaching case without major novel or rare elements.\n"
                "If it is NOT a Case Report, set 'rarity_level' strictly to 'NA'.\n\n"
                f"Title: {rec.get('title', '')}\n"
                f"Abstract: {(rec.get('abstract') or 'None')[:2500]}\n\n"
                "Output ONLY a valid JSON object. No markdown formatting, no explanations outside the JSON. Format:\n"
                "{\n"
                "  \"category\": \"string\",\n"
                "  \"is_case_report\": boolean,\n"
                "  \"rarity_level\": \"string\",\n"
                "  \"reasoning\": \"string\"\n"
                "}"
            )
            yield pipe.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

    saved = 0
    for i, output in enumerate(tqdm(pipe(data_generator(), batch_size=32, max_new_tokens=250, return_full_text=False), total=len(pending_records))):
        rec = pending_records[i]
        raw = output[0]['generated_text'].strip()
        try:
            clean_json = re.search(r'\{.*\}', raw, re.DOTALL).group(0)
            parsed = json.loads(clean_json)
            out_cursor.execute('''
                INSERT INTO publications (
                    file_path, journal, pub_date, year, volume, pages, pmc_id, pmid, license, title, abstract,
                    category, is_case_report, rarity_level, reasoning, raw_response
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rec['file_path'], rec['journal'], rec['pub_date'], rec['year'], rec['volume'], rec['pages'], 
                rec['pmc_id'], rec['pmid'], rec['license'], rec['title'], rec['abstract'],
                parsed.get("category"), 1 if parsed.get("is_case_report") else 0, parsed.get("rarity_level"), 
                parsed.get("reasoning"), raw
            ))
            saved += 1
            if saved % 50 == 0: out_conn.commit()
        except Exception:
            pass # Skipping parse/validation errors silently for the automated updater
            
    out_conn.commit()
    out_conn.close()
    print(f"✅ Processed and saved {saved} new LLM predictions.")

if __name__ == "__main__":
    print("🚀 Starting Incremental Pipeline Sync...")
    check_and_download_oa_list()
    update_metadata()
    update_abstracts()
    update_keywords()
    update_llm()
    print("🏁 Pipeline Sync Complete!")