import sqlite3
import re
import os
from tqdm import tqdm

# --- Configuration ---
INPUT_FILE = "data/oa_file_list.txt"
LOG_DIR = "data"
# Directly defining the database path as requested
DB_PATH = os.path.join(LOG_DIR, "source_metadata.db")

# Pattern for PMC open access file list
LINE_PATTERN = re.compile(
    r'^(?P<path>\S+)\s+'           
    r'(?P<journal>.+?)\.\s+'        
    r'(?P<date>\d{4}[^;]*);\s+'    
    r'(?P<vol>[^:]+):'             
    r'(?P<pages>\S+)\s+'           
    r'PMC(?P<pmc>\d+)\s*'          
    r'(?:PMID:(?P<pmid>\d+))?\s*'  
    r'(?P<license>[\w\s-]+)$'      
)

def setup_db(db_path):
    """Initializes the SQLite database with optimized settings."""
    conn = sqlite3.connect(db_path)
    # WAL mode significantly improves write performance
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL") 
    
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS publications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT,
            journal TEXT,
            pub_date TEXT,
            year INTEGER,
            volume TEXT,
            pages TEXT,
            pmc_id INTEGER,
            pmid INTEGER,
            license TEXT
        )
    ''')
    # Indexing PMC ID and Year for fast lookups
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pmc ON publications(pmc_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_year ON publications(year)")
    conn.commit()
    return conn

def count_lines(filename):
    """Quickly count lines for the progress bar."""
    print(f"📊 Analyzing {filename}...")
    with open(filename, 'rb') as f:
        return sum(1 for _ in f)

def process_file():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    total_lines = count_lines(INPUT_FILE)
    
    # Connect to the single unified database
    conn = setup_db(DB_PATH)
    cursor = conn.cursor()
    
    print(f"🚀 Processing {total_lines} lines into {DB_PATH}...")

    # Start an explicit transaction
    conn.execute("BEGIN")
    
    records_inserted = 0

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        pbar = tqdm(total=total_lines, desc="Indexing PMCs", unit='line')
        
        # Skip header
        next(f, None)
        pbar.update(1)
        
        for line in f:
            pbar.update(1)
            line = line.strip()
            if not line or not line.startswith('oa_package'):
                continue

            match = LINE_PATTERN.search(line)
            if match:
                data = match.groupdict()
                
                try:
                    year_val = int(data['date'][:4])
                    pmc_val = int(data['pmc'])
                    pmid_val = int(data['pmid']) if data['pmid'] else None
                except (ValueError, IndexError):
                    continue

                cursor.execute(
                    '''INSERT INTO publications 
                       (file_path, journal, pub_date, year, volume, pages, pmc_id, pmid, license) 
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (
                        data['path'], 
                        data['journal'].strip(), 
                        data['date'].strip(), 
                        year_val, 
                        data['vol'].strip(), 
                        data['pages'].strip(), 
                        pmc_val, 
                        pmid_val, 
                        data['license'].strip()
                    )
                )
                
                records_inserted += 1

                # Batch commit every 50,000 records for speed
                if records_inserted % 50000 == 0:
                    conn.commit()
                    conn.execute("BEGIN")

        pbar.close()

    conn.commit()
    conn.close()

    print("\n" + "="*30)
    print(f"✅ DONE! Total records indexed: {records_inserted}")
    print(f"📁 Database location: {DB_PATH}")

if __name__ == "__main__":
    process_file()