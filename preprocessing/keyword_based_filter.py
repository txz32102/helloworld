import sqlite3
import os
import re
from datetime import datetime

# --- Configuration ---
DB_DIR = "data/title_abstract_db"
OUTPUT_PATH = "data/keyword_filtered.db"

# Ensure output directory exists
os.makedirs("data", exist_ok=True)

# Keywords and Regex
KEYWORDS = [r"case report", r"case reports", r"case study", r"case studies", r"clinical study"]
PATTERN = re.compile(f"({'|'.join(KEYWORDS)})", re.IGNORECASE)

def setup_output_db():
    conn = sqlite3.connect(OUTPUT_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS publications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT, journal TEXT, pub_date TEXT, year INTEGER,
            volume TEXT, pages TEXT, pmc_id INTEGER, pmid INTEGER,
            license TEXT, title TEXT, abstract TEXT
        )
    """)
    conn.commit()
    return conn

def process_databases():
    out_conn = setup_output_db()
    out_cursor = out_conn.cursor()

    # Get source databases
    db_files = sorted([f for f in os.listdir(DB_DIR) if f.endswith(".db")])

    print(f"🚀 Filtering started at: {datetime.now().strftime('%H:%M:%S')}")
    print(f"💾 Output: {OUTPUT_PATH}\n" + "-" * 60)

    yearly_stats = []

    for db_name in db_files:
        db_path = os.path.join(DB_DIR, db_name)
        
        # Prevent self-processing if the output DB is in the same folder
        if os.path.abspath(db_path) == os.path.abspath(OUTPUT_PATH):
            continue
            
        try:
            with sqlite3.connect(db_path) as source_conn:
                source_cursor = source_conn.cursor()
                
                # Get total count
                source_cursor.execute("SELECT COUNT(*) FROM publications")
                total_in_db = source_cursor.fetchone()[0]
                if total_in_db == 0: continue

                # Stream and Filter
                source_cursor.execute("""
                    SELECT file_path, journal, pub_date, year, volume, 
                           pages, pmc_id, pmid, license, title, abstract 
                    FROM publications
                """)
                
                # Filter rows: title is index 9, abstract is index 10
                batch_data = [
                    row for row in source_cursor 
                    if PATTERN.search(str(row[9] or "")) or PATTERN.search(str(row[10] or ""))
                ]

                if batch_data:
                    out_cursor.executemany("""
                        INSERT INTO publications (
                            file_path, journal, pub_date, year, volume, 
                            pages, pmc_id, pmid, license, title, abstract
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, batch_data)
                    out_conn.commit()

                match_count = len(batch_data)
                percent = (match_count / total_in_db) * 100 if total_in_db > 0 else 0
                year_label = re.search(r"(\d{4})", db_name).group(1) if re.search(r"(\d{4})", db_name) else db_name
                
                yearly_stats.append({'year': year_label, 'matches': match_count, 'total': total_in_db, 'percent': percent})
                print(f"✅ {db_name:25} | Matches: {match_count:<6} | {percent:.2f}%")
            
        except Exception as e:
            print(f"❌ Error in {db_name}: {e}")

    # Summary Table
    print("\n" + "="*65)
    print(f"{'YEAR':<10} | {'MATCHES':<12} | {'TOTAL':<12} | {'PERCENT'}")
    print("-" * 65)
    for s in yearly_stats:
        print(f"{s['year']:<10} | {s['matches']:<12,} | {s['total']:<12,} | {s['percent']:.4f}%")
    
    out_conn.close()

if __name__ == "__main__":
    process_databases()