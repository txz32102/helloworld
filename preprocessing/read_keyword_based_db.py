import sqlite3
import pandas as pd
import random
import os

# --- Configuration ---
DB_PATH = "data/keyword_filtered.db"

def print_random_sample(conn, seed=None):
    """Retrieves and prints every field of a random record from the keyword-filtered DB."""
    if seed is not None:
        random.seed(seed)
        
    print(f"\n🎲 FULL RECORD SAMPLE (Seed: {seed if seed else 'Random'})")
    print("=" * 80)
    
    cursor = conn.cursor()
    
    # 1. Get total count
    cursor.execute("SELECT COUNT(*) FROM publications")
    total = cursor.fetchone()[0]
    
    if total == 0:
        print("⚠️ No data available to sample.")
        return

    # 2. Get Column Names
    cursor.execute("SELECT * FROM publications LIMIT 1")
    columns = [description[0] for description in cursor.description]
    
    # 3. Pick random index
    random_idx = random.randint(0, total - 1)
    
    # 4. Fetch the full row
    cursor.execute(f"SELECT * FROM publications LIMIT 1 OFFSET {random_idx}")
    row = cursor.fetchone()
    record = dict(zip(columns, row))

    # 5. Separate metadata from long text for better display
    long_text_keys = ['title', 'abstract']
    metadata_keys = [k for k in columns if k not in long_text_keys]
    
    for key in metadata_keys:
        val = record.get(key)
        print(f"{key.upper():<15}: {val}")
    
    # 6. Print Title and Abstract with formatting
    for key in long_text_keys:
        print(f"\n{key.upper()}:")
        print("-" * 60)
        text = record.get(key) or "N/A"
        # Simple wrap-around for long abstracts
        print(text if len(text) < 500 else text[:500] + "...") 
        print("-" * 60)
        
    print("=" * 80)

def run_statistics():
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}. Please check the path.")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        
        # 1. Basic Counts
        total_records = conn.execute("SELECT COUNT(*) FROM publications").fetchone()[0]
        
        if total_records == 0:
            print(f"⚠️ The database at {DB_PATH} is empty.")
            return

        print(f"\n📊 Statistical Report for Keyword Filtered Data")
        print(f"Path: {DB_PATH}")
        print(f"Total Matches Found: {total_records:,}")
        
        # --- Random Sample ---
        print_random_sample(conn, seed=42)

        # 2. Top 10 Journals
        print("\n🏥 Top 10 Journals in Filtered Results:")
        query_journals = """
            SELECT journal, COUNT(*) as count 
            FROM publications 
            GROUP BY journal 
            ORDER BY count DESC 
            LIMIT 10
        """
        df_journals = pd.read_sql_query(query_journals, conn)
        for i, row in df_journals.iterrows():
            journal_name = (str(row['journal']) or "Unknown")[:50]
            pct = (row['count'] / total_records) * 100
            print(f"  {i+1:2}. {journal_name:<50} | {row['count']:>6,} ({pct:>5.1f}%)")

        # 3. Yearly Distribution (Top 10)
        print("\n📅 Top 10 Publication Years:")
        query_years = """
            SELECT year, COUNT(*) as count 
            FROM publications 
            WHERE year IS NOT NULL
            GROUP BY year 
            ORDER BY count DESC 
            LIMIT 10
        """
        df_years = pd.read_sql_query(query_years, conn)
        for i, row in df_years.iterrows():
            year_val = str(int(row['year']))
            pct = (row['count'] / total_records) * 100
            print(f"  {i+1:2}. {year_val:<15} | {row['count']:>8,} ({pct:>5.1f}%)")

        # 4. License Type Distribution
        print("\n📜 License Distribution:")
        query_license = """
            SELECT license, COUNT(*) as count 
            FROM publications 
            GROUP BY license 
            ORDER BY count DESC
        """
        df_license = pd.read_sql_query(query_license, conn)
        for _, row in df_license.iterrows():
            lic = str(row['license'] or "Unknown")
            pct = (row['count'] / total_records) * 100
            print(f"  - {lic:<20}: {row['count']:>8,} ({pct:>5.1f}%)")

        conn.close()
        print("\n" + "=" * 60)

    except sqlite3.Error as e:
        print(f"❌ SQLite error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_statistics()