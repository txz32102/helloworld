import sqlite3
import pandas as pd
import random
import os

# --- Configuration ---
DB_PATH = "data/llm_filtered.db"

def print_random_sample(conn, seed=42):
    """Retrieves and prints EVERY field of a random record, formatting long texts nicely."""
    print(f"\n🎲 FULL RECORD SAMPLE (Seed: {seed})")
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
    random.seed(seed)
    random_idx = random.randint(0, total - 1)
    
    # 4. Fetch the full row
    cursor.execute(f"SELECT * FROM publications LIMIT 1 OFFSET {random_idx}")
    row = cursor.fetchone()
    record = dict(zip(columns, row))

    # 5. Print short metadata keys first
    long_text_keys = ['title', 'abstract', 'reasoning', 'raw_response']
    metadata_keys = [k for k in columns if k not in long_text_keys]
    
    for key in metadata_keys:
        val = record.get(key)
        print(f"{key.upper():<15}: {val}")
    
    # 6. Print Long Text Fields with dividers
    for key in long_text_keys:
        print(f"\n{key.upper()}:")
        print("-" * 60)
        print(record.get(key) or "N/A")
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
        # Get total CASE REPORTS for subset calculations
        total_case_reports = conn.execute("SELECT COUNT(*) FROM publications WHERE category = 'Case Report'").fetchone()[0]
        
        if total_records == 0:
            print(f"⚠️ The database at {DB_PATH} is empty.")
            return

        print(f"\n📊 Statistical Report for: {DB_PATH}")
        print(f"Total Filtered Records: {total_records:,}")
        print(f"Total Case Reports:     {total_case_reports:,}")
        
        # --- Random Sample ---
        print_random_sample(conn, seed=17)

        # 2. LLM Category Distribution (Same as before, based on total)
        print("\n🗂️ LLM Category Distribution (of Total Records):")
        df_categories = pd.read_sql_query("SELECT category, COUNT(*) as count FROM publications GROUP BY category ORDER BY count DESC", conn)
        for _, row in df_categories.iterrows():
            percentage = (row['count'] / total_records) * 100
            print(f"  - {str(row['category']):<15}: {row['count']:>8,}  ({percentage:>5.1f}%)")

        # 3. Case Report Rarity Levels (FILTERED)
        print("\n💎 Case Report Rarity Levels (Case Reports Only):")
        df_rarity = pd.read_sql_query("""
            SELECT rarity_level, COUNT(*) as count 
            FROM publications 
            WHERE category = 'Case Report' 
            GROUP BY rarity_level 
            ORDER BY count DESC
        """, conn)
        for _, row in df_rarity.iterrows():
            percentage = (row['count'] / total_case_reports) * 100 if total_case_reports > 0 else 0
            print(f"  - {str(row['rarity_level']):<28}: {row['count']:>8,}  ({percentage:>5.1f}%)")

        # 4. Top 10 Journals (FILTERED)
        print("\n🏥 Top 10 Journals (Case Reports Only):")
        df_journals = pd.read_sql_query("""
            SELECT journal, COUNT(*) as count 
            FROM publications 
            WHERE category = 'Case Report' 
            GROUP BY journal 
            ORDER BY count DESC 
            LIMIT 10
        """, conn)
        for i, row in df_journals.iterrows():
            journal_name = (str(row['journal']) or "Unknown")[:45]
            percentage = (row['count'] / total_case_reports) * 100 if total_case_reports > 0 else 0
            print(f"  {i+1:2}. {journal_name:<45} | {row['count']:>6,}  ({percentage:>5.1f}%)")
            
        # 5. Top 5 Years (FILTERED)
        print("\n📅 Top 5 Publication Years (Case Reports Only):")
        df_years = pd.read_sql_query("""
            SELECT year, COUNT(*) as count 
            FROM publications 
            WHERE year IS NOT NULL AND category = 'Case Report'
            GROUP BY year 
            ORDER BY count DESC 
            LIMIT 5
        """, conn)

        for i, row in df_years.iterrows():
            year_val = str(int(row['year'])) if row['year'] else "Unknown"
            percentage = (row['count'] / total_case_reports) * 100 if total_case_reports > 0 else 0
            print(f"  {i+1:2}. {year_val:<15} | {row['count']:>8,}  ({percentage:>5.1f}%)")

        conn.close()
        print("\n" + "=" * 60)

    except sqlite3.Error as e:
        print(f"❌ SQLite error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_statistics()