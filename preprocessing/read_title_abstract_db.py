import sqlite3
import os
import random

# --- Configuration ---
# You can change this path to any of your year-specific databases
DB_PATH = "data/title_abstract_db/pub_abstracts_2026.db"

def read_samples(db_path, seed=42, sample_size=5):
    if not os.path.exists(db_path):
        print(f"❌ Error: Database not found at:\n   {db_path}")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 1. Get total record count
        cursor.execute("SELECT COUNT(*) FROM publications")
        total_records = cursor.fetchone()[0]
        
        print(f"📊 Database Report")
        print(f"==================================================")
        print(f"File: {os.path.basename(db_path)}")
        print(f"Total records in DB: {total_records:,}")
        
        if total_records == 0:
            print("⚠️ The database is empty.")
            return

        # Handle small databases
        actual_sample_size = min(sample_size, total_records)
        print(f"Randomly sampling {actual_sample_size} records (Seed: {seed})...\n")

        # 2. Get all IDs for seeded random sampling
        cursor.execute("SELECT id FROM publications")
        all_ids = [row[0] for row in cursor.fetchall()]
        
        random.seed(seed)
        sampled_ids = random.sample(all_ids, actual_sample_size)

        # 3. Retrieve full data for the selected IDs
        placeholders = ', '.join(['?'] * len(sampled_ids))
        query = f"SELECT * FROM publications WHERE id IN ({placeholders})"
        cursor.execute(query, sampled_ids)
        
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]

        # 4. Print Results with FULL Title and FULL Abstract
        for i, row in enumerate(rows, 1):
            print(f"--- SAMPLE #{i} (Internal ID: {row[0]}) ---")
            
            # Map column names to values
            record = dict(zip(columns, row))
            
            # Print standard metadata first
            meta_keys = [k for k in columns if k not in ['title', 'abstract']]
            for k in meta_keys:
                print(f"{k:10}: {record[k]}")
            
            print(f"\ntitle:")
            print(f"{record['title']}")
            
            print(f"\nabstract:")
            print(f"{record['abstract']}")
            
            print("\n" + "="*60 + "\n")

    except sqlite3.Error as e:
        print(f"❌ SQLite error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    read_samples(DB_PATH)