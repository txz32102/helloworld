import sqlite3
import os
import random

# --- Configuration ---
DB_PATH = "data/source_metadata.db"

def read_samples(db_path, seed=42, sample_size=5):
    if not os.path.exists(db_path):
        print(f"❌ Error: Database not found at {db_path}")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 1. Get total record count
        cursor.execute("SELECT COUNT(*) FROM publications")
        total_records = cursor.fetchone()[0]
        
        print(f"📊 Database Stats")
        print(f"-------------------------------")
        print(f"Total records found: {total_records:,}")
        print(f"Randomly sampling {sample_size} records (Seed: {seed})...\n")

        if total_records == 0:
            print("⚠️ The database is empty.")
            return

        # 2. Fetch 5 random samples
        # To ensure reproducibility with a seed, we'll fetch all IDs, 
        # seed Python's random, and pick 5 IDs.
        cursor.execute("SELECT id FROM publications")
        all_ids = [row[0] for row in cursor.fetchall()]
        
        random.seed(seed)
        sampled_ids = random.sample(all_ids, min(sample_size, len(all_ids)))

        # 3. Retrieve the full data for those IDs
        placeholders = ', '.join(['?'] * len(sampled_ids))
        query = f"SELECT * FROM publications WHERE id IN ({placeholders})"
        cursor.execute(query, sampled_ids)
        
        rows = cursor.fetchall()
        
        # Get column names for pretty printing
        columns = [description[0] for description in cursor.description]

        # 4. Print results
        for i, row in enumerate(rows, 1):
            print(f"--- Sample #{i} (ID: {row[0]}) ---")
            for col_name, value in zip(columns, row):
                print(f"{col_name:10}: {value}")
            print()

    except sqlite3.Error as e:
        print(f"❌ SQLite error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    read_samples(DB_PATH)