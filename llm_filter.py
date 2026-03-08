import os
import argparse

# --- 1. PRE-INIT GPU (Must be before Torch imports) ---
temp_parser = argparse.ArgumentParser(add_help=False)
temp_parser.add_argument("--gpu", type=str, default="1")
temp_args, _ = temp_parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = temp_args.gpu

import torch
import json
import re
import sqlite3
from tqdm import tqdm
from transformers import pipeline

# --- 2. VALIDATION CONSTANTS ---
VALID_CATEGORIES = {"Case Report", "Group Study", "Review", "Non-Clinical", "Other"}
VALID_RARITIES = {"rare_disease", "unprecedented_presentation", "atypical_presentation", "standard_case", "NA"}

# --- 3. DATABASE SETUP & UTILS ---
def setup_output_db(out_path: str):
    """Creates the output database with all original columns plus LLM results."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    conn = sqlite3.connect(out_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS publications (
            id INTEGER PRIMARY KEY,
            file_path TEXT, journal TEXT, pub_date TEXT, year INTEGER,
            volume TEXT, pages TEXT, pmc_id INTEGER, pmid INTEGER,
            license TEXT, title TEXT, abstract TEXT,
            category TEXT, is_case_report INTEGER, rarity_level TEXT, reasoning TEXT,
            raw_response TEXT
        )
    ''')
    conn.commit()
    return conn

# --- 4. INFERENCE ENGINE ---
def run_filter(args):
    print(f"🚀 Model: {args.model} | Using GPU Mask: {args.gpu}")
    print(f"📥 Input DB: {args.in_db}")
    print(f"📤 Output DB: {args.out_db}")
    
    # --- LOGGING SETUP ---
    os.makedirs("log", exist_ok=True)
    log_file_path = os.path.join("log", "skipped_records.log")
    
    # Initialize Pipeline
    pipe = pipeline(
        "text-generation", 
        model=args.model, 
        device=0,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    pipe.tokenizer.padding_side = "left"

    # Fetch Data
    in_conn = sqlite3.connect(args.in_db)
    in_conn.row_factory = sqlite3.Row
    in_cursor = in_conn.cursor()
    in_cursor.execute("SELECT * FROM publications")
    records = [dict(row) for row in in_cursor.fetchall()]
    in_conn.close()

    total = len(records)
    if total == 0:
        print("⚠️ Input database is empty. Exiting.")
        return
        
    print(f"📊 Loaded {total} records for processing.")

    # Output DB Connection
    out_conn = setup_output_db(args.out_db)
    out_cursor = out_conn.cursor()

    def data_generator():
        for rec in records:
            # STRONGER PROMPT: Emphasize EXACT string matching
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
            messages = [{"role": "user", "content": prompt}]
            yield pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Process Loop
    skipped_count = 0
    saved_count = 0

    # Open the log file for appending before the loop
    log_file = open(log_file_path, "a", encoding="utf-8")

    for i, output in enumerate(tqdm(pipe(data_generator(), batch_size=args.batch_size, max_new_tokens=250, return_full_text=False), 
                                    total=total, desc="Filtering Abstracts")):
        rec = records[i]
        raw_output = output[0]['generated_text'].strip()
        
        # 1. Parse JSON
        try:
            clean_json = re.search(r'\{.*\}', raw_output, re.DOTALL).group(0)
            parsed = json.loads(clean_json)
        except Exception:
            error_msg = f"⚠️ [ID: {rec['id']}] Parse Error. Skipping record. Raw Output: {raw_output}"
            print(f"\n{error_msg}")
            log_file.write(error_msg + "\n")
            log_file.flush() # Ensure it writes to disk immediately
            skipped_count += 1
            continue

        category = parsed.get("category")
        rarity_level = parsed.get("rarity_level")

        # 2. Validate Categories
        if category not in VALID_CATEGORIES or rarity_level not in VALID_RARITIES:
            error_msg = f"⚠️ [ID: {rec['id']}] Invalid LLM Output. Skipping. (Got category: '{category}', rarity: '{rarity_level}')\nRaw Output: {raw_output}"
            print(f"\n{error_msg}")
            log_file.write(error_msg + "\n")
            log_file.flush() # Ensure it writes to disk immediately
            skipped_count += 1
            continue

        # 3. Save to New DB (Preserving all original keys + new LLM keys)
        is_case_report = 1 if parsed.get("is_case_report") else 0
        
        try:
            out_cursor.execute('''
                INSERT INTO publications (
                    id, file_path, journal, pub_date, year, volume, pages, pmc_id, pmid, license, title, abstract,
                    category, is_case_report, rarity_level, reasoning, raw_response
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rec.get('id'), rec.get('file_path'), rec.get('journal'), rec.get('pub_date'), 
                rec.get('year'), rec.get('volume'), rec.get('pages'), rec.get('pmc_id'), 
                rec.get('pmid'), rec.get('license'), rec.get('title'), rec.get('abstract'),
                category, is_case_report, rarity_level, parsed.get("reasoning"), raw_output
            ))
            saved_count += 1
            
            # Commit every 100 records to save progress periodically
            if saved_count % 100 == 0:
                out_conn.commit()
                
        except sqlite3.Error as e:
            db_err_msg = f"❌ [ID: {rec['id']}] DB Insert Error: {e}"
            print(f"\n{db_err_msg}")
            log_file.write(db_err_msg + "\n")
            log_file.flush()

    # Final Commit & Cleanup
    out_conn.commit()
    out_conn.close()
    log_file.close()
    
    print("\n" + "="*50)
    print("🎉 Processing Complete!")
    print(f"Total Processed: {total}")
    print(f"Successfully Saved: {saved_count}")
    print(f"Skipped (Invalid/Errors): {skipped_count}")
    print(f"Output saved to: {args.out_db}")
    print(f"Logs saved to: {log_file_path}")
    print("="*50)

# --- 5. EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--in_db", type=str, default="data/keyword_filtered.db", help="Path to input database")
    parser.add_argument("--out_db", type=str, default="data/llm_filtered.db", help="Path to output database")
    
    args = parser.parse_args()
    run_filter(args)