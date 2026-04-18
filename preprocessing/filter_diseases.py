from vllm import LLM, SamplingParams
import sqlite3
import json

DB_PATH = "data/llm_filtered.db"
MODEL_PATH = "/home/data1/musong/.cache/huggingface/hub/models--google--medgemma-1.5-4b-it/snapshots/e9792da5fb8ee651083d345ec4bce07c3c9f1641"
OUTPUT_PATH = "log/preprocessing/extracted_diseases.jsonl"  # Changed to .jsonl
CHUNK_SIZE = 1000

def build_prompt(title, abstract):
    # Abstract truncation to stay within context limits
    safe_abstract = abstract[:6000] if abstract else ""
    return f"""<|im_start|>system
You are an expert medical AI system. Extract the primary disease. Output ONLY a valid JSON list of strings (Max 3: Superclass, Subtype, Rare Variant).<|im_end|>
<|im_start|>user
Title: {title}
Abstract: {safe_abstract}
Output:<|im_end|>
<|im_start|>assistant
["""

def run_batch_extraction():
    print("Loading model into VRAM...")
    llm = LLM(
        model=MODEL_PATH, 
        dtype="bfloat16", 
        gpu_memory_utilization=0.90,
        max_model_len=4096 
    )
    
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=40, 
        stop=["<|im_end|>", "]"]
    )

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("Executing database query...")
    cursor.execute("SELECT pmc_id, year, title, abstract FROM publications WHERE category = 'Case Report' LIMIT 200000")
    
    print("🚀 Starting chunked inference...")
    
    # Open in 'a' (append) mode if you want to resume, or 'w' to overwrite
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        chunk_count = 0
        while True:
            rows = cursor.fetchmany(CHUNK_SIZE)
            if not rows:
                break
                
            chunk_count += 1
            print(f"\nProcessing Chunk {chunk_count} ({len(rows)} records)...")
            
            prompts = []
            metadata = [] 
            
            for row in rows:
                pmc_id, year, title, abstract = row[0], row[1], row[2], row[3]
                if title and abstract: 
                    prompts.append(build_prompt(title, abstract))
                    metadata.append((pmc_id, year))
            
            outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
            
            for (pmc_id, year), output in zip(metadata, outputs):
                # Reconstruct the JSON list since we provided the opening bracket in the prompt
                raw_text = "[" + output.outputs[0].text.strip()
                if not raw_text.endswith("]"): 
                    raw_text += "]"
                
                try:
                    diseases = json.loads(raw_text)
                    if not isinstance(diseases, list):
                        diseases = [str(diseases)]
                except json.JSONDecodeError:
                    # Fallback: keep the raw text in a list if JSON is malformed
                    diseases = [raw_text.strip("[]").strip()]
                
                # Create the structured record
                record = {
                    "pmc_id": pmc_id,
                    "year": year,
                    "diseases": diseases
                }
                
                # Write as a single line in the JSONL file
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            f.flush() 

    print(f"\n✅ All chunks processed. Data saved to {OUTPUT_PATH}")
    conn.close()

if __name__ == "__main__":
    run_batch_extraction()