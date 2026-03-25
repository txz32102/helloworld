import os
import re
import json
import argparse
import statistics
from pathlib import Path
from collections import defaultdict

def parse_evaluation_markdown(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex patterns to extract the exact text regardless of spacing
    gt_pattern = r'-\s+\*\*(Report [A-Z])\*\*\s+was the Ground Truth'
    gen_pattern = r'-\s+\*\*(Report [A-Z])\*\*\s+was the Generated Report'
    guess_pattern = r'\*\*LLM Guessed Ground Truth is:\*\*\s*(Report [A-Z])'
    header_pattern = r'\|\s*Metric\s*\|\s*(Report [A-Z])\s*\|\s*(Report [A-Z])\s*\|'
    avg_pattern = r'\|\s*\*\*AVERAGE\*\*\s*\|\s*\*\*([0-9.]+)(?:/[0-9.]+)?\*\*\s*\|\s*\*\*([0-9.]+)(?:/[0-9.]+)?\*\*\s*\|'
    
    # Pattern to capture individual metric rows (ignores header and average via logic below)
    row_pattern = r'\|\s*([^|]+?)\s*\|\s*([0-9.]+)(?:/[0-9.]+)?\s*\|\s*([0-9.]+)(?:/[0-9.]+)?\s*\|'

    # Extract identities and tables
    gt_match = re.search(gt_pattern, content, re.IGNORECASE)
    gen_match = re.search(gen_pattern, content, re.IGNORECASE)
    guess_match = re.search(guess_pattern, content, re.IGNORECASE)
    header_match = re.search(header_pattern, content)
    avg_match = re.search(avg_pattern, content)

    # Check if any required piece of information is missing
    missing = []
    if not gt_match: missing.append("Ground Truth Identity")
    if not gen_match: missing.append("Generated Identity")
    if not guess_match: missing.append("LLM Guess")
    if not header_match: missing.append("Table Header")
    if not avg_match: missing.append("Average Scores")

    if missing:
        raise ValueError(f"Missing formatting in markdown: {', '.join(missing)}")

    # Map the identities
    gt_report = gt_match.group(1)
    gen_report = gen_match.group(1)
    llm_guess = guess_match.group(1)

    is_correct = (gt_report == llm_guess)

    # Map the table columns to the extracted identities
    col1_name = header_match.group(1)
    col2_name = header_match.group(2)
    col1_score = float(avg_match.group(1))
    col2_score = float(avg_match.group(2))

    scores = {col1_name: col1_score, col2_name: col2_score}

    if gt_report not in scores or gen_report not in scores:
        raise ValueError("Mismatch between table column headers and Report identities.")

    # Extract individual metrics
    detailed_metrics = {}
    row_matches = re.findall(row_pattern, content)
    for match in row_matches:
        metric_name = match[0].strip()
        # Skip the Header row or Average row if matched
        if 'Metric' in metric_name or 'AVERAGE' in metric_name.upper():
            continue
            
        c1_val = float(match[1])
        c2_val = float(match[2])
        
        # Map back to GT / Gen so it's consistent regardless of which column Report A/B is in
        temp_scores = {col1_name: c1_val, col2_name: c2_val}
        detailed_metrics[metric_name] = {
            "gt_score": temp_scores[gt_report],
            "gen_score": temp_scores[gen_report]
        }

    return {
        "gt_report": gt_report,
        "gen_report": gen_report,
        "llm_guessed_gt": llm_guess,
        "correctly_identified": is_correct,
        "gt_score": scores[gt_report],
        "gen_score": scores[gen_report],
        "metrics": detailed_metrics  # Attach the extracted rows here
    }

def main():
    parser = argparse.ArgumentParser(description="Calculate evaluation metrics from LLM pipeline logs.")
    parser.add_argument(
        "--folder", 
        type=str, 
        default="/home/data1/musong/workspace/2026/03/07/helloworld/log/pipeline_gpt-5.4/20260325_095032"
        
        # qwen3.5-27b api path
        # /home/data1/musong/workspace/2026/03/22/log/pipeline_qwen3.5-27b/20260322_173947
        
        # gpt4.1 path
        # /home/data1/musong/workspace/2026/03/07/helloworld/log/pipeline_gpt-4.1/20260325_095032
        
        # gpt5.4 path
        # /home/data1/musong/workspace/2026/03/07/helloworld/log/pipeline_gpt-5.4/20260325_095032
        
        # qwen3.5-27b fp8 local path
        # /home/data1/musong/workspace/2026/03/22/log/pipeline_Qwen-Qwen3.5-27B-FP8/20260324_161911
    )
    args = parser.parse_args()

    base_dir = Path(args.folder)
    
    if not base_dir.exists() or not base_dir.is_dir():
        print(f"Error: Directory '{base_dir}' does not exist.")
        return

    log_data = {
        "summary": {},
        "details": {}
    }

    stats = {
        "total_subfolders": 0,
        "processed_success": 0,
        "missing_files": 0,
        "parsing_errors": 0,
        "correct_identifications": 0,
        "total_gt_score": 0.0,
        "total_gen_score": 0.0
    }
    
    # Track scores for individual metrics to calculate mean/std dev later
    metric_aggregations = defaultdict(lambda: {"gt": [], "gen": []})

    # Iterate through items in the base directory
    for item in os.listdir(base_dir):
        subfolder_path = base_dir / item
        
        if subfolder_path.is_dir():
            stats["total_subfolders"] += 1
            expected_md_file = subfolder_path / f"{item}_evaluation.md"

            if not expected_md_file.exists():
                print(f"[Warning] Missing file: {expected_md_file.name}")
                stats["missing_files"] += 1
                log_data["details"][item] = {
                    "status": "error", 
                    "reason": "Missing evaluation.md file"
                }
                continue

            try:
                # Parse the file
                parsed_data = parse_evaluation_markdown(expected_md_file)
                parsed_data["status"] = "success"
                log_data["details"][item] = parsed_data
                
                # Update running statistics
                stats["processed_success"] += 1
                if parsed_data["correctly_identified"]:
                    stats["correct_identifications"] += 1
                stats["total_gt_score"] += parsed_data["gt_score"]
                stats["total_gen_score"] += parsed_data["gen_score"]
                
                # Append individual metric scores for later std dev / mean calculation
                for m_name, m_scores in parsed_data["metrics"].items():
                    metric_aggregations[m_name]["gt"].append(m_scores["gt_score"])
                    metric_aggregations[m_name]["gen"].append(m_scores["gen_score"])

            except Exception as e:
                print(f"[Error] Parsing failed for {item}: {str(e)}")
                stats["parsing_errors"] += 1
                log_data["details"][item] = {
                    "status": "error", 
                    "reason": f"Parsing failure - {str(e)}"
                }

    # Calculate final overall averages
    if stats["processed_success"] > 0:
        avg_gt_score = stats["total_gt_score"] / stats["processed_success"]
        avg_gen_score = stats["total_gen_score"] / stats["processed_success"]
        accuracy = (stats["correct_identifications"] / stats["processed_success"]) * 100
    else:
        avg_gt_score = avg_gen_score = accuracy = 0.0

    # Calculate Mean & Standard Deviation for individual row metrics
    calculated_metrics = {}
    for m_name, score_lists in metric_aggregations.items():
        gt_scores = score_lists["gt"]
        gen_scores = score_lists["gen"]
        
        # Mean
        gt_mean = statistics.mean(gt_scores) if gt_scores else 0.0
        gen_mean = statistics.mean(gen_scores) if gen_scores else 0.0
        
        # Stdev (Requires at least 2 data points)
        gt_std = statistics.stdev(gt_scores) if len(gt_scores) > 1 else 0.0
        gen_std = statistics.stdev(gen_scores) if len(gen_scores) > 1 else 0.0
        
        calculated_metrics[m_name] = {
            "gt": f"{gt_mean:.2f} (±{gt_std:.2f})",
            "gen": f"{gen_mean:.2f} (±{gen_std:.2f})"
        }

    # Compile Summary
    log_data["summary"] = {
        "total_folders_checked": stats["total_subfolders"],
        "successfully_parsed": stats["processed_success"],
        "missing_files": stats["missing_files"],
        "format_errors": stats["parsing_errors"],
        "llm_identification_accuracy": f"{accuracy:.2f}% ({stats['correct_identifications']}/{stats['processed_success']})",
        "average_ground_truth_score": round(avg_gt_score, 2),
        "average_generated_score": round(avg_gen_score, 2),
        "individual_metrics_breakdown": calculated_metrics
    }

    # Write out JSON
    output_json_path = base_dir / "metrics_summary.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4)

    # Print to console
    print("\n" + "="*50)
    print("📊 METRICS SUMMARY")
    print("="*50)
    for key, value in log_data["summary"].items():
        if key == "individual_metrics_breakdown":
            print("\n📈 INDIVIDUAL METRICS (Mean ± StdDev):")
            print("-" * 50)
            print(f"{'Metric':<25} | {'Ground Truth':<15} | {'Generated':<15}")
            print("-" * 50)
            for m_name, m_data in value.items():
                print(f"{m_name:<25} | {m_data['gt']:<15} | {m_data['gen']:<15}")
        else:
            formatted_key = key.replace('_', ' ').title()
            print(f"{formatted_key}: {value}")
    print("="*50)
    print(f"✅ Detailed JSON logged to: {output_json_path}")

if __name__ == "__main__":
    main()