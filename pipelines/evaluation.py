import os
import glob
import json
import time
import random
from datetime import datetime

from .utils import get_openai_client, encode_image

class EvaluationPipeline:
    def __init__(self, base_dir: str, model_id: str):
        """
        Initializes the pipeline to evaluate generated case reports.
        Note: Proxy should be set globally before initializing this class.
        """
        self.base_dir = base_dir
        self.model_id = model_id
        
        # 1. OpenAI Client Initialization
        self.client = get_openai_client()

    def evaluate_case_report(self, case_id: str):
        """
        Reads the MD files and images for a case, evaluates them blindly (Round 1),
        performs an unblinded error analysis (Round 2), and logs execution details.
        """
        print(f"Evaluating Case ID: {case_id}...")
        
        case_dir = os.path.join(self.base_dir, case_id)
        gt_path = os.path.join(case_dir, f"{case_id}_gt.md")
        gen_path = os.path.join(case_dir, f"{case_id}_generated.md")
        imgs_dir = os.path.join(case_dir, "imgs")
        log_output_path = os.path.join(case_dir, "eval_execution_log.json")
        
        execution_log = {
            "case_id": case_id,
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "execution_time_seconds": 0.0,
            "blind_setup": {},
            "round_1_results": {},
            "round_2_results": {},
            "error_message": None,
            "tokens": {"prompt": 0, "completion": 0, "total": 0}
        }

        start_time = time.time()
        
        # Read Markdown files
        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_text = f.read()
            with open(gen_path, "r", encoding="utf-8") as f:
                gen_text = f.read()
        except FileNotFoundError as e:
            error_msg = f"Error reading markdown files for {case_id}: {e}"
            print(f"[!] {error_msg}")
            execution_log["error_message"] = error_msg
            return self._save_log(log_output_path, execution_log, start_time)

        # Gather all images
        image_paths = glob.glob(os.path.join(imgs_dir, "*.*"))
        
        # ==========================================
        # ROUND 1: BLINDED A/B TEST (Scoring)
        # ==========================================
        print("   -> Running Round 1: Blinded A/B Test...")
        is_gt_A = random.choice([True, False])
        if is_gt_A:
            report_a_text, report_b_text = gt_text, gen_text
            gt_label, gen_label = "Report A", "Report B"
        else:
            report_a_text, report_b_text = gen_text, gt_text
            gt_label, gen_label = "Report B", "Report A"

        execution_log["blind_setup"] = {"Ground_Truth": gt_label, "Generated": gen_label}

        r1_system_prompt = (
            "You are an expert, highly critical medical reviewer evaluating two case reports (Report A and Report B). "
            "One is a human-written Ground Truth; the other is an AI generation. "
            "Evaluate BOTH strictly on a 1-10 scale across the following 5 criteria:\n\n"
            "1. Citation Depth & Integration: Does the report feature detailed, comprehensive citations integrated heavily throughout the discussion to back up specific clinical claims? (LLMs often provide unnaturally short reference lists and sparse in-text citations).\n"
            "2. Patient History & Timeline Nuances: Does it capture the 'messy reality' of patient care? Look for specific timelines (e.g., patient agreed to intervention after X days, discontinued meds due to Y) versus a flattened, generalized narrative.\n"
            "3. Clinical Coherence & Critical Omissions: Does it provide an in-depth discussion on pathophysiology, specific radiographic criteria, and complex anatomical specifics? (LLMs often omit granular multi-system findings and generalize conditions).\n"
            "4. Image Description, Location & Order: Are the figures strategically placed in chronological order? Do the descriptions provide a comprehensive, multi-part breakdown (e.g., distinguishing 'a' and 'b' sections) of specific anatomical pathology?\n"
            "5. Readability & Structure: Professional medical formatting and logical section transitions.\n\n"
            "ABSOLUTE RULE ON IMAGES: The image filenames/paths are intentionally masked. DO NOT factor the filenames into your scores. "
            "Judge them purely on logical sequence and granular textual integration.\n\n"
            "Output JSON format strictly:\n"
            "{\n"
            '  "scores": {\n'
            '    "Report A": { "Citation_Depth": 0, "Patient_History_Nuance": 0, "Clinical_Coherence": 0, "Image_Integration": 0, "Readability": 0 },\n'
            '    "Report B": { "Citation_Depth": 0, "Patient_History_Nuance": 0, "Clinical_Coherence": 0, "Image_Integration": 0, "Readability": 0 }\n'
            "  },\n"
            '  "comments": { "Report A": "...", "Report B": "..." },\n'
            '  "guess": { "Ground_Truth": "Report A or Report B", "Reasoning": "..." }\n'
            "}"
        )

        r1_payload = [
            {"type": "text", "text": r1_system_prompt},
            {"type": "text", "text": f"--- REPORT A ---\n{report_a_text}\n"},
            {"type": "text", "text": f"--- REPORT B ---\n{report_b_text}\n"},
            {"type": "text", "text": "--- REFERENCED IMAGES ---"}
        ]
        
        for img_path in image_paths:
            base64_img = encode_image(img_path)
            mime = "image/jpeg" if img_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
            r1_payload.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64_img}", "detail": "high"}})

        # ==========================================
        # ROUND 2: UNBLINDED ERROR ANALYSIS
        # ==========================================
        print("   -> Running Round 2: Unblinded Error Analysis...")
        r2_system_prompt = (
            "You are an uncompromising medical auditor performing an Unblinded Error Analysis. "
            "Compare the AI-GENERATED report against the ORIGINAL Ground Truth report. "
            "You must aggressively hunt for the following specific LLM failure modes:\n"
            "- Flattened Timelines: Did the AI miss exact days, exact medication durations, or nuanced patient refusal/consent details?\n"
            "- Clinical Omissions: Did the AI summarize over complex radiographic criteria, precise anatomical measurements, or multi-system secondary diagnoses?\n"
            "- Shallow Citations: Did the AI fail to map specific literature to specific pathophysiological claims made in the Ground Truth?\n"
            "- Weak Image Integration: Did the AI fail to capture the granular breakdown of multi-part figures?\n\n"
            "ABSOLUTE RULE ON IMAGES: You MUST NOT mention the image name, file path, or hash (e.g., 'IMG_4EF82C3') anywhere in your output. "
            "We care strictly about the location and order of the images within the text, not their filenames. "
            "If an image is misplaced or poorly described, refer to it functionally (e.g., 'The first figure', 'Figure 1').\n\n"
            "Output JSON format strictly:\n"
            "{\n"
            '  "hallucinations": [{"issue": "False detail added", "severity": "High/Medium/Low"}],\n'
            '  "omissions": [{"issue": "Critical clinical or timeline detail missing", "severity": "High/Medium/Low"}],\n'
            '  "formatting_issues": ["List illogical structural placements, missing citation mapping, or weak image descriptions. NEVER mention image filenames."],\n'
            '  "improvement_advice": "Specific prompt engineering advice to fix these exact errors."\n'
            "}"
        )

        r2_payload = [
            {"type": "text", "text": r2_system_prompt},
            {"type": "text", "text": f"--- GROUND TRUTH REPORT ---\n{gt_text}\n"},
            {"type": "text", "text": f"--- GENERATED REPORT ---\n{gen_text}\n"},
            {"type": "text", "text": "--- REFERENCED IMAGES ---"}
        ]
        # (Reuse the exact same image payload block from above)
        for img_path in image_paths:
            base64_img = encode_image(img_path)
            mime = "image/jpeg" if img_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
            r2_payload.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64_img}", "detail": "high"}})

        # --- EXECUTE BOTH CALLS ---
        try:
            # R1 Call
            response_r1 = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": r1_payload}],
                max_tokens=2000,
                temperature=0.2,
                response_format={ "type": "json_object" }
            )
            r1_data = json.loads(response_r1.choices[0].message.content)
            
            # R2 Call
            response_r2 = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": r2_payload}],
                max_tokens=2000,
                temperature=0.1, # Lower temp for stricter auditing
                response_format={ "type": "json_object" }
            )
            r2_data = json.loads(response_r2.choices[0].message.content)

            # --- PROCESS RESULTS ---
            # Calculate Averages for R1
            scores_a = r1_data["scores"]["Report A"]
            avg_a = sum(scores_a.values()) / len(scores_a)
            scores_b = r1_data["scores"]["Report B"]
            avg_b = sum(scores_b.values()) / len(scores_b)
            
            guessed_correctly = (r1_data["guess"]["Ground_Truth"] == gt_label)
            
            # Log Data
            execution_log["round_1_results"] = r1_data
            execution_log["round_1_results"]["Averages"] = {"Report A": avg_a, "Report B": avg_b}
            execution_log["round_1_results"]["Guess_Correct"] = guessed_correctly
            execution_log["round_2_results"] = r2_data
            
            # Tokens Total
            execution_log["tokens"]["prompt"] = response_r1.usage.prompt_tokens + response_r2.usage.prompt_tokens
            execution_log["tokens"]["completion"] = response_r1.usage.completion_tokens + response_r2.usage.completion_tokens
            execution_log["tokens"]["total"] = execution_log["tokens"]["prompt"] + execution_log["tokens"]["completion"]
            
            # Generate the unified output
            md_output = self._generate_markdown_report(r1_data, r2_data, avg_a, avg_b, gt_label, gen_label, guessed_correctly)
            eval_output_path = os.path.join(case_dir, f"{case_id}_evaluation.md")
            
            with open(eval_output_path, "w", encoding="utf-8") as f:
                f.write(md_output)

            execution_log["status"] = "success"
            print(f"✅ Evaluation complete. Guessed GT correctly? {guessed_correctly}")
            print(f"   GT Score: {avg_a if is_gt_A else avg_b}/10 | Gen Score: {avg_b if is_gt_A else avg_a}/10\n")

        except Exception as e:
            execution_log["error_message"] = str(e)
            print(f"[X] API Error for {case_id}: {e}\n")

        finally:
            self._save_log(log_output_path, execution_log, start_time)

    def _generate_markdown_report(self, r1_data, r2_data, avg_a, avg_b, gt_label, gen_label, guessed_correctly):
        """Helper function to format both JSON responses into a readable Markdown report."""
        md = ["# Comprehensive Evaluation Report\n"]
        
        # ---------------- ROUND 1 ----------------
        md.append("## Part 1: Blinded A/B Test\n")
        md.append("### Identity Reveal")
        md.append(f"- **{gt_label}** was the Ground Truth.")
        md.append(f"- **{gen_label}** was the Generated Report.\n")
        
        guess_str = "✅ Correctly Identified" if guessed_correctly else "❌ Incorrectly Identified (Fooled!)"
        md.append("### LLM Identification Guess")
        md.append(f"**Status:** {guess_str}")
        md.append(f"**LLM Guessed Ground Truth is:** {r1_data['guess']['Ground_Truth']}")
        md.append(f"**Reasoning:** {r1_data['guess']['Reasoning']}\n")
        
        md.append("### Scores")
        md.append("| Metric | Report A | Report B |")
        md.append("|---|---|---|")
        for key in r1_data['scores']['Report A'].keys():
            val_a = r1_data['scores']['Report A'][key]
            val_b = r1_data['scores']['Report B'][key]
            md.append(f"| {key.replace('_', ' ')} | {val_a}/10 | {val_b}/10 |")
        md.append(f"| **AVERAGE** | **{avg_a:.2f}/10** | **{avg_b:.2f}/10** |\n")
        
        md.append("### Qualitative Feedback")
        md.append(f"**Report A:** {r1_data['comments']['Report A']}\n")
        md.append(f"**Report B:** {r1_data['comments']['Report B']}\n")
        
        md.append("---\n")
        
        # ---------------- ROUND 2 ----------------
        md.append("## Part 2: Unblinded Error Analysis")
        md.append("*(Comparing Generated Report directly against Ground Truth)*\n")
        
        md.append("### 🚩 Medical Hallucinations")
        for item in r2_data.get('hallucinations', []):
            md.append(f"- {item}")
        md.append("\n")
        
        md.append("### 🔍 Critical Omissions")
        for item in r2_data.get('omissions', []):
            md.append(f"- {item}")
        md.append("\n")
        
        md.append("### 📝 Formatting & Image Issues")
        for item in r2_data.get('formatting_issues', []):
            md.append(f"- {item}")
        md.append("\n")
        
        md.append("### 💡 Advice for Generation Pipeline")
        md.append(f"{r2_data.get('improvement_advice', '')}\n")
        
        return "\n".join(md)

    def _save_log(self, path, log_dict, start_time):
        """Helper function to save the execution log."""
        log_dict["execution_time_seconds"] = round(time.time() - start_time, 2)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(log_dict, f, indent=4)

    def run(self):
        """Executes the batch evaluation loop."""
        if not os.path.exists(self.base_dir):
            print(f"Directory not found: {self.base_dir}")
            return

        case_ids = [d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))]
        
        print(f"Starting evaluation for {len(case_ids)} cases...")
        print(f"Model ID: {self.model_id}")
        print("-" * 40)
        
        for case_id in case_ids:
            self.evaluate_case_report(case_id)
                
        print("-" * 40)
        print("Batch evaluation complete.")