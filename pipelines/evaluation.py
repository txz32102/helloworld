import os
import glob
import argparse
import json
import time
from datetime import datetime

from .utils import setup_proxy, get_openai_client, encode_image

class EvaluatorPipeline:
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
        Reads the MD files and images for a case, evaluates them with an LLM, 
        and logs the execution details.
        """
        print(f"Evaluating Case ID: {case_id}...")
        
        case_dir = os.path.join(self.base_dir, case_id)
        gt_path = os.path.join(case_dir, f"{case_id}_gt.md")
        gen_path = os.path.join(case_dir, f"{case_id}_generated.md")
        imgs_dir = os.path.join(case_dir, "imgs")
        log_output_path = os.path.join(case_dir, "eval_execution_log.json")
        
        # Initialize execution log perfectly matching the schema
        execution_log = {
            "case_id": case_id,
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "execution_time_seconds": 0.0,
            "error_message": None,
            "tokens": {
                "prompt": 0,
                "completion": 0,
                "total": 0
            },
            "raw_prompt": None,
            "raw_response": None
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
            
            # Save log before exiting
            execution_log["execution_time_seconds"] = round(time.time() - start_time, 2)
            with open(log_output_path, "w", encoding="utf-8") as f:
                json.dump(execution_log, f, indent=4)
            return

        # Gather all images in the imgs folder
        image_paths = glob.glob(os.path.join(imgs_dir, "*.*"))
        
        # 2. Build the message payload
        content_payload = [
            {
                "type": "text",
                "text": (
                    "You are an expert medical reviewer. I will provide you with a 'Ground Truth' "
                    "case report and a 'Generated' case report. I am also providing all the images "
                    "referenced in these reports.\n\n"
                    "Please evaluate the Generated report against the Ground Truth based on:\n"
                    "1. Clinical Accuracy (Are the medical facts, diagnoses, and treatments consistent?)\n"
                    "2. Image Referencing (Does the generated text accurately describe the attached images compared to the ground truth?)\n"
                    "3. Completeness (Are any critical details omitted in the generation?)\n"
                    "4. Hallucination (Did the generation add false medical details?)\n\n"
                    "Provide a structured evaluation and a final score out of 10."
                )
            },
            {
                "type": "text",
                "text": f"--- GROUND TRUTH REPORT ---\n{gt_text}\n"
            },
            {
                "type": "text",
                "text": f"--- GENERATED REPORT ---\n{gen_text}\n"
            },
            {
                "type": "text",
                "text": "--- REFERENCED IMAGES ---"
            }
        ]

        # Append all images as Base64 encoded strings
        for img_path in image_paths:
            base64_image = encode_image(img_path)
            mime_type = "image/jpeg" if img_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
            
            content_payload.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_image}",
                    "detail": "high"
                }
            })

        # Safely capture the prompt payload for the log (omitting giant base64 strings)
        safe_payload = []
        for item in content_payload:
            if item["type"] == "text":
                safe_payload.append(item)
            elif item["type"] == "image_url":
                safe_payload.append({"type": "image_url", "image_url": "[BASE64_IMAGE_DATA_OMITTED_FOR_LOG]"})
        
        execution_log["raw_prompt"] = safe_payload

        # 3. Call OpenAI API
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": content_payload
                    }
                ],
                max_tokens=1500,
                temperature=0.2
            )
            
            raw_content = response.choices[0].message.content
            usage = response.usage
            
            # Update log with API return
            execution_log["raw_response"] = raw_content
            execution_log["tokens"]["prompt"] = usage.prompt_tokens
            execution_log["tokens"]["completion"] = usage.completion_tokens
            execution_log["tokens"]["total"] = usage.total_tokens

            # 4. Save the evaluation markdown
            eval_output_path = os.path.join(case_dir, f"{case_id}_evaluation.md")
            with open(eval_output_path, "w", encoding="utf-8") as f:
                f.write(raw_content)
                
            execution_log["status"] = "success"
            
            print(f"✅ Evaluation saved to {eval_output_path}")
            print(f"   Tokens - Total: {usage.total_tokens} | Time: {time.time() - start_time:.2f}s\n")

        except Exception as e:
            execution_log["error_message"] = str(e)
            print(f"[X] API Error for {case_id}: {e}\n")

        finally:
            # 5. ALWAYS save the full log file
            execution_log["execution_time_seconds"] = round(time.time() - start_time, 2)
            with open(log_output_path, "w", encoding="utf-8") as f:
                json.dump(execution_log, f, indent=4)

    def run(self):
        """
        Executes the batch evaluation loop.
        """
        if not os.path.exists(self.base_dir):
            print(f"Directory not found: {self.base_dir}")
            return

        # Get all subdirectories (which are the case IDs)
        case_ids = [d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))]
        
        print(f"Starting evaluation for {len(case_ids)} cases...")
        print(f"Model ID: {self.model_id}")
        print("-" * 40)
        
        for case_id in case_ids:
            self.evaluate_case_report(case_id)
                
        print("-" * 40)
        print("Batch evaluation complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated case reports against ground truth.")
    parser.add_argument("--base_dir", type=str, default="/home/data1/musong/workspace/2026/03/07/helloworld/log/generated",
                        help="Base directory containing the generated cases.")
    parser.add_argument("--model_id", type=str, default="gpt-4o",
                        help="The OpenAI multimodal model to use for evaluation.")
    return parser.parse_args()


if __name__ == "__main__":
    # 1. Setup proxy at the VERY BEGINNING of execution
    PROXY = "http://127.0.0.1:7890"
    setup_proxy(PROXY)
    
    # 2. Parse arguments
    args = parse_args()

    # 3. Initialize and run the pipeline
    evaluator = EvaluatorPipeline(
        base_dir=args.base_dir, 
        model_id=args.model_id
    )
    
    evaluator.run()