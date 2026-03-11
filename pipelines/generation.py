import os
import re
import json
import glob
import shutil
import base64
import time
import uuid
import argparse
from datetime import datetime

from .utils import setup_proxy, get_openai_client

class GenerationPipeline:
    def __init__(self, working_dir: str, model_id: str):
        """
        Initializes the pipeline to generate medical journal articles from JSON atoms.
        Note: Proxy should be set globally before initializing this class.
        """
        self.working_dir = working_dir
        self.model_id = model_id
        
        # 1. OpenAI Client Initialization
        self.client = get_openai_client()

    def _encode_image(self, image_path: str) -> str:
        """
        Encodes image to base64 for multimodal input.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _copy_images(self, source_directory: str, destination_directory: str):
        """
        Safely copies images from the source dataset to the generation directory.
        """
        if not os.path.exists(source_directory):
            print(f"[-] Source directory not found for images: {source_directory}")
            return
            
        copied_count = 0
        for ext in ('*.jpg', '*.png', '*.jpeg'):
            for img_path in glob.glob(os.path.join(source_directory, ext)):
                try:
                    shutil.copy2(img_path, destination_directory)
                    copied_count += 1
                except Exception as e:
                    print(f"[!] Warning: Failed to copy image {img_path}: {e}")
        return copied_count

    def _build_prompt(self, case_id: str, case_data: dict, image_list: str, sections_str: str) -> str:
        """
        Constructs the text portion of the multimodal prompt with a highly academic persona.
        """
        # Automatically append a References section to the requirements if it isn't already there
        if "References" not in sections_str and "Citations" not in sections_str:
            sections_str += "\n- References"

        return f"""
        You are an expert medical researcher, clinician, and senior editor for a high-impact, peer-reviewed medical journal. Your task is to synthesize raw clinical data into a comprehensive, highly academic, and publication-ready clinical case report.

        CASE ID: {case_id}
        
        ### RAW CLINICAL DATA ###
        - Clinical History: {case_data.get('history')}
        - Presentation: {case_data.get('presentation')}
        - Diagnostics: {case_data.get('diagnostics')}
        - Management: {case_data.get('management')}
        - Outcome: {case_data.get('outcome')}
        - Available Image References: {image_list}

        ### WRITING GUIDELINES & ACADEMIC STANDARDS ###
        1. Tone & Style: The manuscript must be written in an authoritative, objective, and scholarly medical tone. Use precise clinical terminology. The depth of analysis should reflect a profound understanding of pathophysiology, epidemiology, and contemporary clinical guidelines. Target a comprehensive length (1500+ words).
        
        2. Context & Background Synthesis: Do not simply regurgitate the provided facts. You must embed this case within the broader medical context. 
           - In the Introduction, provide a robust background on the disease entity, its prevalence, and typical presentation. 
           - In the Discussion, extensively analyze the clinical decisions, compare this patient's presentation to standard presentations documented in medical literature, and explicitly state why this specific case adds value to the medical community.

        3. Mandatory Structure: You MUST organize your article using EXACTLY these section headings in this order:
        {sections_str}
        
        4. Title & Abstract: Create a sophisticated, highly specific academic title. The Abstract must be a compelling, unbroken single paragraph summarizing the clinical presentation, key interventions, and the primary educational takeaway.

        5. Figure Integration: Reference the provided images naturally within the text (e.g., "MRI of the cervical spine revealed a hyperintense lesion (Figure 1)"). Immediately following the paragraph where the figure is first mentioned, embed the figure EXACTLY using this Markdown syntax:
           ![Figure n](IMG_XXXXXX)
           > **Figure n:** [Provide a highly detailed, medically accurate radiographic, histological, or clinical descriptive caption].

        6. Academic Referencing: You must compile a formal "References" section at the conclusion of the manuscript. Cite highly relevant medical literature, clinical trials, or landmark papers that support the pathophysiology, diagnostic criteria, and treatment modalities discussed in your text. 
           - Use standard AMA (American Medical Association) citation style. 
           - Integrate these citations naturally using standard bracketed or superscript numbers in the body text (e.g., "...as demonstrated in recent literature [1].").
        """
    def process_case(self, json_path: str):
        """
        Processes a single case, formatting the prompt, passing images to the LLM,
        and logging the execution.
        """
        case_id = os.path.basename(json_path).split('_')[0]
        case_dir = os.path.join(self.working_dir, case_id)
        log_output_path = os.path.join(case_dir, "pipeline_execution_log.json")
        
        print(f"Processing Case ID: {case_id} | JSON Path: {json_path}\n")
        
        # Prepare the comprehensive execution log structure
        execution_log = {
            "step": "generate_journal",
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
            "system_prompt": None,
            "raw_prompt_payload": [], 
            "raw_response": None,
            "mapped_images": {}
        }
        
        start_time = time.time()

        try:
            # 1. Load the atoms JSON data
            with open(json_path, 'r', encoding='utf-8') as f:
                case_data = json.load(f)
                
            source_dir = case_data.get('metadata', {}).get('source_directory', 'Unknown')
            case_img_dir = os.path.join(case_dir, "imgs")
            
            paper_sections = case_data.get('metadata', {}).get('paper_sections_found', [])
            sections_str = "\n".join([f"- {sec}" for sec in paper_sections]) if paper_sections else "- Introduction\n- Case Presentation\n- Discussion"
            
            # Setup directories
            os.makedirs(case_dir, exist_ok=True)
            os.makedirs(case_img_dir, exist_ok=True)
            
            # Copy images to local folder
            self._copy_images(source_dir, case_img_dir)

            # 2. Image Gathering and Natural Sort
            all_imgs = []
            for ext in ('*.jpg', '*.png', '*.jpeg'):
                all_imgs.extend(glob.glob(os.path.join(source_dir, ext)))
            
            # Natural sort (ensures 10.png comes after 2.png)
            all_imgs.sort(key=lambda f: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', f)])

            # 3. Create the Random Mapping and Build Payload
            image_map = {}
            prompt_text = self._build_prompt(case_id, case_data, "HIDDEN_FOR_ANALYSIS", sections_str)
            content_payload = [{"type": "text", "text": prompt_text}]

            for img_p in all_imgs:
                real_filename = os.path.basename(img_p)
                
                # Generate a random 6-character hex string (e.g., IMG_A3F9B2)
                virtual_id = f"IMG_{uuid.uuid4().hex[:6].upper()}"
                image_map[virtual_id] = real_filename
                
                b64 = self._encode_image(img_p)
                mime = "jpeg" if img_p.lower().endswith(('.jpg', '.jpeg')) else "png"
                
                content_payload.append({"type": "text", "text": f"Reference ID: {virtual_id}"})
                content_payload.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{mime};base64,{b64}", "detail": "high"}
                })

            execution_log["mapped_images"] = image_map
            
            # Safely capture the full prompt payload for the log (omitting giant base64 strings)
            safe_payload = []
            for item in content_payload:
                if item["type"] == "text":
                    safe_payload.append(item)
                elif item["type"] == "image_url":
                    safe_payload.append({"type": "image_url", "image_url": "[BASE64_IMAGE_DATA_OMITTED_FOR_LOG]"})
            
            system_instruction = (
                "You are a medical editor. Analyze the provided images. "
                "When embedding images in Markdown, you MUST use the provided random 'Reference ID' "
                "as the source path exactly, like this: ![Figure n](IMG_A1B2C3)"
            )
            
            execution_log["system_prompt"] = system_instruction
            execution_log["raw_prompt_payload"] = safe_payload

            # 4. Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": content_payload}
                ],
                temperature=0.3
            )

            raw_content = response.choices[0].message.content
            usage = response.usage
            
            # Update log with successful API return
            execution_log["raw_response"] = raw_content
            execution_log["tokens"]["prompt"] = usage.prompt_tokens
            execution_log["tokens"]["completion"] = usage.completion_tokens
            execution_log["tokens"]["total"] = usage.total_tokens

            # 5. Post-process with Regex to restore real filenames
            processed_content = raw_content
            for v_id, real_name in image_map.items():
                pattern = rf"\(({re.escape(v_id)})\)"
                processed_content = re.sub(pattern, f"(imgs/{real_name})", processed_content)

            # Save the final Markdown
            output_file = os.path.join(case_dir, f"{case_id}_generated.md")
            with open(output_file, "w", encoding='utf-8') as f:
                f.write(processed_content)
                
            execution_log["status"] = "success"
            
            print(f"[+] {case_id}: Journal generated and saved to {case_dir}")
            print(f"    Tokens - Total: {usage.total_tokens} | Time: {time.time() - start_time:.2f}s")
            print(f"    Images - Randomly mapped and embedded {len(image_map)} files.")

        except Exception as e:
            execution_log["error_message"] = str(e)
            print(f"[X] {case_id}: Error -> {e}")
            
        finally:
            # 6. ALWAYS save the full log file
            execution_log["execution_time_seconds"] = round(time.time() - start_time, 2)
            with open(log_output_path, "w", encoding="utf-8") as f:
                json.dump(execution_log, f, indent=4)

    def run(self):
        """
        Executes the batch processing loop.
        """
        json_files = glob.glob(os.path.join(self.working_dir, "*", "*_atoms.json"))
        
        print(f"Starting journal generation for {len(json_files)} cases...")
        print(f"Model ID: {self.model_id}")
        print("-" * 40)
        
        for path in json_files:
            self.process_case(path)
            
        print("-" * 40)
        print("Batch journal generation complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate multimodal medical journals from JSON facts and images.")
    parser.add_argument("--working_dir", type=str, default="/home/data1/musong/workspace/2026/03/10/log/0310_generated",
                        help="The base directory containing the case folders and their extracted atoms.")
    parser.add_argument("--model_id", type=str, default="gpt-4o",
                        help="The OpenAI multimodal model to use for generation.")
    return parser.parse_args()


if __name__ == "__main__":
    # 1. Setup proxy at the VERY BEGINNING of execution
    PROXY = "http://127.0.0.1:7890"
    setup_proxy(PROXY)
    
    # 2. Parse arguments
    args = parse_args()

    # 3. Initialize and run the pipeline
    pipeline = GenerationPipeline(
        working_dir=args.working_dir, 
        model_id=args.model_id
    )
    
    pipeline.run()