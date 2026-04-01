import os
import re
import json
import glob
import shutil
import base64
import time
import uuid
from datetime import datetime

from .utils import get_openai_client, finalize_prompt, generate_llm_response
from .tools.registry import TOOL_SCHEMAS, AVAILABLE_TOOLS

class GenerationPipeline:
    def __init__(self, working_dir: str, model_id: str, client=None):
        self.working_dir = working_dir
        self.model_id = model_id
        self.client = client if client else get_openai_client()

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _copy_images(self, source_directory: str, destination_directory: str):
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

    def _prepare_multimodal_payload(self, text_prompt: str, source_dir: str) -> tuple:
        all_imgs = []
        for ext in ('*.jpg', '*.png', '*.jpeg'):
            all_imgs.extend(glob.glob(os.path.join(source_dir, ext)))
        all_imgs.sort(key=lambda f: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', f)])

        image_map = {}
        virtual_ids_list = []
        
        for img_p in all_imgs:
            virtual_id = f"IMG_{uuid.uuid4().hex[:6].upper()}"
            image_map[virtual_id] = img_p 
            virtual_ids_list.append(virtual_id)

        # Inject virtual IDs into the prompt so the model knows what it's looking at
        ids_string = ", ".join(virtual_ids_list) if virtual_ids_list else "None"
        final_prompt = text_prompt.replace("{IMAGE_IDS}", ids_string)

        content_payload = [{"type": "text", "text": final_prompt}]
        for virtual_id, img_p in image_map.items():
            b64 = self._encode_image(img_p)
            mime = "jpeg" if img_p.lower().endswith(('.jpg', '.jpeg')) else "png"
            content_payload.append({"type": "text", "text": f"Reference ID: {virtual_id}"})
            content_payload.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/{mime};base64,{b64}", "detail": "high"}
            })
            image_map[virtual_id] = os.path.basename(img_p) 

        return content_payload, image_map, virtual_ids_list

    def _extract_thinking(self, content: str) -> tuple:
        """Extracts <think> tags from reasoning models to save in the trajectory."""
        if not content:
            return None, content
        
        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL | re.IGNORECASE)
        if think_match:
            thinking_process = think_match.group(1).strip()
            # Remove the thinking block from the main content so downstream phases don't get confused
            clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE).strip()
            return thinking_process, clean_content
        return None, content

    def _execute_llm_loop(self, messages: list, allowed_tools: list, case_data: dict, phase_name: str) -> dict:
        """Generic loop for any phase. Returns the trajectory log for this phase."""
        phase_log = {
            "phase": phase_name,
            "turns": [],
            "final_output": None,
            "total_prompt_tokens": 0,
            "total_comp_tokens": 0
        }
        
        print(f"\n    === Starting {phase_name} ===")
        
        for turn in range(15):
            print(f"    [*] Turn {turn + 1}: Generating... ", end="", flush=True)
            
            response_data = generate_llm_response(
                client=self.client,
                model=self.model_id,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                tools=allowed_tools if allowed_tools else None,
                tool_choice="auto" if allowed_tools else None,
                temperature=0.1
            )
            print()

            if response_data["usage"]:
                phase_log["total_prompt_tokens"] += response_data["usage"].prompt_tokens
                phase_log["total_comp_tokens"] += response_data["usage"].completion_tokens

            raw_content = response_data["content"]
            tool_calls = response_data["tool_calls"]
            
            # Extract thinking tags!
            thinking_process, clean_content = self._extract_thinking(raw_content)

            turn_log = {
                "turn": turn + 1,
                "thinking": thinking_process,
                "content": clean_content,
                "tool_calls": []
            }

            if not tool_calls:
                print(f"    [*] {phase_name} complete.")
                phase_log["final_output"] = clean_content
                phase_log["turns"].append(turn_log)
                return phase_log
                
            # Handle Tools
            messages.append({
                "role": "assistant",
                "content": raw_content if raw_content else None,
                "tool_calls": tool_calls
            })
            
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                function_args_str = tool_call["function"]["arguments"]
                function_to_call = AVAILABLE_TOOLS.get(function_name)
                
                tool_exec_log = {"tool_name": function_name, "arguments": None, "raw_result": None, "error": None}
                
                try:
                    function_args = json.loads(function_args_str)
                    tool_exec_log["arguments"] = function_args
                    print(f"        -> Tool: {function_name}({function_args})")
                    
                    # Inject hidden context
                    function_args["case_data"] = case_data
                    
                    if function_to_call:
                        function_response = function_to_call(**function_args)
                        try:
                            tool_exec_log["raw_result"] = json.loads(function_response) if isinstance(function_response, str) else function_response
                        except json.JSONDecodeError:
                            tool_exec_log["raw_result"] = function_response
                    else:
                        function_response = f"Error: Tool '{function_name}' is not registered."
                        tool_exec_log["error"] = function_response
                        
                except Exception as e:
                    print(f"        [X] Tool execution failed: {e}")
                    function_response = f"Error executing tool: {str(e)}"
                    tool_exec_log["error"] = str(e)
                
                turn_log["tool_calls"].append(tool_exec_log)
                messages.append({
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": function_name,
                    "content": str(function_response),
                })
            
            phase_log["turns"].append(turn_log)
            
        return phase_log

    def process_case(self, json_path: str):
        case_id = os.path.basename(json_path).split('_')[0]
        case_dir = os.path.join(self.working_dir, case_id)
        log_output_path = os.path.join(case_dir, "distillation_trajectory_log.json")
        
        print(f"\nProcessing Case ID: {case_id} | JSON Path: {json_path}")
        start_time = time.time()

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                case_data = json.load(f)
                
            source_dir = case_data.get('metadata', {}).get('source_directory', 'Unknown')
            sections_str = "\n".join([f"- {sec}" for sec in case_data.get('metadata', {}).get('paper_sections_found', ["Introduction", "Case Presentation", "Discussion"])])
            
            os.makedirs(case_dir, exist_ok=True)
            case_img_dir = os.path.join(case_dir, "imgs")
            os.makedirs(case_img_dir, exist_ok=True)
            self._copy_images(source_dir, case_img_dir)

            # --- PHASE 1: RESEARCH (Tool heavy) ---
            phase_1_prompt = f"""You are an expert medical researcher. Your goal is to build a 'Research Dossier' for a clinical case report.
            Review the following raw clinical data:
            {json.dumps(case_data, indent=2)}
            
            Images provided: {{IMAGE_IDS}}
            
            TASKS:
            1. Use `analyze_radiology_image` to understand the provided images.
            2. Use `search_pubmed` to find standard-of-care references, similar cases, and epidemiology.
            3. Use genetic tools if applicable.
            4. Once you have found the DOIs you want to cite, use `fetch_ama_citations` to get the formatted citations.
            
            Output a comprehensive 'Research Dossier' summarizing your findings, image analyses, and the finalized list of AMA citations. Do NOT write the paper yet."""
            
            content_payload, image_map, virtual_ids = self._prepare_multimodal_payload(phase_1_prompt, source_dir)
            
            trajectory_log = {
                "case_id": case_id,
                "timestamp": datetime.now().isoformat(),
                "mapped_images": image_map,
                "phases": {}
            }

            p1_messages = [
                {"role": "system", "content": "You are Phase 1: The Researcher. Gather all external knowledge and format citations."},
                {"role": "user", "content": content_payload}
            ]
            
            p1_log = self._execute_llm_loop(p1_messages, TOOL_SCHEMAS, case_data, "Phase_1_Research")
            trajectory_log["phases"]["phase_1"] = p1_log
            research_dossier = p1_log["final_output"]

            # --- PHASE 2: OUTLINING (Reasoning heavy, no tools) ---
            p2_messages = [
                {"role": "system", "content": "You are Phase 2: The Architect. Create a highly detailed structural outline for a medical manuscript."},
                {"role": "user", "content": f"Based on the raw case data and the following Research Dossier, create a detailed, section-by-section outline.\n\nRequired Sections:\n{sections_str}\n\nResearch Dossier:\n{research_dossier}\n\nInclude bullet points for what claims go where, and specifically note where [Citation X] or [Figure Y] will be anchored."}
            ]
            p2_log = self._execute_llm_loop(p2_messages, [], case_data, "Phase_2_Outlining")
            trajectory_log["phases"]["phase_2"] = p2_log
            outline = p2_log["final_output"]

            # --- PHASE 3: DRAFTING (Generation heavy, no tools) ---
            p3_messages = [
                {"role": "system", "content": "You are Phase 3: The Writer. Draft the full academic manuscript in strict Markdown."},
                {"role": "user", "content": f"Write the full manuscript. Use the Outline to guide your structure. Use the Research Dossier for your facts and exact AMA citations.\n\nOutline:\n{outline}\n\nDossier:\n{research_dossier}\n\nRule: You MUST embed images using Markdown: ![Figure X](IMG_ID). Include the full References section at the end."}
            ]
            p3_log = self._execute_llm_loop(p3_messages, [], case_data, "Phase_3_Drafting")
            trajectory_log["phases"]["phase_3"] = p3_log
            draft = p3_log["final_output"]

            # --- PHASE 4: REFINING (Correction heavy, no tools) ---
            p4_messages = [
                {"role": "system", "content": "You are Phase 4: The Editor. Polish the manuscript and ensure zero-omission of images and citations."},
                {"role": "user", "content": f"Review this draft:\n{draft}\n\nEnsure academic tone is perfect, no conversational filler exists, all {len(virtual_ids)} images are embedded, and all citations from the dossier are used. Output ONLY the final perfected Markdown."}
            ]
            p4_log = self._execute_llm_loop(p4_messages, [], case_data, "Phase_4_Refining")
            trajectory_log["phases"]["phase_4"] = p4_log
            final_markdown = p4_log["final_output"]

            # --- Post Processing ---
            for v_id, real_name in image_map.items():
                final_markdown = re.sub(rf"\(({re.escape(v_id)})\)", f"(imgs/{real_name})", final_markdown)

            output_file = os.path.join(case_dir, f"{case_id}_generated.md")
            with open(output_file, "w", encoding='utf-8') as f:
                f.write(final_markdown.strip())

            print(f"[+] {case_id}: Multi-stage generation complete.")

        except Exception as e:
            trajectory_log["error"] = str(e)
            print(f"[X] {case_id}: Error -> {e}")
            
        finally:
            trajectory_log["total_time_seconds"] = round(time.time() - start_time, 2)
            with open(log_output_path, "w", encoding="utf-8") as f:
                json.dump(trajectory_log, f, indent=4)

    def run(self):
        json_files = glob.glob(os.path.join(self.working_dir, "*", "*_atoms.json"))
        for path in json_files:
            self.process_case(path)