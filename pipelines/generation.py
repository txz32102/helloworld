import os
import re
import json
import glob
import shutil
import base64
import time
import uuid
from datetime import datetime

from .utils import get_openai_client, finalize_prompt, generate_llm_response, format_clinical_section
from .tools.registry import TOOL_SCHEMAS, AVAILABLE_TOOLS

class GenerationPipeline:
    def __init__(self, working_dir: str, model_id: str, mode: str = "multi", client=None):
        """
        Initializes the pipeline to generate medical journal articles from JSON atoms.
        :param mode: "single" for one-shot generation, "multi" for the 4-phase distillation pipeline.
        """
        self.working_dir = working_dir
        self.model_id = model_id
        self.mode = mode.lower()
        self.client = client if client else get_openai_client()

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _copy_images(self, source_directory: str, destination_directory: str):
        if not os.path.exists(source_directory):
            print(f"[-] Source directory not found for images: {source_directory}")
            return 0
        copied_count = 0
        for ext in ('*.jpg', '*.png', '*.jpeg'):
            for img_path in glob.glob(os.path.join(source_directory, ext)):
                try:
                    shutil.copy2(img_path, destination_directory)
                    copied_count += 1
                except Exception as e:
                    print(f"[!] Warning: Failed to copy image {img_path}: {e}")
        return copied_count

    def _extract_thinking(self, content: str) -> tuple:
        """Extracts <think> tags from reasoning models to save in the trajectory."""
        if not content:
            return None, content
        
        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL | re.IGNORECASE)
        if think_match:
            thinking_process = think_match.group(1).strip()
            clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE).strip()
            return thinking_process, clean_content
        return None, content

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

        ids_string = ", ".join(virtual_ids_list) if virtual_ids_list else "None"
        final_prompt = text_prompt.replace("{IMAGE_IDS}", ids_string)
        final_prompt = final_prompt.replace("{NUM_IMAGES}", str(len(virtual_ids_list)))

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

    def _execute_llm_loop(self, messages: list, allowed_tools: list, case_data: dict, phase_name: str) -> dict:
        """Generic loop for handling LLM generation, tool calls, and thought extraction."""
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

            if response_data.get("usage"):
                phase_log["total_prompt_tokens"] += response_data["usage"].prompt_tokens
                phase_log["total_comp_tokens"] += response_data["usage"].completion_tokens

            raw_content = response_data["content"]
            tool_calls = response_data["tool_calls"]
            
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
                    
                    # Inject hidden context
                    function_args["case_data"] = case_data
                    function_args["execution_log"] = phase_log # Optional context for tools
                    
                    log_args = {k: v for k, v in function_args.items() if k not in ["execution_log", "case_data"]}
                    tool_exec_log["arguments"] = log_args
                    print(f"        -> Tool: {function_name}({log_args})")
                    
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

    def _post_process_markdown(self, raw_markdown: str, image_map: dict) -> str:
        """Replaces virtual image IDs with real local paths."""
        processed = raw_markdown.strip()
        for v_id, real_name in image_map.items():
            processed = re.sub(rf"\(({re.escape(v_id)})\)", f"(imgs/{real_name})", processed)
        return processed.strip()

    def process_case_single(self, case_id: str, case_data: dict, source_dir: str, sections_str: str, case_dir: str) -> dict:
        """Executes the single-step generation mode."""
        if "References" not in sections_str and "Citations" not in sections_str:
            sections_str += "\n- References"

        prompt_template = f"""You are an expert medical researcher, clinician, and senior editor for a high-impact, peer-reviewed medical journal. Your task is to synthesize raw clinical data into a comprehensive, highly academic, and publication-ready clinical case report. Pure Markdown: Output the final report in strict Markdown format only. No AI Disclaimers: Never mention, acknowledge, or hint that this text was generated by an AI. No Conversational Filler: Provide only the medical report itself.
        
        ### RAW CLINICAL DATA ###
        - Clinical History: {case_data.get('history')}
        - Presentation: {case_data.get('presentation')}
        - Diagnostics: {case_data.get('diagnostics')}
        - Management: {case_data.get('management')}
        - Outcome: {case_data.get('outcome')}
        
        ### WRITING GUIDELINES & ACADEMIC STANDARDS ###
        1. Tone & Style: Authoritative, objective, and scholarly medical tone. Use precise clinical terminology. Target 1500+ words.
        2. Context & Background Synthesis: Embed this case within the broader medical context in the Introduction and Discussion.
        3. Mandatory Structure: Use EXACTLY these section headings:
        {sections_str}
        
        4. Figure Integration (CRITICAL REQUIREMENT): 
           You have been provided with EXACTLY {{NUM_IMAGES}} images. Reference IDs: [{{IMAGE_IDS}}].
           You MUST analyze and embed EVERY SINGLE IMAGE into the manuscript using this Markdown syntax:
           ![Figure n](IMG_XXXXXX)
           > **Figure n:** [Caption]
           
        ### CITATION & REFERENCE RULES ###
        1. NO MEMORY CITATIONS: Zero tolerance for hallucination.
        2. STEP 1: Use `search_pubmed` to find recent, relevant landmark papers.
        3. STEP 2: Use `fetch_ama_citations` with the DOIs. 
        4. EXACT COPY: Copy and paste the EXACT output string into the References list.
        5. CITATION VOLUME: Cite at least 10 distinct, peer-reviewed sources inline.
        """
        
        prompt_template = finalize_prompt(prompt_template)
        content_payload, image_map, virtual_ids = self._prepare_multimodal_payload(prompt_template, source_dir)
        
        system_instruction = (
            "You are an expert medical researcher and senior editor. "
            "STEP 1: `search_pubmed`. STEP 2: Find DOIs. STEP 3: `fetch_ama_citations`. "
            "STEP 4: Write manuscript. NO HALLUCINATIONS."
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": content_payload}
        ]

        log_data = self._execute_llm_loop(messages, TOOL_SCHEMAS, case_data, "Single_Step_Generation")
        final_markdown = self._post_process_markdown(log_data["final_output"], image_map)
        
        output_file = os.path.join(case_dir, f"{case_id}_generated.md")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(final_markdown)

        return {"mapped_images": image_map, "execution": log_data}

    def process_case_multi(self, case_id: str, case_data: dict, source_dir: str, sections_str: str, case_dir: str) -> dict:
        """Executes the multi-step (distillation) generation mode."""
        trajectory_log = {"mapped_images": {}, "phases": {}}

        # PHASE 1: RESEARCH
        hist_str = format_clinical_section(case_data.get('history'))
        pres_str = format_clinical_section(case_data.get('presentation'))
        diag_str = format_clinical_section(case_data.get('diagnostics'))
        mgmt_str = format_clinical_section(case_data.get('management'))
        out_str = format_clinical_section(case_data.get('outcome'))

        p1_prompt = f"""You are an expert medical researcher. Build a 'Research Dossier' for a clinical case report.
        
        Images provided: {{IMAGE_IDS}} (Total: {{NUM_IMAGES}})

        <clinical_data>
        <history>
        {hist_str}
        </history>
        
        <presentation>
        {pres_str}
        </presentation>
        
        <diagnostics>
        {diag_str}
        </diagnostics>
        
        <management>
        {mgmt_str}
        </management>
        
        <outcome>
        {out_str}
        </outcome>
        </clinical_data>
        
        TASKS (STRICT NO MEMORY CITATIONS):
        1. Assess Visuals: Use available imaging tools (e.g., `analyze_radiology_image`) to analyze the provided images.
        2. Gather Literature: Use available search tools (e.g., `search_pubmed`) to find at least 10 standard-of-care references. 
           *CRITICAL:* You must use strict PubMed search style (short keywords, MeSH terms, Boolean AND/OR). Do NOT use natural language sentences.
        3. Deep Dive: Deploy any other relevant specialized tools (like ClinGen or other databases) if the specific case context demands it.
        4. Format Citations: Process your findings through citation formatting tools (e.g., `fetch_ama_citations`) to get exact strings based on DOIs.
        
        Output a comprehensive 'Research Dossier' summarizing clinical findings, image insights, and a finalized list of EXACT citations. Do NOT write the paper yet."""
        
        content_payload, image_map, virtual_ids = self._prepare_multimodal_payload(p1_prompt, source_dir)
        trajectory_log["mapped_images"] = image_map
        
        p1_messages = [
            {"role": "system", "content": "Phase 1: The Researcher. Gather external knowledge and format citations. Zero tolerance for hallucinations."},
            {"role": "user", "content": content_payload}
        ]
        p1_log = self._execute_llm_loop(p1_messages, TOOL_SCHEMAS, case_data, "Phase_1_Research")
        trajectory_log["phases"]["phase_1"] = p1_log
        research_dossier = p1_log["final_output"]

        # PHASE 2: OUTLINING
        p2_messages = [
            {"role": "system", "content": "Phase 2: The Architect. Create a highly detailed structural outline."},
            {"role": "user", "content": f"Based on the raw case data and Research Dossier, create a detailed outline.\n\nRequired Sections:\n{sections_str}\n\nDossier:\n{research_dossier}\n\nInclude bullet points for claims, and explicitly note where [Citation X] or [Figure Y] will be anchored to ensure broad medical context."}
        ]
        p2_log = self._execute_llm_loop(p2_messages, [], case_data, "Phase_2_Outlining")
        trajectory_log["phases"]["phase_2"] = p2_log
        outline = p2_log["final_output"]

        # PHASE 3: DRAFTING
        p3_messages = [
            {"role": "system", "content": "Phase 3: The Writer. Draft the full academic manuscript in strict Markdown."},
            {"role": "user", "content": f"Write the full manuscript (1500+ words) using authoritative, objective medical tone. No conversational filler or AI disclaimers.\n\nOutline:\n{outline}\n\nDossier:\n{research_dossier}\n\nCRITICAL: Embed EXACTLY {len(virtual_ids)} images using Markdown: ![Figure X](IMG_ID). Include the full References list at the end matching the Dossier exactly."}
        ]
        p3_log = self._execute_llm_loop(p3_messages, [], case_data, "Phase_3_Drafting")
        trajectory_log["phases"]["phase_3"] = p3_log
        draft = p3_log["final_output"]

        # PHASE 4: REFINING
        p4_messages = [
            {"role": "system", "content": "Phase 4: The Editor. Polish the manuscript and ensure zero-omission of images and citations."},
            {"role": "user", "content": f"Review this draft:\n{draft}\n\nEnsure perfect academic tone, no filler, ALL {len(virtual_ids)} images are embedded via standard Markdown, and all citations are used. Output ONLY the final perfected Markdown."}
        ]
        p4_log = self._execute_llm_loop(p4_messages, [], case_data, "Phase_4_Refining")
        trajectory_log["phases"]["phase_4"] = p4_log
        
        final_markdown = self._post_process_markdown(p4_log["final_output"], image_map)
        output_file = os.path.join(case_dir, f"{case_id}_generated.md")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(final_markdown)

        return trajectory_log

    def process_case(self, json_path: str):
        case_id = os.path.basename(json_path).split('_')[0]
        case_dir = os.path.join(self.working_dir, case_id)
        log_output_path = os.path.join(case_dir, f"generation_log_{self.mode}.json")
        
        print(f"\nProcessing Case ID: {case_id} | Mode: {self.mode.upper()} | JSON Path: {json_path}")
        start_time = time.time()
        
        log_envelope = {
            "case_id": case_id,
            "mode": self.mode,
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "error_message": None
        }

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                case_data = json.load(f)
                
            source_dir = case_data.get('metadata', {}).get('source_directory', 'Unknown')
            paper_sections = case_data.get('metadata', {}).get('paper_sections_found', ["Introduction", "Case Presentation", "Discussion"])
            sections_str = "\n".join([f"- {sec}" for sec in paper_sections])
            
            os.makedirs(case_dir, exist_ok=True)
            case_img_dir = os.path.join(case_dir, "imgs")
            os.makedirs(case_img_dir, exist_ok=True)
            self._copy_images(source_dir, case_img_dir)

            if self.mode == "single":
                result_data = self.process_case_single(case_id, case_data, source_dir, sections_str, case_dir)
            else:
                result_data = self.process_case_multi(case_id, case_data, source_dir, sections_str, case_dir)
                
            log_envelope.update(result_data)
            log_envelope["status"] = "success"
            print(f"[+] {case_id}: {self.mode.capitalize()}-stage generation complete.")

        except Exception as e:
            log_envelope["error_message"] = str(e)
            print(f"[X] {case_id}: Error -> {e}")
            
        finally:
            log_envelope["total_time_seconds"] = round(time.time() - start_time, 2)
            with open(log_output_path, "w", encoding="utf-8") as f:
                json.dump(log_envelope, f, indent=4)

    def run(self):
        json_files = glob.glob(os.path.join(self.working_dir, "*", "*_atoms.json"))
        print(f"Starting journal generation for {len(json_files)} cases...")
        print(f"Model ID: {self.model_id} | Mode: {self.mode.upper()}")
        print("-" * 50)
        
        for path in json_files:
            self.process_case(path)
            
        print("-" * 50)
        print("Batch journal generation complete.")