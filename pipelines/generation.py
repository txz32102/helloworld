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
    def __init__(self, working_dir: str, model_id: str, mode: str = "single", client=None, tools_config: dict = None):
        """
        Initializes the pipeline to generate medical journal articles from JSON atoms.
        :param mode: "single" for one-shot generation, "multi" for the 4-phase distillation pipeline.
        """
        self.working_dir = working_dir
        self.model_id = model_id
        self.mode = mode.lower()
        self.client = client if client else get_openai_client()
        self.tools_config = tools_config or {} # <--- NEW: Store the tools config

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

    def _execute_llm_loop(self, messages: list, allowed_tools: list, case_data: dict, phase_name: str, image_map: dict = None) -> dict:
        """Generic loop for handling LLM generation, tool calls, and thought extraction."""
        phase_start_time = datetime.now()
        
        # 1. Dynamically extract the system prompt from the messages payload
        system_prompt = next((msg["content"] for msg in messages if msg.get("role") == "system"), None)
        
        # NEW: Extract the user prompt, ignoring base64 image data
        user_prompt_text = None
        user_msg = next((msg["content"] for msg in messages if msg.get("role") == "user"), None)
        
        if isinstance(user_msg, list):
            # Extract only the text payloads, skipping "image_url" dictionaries
            text_parts = [item["text"] for item in user_msg if item.get("type") == "text"]
            user_prompt_text = "\n".join(text_parts)
        elif isinstance(user_msg, str):
            # Fallback for standard string prompts
            user_prompt_text = user_msg

        # 2. Extract the image paths/filenames used in this phase
        images_used = list(image_map.values()) if image_map else []

        phase_log = {
            "phase": phase_name,
            "start_time": phase_start_time.isoformat(),
            "end_time": None,
            "system_prompt": system_prompt,  # <--- Logs the system instruction
            "user_prompt": user_prompt_text, # <--- NEW: Logs the user instruction
            "images_used": images_used,      # <--- Logs the local image filenames
            "api_call_count": 0,  
            "turns": [],
            "final_output": None,
            "total_prompt_tokens": 0,
            "total_comp_tokens": 0,
            "mapped_images": image_map or {}
        }
        
        print(f"\n    === Starting {phase_name} ===")
        
        for turn in range(10):
            phase_log["api_call_count"] += 1  
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

            raw_content = response_data.get("content", "")
            reasoning_content = response_data.get("reasoning_content", "") 
            tool_calls = response_data.get("tool_calls", [])
            
            extracted_thinking, clean_content = self._extract_thinking(raw_content)
            final_thinking = reasoning_content if reasoning_content else extracted_thinking

            turn_log = {
                "turn": turn + 1,
                "thinking": final_thinking,
                "content": clean_content,
                "tool_calls": []
            }

            if not tool_calls:
                print(f"    [*] {phase_name} complete.")
                phase_log["final_output"] = clean_content
                phase_log["turns"].append(turn_log)
                phase_log["end_time"] = datetime.now().isoformat() 
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
                    function_args["case_data"] = case_data
                    function_args["execution_log"] = phase_log 
                    
                    # --- NEW: INJECT YAML CONFIG FOR MEDGEMMA ---
                    if function_name == "analyze_radiology_image":
                        medgemma_cfg = self.tools_config.get("medgemma", {})
                        if "use_vllm" in medgemma_cfg:
                            function_args["use_vllm"] = medgemma_cfg["use_vllm"]
                        if "vllm_url" in medgemma_cfg:
                            function_args["vllm_url"] = medgemma_cfg["vllm_url"]
                        if "vllm_model" in medgemma_cfg:
                            function_args["vllm_model"] = medgemma_cfg["vllm_model"]
                    # --------------------------------------------
                    
                    # Keep internal kwargs out of the terminal print
                    excluded_keys = ["execution_log", "case_data", "use_vllm", "vllm_url", "vllm_model"]
                    log_args = {k: v for k, v in function_args.items() if k not in excluded_keys}
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
            
        phase_log["end_time"] = datetime.now().isoformat() 
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

        # Passed image_map here
        log_data = self._execute_llm_loop(messages, TOOL_SCHEMAS, case_data, "Single_Step_Generation", image_map=image_map)
        final_markdown = self._post_process_markdown(log_data["final_output"], image_map)
        
        output_file = os.path.join(case_dir, f"{case_id}_generated.md")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(final_markdown)

        return {"mapped_images": image_map, "execution": log_data}

    def process_case_multi(self, case_id: str, case_data: dict, source_dir: str, sections_str: str, case_dir: str) -> dict:
        """Executes the 3-step (Planning, Drafting, Refining) generation mode."""
        trajectory_log = {"mapped_images": {}, "phases": {}}

        # Ensure References section is included
        if "References" not in sections_str and "Citations" not in sections_str:
            sections_str += "\n- References"

        # Extract atoms early so they can be used in the phases
        hist_str = format_clinical_section(case_data.get('history'))
        pres_str = format_clinical_section(case_data.get('presentation'))
        diag_str = format_clinical_section(case_data.get('diagnostics'))
        mgmt_str = format_clinical_section(case_data.get('management'))
        out_str = format_clinical_section(case_data.get('outcome'))

        # ------------------------------------------------------------------
        # PHASE 1: PLANNING (Research + Outlining)
        # ------------------------------------------------------------------
        system_prompt = (
            "Phase 1: The Planner. Gather external knowledge, format citations, and create a highly detailed structural outline. "
            "Zero tolerance for hallucinations.\n\n"
            "You are an expert medical researcher and architect. Your core objective is to build a comprehensive "
            "'Strategic Plan & Outline' for a clinical case report destined for a high-impact peer-reviewed medical journal."
        )
        p1_prompt = f"""### Clinical Data
            #### History
            {hist_str}
            #### Presentation
            {pres_str}
            #### Diagnostics
            {diag_str}
            #### Management
            {mgmt_str}
            #### Outcome
            {out_str}
            ---
            ### TASKS (STRICT NO MEMORY CITATIONS)
            1. **Assess Visuals:** Use available tools (e.g., `analyze_radiology_image`) to analyze the provided images. 
            You have been provided with EXACTLY {{NUM_IMAGES}} images. Reference IDs: [{{IMAGE_IDS}}].
            2. **Gather Literature:** Use search tools (e.g., `search_pubmed`, `fetch_ama_citations`) to compile a robust bibliography of exact citations based on DOIs.
            3. **Structural Outline:** Synthesize the literature and clinical data into a detailed outline using EXACTLY these required sections:
            {sections_str}
            ### OUTLINE REQUIREMENTS
            * Include bullet points for clinical claims.
            * Explicitly note EXACTLY where each citation (e.g., [Citation 1]) will be anchored.
            * Explicitly map EXACTLY where EVERY provided image (e.g., [Insert IMG_XXXXXX here]) will be placed to support the narrative.
            """
        # Payload preparation remains the same
        content_payload, image_map, virtual_ids = self._prepare_multimodal_payload(p1_prompt, source_dir)
        trajectory_log["mapped_images"] = image_map

        # 3. Message Assembly: Injecting the new system prompt
        p1_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_payload}
        ]
        
        p1_log = self._execute_llm_loop(p1_messages, TOOL_SCHEMAS, case_data, "Phase_1_Planning", image_map=image_map)
        trajectory_log["phases"]["phase_1"] = p1_log
        planning_doc = p1_log["final_output"]

        # ------------------------------------------------------------------
        # PHASE 2: DRAFTING
        # ------------------------------------------------------------------
        ids_string = ", ".join(virtual_ids) if virtual_ids else "None"
        num_images = len(virtual_ids)
        
        p2_messages = [
            {"role": "system", "content": "Phase 2: The Writer. Draft a full, publication-ready academic manuscript in strict Markdown."},
            {"role": "user", "content": f"""You are an expert medical clinician and senior editor. Write the full manuscript based on the Strategic Plan & Outline provided below.

Strategic Plan & Outline:
{planning_doc}

### CLINICAL AUTHENTICITY & FIDELITY RULES (CRITICAL) ###
1. Zero Timeline Sanitization: Preserve the "messy reality" of the patient history (missed appointments, unrelated surgeries, specific intervals). Do NOT smooth over the timeline.
2. Procedural Precision: Retain the exact clinical and mechanical terminology used in the source data.
3. Granular Specifics: Include all specific statistical claims, precise anatomical measurements, and multi-system secondary diagnoses present in the raw atoms.

### WRITING GUIDELINES & ACADEMIC STANDARDS ###
1. Tone & Style: Authoritative, objective, and scholarly medical tone. Target 1500 to 2500 words.
2. Mandatory Structure (STRICT): You MUST ONLY use the section headings provided below. Do NOT generate any other sections:
{sections_str}
3. Abstract Formatting: The abstract MUST be written as a single, continuous paragraph. Strictly avoid using structured subheadings (e.g., "Background:", "Case Presentation:", "Conclusion:") within the abstract body. No more than 200 words.
4. Sparsely integrate at least 10 relevant references (e.g., academic papers, official guidelines) throughout the main text. Do not include any citations in the abstract.
5. Figure Integration: 
   You have been provided with EXACTLY {num_images} images. Reference IDs: [{ids_string}].
   Embed EVERY SINGLE IMAGE using this Markdown syntax:
   ![Figure n](IMG_XXXXXX)
   > **Figure n:** [Detailed clinical caption]

Pure Markdown: Output the final report in strict Markdown format only. 
No AI Disclaimers: Never mention that this text was generated by an AI. 
No Conversational Filler: Provide only the medical report itself."""}
      ]
        p2_log = self._execute_llm_loop(p2_messages, [], case_data, "Phase_2_Drafting", image_map=image_map)
        trajectory_log["phases"]["phase_2"] = p2_log
        draft = p2_log["final_output"]

        # ------------------------------------------------------------------
        # PHASE 3: REFINING
        # ------------------------------------------------------------------
        p3_messages = [
            {"role": "system", "content": "Phase 3: The Editor. Polish the manuscript. Ensure perfect academic tone, strict adherence to allowed sections, factual alignment with original data, and sequential citation numbering."},
            {"role": "user", "content": f"""Review and refine this draft for final publication.

### DRAFT TO REFINE ###
{draft}

### ORIGINAL CLINICAL ATOMS (FOR FACT-CHECKING) ###
- History: {hist_str}
- Presentation: {pres_str}
- Diagnostics: {diag_str}
- Management: {mgmt_str}
- Outcome: {out_str}

### EDITORIAL STANDARDS (CRITICAL) ###
1. Strict Sections: You MUST ONLY include the exact sections listed below:
{sections_str}

2. Citation Ordering: Renumber ALL in-text citations and the final References list so they appear in strict chronological sequence (e.g., [1], then [2]). 
3. Fact-Check against Atoms: Cross-reference the DRAFT against the ORIGINAL CLINICAL ATOMS provided above. Correct any clinical values, timelines, or facts that drifted during the drafting phase.
4. Verify Figures: Ensure ALL {num_images} images remain embedded via standard Markdown: ![Figure X](IMG_XXXXXX).
5. Perfect Formatting: Output ONLY the final perfected Markdown manuscript. Do not include any preambles or AI disclaimers."""}
        ]
        p3_log = self._execute_llm_loop(p3_messages, [], case_data, "Phase_3_Refining", image_map=image_map)
        trajectory_log["phases"]["phase_3"] = p3_log
        
        # ------------------------------------------------------------------
        # FINALIZATION
        # ------------------------------------------------------------------
        # Process the final output to replace the virtual IDs with real paths
        final_markdown = self._post_process_markdown(p3_log["final_output"], image_map)
        
        # Save the final text
        output_file = os.path.join(case_dir, f"{case_id}_generated.md")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(final_markdown)

        return trajectory_log

    def process_case(self, json_path: str):
        case_id = os.path.basename(json_path).split('_')[0]
        case_dir = os.path.join(self.working_dir, case_id)
        log_output_path = os.path.join(case_dir, f"generation_log_{self.mode}.json")
        
        print(f"\nProcessing Case ID: {case_id} | Mode: {self.mode.upper()} | JSON Path: {json_path}")
        
        # Track master timestamps
        start_dt = datetime.now()
        start_time_real = time.time()
        
        log_envelope = {
            "case_id": case_id,
            "mode": self.mode,
            "start_time": start_dt.isoformat(),
            "end_time": None,
            "total_time_seconds": 0,
            "total_api_calls": 0,
            "total_prompt_tokens": 0,
            "total_comp_tokens": 0,
            "status": "failed",
            "error_message": None
        }

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                case_data = json.load(f)
                
            source_dir = case_data.get('metadata', {}).get('source_directory', 'Unknown')
            paper_sections = case_data.get('metadata', {}).get('paper_sections_found', ["Introduction", "Case Presentation", "Discussion"])
            if "Abstract" not in paper_sections:
                paper_sections.insert(0, "Abstract")
            if "Title" not in paper_sections:
                paper_sections.insert(0, "Title")
            sections_str = "\n".join([f"- {sec}" for sec in paper_sections])
            
            os.makedirs(case_dir, exist_ok=True)
            case_img_dir = os.path.join(case_dir, "imgs")
            os.makedirs(case_img_dir, exist_ok=True)
            self._copy_images(source_dir, case_img_dir)

            if self.mode == "single":
                result_data = self.process_case_single(case_id, case_data, source_dir, sections_str, case_dir)
                phases = [result_data.get("execution", {})]
            else:
                result_data = self.process_case_multi(case_id, case_data, source_dir, sections_str, case_dir)
                phases = list(result_data.get("phases", {}).values())
                
            # Aggregate totals across all phases
            for p in phases:
                log_envelope["total_api_calls"] += p.get("api_call_count", 0)
                log_envelope["total_prompt_tokens"] += p.get("total_prompt_tokens", 0)
                log_envelope["total_comp_tokens"] += p.get("total_comp_tokens", 0)

            log_envelope.update(result_data)
            log_envelope["status"] = "success"
            print(f"[+] {case_id}: {self.mode.capitalize()}-stage generation complete.")

        except Exception as e:
            log_envelope["error_message"] = str(e)
            print(f"[X] {case_id}: Error -> {e}")
            
        finally:
            # Mark the global end times
            log_envelope["end_time"] = datetime.now().isoformat()
            log_envelope["total_time_seconds"] = round(time.time() - start_time_real, 2)
            
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