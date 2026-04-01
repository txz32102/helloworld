import os
import random
import time
from datetime import datetime
import glob
import json
import xml.etree.ElementTree as ET
import pubmed_parser as pp
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from .utils import get_openai_client, extract_json_from_text, generate_llm_response

class AtomsExtractorPipeline:
    # Added 'bs' (batch size/workers) parameter
    def __init__(self, data_dir: str, out_dir: str, num_folders: int, model_id: str, included_sections: list, char_limit: int=100000, seed: int=42, client=None, bs: int=1):
        """
        Initializes the pipeline with configuration. 
        Accepts an optional injected client; defaults to standard OpenAI.
        """
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.num_folders = num_folders
        self.seed = seed
        self.char_limit = char_limit
        self.model_id = model_id
        self.included_sections = included_sections
        self.bs = bs
        
        # 1. Seed & Environment Setup
        random.seed(self.seed)
        os.makedirs(self.out_dir, exist_ok=True)
        
        # 2. Assign injected client, or fallback to default OpenAI
        if client:
            self.client = client
        else:
            self.client = get_openai_client()

    def _extract_text_pubmed_parser(self, xml_path: str) -> tuple[str, list]:
        """
        Extracts plain text and metadata from PubMed XML files robustly using pubmed_parser.
        Combines title, abstract, main content, and optionally authors, year, figures, tables, and citations.
        RETURNS: A tuple of (extracted_text, extracted_headers_list) to remain thread-safe.
        """
        local_extracted_headers = [] 
        try:
            full_text_parts = []
            
            # 1. Extract Basic Info
            basic_info = pp.parse_pubmed_xml(xml_path)
            title = basic_info.get('full_title', '')
            abstract = basic_info.get('abstract', '')

            if title:
                full_text_parts.append(f"TITLE: {title}")
                
            if "authors" in self.included_sections:
                authors = basic_info.get('author_list', [])
                if authors:
                    author_str = "; ".join([f"{a[0]} {a[1]}" if isinstance(a, tuple) or isinstance(a, list) else str(a) for a in authors])
                    full_text_parts.append(f"AUTHORS: {author_str}")
                    
            if "year" in self.included_sections:
                year = basic_info.get('publication_year', '')
                if year:
                    full_text_parts.append(f"PUBLICATION YEAR: {year}")

            if abstract:
                full_text_parts.append(f"ABSTRACT:\n{abstract}")

            # 2. Extract Sections (Grouping paragraphs by their heading)
            paragraphs = pp.parse_pubmed_paragraph(xml_path)
            
            sections_dict = {}
            for p in paragraphs:
                sec_title = p.get('section', '').strip()
                
                if not sec_title:
                    continue
                
                raw_text = p.get('text', '')
                clean_text = " ".join(raw_text.split())
                
                if "Publisher's Note" in clean_text or "Springer Nature remains neutral" in clean_text:
                    continue
                
                if clean_text:
                    if sec_title not in sections_dict:
                        sections_dict[sec_title] = []
                    sections_dict[sec_title].append(clean_text)

            for sec_title, texts in sections_dict.items():
                local_extracted_headers.append(sec_title) 
                section_text = "\n\n".join(texts)
                full_text_parts.append(f"--- {sec_title.upper()} ---\n{section_text}")

            # 3. Extract Figures
            if "figures" in self.included_sections:
                try:
                    captions = pp.parse_pubmed_caption(xml_path)
                    if captions:
                        fig_texts = [f"{c.get('fig_label', 'Figure')}: {c.get('fig_caption', 'No caption').strip()}" for c in captions]
                        full_text_parts.append("--- FIGURE DESCRIPTIONS ---\n" + "\n\n".join(fig_texts))
                except Exception as e:
                    # CHANGED: print -> tqdm.write
                    tqdm.write(f"    [!] Warning: Could not parse figures for {xml_path}: {e}")

            # 4. Extract Tables
            if "tables" in self.included_sections:
                try:
                    tables = pp.parse_pubmed_table(xml_path)
                    if tables:
                        table_texts = [f"{t.get('label', 'Table')}: {t.get('caption', '').strip()}" for t in tables]
                        full_text_parts.append("--- TABLE DESCRIPTIONS ---\n" + "\n\n".join(table_texts))
                except Exception as e:
                    # CHANGED: print -> tqdm.write
                    tqdm.write(f"    [!] Warning: Could not parse tables for {xml_path}: {e}")

            # 5. Extract Citations (Custom ElementTree Parser)
            if "citations" in self.included_sections:
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    cite_texts = []
                    
                    for i, ref in enumerate(root.findall('.//ref')):
                        authors = []
                        for name in ref.findall('.//name'):
                            surname = name.findtext('surname', '')
                            given = name.findtext('given-names', '')
                            if surname or given:
                                authors.append(f"{surname} {given}".strip())
                        
                        author_str = ", ".join(authors) if authors else "Unknown Author"
                        year = ref.findtext('.//year', 'n.d.')
                        title = ref.findtext('.//article-title', 'No title')
                        journal = ref.findtext('.//source', 'Unknown Journal')
                        
                        doi_node = ref.find('.//pub-id[@pub-id-type="doi"]')
                        doi_str = f" https://doi.org/{doi_node.text}" if doi_node is not None and doi_node.text else ""
                        
                        cite_texts.append(f"[{i+1}] {author_str} ({year}). {title}. {journal}.{doi_str}")
                        
                    if cite_texts:
                        full_text_parts.append("--- CITATIONS / REFERENCES ---\n" + "\n".join(cite_texts))
                except Exception as e:
                    # CHANGED: print -> tqdm.write
                    tqdm.write(f"    [!] Warning: Could not parse citations manually for {xml_path}: {e}")

            # Combine everything
            full_text = "\n\n---------------------------\n\n".join(full_text_parts)

            if not full_text.strip():
                return "XML_PARSE_ERROR: No valid text extracted by pubmed_parser.", local_extracted_headers
                
            return full_text, local_extracted_headers
            
        except Exception as e:
            return f"XML_PARSE_ERROR: {str(e)}", local_extracted_headers

    def _build_prompt(self, raw_text: str) -> str:
        """
        Constructs the extraction prompt.
        """
        return f"""
        Role: Senior Clinical Data Architect.
        Task: Extract clinical facts into structured JSON.
        
        JSON SCHEMA:
        {{
          "history": ["fact1", "fact2"],
          "presentation": ["fact1"],
          "diagnostics": ["fact1"],
          "management": ["fact1"],
          "outcome": ["fact1"]
        }}

        STRICT RULES:
        1. "outcome" must ONLY contain patient-specific results (e.g. "Lesion reduced").
        2. DELETE any sentences about "this report highlights", "first case", etc.
        3. Do not include general disease facts.
        4. Use all provided context (sections, tables, figures, etc.) to extract accurate facts.
        5. Return ONLY the raw JSON object. No conversational text.
        
        TEXT AND METADATA:
        {raw_text}
        """

    def process_case(self, folder_id: str):
        """
        Processes a single case folder, extracting facts and logging full execution details.
        Skips invalid XMLs and tries the next file if an XML parser error occurs.
        """
        folder_full_path = os.path.join(self.data_dir, folder_id)
        xml_files = glob.glob(os.path.join(folder_full_path, "*.xml"))
        
        tqdm.write(f"Processing Case ID: {folder_id} | Found XML files: {len(xml_files)}")
        
        execution_log = {
            "case_id": folder_id,
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "execution_time_seconds": 0.0,
            "error_message": None,
            "tokens": {"prompt": 0, "completion": 0, "total": 0},
            "raw_prompt": None,
            "raw_response": None
        }

        case_dir = os.path.join(self.out_dir, folder_id)
        os.makedirs(case_dir, exist_ok=True)
        log_output_path = os.path.join(case_dir, "atoms_execution_log.json")

        if not xml_files:
            error_msg = "No XML files found. Skipping."
            tqdm.write(f"    [-] {folder_id}: {error_msg}")
            execution_log["error_message"] = error_msg
            with open(log_output_path, "w", encoding="utf-8") as f:
                json.dump(execution_log, f, indent=4)
            return False

        start_time = time.time()

        try:
            # Iterate through all XML files in the folder
            for source_xml_path in xml_files:
                raw_text, headers_found = self._extract_text_pubmed_parser(source_xml_path)
                
                # 1. Check for parser errors immediately
                if raw_text.startswith("XML_PARSE_ERROR:"):
                    clean_err = raw_text.replace("XML_PARSE_ERROR:", "").strip()
                    error_msg = f"Parse error in {os.path.basename(source_xml_path)}: {clean_err}"
                    tqdm.write(f"    [!] {folder_id}: {error_msg} -> Skipping to next file...")
                    
                    # Record the error, but keep the loop going to try the next XML
                    execution_log["error_message"] = error_msg 
                    continue 

                # 2. If parsing succeeded, proceed with building the prompt
                raw_text = raw_text[:self.char_limit]
                prompt = self._build_prompt(raw_text)
                execution_log["raw_prompt"] = prompt  

                use_streaming = self.bs <= 1
                status_text = "(streaming)" if use_streaming else "(batching)"
                tqdm.write(f"    [*] Extracting data for {folder_id} {status_text}...")

                llm_kwargs = {
                    "client": self.client,
                    "model": self.model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "stream": use_streaming,
                    "timeout": 360.0  
                }
                if use_streaming:
                    llm_kwargs["stream_options"] = {"include_usage": True}

                # Call LLM
                response_data = generate_llm_response(**llm_kwargs)
                
                if use_streaming:
                    tqdm.write(f"    [~] Stream finished for {folder_id}") 

                # Capture Raw Outputs and Tokens
                content = response_data["content"]
                usage = response_data["usage"]
                
                execution_log["raw_response"] = content
                if usage:
                    execution_log["tokens"]["prompt"] = usage.prompt_tokens
                    execution_log["tokens"]["completion"] = usage.completion_tokens
                    execution_log["tokens"]["total"] = usage.total_tokens

                # Parse JSON
                data = extract_json_from_text(content)
                
                if data:
                    data["metadata"] = {
                        "folder_id": folder_id,
                        "original_xml_path": source_xml_path,
                        "source_directory": folder_full_path,
                        "optional_sections_requested": self.included_sections,
                        "paper_sections_found": headers_found 
                    }
                    
                    atoms_output_path = os.path.join(case_dir, f"{folder_id}_atoms.json")
                    with open(atoms_output_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    
                    execution_log["status"] = "success"
                    # Clear out earlier parsing errors if we eventually succeeded
                    execution_log["error_message"] = None 
                    tqdm.write(f"    [+] {folder_id}: Extracted and saved to {case_dir} | Time: {time.time() - start_time:.2f}s")
                    return True  
                    
                else:
                    error_msg = "Could not find valid JSON in LLM response."
                    execution_log["error_message"] = error_msg
                    tqdm.write(f"    [X] {folder_id}: {error_msg}")
                    return False  

            # 3. If the loop finishes and we never hit `return True`, all files failed
            error_msg = "All XML files in this folder failed to parse."
            execution_log["error_message"] = error_msg
            tqdm.write(f"    [X] {folder_id}: {error_msg}")
            return False

        except Exception as e:
            execution_log["error_message"] = str(e)
            tqdm.write(f"    [X] {folder_id}: Error/Timeout -> {e}")
            return False  
            
        finally:
            # 4. Save the log file once, regardless of how we exited the block
            execution_log["execution_time_seconds"] = round(time.time() - start_time, 2)
            with open(log_output_path, "w", encoding="utf-8") as f:
                json.dump(execution_log, f, indent=4)

    def run(self):
        """
        Executes the batch processing loop, supporting concurrent API calls.
        """
        all_folders = [f for f in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, f))]
        selected_folders = random.sample(all_folders, min(len(all_folders), self.num_folders))
        
        print(f"Starting extraction for {len(selected_folders)} cases...")
        print(f"Included Sections: {', '.join(self.included_sections)}")
        print(f"Batch Size (Workers): {self.bs}")
        print("-" * 40)
        
        if self.bs > 1:
            with ThreadPoolExecutor(max_workers=self.bs) as executor:
                futures = [executor.submit(self.process_case, folder_id) for folder_id in selected_folders]
                
                successes = 0
                failures = 0
                
                # Added dynamic_ncols and smoothing for a cleaner visual bar that handles multiple threads
                with tqdm(as_completed(futures), total=len(futures), desc="Extraction Progress", unit="case", dynamic_ncols=True, smoothing=0) as pbar:
                    for future in pbar:
                        try:
                            # future.result() receives True or False from process_case
                            if future.result():
                                successes += 1
                            else:
                                failures += 1
                        except Exception as e:
                            tqdm.write(f"\nUnhandled thread exception: {e}")
                            failures += 1
                        
                        # Tell tqdm to display the live stats at the end of the bar!
                        pbar.set_postfix(Success=successes, Failed=failures)
        else:
            # Fallback for sequential processing (when batch size is 1)
            for folder_id in tqdm(selected_folders, desc="Extraction Progress", unit="case", dynamic_ncols=True):
                self.process_case(folder_id)
            
        print("-" * 40)
        print("Batch extraction complete.")