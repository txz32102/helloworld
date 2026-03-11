import os
import random
import time
from datetime import datetime
import glob
import argparse
import json
import xml.etree.ElementTree as ET
import pubmed_parser as pp
from .utils import setup_proxy, get_openai_client, extract_json_from_text

class AtomsExtractorPipeline:
    def __init__(self, data_dir: str, out_dir: str, num_folders: int, seed: int, char_limit: int, model_id: str, included_sections: list):
        """
        Initializes the pipeline with configuration and sets up the OpenAI client.
        Note: Proxy should be set globally before initializing this class.
        """
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.num_folders = num_folders
        self.seed = seed
        self.char_limit = char_limit
        self.model_id = model_id
        self.included_sections = included_sections
        self.extracted_headers = [] # Tracks the actual structural headers found in the text
        
        # 1. Seed & Environment Setup
        random.seed(self.seed)
        os.makedirs(self.out_dir, exist_ok=True)
        
        # 2. OpenAI Client Initialization
        self.client = get_openai_client()

    def _extract_text_pubmed_parser(self, xml_path: str) -> str:
        """
        Extracts plain text and metadata from PubMed XML files robustly using pubmed_parser.
        Combines title, abstract, main content, and optionally authors, year, figures, tables, and citations.
        """
        self.extracted_headers = [] # Reset for each new file
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
                
                # If the paragraph doesn't have a formal section header, skip it entirely!
                if not sec_title:
                    continue
                
                # CLEAN THE TEXT: This fixes the "messy" paragraph breaks
                raw_text = p.get('text', '')
                clean_text = " ".join(raw_text.split())
                
                # Skip useless publisher notes explicitly
                if "Publisher's Note" in clean_text or "Springer Nature remains neutral" in clean_text:
                    continue
                
                if clean_text:
                    if sec_title not in sections_dict:
                        sections_dict[sec_title] = []
                    sections_dict[sec_title].append(clean_text)

            # Append each section with clean, double-newline paragraph breaks
            for sec_title, texts in sections_dict.items():
                self.extracted_headers.append(sec_title) # Save the header name
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
                    print(f"  [!] Warning: Could not parse figures for {xml_path}: {e}")

            # 4. Extract Tables
            if "tables" in self.included_sections:
                try:
                    tables = pp.parse_pubmed_table(xml_path)
                    if tables:
                        table_texts = [f"{t.get('label', 'Table')}: {t.get('caption', '').strip()}" for t in tables]
                        full_text_parts.append("--- TABLE DESCRIPTIONS ---\n" + "\n\n".join(table_texts))
                except Exception as e:
                    print(f"  [!] Warning: Could not parse tables for {xml_path}: {e}")

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
                    print(f"  [!] Warning: Could not parse citations manually for {xml_path}: {e}")

            # Combine everything
            full_text = "\n\n---------------------------\n\n".join(full_text_parts)

            if not full_text.strip():
                return "XML_PARSE_ERROR: No valid text extracted by pubmed_parser."
                
            return full_text
            
        except Exception as e:
            return f"XML_PARSE_ERROR: {str(e)}"

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
        """
        folder_full_path = os.path.join(self.data_dir, folder_id)
        xml_files = glob.glob(os.path.join(folder_full_path, "*.xml"))
        print(f"Processing Case ID: {folder_id} | Found XML files: {len(xml_files)}\n")
        
        execution_log = {
            "case_id": folder_id,
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

        case_dir = os.path.join(self.out_dir, folder_id)
        os.makedirs(case_dir, exist_ok=True)
        log_output_path = os.path.join(case_dir, "atoms_execution_log.json")

        if not xml_files:
            error_msg = "No XML files found. Skipping."
            print(f"[-] {folder_id}: {error_msg}")
            execution_log["error_message"] = error_msg
            with open(log_output_path, "w", encoding="utf-8") as f:
                json.dump(execution_log, f, indent=4)
            return

        source_xml_path = xml_files[0]
        raw_text = self._extract_text_pubmed_parser(source_xml_path)
        raw_text = raw_text[:self.char_limit]

        prompt = self._build_prompt(raw_text)
        execution_log["raw_prompt"] = prompt  

        start_time = time.time()

        try:
            # 1. Call LLM
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            # 2. Capture Raw Outputs and Tokens immediately
            content = response.choices[0].message.content
            usage = response.usage
            
            execution_log["raw_response"] = content
            execution_log["tokens"]["prompt"] = usage.prompt_tokens
            execution_log["tokens"]["completion"] = usage.completion_tokens
            execution_log["tokens"]["total"] = usage.total_tokens

            # 3. Parse JSON using the shared utility function
            data = extract_json_from_text(content)
            
            if data:
                # Metadata for the extracted data
                data["metadata"] = {
                    "folder_id": folder_id,
                    "original_xml_path": source_xml_path,
                    "source_directory": folder_full_path,
                    "optional_sections_requested": self.included_sections,
                    "paper_sections_found": self.extracted_headers 
                }
                
                # Save atoms JSON
                atoms_output_path = os.path.join(case_dir, f"{folder_id}_atoms.json")
                with open(atoms_output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                # Mark as success
                execution_log["status"] = "success"
                print(f"[+] {folder_id}: Extracted and saved to {case_dir}")
                print(f"Tokens - Total: {usage.total_tokens} | Time: {time.time() - start_time:.2f}s")
                
            else:
                error_msg = "Could not find valid JSON in LLM response."
                execution_log["error_message"] = error_msg
                print(f"[X] {folder_id}: {error_msg}")
            
        except Exception as e:
            execution_log["error_message"] = str(e)
            print(f"[X] {folder_id}: Error -> {e}")
            
        finally:
            # 4. ALWAYS save the log file
            execution_log["execution_time_seconds"] = round(time.time() - start_time, 2)
            with open(log_output_path, "w", encoding="utf-8") as f:
                json.dump(execution_log, f, indent=4)

    def run(self):
        """
        Executes the batch processing loop.
        """
        all_folders = [f for f in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, f))]
        selected_folders = random.sample(all_folders, min(len(all_folders), self.num_folders))
        
        print(f"Starting extraction for {len(selected_folders)} cases...")
        print(f"Included Sections: {', '.join(self.included_sections)}")
        print("-" * 40)
        
        for folder_id in selected_folders:
            self.process_case(folder_id)
            
        print("-" * 40)
        print("Batch extraction complete.")

def parse_args():
    parser = argparse.ArgumentParser(description="Extract atomic clinical facts from PubMed XML to JSON.")
    parser.add_argument("--data_dir", type=str, default="/home/data1/musong/workspace/2026/03/10/data/0310_pipeline_bench")
    parser.add_argument("--out_dir", type=str, default="/home/data1/musong/workspace/2026/03/10/log/0310_generated")
    parser.add_argument("--num_folders", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--char_limit", type=int, default=100000) 
    parser.add_argument("--model_id", type=str, default="gpt-4o")
    parser.add_argument(
        "--sections", 
        nargs="+", 
        default=["authors", "year", "figures", "tables", "citations"], 
        help="Specify which additional metadata to feed to the LLM. Options: authors year figures tables citations"
    )
    return parser.parse_args()

if __name__ == "__main__":
    # 1. Setup proxy at the VERY BEGINNING of execution
    PROXY = "http://127.0.0.1:7890"
    setup_proxy(PROXY)
    
    # 2. Parse arguments and run the pipeline
    args = parse_args()

    extractor = AtomsExtractorPipeline(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        num_folders=args.num_folders,
        seed=args.seed,
        char_limit=args.char_limit,
        model_id=args.model_id,
        included_sections=args.sections
    )
    
    extractor.run()