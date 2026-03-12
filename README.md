# Case Report Benchmark Pipeline

A multi-stage, LLM-powered pipeline designed to evaluate the capability of multimodal Vision-Language Models (VLMs) in synthesizing, generating, and evaluating clinical case reports.

## Overview

This project processes published medical case reports (in PMC XML format) and tests an AI's ability to reconstruct a highly academic, publication-ready case report using only the raw "atomic" facts and the associated medical images.

The pipeline consists of four distinct stages executed sequentially:

1. **Extraction (`extraction.py`)** - Parses source XMLs and uses an LLM to distill the paper into atomic clinical facts.
2. **Generation (`generation.py`)** - Tasks a multimodal LLM to write a brand new, academic case report using the extracted facts and raw images.
3. **Ground Truth Conversion (`convertion.py`)** - Parses the original XML into a clean, human-readable Markdown file (interleaving text and images) to serve as the benchmark.
4. **Evaluation (`evaluation.py`)** - Uses an LLM-as-a-judge to compare the generated report against the ground truth.

## Directory Structure

```text
├── demo_data/                   # Input data directory 
│   ├── {paper_id}/              # Individual paper directory (e.g., 32451719)
│   │   ├── *.xml                # Source PMC XML file
│   │   └── *.jpg/*.png          # Source medical images
├── log/                         
│   └── pipelines/               # Pipeline execution outputs
│       └── {paper_id}/          
│           ├── *_atoms.json     # Extracted clinical facts
│           ├── *_gt.md          # Ground Truth markdown 
│           ├── *_generated.md   # LLM-generated case report
│           ├── *_evaluation.md  # LLM-as-a-judge scorecard
│           ├── imgs/            # Local copy of referenced images
│           └── *log.json        # Execution logs and token usage
├── pipelines/                   # Core pipeline modules
│   ├── extraction.py            # XML Parsing -> JSON Atoms
│   ├── generation.py            # JSON Atoms + Images -> Generated MD
│   ├── convertion.py            # XML -> Ground Truth MD
│   ├── evaluation.py            # Generated MD vs GT MD -> Scorecard
│   └── utils.py                 # Shared utilities (Proxy, API clients)
└── scripts/                     
    └── pipeline.py              # Main execution orchestrator

```

## How the Pipeline Works (Under the Hood)

### 1. Data Extraction (`extraction.py`)

This stage acts as a "Senior Clinical Data Architect," breaking down a full medical paper into raw data points.

* **Parsing:** It uses `pubmed_parser` and custom `ElementTree` logic to extract the title, abstract, authors, sectioned paragraphs, figure captions, and citations from the source XML. It cleans the text by removing publisher notes and formatting artifacts.
* **LLM Structuring:** The cleaned text is fed to the LLM (e.g., GPT-4o) with a strict prompt to extract patient-specific facts into a predefined JSON schema.
* **Output:** A `{id}_atoms.json` file containing arrays of facts categorized into: `history`, `presentation`, `diagnostics`, `management`, and `outcome`. It deliberately strips out general disease facts to leave only the raw patient data.

### 2. Content Generation (`generation.py`)

This stage acts as an "Expert Medical Researcher," tasked with writing a comprehensive case report from scratch.

* **Image Handling & Virtual Mapping:** It gathers all relevant images and encodes them in Base64. To prevent the LLM from struggling with complex original filenames, the pipeline generates a random 6-character hex ID (e.g., `IMG_A3F9B2`) for each image.
* **Multimodal Prompting:** The LLM receives the atomic JSON facts, the Base64 images, and strict academic writing guidelines (requiring specific sections, AMA citations, and an authoritative tone). The LLM is instructed to embed images using the virtual IDs.
* **Post-Processing:** Once the LLM generates the Markdown text, a regex function sweeps through and replaces the virtual `IMG_XXXXXX` IDs back to the actual local image paths (e.g., `imgs/actual_filename.jpg`).
* **Output:** A `{id}_generated.md` file containing the synthesized case report.

### 3. Ground Truth Conversion (`convertion.py`)

To evaluate the generation, we need a clean benchmark. This stage converts the source PMC XML into a readable Markdown file.

* **Data Aggregation:** It parses the XML for abstracts, authors, and paragraph bodies. Crucially, it maps figure IDs to their actual graphic filenames.
* **Smart Interleaving:** Instead of dumping all images at the end, the script uses regular expressions to hunt for figure citations in the text (e.g., "fig. 1a"). It then dynamically injects the correct image and caption markdown immediately after the referencing paragraph.
* **Output:** A `{id}_gt.md` file representing the original published paper perfectly formatted in Markdown.

### 4. Evaluation (`evaluation.py`)

The final stage employs the LLM as an expert medical reviewer to score the generation against the ground truth.

* **Multimodal Context:** The evaluator LLM is fed the Ground Truth Markdown, the Generated Markdown, and all referenced images (Base64 encoded).
* **Scoring Criteria:** The LLM reviews the generated report based on four strict pillars:
1. **Clinical Accuracy:** Are facts, diagnoses, and treatments consistent?
2. **Image Referencing:** Does the text accurately describe the attached images compared to the ground truth?
3. **Completeness:** Were any critical details from the ground truth omitted?
4. **Hallucination:** Did the model invent false medical details?


* **Output:** A structured `{id}_evaluation.md` file containing qualitative feedback on each pillar and a final numerical score out of 10.

## Execution Logs & Telemetry

Every stage of the pipeline generates an `execution_log.json`. These logs capture comprehensive telemetry, including:

* Execution time (in seconds)
* Success/Failure status and error tracebacks
* Exact token usage (prompt, completion, and total)
* The raw prompt payloads and raw LLM responses, allowing for deep auditing of the model's behavior.
