import sys
import os
import yaml
from datetime import datetime
from openai import OpenAI
import re

# ---------------------------------------------------------
# 1. SETUP LOGGER (Catches all prints and saves to file)
# ---------------------------------------------------------
class DualLogger:
    """Writes output to both the terminal (with colors) and a log file (plain text)."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        self.is_tqdm = False 

    def write(self, message):
        self.terminal.write(message)
        if '\r' in message:
            self.is_tqdm = True
            return 
        if self.is_tqdm and message == '\n':
            self.is_tqdm = False
            return
        self.is_tqdm = False
        clean_message = self.ansi_escape.sub('', message)
        self.log.write(clean_message)
        self.log.flush() 

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ---------------------------------------------------------
# 2. LOAD CONFIGURATION
# ---------------------------------------------------------
def load_config(config_file="/home/data1/musong/workspace/2026/03/07/helloworld/configs/config.yaml"):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file {config_file} not found.")
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------
# 3. MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    config = load_config()

    # Extract settings from config
    paths = config.get("paths", {})
    llm_config = config.get("llm", {})
    sys_config = config.get("system", {})
    pipeline_config = config.get("pipelines", {})
    tools_config = config.get("tools", {})

    data_dir = paths.get("data_dir", "data/default")
    model_id = llm_config.get("model_id", "gpt-4.1")
    
    # Setup dynamic directories
    cleaned_model_id = model_id.replace("/", "_") # Prevents directory creation errors if model_id has slashes
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(paths.get("base_log_dir", "log"), f"pipeline_{cleaned_model_id}", timestamp)
    
    os.makedirs(out_dir, exist_ok=True)
    log_filepath = os.path.join(out_dir, f"{cleaned_model_id}_pipeline_execution.log")

    # Initialize Dual Logger
    sys.stdout = DualLogger(log_filepath)
    sys.stderr = sys.stdout  

    print(f"🚀 Starting pipeline execution. Logging terminal output to: {log_filepath}")
    print("-" * 60)

    # Setup Proxy
    proxy_url = sys_config.get("proxy")
    if proxy_url:
        from pipelines.utils import setup_proxy
        setup_proxy(proxy_url)
        print(f"🔗 Proxy configured: {proxy_url}")

    # Setup Client
    # Only pass base_url and api_key if they are defined in the yaml
    client_kwargs = {}
    if llm_config.get("base_url"):
        client_kwargs["base_url"] = llm_config["base_url"]
    if llm_config.get("api_key"):
        client_kwargs["api_key"] = llm_config["api_key"]

    client = OpenAI(**client_kwargs) if client_kwargs else OpenAI()
    print(f"🤖 Initialized OpenAI client. Target model: {model_id}")

    # Import pipelines
    from pipelines.extraction import AtomsExtractorPipeline
    from pipelines.generation import GenerationPipeline
    from pipelines.evaluation import EvaluationPipeline
    from pipelines.convertion import MDConversionPipeline
    from pipelines.convertion import PDFConversionPipeline

    # ---------------------------------------------------------
    # 4. RUN PIPELINES
    # ---------------------------------------------------------
    
    extractor_cfg = pipeline_config.get("extractor", {})
    extractor = AtomsExtractorPipeline(
        data_dir=data_dir,
        out_dir=out_dir,
        num_folders=extractor_cfg.get("num_folders", 20),
        model_id=model_id,
        # client=client,  # Uncomment if your pipeline accepts the custom client
        included_sections=extractor_cfg.get("included_sections", [])
    )
    extractor.run()

    generator_cfg = pipeline_config.get("generator", {})
    generator = GenerationPipeline(
        working_dir=out_dir, 
        model_id=model_id,
        mode=generator_cfg.get("mode", "single"),
        tools_config=tools_config
    )
    generator.run()

    md_converter = MDConversionPipeline(
        data_dir=data_dir,
        out_dir=out_dir
    )
    md_converter.run()

    evaluator = EvaluationPipeline(
        base_dir=out_dir, 
        model_id=model_id,
        # client=client,
    )
    evaluator.run()

    pdf_converter = PDFConversionPipeline(
        working_dir=out_dir,
        fonts_dir=paths.get("fonts_dir", "fonts")
    )
    pdf_converter.run()
    
    print("-" * 60)
    print("✅ All pipelines executed successfully.")