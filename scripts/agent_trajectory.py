import sys
import os
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
        # Pre-compile the regex to strip ANSI escape codes efficiently
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, message):
        # 1. Write the raw message (with color codes) to the terminal
        self.terminal.write(message)
        
        # 2. Strip the color codes to create clean text
        clean_message = self.ansi_escape.sub('', message)
        
        # 3. Write the clean text to the log file
        self.log.write(clean_message)
        self.log.flush()  # Forces write to disk immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()

qwen_client = OpenAI(
    api_key=os.environ.get("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# gpt-4.1, gpt-5.4, Qwen/Qwen3.5-27B-FP8
openai_model_id = 'Qwen/Qwen3.5-27B-FP8'

# qwen3.5-27b-fp8
cleaned_model_id = "qwen3.5-27b-fp8"
qwen_model_id = 'Qwen/Qwen3.5-27B-FP8'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
timestamp = "20260326_143348"
data_dir = "demo_data"
out_dir = f"log/pipeline_{cleaned_model_id}/{timestamp}"
os.makedirs("log", exist_ok=True)
os.makedirs(f"log/pipeline_{cleaned_model_id}/{timestamp}", exist_ok=True)
log_filepath = os.path.join(f"log/pipeline_{cleaned_model_id}/{timestamp}", f"{cleaned_model_id}_pipeline_execution.log")

sys.stdout = DualLogger(log_filepath)
sys.stderr = sys.stdout  # This ensures error tracebacks are also logged

print(f"🚀 Starting pipeline execution. Logging terminal output to: {log_filepath}")
print("-" * 60)

from pipelines.utils import setup_proxy
from pipelines.extraction import AtomsExtractorPipeline
from pipelines.generation import GenerationPipeline
from pipelines.evaluation import EvaluationPipeline
from pipelines.convertion import MDConversionPipeline
from pipelines.convertion import PDFConversionPipeline

PROXY = "http://127.0.0.1:7890"
setup_proxy(PROXY)

# extractor = AtomsExtractorPipeline(
#     data_dir=data_dir,
#     out_dir=out_dir,
#     num_folders=20,
#     model_id=qwen_model_id,
#     client=qwen_client,
#     included_sections=["authors", "year", "figures", "tables", "citations"]
# )
# extractor.run()

generator = GenerationPipeline(
    working_dir=out_dir, 
    model_id=openai_model_id,
    client=qwen_client,
)
generator.run()

md_converter = MDConversionPipeline(
    data_dir=data_dir,
    out_dir=out_dir
)
md_converter.run()

evaluator = EvaluationPipeline(
    base_dir=out_dir, 
    model_id=qwen_model_id,
    client=qwen_client,
)
evaluator.run()

pdf_converter = PDFConversionPipeline(
    working_dir=out_dir,
    fonts_dir="fonts"
)
pdf_converter.run()