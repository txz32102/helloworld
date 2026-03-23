import sys
import os
from datetime import datetime

# ---------------------------------------------------------
# 1. SETUP LOGGER (Catches all prints and saves to file)
# ---------------------------------------------------------
class DualLogger:
    """Writes output to both the terminal and a log file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Forces write to disk immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()


model_id = 'gpt-4.1'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
data_dir = "demo_data"
out_dir = f"log/pipeline_{model_id}/{timestamp}"
os.makedirs("log", exist_ok=True)
os.makedirs(f"log/pipeline_{model_id}/{timestamp}", exist_ok=True)
log_filepath = os.path.join(f"log/pipeline_{model_id}/{timestamp}", f"{model_id}_pipeline_execution.log")

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

extractor = AtomsExtractorPipeline(
    data_dir=data_dir,
    out_dir=out_dir,
    num_folders=20,
    model_id=model_id,
    included_sections=["authors", "year", "figures", "tables", "citations"]
)
extractor.run()

generator = GenerationPipeline(
    working_dir=out_dir, 
    model_id=model_id
)
generator.run()

md_converter = MDConversionPipeline(
    data_dir=data_dir,
    out_dir=out_dir
)
md_converter.run()

evaluator = EvaluationPipeline(
    base_dir=out_dir, 
    model_id=model_id
)
evaluator.run()

pdf_converter = PDFConversionPipeline(
    working_dir=out_dir,
    fonts_dir="fonts"
)
pdf_converter.run()