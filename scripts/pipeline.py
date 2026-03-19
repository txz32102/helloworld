from pipelines.utils import setup_proxy
from pipelines.extraction import AtomsExtractorPipeline
from pipelines.generation import GenerationPipeline
from pipelines.evaluation import EvaluationPipeline
from pipelines.convertion import MDConversionPipeline
from pipelines.convertion import PDFConversionPipeline

PROXY = "http://127.0.0.1:7890"
setup_proxy(PROXY)
model_id = 'gpt-4.1'

data_dir = "demo_data"
out_dir = "log/pipelines"

extractor = AtomsExtractorPipeline(
    data_dir=data_dir,
    out_dir=out_dir,
    num_folders=2,
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