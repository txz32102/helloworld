from pipelines.utils import setup_proxy
from pipelines.extraction import AtomsExtractorPipeline
from pipelines.generation import GenerationPipeline
from pipelines.evaluation import EvaluationPipeline
from pipelines.convertion import XMLConversionPipeline

PROXY = "http://127.0.0.1:7890"
setup_proxy(PROXY)

data_dir = "demo_data"
out_dir = "log/pipelines"

extractor = AtomsExtractorPipeline(
    data_dir=data_dir,
    out_dir=out_dir,
    num_folders=2,
    seed=42,
    char_limit=100000,
    model_id="gpt-4o",
    included_sections=["authors", "year", "figures", "tables", "citations"]
)
extractor.run()

generator = GenerationPipeline(
    working_dir=out_dir, 
    model_id="gpt-4o"
)
generator.run()

converter = XMLConversionPipeline(
    data_dir=data_dir,
    out_dir=out_dir
)
converter.run()

evaluator = EvaluationPipeline(
    base_dir=out_dir, 
    model_id="gpt-4o"
)
evaluator.run()
