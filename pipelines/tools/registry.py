from .pubmed_tools import search_pubmed, fetch_ama_citations
from .clingen_tools import search_clingen_by_keyword, fetch_clingen_variant_data
from .medgemma_tools import analyze_radiology_image

# 1. Define the schema for the LLM
PUBMED_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_pubmed",
        "description": (
            "CRITICAL: You MUST use this tool to find real, recent, peer-reviewed medical literature. "
            "DO NOT HALLUCINATE OR INVENT CITATIONS. All references in your final output MUST be verified "
            "using this tool and MUST include a valid DOI or PMID. "
            "Use this tool to find scientific context, epidemiology, and standard treatment guidelines."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The medical search query. Use strict Boolean keyword matching (e.g., 'sarcoidosis AND arrhythmias'). "
                        "Do not use natural language sentences."
                    )
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of papers to return. Default is 5.",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}

FETCH_CITATION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "fetch_ama_citations", # Updated to plural to match the new function name
        "description": (
            "Given a list of DOIs (Digital Object Identifiers) obtained from the search_pubmed tool, "
            "this tool returns perfectly formatted, sequentially numbered AMA (American Medical Association) citation strings. "
            "You can pass a single DOI or multiple DOIs at once. "
            "You MUST use this tool to build your References section to ensure perfect formatting."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dois": { # Changed from 'doi' to 'dois'
                    "type": "array", # Changed type to array so it can accept a list
                    "items": {
                        "type": "string"
                    },
                    "description": "A list of exact DOIs for the papers (e.g., ['10.1001/jama.2023.1', '10.1056/NEJMoa123']). Do not include URLs."
                }
            },
            "required": ["dois"]
        }
    }
}

CLINGEN_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_clingen_by_keyword",
        "description": (
            "STEP 1 OF GENETIC SEARCH: Use this tool when the patient data mentions a specific genetic variant or mutation "
            "(e.g., 'RPGR c.1512_1513del' or 'BRAF V600E'). This tool searches the NCBI database to find the official ClinGen UUID. "
            "You MUST use this tool first to get the UUID before you can fetch the detailed genetic evidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The gene and variant name. e.g., 'RPGR c.1512_1513del'. Do not include extra text."
                }
            },
            "required": ["query"]
        }
    }
}

CLINGEN_FETCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "fetch_clingen_variant_data",
        "description": (
            "STEP 2 OF GENETIC SEARCH: Takes a ClinGen UUID (obtained from the search_clingen_by_keyword tool) and returns "
            "expert-curated medical evidence, disease condition, and pathogenicity classification. Use this data to write "
            "the 'Genetic Findings' section of the case report."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "uuid": {
                    "type": "string",
                    "description": "The exact UUID string returned by the search_clingen_by_keyword tool."
                }
            },
            "required": ["uuid"]
        }
    }
}

MEDGEMMA_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "analyze_radiology_image",
        "description": (
            "Analyzes a medical image (X-ray, CT, MRI, histology) to identify notable clinical findings. "
            "If the image is a composite figure (e.g., panels A, B, C), this tool automatically detects and splits it, "
            "returning a separate analysis for each sub-panel. Use this to understand the provided image context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_reference_id": {
                    "type": "string",
                    "description": "The virtual Reference ID of the image exactly as provided in the prompt (e.g., 'IMG_A1B2C3')."
                },
                "query": {
                    "type": "string",
                    "description": "Specific question about the image. Leave default for a general clinical analysis.",
                    "default": "Can you analyze this medical image and describe any notable clinical findings?"
                }
            },
            "required": ["image_reference_id"]
        }
    }
}

# ==========================================
# TOOL REGISTRY AND MAPPING
# ==========================================

AVAILABLE_TOOLS = {
    "search_pubmed": search_pubmed,
    "search_clingen_by_keyword": search_clingen_by_keyword,
    "fetch_clingen_variant_data": fetch_clingen_variant_data,
    "analyze_radiology_image": analyze_radiology_image,
    "fetch_ama_citations": fetch_ama_citations
}

TOOL_SCHEMAS = [
    PUBMED_TOOL_SCHEMA,
    CLINGEN_SEARCH_SCHEMA,
    CLINGEN_FETCH_SCHEMA,
    MEDGEMMA_TOOL_SCHEMA,
    FETCH_CITATION_SCHEMA
]