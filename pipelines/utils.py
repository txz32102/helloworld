import os
import re
import json
import base64
from openai import OpenAI

def setup_proxy(proxy_url: str):
    """
    Sets up the environment proxy variables globally.
    MUST be called at the very beginning of the script execution.
    """
    if proxy_url:
        os.environ["http_proxy"] = proxy_url
        os.environ["https_proxy"] = proxy_url
        os.environ["all_proxy"] = proxy_url 

def get_openai_client() -> OpenAI:
    """
    Returns an initialized OpenAI client.
    Assumes setup_proxy() has already been executed.
    """
    return OpenAI()

def extract_json_from_text(content: str):
    """
    Extracts and parses a JSON object from a raw LLM text response.
    Returns the parsed dictionary, or None if parsing fails.
    """
    json_match = re.search(r'(\{.*\})', content, re.DOTALL)
    if json_match:
        clean_json_str = json_match.group(1)
        try:
            return json.loads(clean_json_str)
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            return None
    return None

def encode_image(image_path: str) -> str:
    """
    Encodes an image to a base64 string. 
    Useful for multimodal LLM evaluation and generation.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')