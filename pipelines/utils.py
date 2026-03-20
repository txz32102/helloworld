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
    
def finalize_prompt(prompt_text: str) -> str:
    # 1. Split into lines and strip EVERY line of leading/trailing whitespace
    lines = [line.strip() for line in prompt_text.splitlines()]
    
    # 2. Join back with newlines
    cleaned = "\n".join(lines)
    
    # 3. Optional: Replace 3+ consecutive newlines with just 2 (standardizes spacing)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    return cleaned.strip()

def generate_llm_response(client: OpenAI, model: str, messages: list, stream: bool = False, **kwargs) -> dict:
    """
    Advanced utility to handle OpenAI calls with an easy toggle for streaming.
    Returns a normalized dictionary containing "content", "tool_calls", and "usage".
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=stream,
        **kwargs
    )

    # 1. Handle Non-Streaming
    if not stream:
        msg = response.choices[0].message
        formatted_tools = None
        if msg.tool_calls:
            # Convert objects to dicts to match streaming format
            formatted_tools = [{
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments}
            } for tc in msg.tool_calls]
            
        return {
            "content": msg.content or "",
            "tool_calls": formatted_tools,
            "usage": response.usage
        }

    # 2. Handle Streaming
    collected_content = ""
    tool_calls_dict = {}
    final_usage = None

    for chunk in response:
        # Track token usage
        if hasattr(chunk, 'usage') and chunk.usage:
            final_usage = chunk.usage

        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        # Accumulate Text
        if delta.content:
            collected_content += delta.content
            print(delta.content, end="", flush=True)

        # Accumulate Tool Calls
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls_dict:
                    tool_calls_dict[idx] = {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name or "", "arguments": ""}
                    }
                if tc.function.arguments:
                    tool_calls_dict[idx]["function"]["arguments"] += tc.function.arguments

    # Format tool calls into a list
    formatted_tool_calls = None
    if tool_calls_dict:
        formatted_tool_calls = [tool_calls_dict[idx] for idx in sorted(tool_calls_dict.keys())]

    return {
        "content": collected_content,
        "tool_calls": formatted_tool_calls,
        "usage": final_usage
    }