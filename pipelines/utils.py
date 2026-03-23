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

def generate_llm_response(client, model: str, messages: list, stream: bool = False, **kwargs) -> dict:
    """
    Advanced utility to handle OpenAI and Qwen calls with an easy toggle for streaming.
    Returns a normalized dictionary containing "content", "reasoning_content", "tool_calls", and "usage".
    """
    
    # 1. Determine if the model strictly rejects temperature/sampling parameters
    # Catches o1, o3, and gpt-5, BUT explicitly allows gpt-5.4 and 'chat' variants
    is_temp_unsupported = (
        model.startswith(("o1", "o3", "gpt-5")) 
        and not model.startswith("gpt-5.4") 
        and "chat" not in model
    )

    if is_temp_unsupported:
        # Remove unsupported sampling parameters silently
        unsupported_params = ["temperature", "top_p", "presence_penalty", "frequency_penalty", "logprobs", "logit_bias"]
        for param in unsupported_params:
            kwargs.pop(param, None)
            
        # Reasoning models strictly require 'max_completion_tokens' instead of 'max_tokens'
        if "max_tokens" in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")

    # 2. Inject stream_options for token tracking (Required for Qwen/DashScope)
    if stream and "stream_options" not in kwargs:
        kwargs["stream_options"] = {"include_usage": True}

    # 3. Make the API Call
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=stream,
        **kwargs
    )

    # 4. Handle Non-Streaming
    if not stream:
        # Use getattr to safely handle unexpected response structures
        msg = response.choices[0].message if getattr(response, 'choices', None) else None
        
        if not msg:
            return {"content": "", "reasoning_content": "", "tool_calls": None, "usage": None}

        formatted_tools = None
        if getattr(msg, 'tool_calls', None):
            # Convert objects to dicts to match streaming format
            formatted_tools = [{
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments}
            } for tc in msg.tool_calls]
            
        return {
            "content": msg.content or "",
            "reasoning_content": getattr(msg, 'reasoning_content', ""), # Captures Qwen thinking
            "tool_calls": formatted_tools,
            "usage": getattr(response, 'usage', None)
        }

    # 5. Handle Streaming
    collected_content = ""
    collected_reasoning = ""
    tool_calls_dict = {}
    final_usage = None
    started_final_answer = False

    for chunk in response:
        # 🚨 CRITICAL FIX: Convert the strict Pydantic object to a raw dictionary 
        # so the SDK stops hiding non-standard fields like 'reasoning_content'!
        chunk_dict = chunk.model_dump()

        # Track token usage 
        if chunk_dict.get('usage'):
            final_usage = chunk_dict['usage']

        if not chunk_dict.get('choices'):
            continue

        delta = chunk_dict['choices'][0].get('delta', {})

        # A. Accumulate Reasoning (Qwen/Reasoning models)
        # Check both 'reasoning_content' and 'reasoning' just in case vLLM changes the key
        reasoning = delta.get('reasoning_content', "") or delta.get('reasoning', "")
        if reasoning:
            collected_reasoning += reasoning
            # Print thinking process in gray (ANSI escape code)
            print(f"\033[90m{reasoning}\033[0m", end="", flush=True)

        # B. Accumulate Content (Standard text)
        content = delta.get('content', "")
        if content:
            if collected_reasoning and not started_final_answer:
                print("\n\n✅ Final Answer:\n" + "-"*30)
                started_final_answer = True
            
            collected_content += content
            print(content, end="", flush=True)

        # C. Accumulate Tool Calls (Updated for dictionary access)
        if delta.get('tool_calls'):
            for tc in delta['tool_calls']:
                idx = tc.get('index')
                if idx not in tool_calls_dict:
                    tool_calls_dict[idx] = {
                        "id": tc.get('id', ""),
                        "type": "function",
                        "function": {"name": tc.get('function', {}).get('name', ""), "arguments": ""}
                    }
                
                # Append arguments as they stream in
                args = tc.get('function', {}).get('arguments', "")
                if args:
                    tool_calls_dict[idx]["function"]["arguments"] += args

    # Format tool calls into a list
    formatted_tool_calls = None
    if tool_calls_dict:
        formatted_tool_calls = [tool_calls_dict[idx] for idx in sorted(tool_calls_dict.keys())]

    return {
        "content": collected_content,
        "reasoning_content": collected_reasoning,
        "tool_calls": formatted_tool_calls,
        "usage": final_usage
    }