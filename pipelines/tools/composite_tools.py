import os
import base64
from openai import OpenAI

def analyze_composite_figure(
    image_reference_id: str, 
    query: str = None,
    case_data: dict = None, 
    execution_log: dict = None, 
    model_id: str = "gpt-4o",                 # <--- INJECTED FROM YAML
    base_url: str = "https://api.openai.com/v1", # <--- INJECTED FROM YAML
    api_key_env: str = "OPENAI_API_KEY",      # <--- INJECTED FROM YAML
    **kwargs
) -> str:
    """
    Analyzes complex, multi-panel composite medical images and generates 
    a comprehensive, panel-by-panel description using clinical context.
    """
    if not case_data or not execution_log:
        return "Error: Missing case_data or execution_log context from pipeline."

    # 1. Resolve the physical image path using the pipeline's execution log
    mapped_images = execution_log.get("mapped_images", {})
    image_filename = mapped_images.get(image_reference_id)
    
    if not image_filename:
        return f"Error: Image Reference ID '{image_reference_id}' not found in current image map."

    source_dir = case_data.get('metadata', {}).get('source_directory', '')
    image_path = os.path.join(source_dir, image_filename)

    if not os.path.exists(image_path):
        return f"Error: Image file not found at {image_path}"

    # 2. Safely format the clinical context
    def _format_section(data):
        if isinstance(data, list):
            return "\n".join(f"- {item}" for item in data)
        return str(data) if data else "Not provided."

    clinical_context = f"""HISTORY:
{_format_section(case_data.get('history'))}

PRESENTATION:
{_format_section(case_data.get('presentation'))}

DIAGNOSTICS:
{_format_section(case_data.get('diagnostics'))}

MANAGEMENT:
{_format_section(case_data.get('management'))}

OUTCOME:
{_format_section(case_data.get('outcome'))}"""

    # 3. Encode image to base64
    try:
        with open(image_path, 'rb') as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        mime_type = "image/jpeg" if image_path.lower().endswith(('.jpg', '.jpeg')) else "image/png"
    except Exception as e:
        return f"Error reading image file: {str(e)}"

    # 4. Construct prompt
    prompt = f"""You are an expert medical researcher and writer. Your task is to write a professional figure caption for a scientific medical case report.

Below is the structured clinical information from the case:

{clinical_context}

Please analyze the provided composite clinical image and write a concise, publication-ready figure caption.

IMPORTANT GUIDELINES:
- Begin the caption with a single, comprehensive sentence summarizing the entire figure.
- Following the summary sentence, describe each panel individually using its corresponding label exactly as it appears in the image (e.g., (A), (B), (C)... or (I), (II)...).
- You must read the visible panel labels directly from the image itself. Do not invent labels, and do not replace them with panel_1, panel_2, or spatial order.
- If the figure contains two panels, only use left/right or upper/lower wording when that agrees with the visible labels.
- The caption should describe ONLY what is visible in the image.
- State the image type/modality for each panel.
- Mention relevant anatomical structures and pathological/radiographic findings shown.
- Do NOT include treatment rationale, clinical decision-making, or future treatment plans.
- Do NOT write "These findings supported..." or similar interpretive conclusions.
- Keep it factual and descriptive, as if annotating what a reader sees."""

    if query and query.strip():
        prompt += f"\n\nSPECIFIC USER INSTRUCTIONS TO FOLLOW STRICTLY:\n{query}\n\nProvide ONLY the figure caption text."
    else:
        prompt += "\n\nProvide ONLY the figure caption text."

    # 5. Call the API
    try:
        # 1. Attempt to fetch it as an environment variable
        actual_api_key = os.environ.get(api_key_env)
        
        # 2. If it's not in the environment, assume the YAML string IS the literal key
        if not actual_api_key:
            actual_api_key = api_key_env
            
        if not actual_api_key:
            return "Error: API Key is missing. Provide a valid key or environment variable name in the YAML config."

        client = OpenAI(
            api_key=actual_api_key,
            base_url=base_url
        )

        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1
        )
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error executing composite vision model: {str(e)}"