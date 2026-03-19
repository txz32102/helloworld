import os
import json
from typing import Union, Dict, Any
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline


# 1. Global pipeline variable to prevent reloading the model on every tool call
MEDGEMMA_PIPE = None

def extract_image_panels(
    image_input: Union[np.ndarray, Image.Image], 
    separation_iters: int = 2,
    thresh_val: int = 240,
    min_area_ratio: float = 0.005,
    max_area_ratio: float = 0.90
) -> Dict[str, Dict[str, Any]]:
    """
    Splits a composite image into its sub-panels IN MEMORY.
    
    Args:
        image_input: The input image (Numpy array or PIL Image).
        separation_iters: Controls aggressive splitting of close panels.
        thresh_val: Pixel intensity threshold (0-255).
        min_area_ratio: Minimum area a panel must have to be kept.
        max_area_ratio: Maximum area a panel can have (filters out full image boundaries).

    Returns:
        dict: A dictionary where keys are panel names (e.g., 'panel_1') and values 
              are dicts containing the cropped 'image' (np.ndarray) and its 'box' (x,y,w,h).
    """
    # 1. Standardize Input to OpenCV format (Numpy BGR)
    if isinstance(image_input, Image.Image):
        image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        image = image_input.copy()
    else:
        raise TypeError("Input must be a PIL Image or a numpy array.")

    img_h, img_w = image.shape[:2]
    total_area = img_h * img_w

    # 2. Image Preprocessing & Separation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

    if separation_iters > 0:
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=separation_iters)

    # 3. Find and Filter Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    panel_boxes = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if (total_area * min_area_ratio) < area < (total_area * max_area_ratio):
            pad = separation_iters
            x_pad, y_pad = max(0, x - pad), max(0, y - pad)
            w_pad, h_pad = min(img_w - x_pad, w + pad * 2), min(img_h - y_pad, h + pad * 2)
            panel_boxes.append((x_pad, y_pad, w_pad, h_pad))

    # FALLBACK: If no panels pass, use the whole image
    if not panel_boxes:
        panel_boxes.append((0, 0, img_w, img_h))

    # 4. Sort roughly Top-to-Bottom, Left-to-Right
    panel_boxes = sorted(panel_boxes, key=lambda b: b[1]) 
    avg_height = sum([b[3] for b in panel_boxes]) / len(panel_boxes)
    y_tolerance = avg_height * 0.4 

    rows = []
    current_row = [panel_boxes[0]]

    for box in panel_boxes[1:]:
        if abs(box[1] - current_row[-1][1]) <= y_tolerance:
            current_row.append(box)
        else:
            rows.append(current_row)
            current_row = [box]
    rows.append(current_row)

    sorted_boxes = []
    for row in rows:
        sorted_row = sorted(row, key=lambda b: b[0])
        sorted_boxes.extend(sorted_row)

    # 5. Crop and Build Output Dictionary
    results = {}
    for i, (x, y, w, h) in enumerate(sorted_boxes):
        chunk = image[y:y+h, x:x+w]
        results[f'panel_{i+1}'] = {
            'image': chunk,
            'box': (x, y, w, h)
        }

    return results

def get_medgemma_pipe():
    global MEDGEMMA_PIPE
    if MEDGEMMA_PIPE is None:
        print("    [*] Initializing MedGemma pipeline (this may take a moment)...")
        # Ensure it loads on the correct device (GPU if available)
        device = 0 if torch.cuda.is_available() else -1
        MEDGEMMA_PIPE = pipeline(
            "image-text-to-text", 
            model="google/medgemma-1.5-4b-it", 
            device=device
        )
    return MEDGEMMA_PIPE

def analyze_radiology_image(
    image_reference_id: str, 
    query: str = "Can you analyze this medical image and describe any notable clinical findings?",
    execution_log: dict = None,
    case_data: dict = None,
    **kwargs
) -> str:
    """Loads an image, splits it into panels, and runs MedGemma."""
    
    # 1. Secretly resolve the real path using the injected data
    mapped_images = execution_log.get("mapped_images", {}) if execution_log else {}
    if image_reference_id not in mapped_images:
        return json.dumps({"error": f"LLM hallucinated image ID: {image_reference_id}"})
        
    real_filename = mapped_images[image_reference_id]
    source_dir = case_data.get('metadata', {}).get('source_directory', '') if case_data else ''
    image_path = os.path.join(source_dir, real_filename)
    
    print(f"    [!] Internal Tool Logic resolved {image_reference_id} to local path: {real_filename}")
    print(f"    [*] MedGemma analyzing image: {image_path}")
    
    # 1. Load the original image
    original_img = cv2.imread(image_path)
    if original_img is None:
        return json.dumps({"error": f"Failed to load image at {image_path}. Check file path."})

    # 2. Extract panels using your existing split logic
    try:
        extracted_data = extract_image_panels(
            image_input=original_img, 
            separation_iters=3, 
            thresh_val=240
        )
    except Exception as e:
        return json.dumps({"error": f"Image splitting failed: {str(e)}"})

    # 3. Initialize MedGemma
    pipe = get_medgemma_pipe()
    results = {}

    # 4. Process each panel
    for panel_name, data in extracted_data.items():
        # Convert OpenCV BGR array to PIL RGB Image for MedGemma
        panel_bgr = data['image']
        panel_rgb = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(panel_rgb)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": query}
                ]
            }
        ]

        try:
            # Run inference
            output = pipe(text=messages, max_new_tokens=256)
            
            # --- UPDATED EXTRACTION LOGIC ---
            text_response = "Error: Could not parse MedGemma output."
            
            if isinstance(output, list) and len(output) > 0 and 'generated_text' in output[0]:
                generated_sequence = output[0]['generated_text']
                
                # The pipeline returns the conversation history array.
                # Loop backwards to find the latest 'assistant' response.
                if isinstance(generated_sequence, list):
                    for msg in reversed(generated_sequence):
                        if msg.get('role') == 'assistant':
                            text_response = msg.get('content', '')
                            break
                elif isinstance(generated_sequence, str):
                    text_response = generated_sequence
            else:
                text_response = str(output)
                
            results[panel_name] = text_response
            print(f"        -> {panel_name} analyzed successfully.")
            
        except Exception as e:
            results[panel_name] = f"Inference error: {str(e)}"
            print(f"        -> [X] Failed to analyze {panel_name}: {e}")

    # 5. Return JSON string to the LLM (Now safely stripped of PIL objects)
    return json.dumps({
        "status": "success",
        "panels_detected": len(extracted_data),
        "analysis": results
    })