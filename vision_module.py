import os
import json
import requests
import warnings
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

warnings.filterwarnings("ignore", category=UserWarning)

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_CHECK_URL = "http://127.0.0.1:11434/api/tags"
STRUCTURING_MODEL = "medgemma:4b"

# Initialize DocTR predictor (loads models into memory once)
print("[Vision] Loading DocTR OCR model...")
_doctr_model = ocr_predictor(
    det_arch='db_resnet50',
    reco_arch='crnn_vgg16_bn',
    pretrained=True,
    assume_straight_pages=True
)
print("[Vision] DocTR ready.")

def clean_json_response(raw_text):
    """ Removes markdown formatting if the model hallucinates it. """
    raw_text = raw_text.strip()
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    if raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
    return raw_text.strip()

def extract_text_from_image(image_path):
    """Extract text from an image using DocTR with layout awareness."""
    if not os.path.exists(image_path):
        print(f"[Error] Image not found for OCR: {image_path}")
        return ""
        
    print(f"[Vision] Extracting text from {os.path.basename(image_path)} using DocTR...")
    try:
        doc = DocumentFile.from_images(image_path)
        result = _doctr_model(doc)
        
        lines = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = " ".join([word.value for word in line.words])
                    lines.append(line_text)
        
        text = "\n".join(lines)
        print(f"[Vision] DocTR extracted {len(text)} characters across {len(lines)} lines.")
        print("-" * 30)
        print(f"[Vision] Raw OCR text (first 500 chars):\n{text[:500]}...")
        print("-" * 30)
        return text
    except Exception as e:
        print(f"[Error] DocTR failed: {e}")
        return ""

def structure_data_with_medgemma(raw_text):
    """Send OCR text to MedGemma to structure it into a strict JSON schema."""
    if not raw_text.strip():
        print("[Error] No text provided for structuring.")
        return None

    print(f"[Vision] Structuring data using {STRUCTURING_MODEL}...")
    
    # Pre-check if Ollama is responsive
    try:
        requests.get(OLLAMA_CHECK_URL, timeout=5)
    except Exception:
        print("[Error] Ollama server is not responding at 127.0.0.1:11434. Please ensure Ollama is running.")
        return None

    prompt = f"""You are a medical data extraction AI. Parse the following OCR text from a Romanian blood test report and return ONLY a valid JSON object.

RAW OCR TEXT:
{raw_text}

INSTRUCTIONS:
1. Extract patient age (as a string like "48 ani") and sex (M or F).
2. DO NOT include the patient's Name or CNP in the output.
3. Extract every blood test with its name, numeric value, unit, and reference range.
4. Return ONLY the JSON below, no extra text.

REQUIRED JSON FORMAT:
{{
  "patient_age": "48 ani 5 luni",
  "patient_sex": "M",
  "blood_tests": [
    {{"test_name": "Colesterol HDL", "value": "40.9", "unit": "mg/dL", "reference": "scazut: <= 40, factor protector: >= 60"}},
    {{"test_name": "Creatinina serica", "value": "1", "unit": "mg/dL", "reference": "< 1.2"}}
  ]
}}"""
    
    payload = {
        "model": STRUCTURING_MODEL, 
        "prompt": prompt, 
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }
    
    try:
        print(f"[Vision] Full Prompt sent to {STRUCTURING_MODEL}:\n{prompt}")
        print("-" * 30)
        print("[Vision] Sending request to Ollama (this may take a minute)...")
        response = requests.post(OLLAMA_URL, json=payload, timeout=None)
        response.raise_for_status()
        
        response_json = response.json()
        raw_response_text = response_json.get("response", "")
        print(f"[Vision] Raw Response from {STRUCTURING_MODEL}:\n{raw_response_text}")
        print("-" * 30)
        
        if not raw_response_text:
            print("[Error] Ollama returned an empty response.")
            return None
            
        cleaned_json = clean_json_response(raw_response_text)
        return json.loads(cleaned_json)
    except json.JSONDecodeError as e:
        print(f"[Error] Failed to parse JSON from model response: {e}")
        print(f"[Debug] Raw response was: {raw_response_text[:500]}")
        return None
    except Exception as e:
        print(f"[Error] Structuring failed: {e}")
        return None

if __name__ == "__main__":
    import sys
    
    # Allow passing image path via command line
    input_image = sys.argv[1] if len(sys.argv) > 1 else "analize.jpg"
    
    if os.path.exists(input_image):
        print(f"[Debug] Starting standalone test with: {input_image}")
        raw_text = extract_text_from_image(input_image)
        print(f"[Debug] Raw text extracted ({len(raw_text)} chars).")
        
        extracted_json = structure_data_with_medgemma(raw_text)
        
        if extracted_json:
            print("\n[Success] Data extracted and structured:")
            print(json.dumps(extracted_json, indent=4))
            with open("extracted_data.json", "w", encoding="utf-8") as f:
                json.dump(extracted_json, f, indent=4)
        else:
            print("\n[Failure] Structuring failed.")
    else:
        print(f"[Error] Could not find image '{input_image}'.")
        print("Usage: python vision_module.py <path_to_image>")