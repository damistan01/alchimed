import gradio as gr
import json
import csv
import os
import shutil
import datetime
import io

# Import functions from your existing modules
from vision_module import extract_text_from_image, structure_data_with_medgemma, STRUCTURING_MODEL
from reasoning_module import generate_final_prescription, load_json, load_csv, GOALS

def get_next_customer_id():
    data_dir = "data"
    if not os.path.exists(data_dir):
        return "0001"
    
    existing_ids = []
    for folder in os.listdir(data_dir):
        # Check for both new 'XXXX' and legacy 'id_XXXX' formats
        if folder.isdigit():
            existing_ids.append(int(folder))
        elif folder.startswith("id_") and len(folder) > 3 and folder[3:].isdigit():
            existing_ids.append(int(folder[3:]))
            
    if not existing_ids:
        return "0001"
        
    return f"{max(existing_ids) + 1:04d}"

def process_blood_test(customer_id, image_filepath, selected_goal):
    if not image_filepath:
        return "Error: Please upload an image.", "No data."
    if not selected_goal:
        return "Error: Please select a health goal.", "No data."
        
    customer_id = str(customer_id).strip()
    if not customer_id:
        customer_id = get_next_customer_id()
    elif customer_id.startswith("id_"):
        customer_id = customer_id[3:]
    
    # Extract the numeric part of the customer ID (e.g., '0001' from 'id_0001')
    id_num = customer_id.split('_')[-1] if '_' in customer_id else customer_id
    
    # Structure Folders: data/XXXX/YYYY-MM-DD_HH-MM-SS
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    date_str = now.strftime("%Y-%m-%d")
    
    customer_dir = os.path.abspath(os.path.join("data", customer_id))
    session_dir = os.path.join(customer_dir, timestamp)
    os.makedirs(session_dir, exist_ok=True)
    
    # Save original image: original_<ID>_<DATE>.jpg
    original_image_dest = os.path.abspath(os.path.join(session_dir, f"original_{id_num}_{date_str}.jpg"))
    try:
        shutil.copy2(image_filepath, original_image_dest)
        print(f"[App] Saved original image to: {original_image_dest}")
    except Exception as e:
        yield f"Error: Failed to save image. {e}", "Error"
        return

    yield "Extracting text with DocTR...", "Waiting..."
    
    # 1. Vision Pipeline
    raw_text = extract_text_from_image(original_image_dest)
    if not raw_text:
        yield "Error: OCR failed to extract any text.", "Error"
        return
    
    yield f"Extracted {len(raw_text)} characters. Structuring with {STRUCTURING_MODEL}...", "Waiting..."
    extracted_json = structure_data_with_medgemma(raw_text)
    
    if not extracted_json:
        yield "Error: Structuring with MedGemma failed. Check Ollama status.", "Error"
        return

    # Save the extracted data locally as CSV
    extracted_data_path = os.path.join(session_dir, f"extracted_{id_num}_{date_str}.csv")
    
    # Create CSV content
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["patient_age", "patient_sex", "test_name", "value", "unit", "reference"])
    
    age = extracted_json.get("patient_age", "")
    sex = extracted_json.get("patient_sex", "")
    tests = extracted_json.get("blood_tests", [])
    
    for t in tests:
        writer.writerow([
            age, 
            sex, 
            t.get("test_name", ""),
            t.get("value", ""),
            t.get("unit", ""),
            t.get("reference", "")
        ])
        
    csv_string = csv_buffer.getvalue()
    
    with open(extracted_data_path, "w", encoding="utf-8", newline="") as f:
        f.write(csv_string)

    yield csv_string, "Analyzing deficits and calculating prescription..."

    # 2. Reasoning Pipeline
    supplements = load_csv("supplements_db.csv")
    if not supplements:
        yield csv_string, "Error: supplements_db.csv not found."
        return
        
    prescription = generate_final_prescription(extracted_json, supplements, selected_goal)
    print(f"[App] Final Prescription Generated:\n{prescription}")
    print("-" * 30)

    # Save final prescription
    prescription_path = os.path.join(session_dir, f"prescription_{id_num}_{date_str}.txt")
    with open(prescription_path, "w", encoding="utf-8") as f:
        f.write(prescription)

    # Final Output
    yield csv_string, prescription

# --- Gradio Web Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🩺 AI Medical Assistant (Local-First MVP)")
    gr.Markdown("Upload a blood test, select a goal, and get a personalized, pharmacologically-checked supplement protocol. Runs 100% locally.")
    
    with gr.Row():
        with gr.Column():
            customer_input = gr.Textbox(label="Customer ID (Optional)", placeholder="Leave blank to auto-generate XXXX")
            img_input = gr.Image(type="filepath", label="Upload Blood Test (Image)")
            goal_input = gr.Dropdown(choices=GOALS, label="Select Primary Health Goal")
            btn = gr.Button("Generate Protocol", variant="primary")
            
        with gr.Column():
            csv_output = gr.Code(language="markdown", label="1. Extracted Medical Data (CSV)")
            text_output = gr.Textbox(label="2. Final Prescription (TxGemma & Python Math)", lines=20)

    # Connect the button to the pipeline function
    btn.click(
        fn=process_blood_test, 
        inputs=[customer_input, img_input, goal_input], 
        outputs=[csv_output, text_output]
    )

if __name__ == "__main__":
    print("Starting Web UI...")
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)