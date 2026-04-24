import json
import requests
import os
import re
import csv

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MEDGEMMA_MODEL = "medgemma:4b"
TXGEMMA_MODEL = "txgemma" 

GOALS =[
    "Sleep and Circadian Rhythm Optimization",
    "Biological Aging and Longevity (Geroscience)",
    "Metabolic Health and Glucose Homeostasis",
    "Immunological Homeostasis and Inflammatory Response",
    "Skin Longevity (Skin Health)",
    "Endocrine Axis (Thyroid and Reproductive Hormones)",
    "Muscle Protein Synthesis and Hypertrophy",
    "Cognitive Performance (Focus) and Stress Response"
]

def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"[Error] File {filepath} does not exist.")
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_csv(filepath):
    if not os.path.exists(filepath):
        print(f"[Error] File {filepath} does not exist.")
        return None
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'concentration_mg' in row:
                try:
                    row['concentration_mg'] = float(row['concentration_mg'])
                except ValueError:
                    pass
            if 'indications' in row:
                row['indications'] = [i.strip() for i in row['indications'].split(';')]
            data.append(row)
    return data

def extract_json_array(text):
    match = re.search(r'\[.*\]', text, re.DOTALL)
    return json.loads(match.group(0)) if match else[]

def extract_json_object(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return json.loads(match.group(0)) if match else {}



def calculate_pill_fraction(required_mg, pill_concentration_mg):
    return round(required_mg / pill_concentration_mg, 2)

def detect_abnormalities(blood_tests):
    abnormalities =[]
    for test in blood_tests:
        val_str = str(test.get("value", ""))
        ref_str = str(test.get("reference", ""))
        
        refs = re.findall(r'\d+\.\d+|\d+', ref_str)
        vals = re.findall(r'\d+\.\d+|\d+', val_str)
        
        if vals and len(refs) >= 2:
            try:
                val = float(vals[0])
                min_ref = float(refs[0])
                max_ref = float(refs[1])
                
                if min_ref > max_ref:
                    min_ref, max_ref = max_ref, min_ref
                    
                if val < min_ref:
                    abnormalities.append(f"{test['test_name']} is LOW ({val} vs min {min_ref})")
                elif val > max_ref:
                    abnormalities.append(f"{test['test_name']} is HIGH ({val} vs max {max_ref})")
            except:
                pass
    return abnormalities

def match_supplement(requested, db_name):
    req = str(requested).lower() if requested else ""
    db = str(db_name).lower() if db_name else ""
    
    if req in db: return True
    
    if "vitamin d" in req and "vitamin d3" in db: return True
    if "omega" in req and "omega" in db: return True
    if "magnesium" in req and "magnesium" in db: return True
    if "zinc" in req and "zinc" in db: return True
    if "iron" in req and "iron" in db: return True
    if "probiotic" in req and "probiotic" in db: return True
    if "vitamin c" in req and "vitamin c" in db: return True
    if "vitamin b12" in req and "vitamin b12" in db: return True
    
    req_words = set(re.findall(r'\b\w+\b', req))
    db_words = set(re.findall(r'\b\w+\b', db))
    if req_words.issubset(db_words): return True
    
    return False

def get_clinical_reasoning_medgemma(medical_data, user_goal):
    print(f"\n[Agent 1] Analyzing clinical data using {MEDGEMMA_MODEL}...")
    
    abnormalities = detect_abnormalities(medical_data.get("blood_tests",[]))
    print(f"[Agent 1] Detected abnormalities: {abnormalities}")
    
    prompt = f"""
    You are an expert hematologist and functional medicine AI.
    
    SYSTEM DETECTED ABNORMALITIES:
    {json.dumps(abnormalities, indent=2)}
    
    PRIMARY GOAL: "{user_goal}"
    
    CRITICAL INSTRUCTIONS:
    1. Recommend supplements to address the ABNORMALITIES listed above.
    2. Recommend 1 or 2 supplements for the PRIMARY GOAL.
    3. Return STRICTLY a JSON array.
    
    FORMAT EXAMPLE:[
      {{"substance": "Iron", "daily_need_mg": 50, "reason": "Hemoglobin is LOW"}},
      {{"substance": "Vitamin C", "daily_need_mg": 1000, "reason": "Supports Immunological Homeostasis goal"}}
    ]
    """
    
    print(f"[Agent 1] Full Clinical Reasoning Prompt:\n{prompt}")
    print("-" * 30)
    
    payload = {
        "model": MEDGEMMA_MODEL, 
        "prompt": prompt, 
        "stream": False, 
        "options": {"temperature": 0.1, "num_ctx": 4096}
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        raw_text = response.json().get("response", "")
        print(f"[Agent 1] Raw Clinical Reasoning Response:\n{raw_text}")
        print("-" * 30)
        return extract_json_array(raw_text)
    except Exception as e:
        print(f"[Error] MedGemma failed: {e}")
        return[]

def get_pharmacology_txgemma(selected_supplements):
    print(f"[Agent 2] Checking interactions and scheduling via {TXGEMMA_MODEL}...")
    
    prompt = f"""
    You are a clinical pharmacologist AI.
    Review this list of supplements:
    {json.dumps(selected_supplements, indent=2)}
    
    1. Check for severe drug-supplement interactions.
    2. Assign each supplement to "Morning" or "Evening" based on optimal absorption.
    
    Return STRICTLY a JSON object.
    FORMAT EXAMPLE:
    {{
        "interactions_warning": "None",
        "schedule":[
            {{"name": "Vitamin C", "timing": "Morning", "reason": "Energy boost"}}
        ]
    }}
    """
    
    print(f"[Agent 2] Full Pharmacology Prompt:\n{prompt}")
    print("-" * 30)
    
    payload = {
        "model": TXGEMMA_MODEL, 
        "prompt": prompt, 
        "stream": False, 
        "options": {"temperature": 0.1, "num_ctx": 4096}
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=None)
        response.raise_for_status()
        raw_text = response.json().get("response", "")
        print(f"[Agent 2] Raw Pharmacology Response:\n{raw_text}")
        print("-" * 30)
        return extract_json_object(raw_text)
    except Exception as e:
        print(f"[Error] TxGemma failed: {e}")
        return {"interactions_warning": "Check failed.", "schedule":[]}

def generate_final_prescription(medical_data, supplements_db, user_goal):
    ai_recommendations = get_clinical_reasoning_medgemma(medical_data, user_goal)
    
    if not ai_recommendations:
        return "Could not generate recommendations. AI returned empty data."

    calculated_supplements =[]
    for rec in ai_recommendations:
        if not isinstance(rec, dict):
            continue
        requested_substance = rec.get("substance", "")
        try:
            required_mg = float(rec.get("daily_need_mg", 0))
        except (ValueError, TypeError):
            required_mg = 0.0
        
        found_product = None
        for product in supplements_db:
            if match_supplement(requested_substance, product["name"]):
                found_product = product
                break
        
        if found_product and required_mg > 0:
            # IN STOCK
            if not any(s["name"] == found_product["name"] for s in calculated_supplements):
                pills = calculate_pill_fraction(required_mg, found_product["concentration_mg"])
                calculated_supplements.append({
                    "name": found_product['name'],
                    "dosage_pills": pills,
                    "target_mg": required_mg,
                    "clinical_reason": rec.get("reason", ""),
                    "in_stock": True
                })
        elif required_mg > 0:
            # OUT OF STOCK (Not in DB)
            if not any(s["name"].lower() == str(requested_substance).lower() for s in calculated_supplements):
                calculated_supplements.append({
                    "name": str(requested_substance).title(),
                    "dosage_pills": "N/A",
                    "target_mg": required_mg,
                    "clinical_reason": rec.get("reason", ""),
                    "in_stock": False
                })

    if not calculated_supplements:
        return "No recommendations generated."

    # TxGemma now receives BOTH in-stock and out-of-stock items to check interactions
    tx_data = get_pharmacology_txgemma(calculated_supplements)
    
    prescription_text = "\n💊 ADVANCED PERSONALIZED PRESCRIPTION\n"
    prescription_text += "="*60 + "\n"
    prescription_text += f"Goal: {user_goal}\n"
    prescription_text += f"Interactions Warning: {tx_data.get('interactions_warning', 'None')}\n"
    prescription_text += "="*60 + "\n\n"
    
    for supp in calculated_supplements:
        timing = "Unknown"
        timing_reason = ""
        schedule_list = tx_data.get("schedule", [])
        if not isinstance(schedule_list, list):
            schedule_list = []
            
        for schedule_item in schedule_list:
            if not isinstance(schedule_item, dict):
                continue
            if match_supplement(supp["name"], schedule_item.get("name", "")):
                timing = schedule_item.get("timing", "Unknown")
                timing_reason = schedule_item.get("reason", "")
                break
        
        if supp["in_stock"]:
            prescription_text += f"🔹 {supp['name']}\n"
            prescription_text += f"   - Dosage: {supp['dosage_pills']} pill(s) (Target: {supp['target_mg']} mg/IU)\n"
        else:
            prescription_text += f"⚠️ {supp['name']} [NOT IN DATABASE]\n"
            prescription_text += f"   - Dosage: Please source externally (Target: {supp['target_mg']} mg/IU)\n"
            
        prescription_text += f"   - Timing: {timing} ({timing_reason})\n"
        prescription_text += f"   - Clinical Reason: {supp['clinical_reason']}\n\n"
            
    return prescription_text



if __name__ == "__main__":
    extracted_data = load_json("extracted_data.json")
    supplements = load_csv("supplements_db.csv")
    
    if extracted_data and supplements:
        selected_goal = get_user_goal()
        prescription = generate_final_prescription(extracted_data, supplements, selected_goal)
        
        print(prescription)
        with open("final_prescription.txt", "w", encoding="utf-8") as f:
            f.write(prescription)