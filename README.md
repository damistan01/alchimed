# 🩺 Alchimed: AI Medical Assistant

Alchimed is a local-first, privacy-focused medical assistant that analyzes blood test reports to provide personalized health protocols. It uses DocTR for OCR and local LLMs (via Ollama) for clinical reasoning and pharmacological checking.

## 🚀 Getting Started

Follow these steps to set up the project on your local machine.

### 1. Prerequisites

- **Python 3.10+**: Ensure you have Python installed.
- **Ollama**: Download and install from [ollama.com](https://ollama.com/).
- **System Dependencies**:
  - **Windows**: Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (optional but recommended for some backends) and ensure `git` is installed.
  - **Linux/Mac**: `sudo apt install libgl1-mesa-glx` (required for OpenCV/DocTR).

### 2. Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/damistan01/alchimed.git
   cd alchimed
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If `requirements.txt` is missing, run:*
   `pip install gradio requests python-doctr opencv-python tf2-backend-meta` (or `torch` instead of `tf2` depending on your preference).

### 3. Setup Ollama Models

Ensure Ollama is running, then pull the required models:

```bash
ollama pull medgemma:4b
# If txgemma is a custom model, ensure you have the Modelfile or pull the appropriate version
ollama pull gemma:7b 
```

### 4. Running the App

Start the Gradio web interface:

```bash
python app.py
```

The app will be available at [http://127.0.0.1:7860](http://127.0.0.1:7860).

## 🛠️ How it Works

1. **Vision Module**: Uses **DocTR** (Document Text Recognition) to extract text from Romanian blood test reports with layout awareness.
2. **Structuring**: **MedGemma:4b** parses the raw OCR text into a clean JSON format.
3. **Reasoning**:
   - **Clinical Analysis**: Identifies abnormalities (LOW/HIGH values) based on reference ranges.
   - **Supplement Matching**: Matches needs against `supplements_db.csv`.
   - **Pharmacology Check**: Uses **TxGemma** to check for interactions and suggest optimal timing (Morning/Evening).

## 📁 Project Structure

- `app.py`: Main entry point (Gradio UI).
- `vision_module.py`: OCR and data structuring.
- `reasoning_module.py`: Clinical logic and pharmacological scheduling.
- `supplements_db.csv`: Local database of available supplements and concentrations.
- `data/`: (Ignored by Git) Stores session images and results locally.

## 🤝 Contributing

1. Create a branch: `git checkout -b feature/your-feature`.
2. Commit changes: `git commit -m "Add some feature"`.
3. Push to branch: `git push origin feature/your-feature`.
4. Open a Pull Request.

---
*Disclaimer: This tool is for educational/experimental purposes only and does not provide medical advice. Always consult with a doctor.*