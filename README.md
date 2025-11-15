AI-powered Prescription Parsing (OCR + Regex + Local LLM Refinement)
This project is a fully offline, privacy-safe, and high-accuracy medical prescription extractor.
It combines:
Doctr OCR (ResNet50 + CRNN)
Advanced medical regex library (PatternLibrary v3.0)
Local LLM refinement (Phi-3-Mini GGUF)
Structured extraction: Patient, Prescriber, Medications, Metadata
Smart routing (AUTO / REVIEW / CONTROLLED SUBSTANCE)
Built for clinical digitization, hospital automation, and EHR ingestion.
🚀 Features
🔍 OCR Engine
Supports PDF, PNG, JPEG, TIFF, etc.
GPU-accelerated on Apple MPS and CUDA
Extracts line-level confidence scores + raw text
🤖 Extraction Engine
250+ medical regex patterns
Medication parsing (strength, frequency, duration, route)
Clinical info
Diagnosis detection
Prescriber + DEA/NPI extraction
Pharmacy extraction
🧠 Local LLM Refinement
Uses llama-cpp-python + Phi-3-mini.gguf
Refines the OCR text to generate a corrected human-readable output.
🛡 Privacy
Runs fully offline.
No API calls.
No cloud dependencies.
📦 Installation
git clone https://github.com/<your-username>/universal-prescription-ocr.git
cd universal-prescription-ocr

python3 -m venv doctr_env
source doctr_env/bin/activate

pip install -r requirements.txt
Make sure to place your GGUF model inside model/:
model/phi-3-mini.gguf
▶️ Running the Server
python3 app.py
Server starts on:
http://localhost:5001
📝 API Endpoints
POST /process
Upload a prescription PDF or image.
Example:
curl -X POST "http://127.0.0.1:5001/process" \
  -F "file=@prescription.pdf" \
  -F "mrn=MRN-007" \
  -F "document_type=Clinical_Prescription"
Returns:
OCR text
Extracted patient, prescriber, medications
Metadata
LLM refined text
Validation results
Routing decision
🧩 Project Structure
app.py                 ← Flask server
extraction.py          ← OCR + regex + LLM extraction engine
llm_refiner.py         ← Local LLM inference
ner_model.py           ← Optional biomedical NER
models.py              ← Pydantic models
requirements.txt       ← Dependencies
sample_data/           ← Example PDFs + outputs
model/                 ← Local GGUF model
uploads/               ← Temporary saved inputs
📘 Roadmap
 Add biomedical NER integration
 Add medication normalization (RxNorm)
 Export to FHIR resources
 Add multi-language OCR
 Add UI dashboard
🛡 License
MIT License — open source, free to use.
