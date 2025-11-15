📄 README.md (Final — polished, production-ready)
# 🩺 Medimitra – Universal Prescription OCR System v3.0  
### **AI-powered OCR + Regex Extraction + Local LLM Refinement (Phi-3 Mini GGUF)**

Medimitra is a fully offline, high-accuracy prescription digitization engine designed for hospitals, clinics, pharmacies, and healthcare automation platforms.

It combines:

- **Doctr OCR (ResNet50 + CRNN)**
- **Advanced Regex Engine (250+ medical patterns)**
- **Local LLM Refinement (Phi-3 Mini via llama-cpp-python)**
- **Full structured extraction** (Patient, Prescriber, Medications, Diagnosis, Metadata)
- **Smart validation & routing logic**
- **Zero cloud dependency → 100% privacy-safe**

---

## 🚀 Key Features

### 🔍 **OCR Engine (Doctr)**
- Supports PDFs, PNG, JPEG, TIFF
- GPU acceleration (Apple MPS / CUDA)
- Extracts line confidence, bounding-box text, multi-page output

### 🧠 **Extraction Engine**
- 250+ advanced medical regex patterns (PatternLibrary v3.0)
- Patient info (Name, MRN, Age, Gender, Phone)
- Prescriber info (Name, License, DEA, NPI)
- Medication parsing:
  - Name
  - Strength
  - Form
  - Frequency
  - Duration
  - Quantity
  - Route
- Clinical info (Diagnosis, ICD codes, vitals)
- Pharmacy details
- Metadata extraction

### 🤖 **Local LLM Refinement (Offline)**
Uses **Phi-3 Mini GGUF** (through `llama-cpp-python`) to produce a refined, clean text version of the OCR.

### 🛡 Offline & Privacy-Safe  
Runs 100% locally.  
No external API calls.  
HIPAA-friendly architecture.

---

## 📁 Project Structure

medimitra/
│
├── app.py # Flask API server
├── extraction.py # OCR + regex + LLM extraction logic
├── llm_refiner.py # Local LLM (Phi-3 Mini GGUF) inference
├── ner_model.py # Optional biomedical NER
├── models.py # Pydantic models for structured output
├── requirements.txt # Dependencies
├── README.md
├── .gitignore
│
├── uploads/ # Auto-created for temporary files
├── model/ # Place Phi-3 GGUF model here
│ └── phi-3-mini.gguf
│
└── sample_data/
├── sample_prescription.pdf
└── sample_output.json

---

## 🧩 Installation

```bash
git clone https://github.com/<your-username>/medimitra
cd medimitra

python3 -m venv doctr_env
source doctr_env/bin/activate

pip install -r requirements.txt
Place your Phi-3 Mini GGUF model here:
model/phi-3-mini.gguf
▶️ Running the API Server
python3 app.py
Server will start at:
http://127.0.0.1:5001
🧪 API Endpoints
POST /process
Upload a PDF/image for full OCR + extraction + LLM refinement.
Example:
curl -X POST "http://127.0.0.1:5001/process" \
  -F "file=@prescription.pdf" \
  -F "mrn=MRN-001" \
  -F "document_type=Clinical_Prescription"
Response includes:
OCR output
Structured extraction
Medication list
Metadata
Refined text (LLM)
Validation
Routing decision
Raw Doctr OCR output
🔍 Example JSON Output
Includes:
patient
prescriber
medications[]
clinical_info
pharmacy
metadata
refined_text (LLM)
validation
routing_decision
ocr_raw
(See sample_data/sample_output.json for a full reference.)
🧠 Optional: Biomedical NER
You can enable transformers-based medical NER (d4data/biomedical-ner-all):
from ner_model import load_biomedical_ner
ner_pipeline = load_biomedical_ner()
(Currently disabled by default.)
🧭 Roadmap
 RxNorm medication normalization
 Transformer-based NER integration
 Handwriting recognition enhancement
 FHIR-compatible JSON export
 Docker container + CI workflows
 Simple web UI dashboard
📜 License
Open source under the MIT License.
Free to use for commercial & research purposes.
⭐ Acknowledgements
Doctr by Mindee
Phi-3 Mini LLM
Llama-Cpp-Python
Pydantic
All open-source contributors helping healthcare digitization ❤️
