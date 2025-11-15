# 🩺 Medimitra – Universal Prescription OCR System v3.0  
### **AI-powered OCR + Regex Extraction + Local LLM Refinement (Phi-3 Mini GGUF)**

Medimitra is a fully offline, high-accuracy prescription digitization engine designed for hospitals, clinics, pharmacies, and healthcare automation platforms.

It combines:
- **Doctr OCR (ResNet50 + CRNN)**
- **Advanced Regex Engine (250+ medical patterns)**
- **Local LLM Refinement (Phi-3 Mini via llama-cpp-python)**
- **Full structured extraction** (Patient, Prescriber, Medications, Diagnosis, Metadata)
- **Smart validation & routing logic**
- **Zero cloud dependency — 100% privacy-safe**

---

## 🚀 Key Features
### 🔍 OCR Engine (Doctr)
- Supports PDFs, PNG, JPEG, TIFF  
- GPU acceleration (Apple MPS / CUDA)  
- Multi-page extraction with confidence scores  

### 🧠 Extraction Engine
- 250+ medical regex patterns  
- Patient, Prescriber, Pharmacy extraction  
- Medication parsing (strength, dose, frequency, duration, route)  
- Clinical info + ICD codes  
- Metadata (date written, prescription type, etc.)

### 🤖 Local LLM Refinement  
- Offline Phi-3 Mini (GGUF) via llama-cpp-python  
- Produces clean, corrected medical text  

---

## 📁 Project Structure

```
medimitra/
├── app.py
├── extraction.py
├── llm_refiner.py
├── ner_model.py
├── models.py
├── requirements.txt
├── README.md
├── .gitignore
├── uploads/
├── model/
│   └── phi-3-mini.gguf
└── sample_data/
```

---

## 🧩 Installation

```bash
git clone https://github.com/<your-user>/medimitra
cd medimitra

python3 -m venv doctr_env
source doctr_env/bin/activate

pip install -r requirements.txt
```

Place your GGUF model inside:

```
model/phi-3-mini.gguf
```

---

## ▶️ Run the Server

```bash
python3 app.py
```

Server starts at:

```
http://127.0.0.1:5001
```

---

## 🧪 API: `/process`

```bash
curl -X POST "http://127.0.0.1:5001/process"   -F "file=@prescription.pdf"   -F "mrn=MRN-001"   -F "document_type=Clinical_Prescription"
```

---

## 📜 License  
MIT License – free for commercial use.

---

# ⭐ If you use Medimitra, please ⭐ the repo!
