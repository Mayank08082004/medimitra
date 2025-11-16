"""
extraction.py

Core extraction logic for the Universal Prescription OCR System.

Includes:
- PatternLibrary (Regex rules)
- UniversalExtractor (Regex engine + optional LLM refinement)
- Device/OCR initialization helpers
- OCR processing function
"""

import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

from llm_refiner import refine_text_with_llm
from models import (
    Medication,
    Patient,
    Prescriber,
    Pharmacy,
    PrescriptionMetadata,
    ClinicalInfo,
    PrescriptionType,
)

# -------- ADVANCED REGEX PATTERNS --------
class PatternLibrary:
    """Comprehensive regex patterns for medical document extraction"""

    # Patient Information
    PATIENT_NAME = [
        r'(?:Patient\s*Name|Name|Pt\.?\s*Name|Patient)\s*[:\-]?\s*([A-Z][a-zA-Z\s,.\'-]{2,100})',
        r'Name\s*[:\-]\s*([A-Z][a-zA-Z\s,.\'-]{2,100})',
        r'^([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})$',
    ]

    PATIENT_ID = [
        r'(?:MRN|Patient\s*ID|Medical\s*Record|ID|Chart\s*#)\s*[:\-#]?\s*([A-Za-z0-9\-_/]{3,40})',
        r'(?:Pt\s*ID|Record\s*#|Reg\.\s*No)\s*[:\-#]?\s*([A-Za-z0-9\-_/]{3,40})',
    ]

    DOB = [
        r'(?:DOB|Date\s*of\s*Birth|Birth\s*Date)\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
        r'(?:DOB|Birth)\s*[:\-]?\s*(\d{4}\-\d{2}\-\d{2})',
        r'(?:Born|Birth)\s*[:\-]?\s*(\w+\s+\d{1,2},?\s*\d{4})',
    ]

    AGE = [
        r'Age,\s*Gender\s*[:\-]?\s*(\d{1,3})',
        r'(?:Age|Years|Yrs?)\s*[:\-]?\s*(\d{1,3})\s*(?:years?|yrs?|y)?',
        r'(\d{1,3})\s*(?:year|yr)s?\s*old',
    ]

    GENDER = [
        r'(?:Sex|Gender)\s*[:\-]?\s*(Male|Female|M|F|Other|Non-binary)',
        r'\b(Male|Female)\b',
    ]

    PHONE = [
        r'(?:Phone|Tel|Contact|Mobile|Cell|Ph)\s*[:\-]?\s*([\d\s\-\(\)\.]{10,20})',
        r'(\d{3}[\-\.\s]?\d{3}[\-\.\s]?\d{4})',
        r'\((\d{3})\)\s*(\d{3})[\-\.]?(\d{4})',
    ]

    # Doctor Information
    DOCTOR_NAME = [
        r'(?:Dr\.?|Doctor|Physician|Provider|Prescriber)\s+([A-Z][a-zA-Z\s,.\'-]{2,80})',
        r'(?:Signed|Signature)\s*[:\-]?\s*Dr\.?\s+([A-Z][a-zA-Z\s,.\'-]{2,80})',
        r'([A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+),?\s*(?:MD|DO|MBBS|PhD|DPM|DDS|PharmD|NP|PA)',
    ]

    LICENSE = [
        r'(?:License|Lic\.?|Medical\s*License)\s*[:\-#]?\s*([A-Z0-9\-]{5,30})',
        r'(?:State\s*Lic|Med\s*Lic|LIC\s*#)\s*[:\-#]?\s*([A-Z0-9\-]{5,30})',
    ]

    NPI = [r'(?:NPI|National\s*Provider)\s*[:\-#]?\s*(\d{10})']

    DEA = [r'(?:DEA|Drug\s*Enforcement|DEA#)\s*[:\-#]?\s*([A-Z]{2}\d{7})']

    # Prescription Numbers
    RX_NUMBER = [
        r'(?:Rx|RX|Prescription)\s*[#:\-]?\s*([A-Z0-9\-]{5,30})',
        r'(?:Script\s*#|Presc\s*#)\s*([A-Z0-9\-]{5,30})',
    ]

    # Dates
    DATE_PATTERNS = [
        r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
        r'(\d{4}\-\d{1,2}\-\d{1,2})',
        r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s*\d{2,4})',
        r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{2,4})',
    ]

    DATE_WRITTEN = [
        r'(?:Date|Written|Prescribed|Issued)\s*[:\-]?\s*' + DATE_PATTERNS[0],
        r'(?:Date|Written|Prescribed|Issued)\s*[:\-]?\s*' + DATE_PATTERNS[1],
        r'(?:Date|Written|Prescribed|Issued)\s*[:\-]?\s*' + DATE_PATTERNS[2],
    ]

    # Medications
    MEDICATION_KEYWORDS = (
        r'\b(mg|mcg|g|ml|units?|iu|mEq|%|tablet|cap(?:sule)?|syrup|suspension|injection|cream|'
        r'ointment|gel|patch|inhaler|drops?|spray|powder|suppository|lozenge|solution|once|twice|'
        r'three times?|four times?|daily|bid|tid|qid|qd|qh|q\d+h|prn|as needed|every \d+ hours?|'
        r'morning|evening|night|bedtime|before meals?|after meals?|with food|oral|topical|IV|IM|SC|'
        r'sublingual|rectal|vaginal|ophthalmic|otic|nasal|transdermal|inhalation)\b'
    )

    STRENGTH = [
        r'(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?|iu|mEq|%)',
        r'(\d+(?:\.\d+)?)\s*(?:mg|mcg|g|ml|units?)/(\d+(?:\.\d+)?)\s*(?:mg|mcg|g|ml)',
    ]

    DOSAGE_FORM = [
        r'\b(tablet|cap(?:sule)?|syrup|suspension|injection|cream|ointment|gel|patch|inhaler|drops?|'
        r'spray|powder|suppository|lozenge|solution)s?\b'
    ]

    FREQUENCY = [
        r'\b(once|twice|three times?|four times?|daily|bid|tid|qid|qd|qh|q\d+h|prn|as needed|every \d+ hours?)\b',
        r'\b(\d+x|x\d+)\s*(?:daily|per day|a day)\b',
        r'\b(morning|evening|night|bedtime|before meals?|after meals?|with food)\b',
    ]

    DURATION = [
        r'(?:for|take for|duration)\s*(\d+)\s*(day|week|month)s?',
        r'(\d+)\s*(day|week|month)s?\s*(?:supply|course)',
        r'x\s*(\d+)\s*(days?|weeks?|months?)',
    ]

    QUANTITY = [
        r'(?:Qty|Quantity|Disp|Dispense)\s*[:\-]?\s*(\d+)',
        r'#\s*(\d+)',
    ]

    REFILLS = [
        r'(?:Refill|Refills?|Repeats?)\s*[:\-]?\s*(\d+|none|no|zero)',
        r'(?:Ref|Rpts?)\s*[:\-]?\s*(\d+)',
    ]

    ROUTE = [
        r'\b(oral|topical|IV|IM|SC|sublingual|rectal|vaginal|ophthalmic|otic|nasal|transdermal|inhalation)\b'
    ]

    # Diagnosis
    DIAGNOSIS = [
        r'(?:Diagnosis|Dx|Impression|Condition|Clinical Description)\s*[:\-]?\s*(.{5,200})',
        r'(?:ICD(?:\-10)?)\s*[:\-]?\s*([A-Z]\d{2}(?:\.\d{1,4})?)',
    ]

    # Pharmacy
    PHARMACY_NAME = [
        r'(?:Pharmacy|Rx|Dispense(?:d)?\s*(?:at|by))\s*[:\-]?\s*([A-Za-z\s&\'-]{3,100})'
    ]

    # Allergies
    ALLERGIES = [
        r'(?:Allerg(?:y|ies)|Allergic\s*to)\s*[:\-]?\s*(.{3,200})',
        r'(?:NKDA|No\s*Known\s*Drug\s*Allergies)',
    ]


# -------- EXTRACTION ENGINE --------
class UniversalExtractor:
    """Advanced extraction engine (Regex + optional LLM refinement)."""

    def __init__(self, ner_pipeline: Optional[Any] = None):
        # Currently regex-only; NER pipeline kept for future extension.
        self.ner_pipeline = None
        self.patterns = PatternLibrary()

    def extract_with_confidence(
        self, text: str, patterns: List[str], flags=re.IGNORECASE
    ) -> Tuple[Optional[str], float]:
        """Try multiple patterns and return best match with confidence."""
        for pattern in patterns:
            match = re.search(pattern, text, flags)
            if match:
                value = match.group(1).strip().split("\n")[0]
                return value, 0.9
        return None, 0.0

    def extract_patient(self, text: str, ml_entities: List[Dict]) -> Patient:
        """Extract comprehensive patient information (Regex-Only)."""
        patient = Patient()

        patient.name, _ = self.extract_with_confidence(text, self.patterns.PATIENT_NAME)
        patient.id, _ = self.extract_with_confidence(text, self.patterns.PATIENT_ID)
        patient.dob, _ = self.extract_with_confidence(text, self.patterns.DOB)
        patient.age, _ = self.extract_with_confidence(text, self.patterns.AGE)
        patient.gender, _ = self.extract_with_confidence(text, self.patterns.GENDER)
        patient.phone, _ = self.extract_with_confidence(text, self.patterns.PHONE)

        # Allergies
        allergy_match = re.search(self.patterns.ALLERGIES[0], text, re.IGNORECASE)
        if allergy_match:
            allergy_text = allergy_match.group(1).strip()
            patient.allergies = [a.strip() for a in re.split(r"[,;]", allergy_text) if a.strip()]
        elif re.search(self.patterns.ALLERGIES[1], text, re.IGNORECASE):
            patient.allergies = ["NKDA"]

        return patient

    def extract_prescriber(self, text: str, ml_entities: List[Dict]) -> Prescriber:
        """Extract prescriber information (Regex-Only)."""
        prescriber = Prescriber()

        prescriber.name, _ = self.extract_with_confidence(text, self.patterns.DOCTOR_NAME)
        prescriber.license, _ = self.extract_with_confidence(text, self.patterns.LICENSE)
        prescriber.npi, _ = self.extract_with_confidence(text, self.patterns.NPI)
        prescriber.dea, _ = self.extract_with_confidence(text, self.patterns.DEA)

        phone_text = text
        if prescriber.name:
            name_pos = text.find(prescriber.name)
            if name_pos != -1:
                phone_text = text[name_pos:]
        prescriber.phone, _ = self.extract_with_confidence(phone_text, self.patterns.PHONE)

        if re.search(r"(?:signed|signature|/s/)", text, re.IGNORECASE):
            prescriber.signature_present = True

        return prescriber

    def extract_medications(self, text: str, ml_entities: List[Dict]) -> List[Medication]:
        """Extract detailed medication information using a reliable line-based approach."""
        medications: List[Medication] = []

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            # Must have medication keywords or look like a bullet/line item
            if not (
                re.search(self.patterns.MEDICATION_KEYWORDS, line, re.IGNORECASE)
                or re.match(r"^\s*[\-\•\*]\s*\w+", line)
            ):
                continue

            # Ignore obvious header / non-med lines
            lower = line.lower()
            if (
                lower.startswith("name:")
                or lower.startswith("date:")
                or lower.startswith("address:")
                or lower.startswith("age:")
                or lower.startswith("dea#")
                or lower.startswith("lic #")
                or lower.startswith("mbbs")
                or lower.startswith("reg. no:")
                or lower.startswith("chc,")
                or lower.startswith("weight:")
                or lower.startswith("clinical description:")
                or lower.startswith("advice:")
                or lower.startswith("dlabel")
                or lower.startswith("refill")
                or lower.startswith("wtx-n-presc-t")
                or lower.startswith("medical - centre")
                or line.strip().isdigit()
            ):
                continue

            med = Medication(name="", instructions=line, confidence=0.85)

            # Try to extract a reasonably capitalized name
            name_match = re.search(
                r"^\s*[\-\•\*]?\s*(?:[A-Za-z0-9]+\s+)?([A-Z]{3,}[a-zA-Z0-9\-]*)", line
            )
            if name_match:
                med.name = name_match.group(1)
            else:
                name_match_fallback = re.match(
                    r"^\s*[\-\•\*]?\s*([A-Za-z]{3,}[a-zA-Z0-9\-]*)", line
                )
                if name_match_fallback:
                    med.name = name_match_fallback.group(1)

            if not med.name or med.name.isdigit():
                # Skip junk lines
                continue

            strength_match = re.search(self.patterns.STRENGTH[0], line, re.IGNORECASE)
            if strength_match:
                med.strength = f"{strength_match.group(1)}{strength_match.group(2)}"

            form_match = re.search(self.patterns.DOSAGE_FORM[0], line, re.IGNORECASE)
            if form_match:
                med.form = form_match.group(1).lower()

            freq_match = re.search(self.patterns.FREQUENCY[0], line, re.IGNORECASE)
            if freq_match:
                med.frequency = freq_match.group(1)

            dur_match = re.search(self.patterns.DURATION[0], line, re.IGNORECASE)
            if dur_match:
                med.duration = f"{dur_match.group(1)} {dur_match.group(2)}"

            qty_match = re.search(self.patterns.QUANTITY[0], line, re.IGNORECASE)
            if qty_match:
                med.quantity = qty_match.group(1)

            ref_match = re.search(self.patterns.REFILLS[0], line, re.IGNORECASE)
            if ref_match:
                med.refills = ref_match.group(1)

            route_match = re.search(self.patterns.ROUTE[0], line, re.IGNORECASE)
            if route_match:
                med.route = route_match.group(1).lower()

            medications.append(med)

        return medications[:50]

    def extract_clinical_info(self, text: str, ml_entities: List[Dict]) -> ClinicalInfo:
        """Extract clinical information (Regex-Only)."""
        clinical = ClinicalInfo()

        diag_match = re.search(self.patterns.DIAGNOSIS[0], text, re.IGNORECASE)
        if diag_match:
            diag_text = diag_match.group(1).strip().split("\n")[0]
            if not any(d.lower() == diag_text.lower() for d in clinical.diagnosis):
                clinical.diagnosis.append(diag_text)

        keyword_match = re.search(
            r"\b(URTI|fever|cough|pain|infection|diabetes|hypertension)\b",
            text,
            re.IGNORECASE,
        )
        if keyword_match and not clinical.diagnosis:
            clinical.diagnosis.append(keyword_match.group(0))

        icd_matches = re.findall(self.patterns.DIAGNOSIS[1], text)
        clinical.icd_codes = list(set(icd_matches))

        vital_patterns = {
            "bp": r"(?:BP|Blood\s*Pressure)\s*[:\-]?\s*(\d{2,3}/\d{2,3})",
            "hr": r"(?:HR|Heart\s*Rate|Pulse)\s*[:\-]?\s*(\d{2,3})",
            "temp": r"(?:Temp|Temperature)\s*[:\-]?\s*(\d{2,3}(?:\.\d)?)",
            "rr": r"(?:RR|Respiratory\s*Rate)\s*[:\-]?\s*(\d{1,2})",
            "spo2": r"(?:SpO2|O2\s*Sat)\s*[:\-]?\s*(\d{2,3})%?",
        }

        for key, pattern in vital_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                clinical.vital_signs[key] = match.group(1)

        return clinical

    def extract_metadata(self, text: str) -> PrescriptionMetadata:
        """Extract prescription metadata (Regex-Only)."""
        metadata = PrescriptionMetadata()

        metadata.prescription_number, _ = self.extract_with_confidence(
            text, self.patterns.RX_NUMBER
        )

        metadata.date_written, _ = self.extract_with_confidence(
            text, self.patterns.DATE_WRITTEN
        )
        if not metadata.date_written:
            metadata.date_written, _ = self.extract_with_confidence(
                text, self.patterns.DATE_PATTERNS
            )

        if re.search(r"\b(Schedule\s*(?:II|III|IV|V)|controlled\s*substance|DEA)\b", text, re.IGNORECASE):
            metadata.prescription_type = PrescriptionType.CONTROLLED_SUBSTANCE
        elif re.search(r"\b(discharge|hospital discharge|inpatient)\b", text, re.IGNORECASE):
            metadata.prescription_type = PrescriptionType.HOSPITAL_DISCHARGE
        elif re.search(r"\b(e-?prescri|electronic|digital)\b", text, re.IGNORECASE):
            metadata.prescription_type = PrescriptionType.E_PRESCRIPTION
            metadata.is_electronic = True
        elif re.search(r"\b(compound|compounded)\b", text, re.IGNORECASE):
            metadata.prescription_type = PrescriptionType.COMPOUND
        else:
            metadata.prescription_type = PrescriptionType.STANDARD

        return metadata

    def extract_pharmacy(self, text: str) -> Pharmacy:
        """Extract pharmacy information (Regex-Only)."""
        pharmacy = Pharmacy()

        pharmacy.name, _ = self.extract_with_confidence(text, self.patterns.PHARMACY_NAME)

        phone_text = text
        if pharmacy.name:
            name_pos = text.find(pharmacy.name)
            if name_pos != -1:
                phone_text = text[name_pos:]

        pharmacy.phone, _ = self.extract_with_confidence(phone_text, self.patterns.PHONE)

        return pharmacy

    def extract_all(self, text: str) -> Dict[str, Any]:
            """Perform complete extraction using Refine-then-Extract."""
            try:
                ml_entities: List[Dict] = []
                
                # --- 1️⃣ LLM REFINEMENT (THE FIX) ---
                # Run the LLM to clean the raw text *first*.
                logging.info("🧠 Passing raw text to LLM refiner...")
                refined_text = refine_text_with_llm(text)
                
                if not refined_text:
                    logging.warning("LLM refinement failed or returned empty, falling back to raw text.")
                    refined_text = text

                # --- 2️⃣ Regex-only extraction baseline ---
                # Now, run all regex extractors on the *clean, refined* text.
                logging.info("⚙️ Running regex engine on refined text...")
                patient = self.extract_patient(refined_text, ml_entities)
                prescriber = self.extract_prescriber(refined_text, ml_entities)
                medications = self.extract_medications(refined_text, ml_entities)
                clinical = self.extract_clinical_info(refined_text, ml_entities)
                metadata = self.extract_metadata(refined_text)
                pharmacy = self.extract_pharmacy(refined_text)

                metadata_dict = asdict(metadata)
                if isinstance(metadata_dict.get("prescription_type"), PrescriptionType):
                    metadata_dict["prescription_type"] = metadata_dict["prescription_type"].value

                final_result: Dict[str, Any] = {
                    "patient": asdict(patient),
                    "prescriber": asdict(prescriber),
                    "medications": [asdict(m) for m in medications],
                    "clinical_info": asdict(clinical),
                    "metadata": metadata_dict,
                    "pharmacy": asdict(pharmacy),
                    "ml_entities_count": 0,
                    "refined_text": refined_text,  # Keep this for debugging
                }

                return final_result
                
            except Exception as e:
                logging.error(f"Extraction error: {e}", exc_info=True)
                return {"error": str(e)}


# -------- DEVICE / OCR HELPERS --------
def init_device():
    if torch.backends.mps.is_available():
        logging.info("🔧 Apple MPS (Metal) available — using GPU acceleration.")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        logging.info("🔧 NVIDIA GPU available — using CUDA.")
        device = torch.device("cuda")
    else:
        logging.warning("⚠️ No GPU — using CPU fallback.")
        device = torch.device("cpu")
    return device


def init_ocr():
    device = init_device()
    try:
        logging.info("Initializing Doctr OCR model...")
        ocr_instance = ocr_predictor(
            det_arch="db_resnet50",
            reco_arch="crnn_vgg16_bn",
            pretrained=True,
        ).to(device)
        logging.info("✅ OCR initialized successfully.")
        return ocr_instance
    except Exception as e:
        logging.critical(f"❌ OCR initialization failed: {e}")
        return None


def process_document_ocr(file_path: str, ocr_model: Any) -> Dict[str, Any]:
    """Processes PDF or image file and returns OCR text and stats."""
    if ocr_model is None:
        return {"success": False, "error": "OCR not initialized"}

    logging.info(f"📄 OCR processing: {file_path}")
    try:
        ext = Path(file_path).suffix.lower()
        doc = DocumentFile.from_pdf(file_path) if ext == ".pdf" else DocumentFile.from_images(file_path)
        result = ocr_model(doc)
        export = result.export()

        all_results: Dict[str, List[Dict[str, Any]]] = {}
        all_confidences: List[float] = []

        for page_idx, page in enumerate(export.get("pages", [])):
            page_regions: List[Dict[str, Any]] = []
            for block in page.get("blocks", []):
                for line in block.get("lines", []):
                    line_text = " ".join([w.get("value", "") for w in line.get("words", [])])
                    line_conf = (
                        np.mean([w.get("confidence", 0.0) for w in line.get("words", [])])
                        if line.get("words")
                        else 0.0
                    )
                    page_regions.append(
                        {
                            "text": line_text,
                            "confidence": float(line_conf),
                        }
                    )
                    all_confidences.append(line_conf)
            all_results[f"page_{page_idx}"] = page_regions

        page_texts = [page.render() for page in result.pages]
        avg_conf = float(np.mean(all_confidences)) if all_confidences else 0.0

        return {
            "success": True,
            "ocr_results": all_results,
            "page_texts": page_texts,
            "full_text": "\n\n".join(page_texts),
            "average_confidence": round(avg_conf, 6),
            "page_count": len(page_texts),
            "total_text_regions": sum(len(v) for v in all_results.values()),
        }
    except Exception as e:
        logging.error(f"OCR error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
