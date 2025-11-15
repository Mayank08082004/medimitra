"""
app.py

Runs the Flask web server for the Universal Prescription OCR System.
v3.0: Regex + local Phi-3-mini (GGUF) refinement.
"""

import os
import uuid
import shutil
import json
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging

from models import PrescriptionType
from extraction import (
    init_device,
    init_ocr,
    process_document_ocr,
    UniversalExtractor,
)

# Optional NER (not required for v3.0)
# from ner_model import load_biomedical_ner


# -------- CONFIGURATION --------
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB

USE_LLM_REFINEMENT = True  # global toggle (LLM is called inside extractor)


# -------- FLASK APP --------
app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

ocr_model = None
device = None
# ner_pipeline = None  # if you decide to enable NER later


# -------- HELPER FUNCTIONS --------
def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXT


def safe_remove(path: str):
    """Safely delete temporary files."""
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logging.warning(f"Could not remove {path}: {e}")


# -------- API ROUTES --------
@app.route("/health", methods=["GET"])
def health():
    global ocr_model
    ocr_ok = ocr_model is not None
    return jsonify(
        {
            "status": "healthy" if ocr_ok else "degraded",
            "service": "Universal-Prescription-OCR v3.0 (Regex + LLM)"
            if USE_LLM_REFINEMENT
            else "Universal-Prescription-OCR v3.0 (Regex-Only)",
            "ocr_available": ocr_ok,
            "llm_enabled": USE_LLM_REFINEMENT,
            # "ner_available": ner_pipeline is not None if ner_pipeline else False,
            "version": "3.0",
            "supported_types": [t.value for t in PrescriptionType],
        }
    )


@app.route("/process", methods=["POST"])
def process():
    global ocr_model

    if ocr_model is None:
        return jsonify({"success": False, "error": "OCR not initialized"}), 500

    logging.info("📥 Received /process request")

    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400

        upload = request.files["file"]
        if not upload.filename or not allowed_file(upload.filename):
            return jsonify({"success": False, "error": "Invalid file type"}), 400

        filename = secure_filename(upload.filename)
        mrn = request.form.get("mrn") or request.args.get("mrn")
        document_type = request.form.get("document_type", "prescription")

        logging.info(f"Processing file: {filename}, Type: {document_type}")

        unique_name = f"{uuid.uuid4().hex}_{filename}"
        saved_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        upload.save(saved_path)

        # --- Step 1: OCR ---
        ocr_out = process_document_ocr(saved_path, ocr_model)
        safe_remove(saved_path)

        if not ocr_out.get("success"):
            return (
                jsonify({"success": False, "error": ocr_out.get("error", "OCR failed")}),
                500,
            )

        # --- Step 2: Extraction (Regex + LLM) ---
        extractor = UniversalExtractor()
        all_page_extractions = []
        page_texts = ocr_out.get("page_texts", [])

        for page_num, page_text in enumerate(page_texts):
            logging.debug(f"Extracting entities from page {page_num}...")
            extraction = extractor.extract_all(page_text)
            all_page_extractions.append({"page": page_num, "extraction": extraction})

        primary_extraction = all_page_extractions[0]["extraction"] if all_page_extractions else {}

        # Inject MRN if provided
        if mrn and primary_extraction.get("patient"):
            primary_extraction["patient"]["id"] = mrn

        # --- Step 3: Validation & Routing ---
        quality_score = ocr_out.get("average_confidence", 0.0)
        critical_fields = {
            "patient.name": primary_extraction.get("patient", {}).get("name"),
            "patient.id": primary_extraction.get("patient", {}).get("id"),
            "prescriber.name": primary_extraction.get("prescriber", {}).get("name"),
            "medications": primary_extraction.get("medications", []),
        }

        missing_fields = [
            field
            for field, value in critical_fields.items()
            if not value or (isinstance(value, list) and len(value) == 0)
        ]

        # If MRN was supplied, don't treat patient.id as missing
        if mrn and not critical_fields["patient.id"]:
            if "patient.id" in missing_fields:
                missing_fields.remove("patient.id")
            primary_extraction.setdefault("patient", {})
            primary_extraction["patient"]["id"] = mrn

        routing_action = "AUTO_PROCESS"
        routing_reason = "Complete extraction with high confidence"
        priority = "normal"

        if quality_score < 0.7:
            routing_action = "MANUAL_REVIEW"
            routing_reason = "Low OCR confidence"
            priority = "high"
        elif len(missing_fields) > 0:
            routing_action = "MANUAL_REVIEW"
            routing_reason = f"Missing critical fields: {', '.join(missing_fields)}"
            priority = "high"
        elif (
            primary_extraction.get("metadata", {}).get("prescription_type")
            == PrescriptionType.CONTROLLED_SUBSTANCE.value
        ):
            routing_action = "MANUAL_REVIEW"
            routing_reason = "Controlled substance requires verification"
            priority = "urgent"

        # --- Step 4: Final Response ---
        response_json = {
            "document_id": f"doc_{uuid.uuid4().hex}",
            "success": True,
            "classification": {
                "document_category": document_type,
                "prescription_type": primary_extraction.get("metadata", {}).get(
                    "prescription_type", "unknown"
                ),
                "is_electronic": primary_extraction.get("metadata", {}).get(
                    "is_electronic", False
                ),
            },
            "ocr_confidence": quality_score,
            "ocr_text": ocr_out.get("full_text", ""),
            "ocr_text_length": len(ocr_out.get("full_text", "")),
            "ocr_raw": ocr_out.get("ocr_results", {}),
            "extraction": primary_extraction,
            "extraction_per_page": all_page_extractions,
            "validation": {
                "validation_status": "PENDING_REVIEW"
                if routing_action == "MANUAL_REVIEW"
                else "PASSED",
                "quality_score": quality_score,
                "missing_critical_fields": missing_fields,
                "completeness_score": 1.0 - (len(missing_fields) / len(critical_fields)),
            },
            "routing_decision": {
                "action": routing_action,
                "reason": routing_reason,
                "priority": priority,
                "requires_pharmacist_review": primary_extraction.get(
                    "metadata", {}
                ).get("prescription_type")
                in [PrescriptionType.CONTROLLED_SUBSTANCE.value, PrescriptionType.COMPOUND.value],
            },
            "database_result": {
                "patient_id": primary_extraction.get("patient", {}).get("id") or mrn,
                "record_id": f"rec_{uuid.uuid4().hex}",
                "success": True,
                "matched_existing_patient": bool(mrn),
            },
            "meta": {
                "page_count": ocr_out.get("page_count", 0),
                "total_text_regions": ocr_out.get("total_text_regions", 0),
                "ml_entities_detected": primary_extraction.get("ml_entities_count", 0),
                "processing_timestamp": datetime.utcnow().isoformat() + "Z",
                "api_version": "3.0 + LLM" if USE_LLM_REFINEMENT else "3.0 (Regex-Only)",
            },
        }

        # Save response locally for debugging
        try:
            output_path = os.path.join(
                app.config["UPLOAD_FOLDER"], f"{unique_name}_response.json"
            )
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(response_json, f, ensure_ascii=False, indent=2)
            logging.info(f"Response saved to {output_path}")
        except Exception as e:
            logging.warning(f"Could not save response: {e}")

        logging.info(f"✅ Successfully processed {filename}")
        return jsonify(response_json)

    except Exception as e:
        logging.error(f"Error in /process: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"Internal error: {str(e)}"}), 500


@app.route("/extract", methods=["POST"])
def extract_text():
    """Field-level extraction endpoint (regex-only)."""
    try:
        from extraction import PatternLibrary  # local import to avoid cycles

        data = request.get_json()
        text = data.get("text")
        field_type = data.get("field_type")

        if not text or not field_type:
            return jsonify({"success": False, "error": "Missing text or field_type"}), 400

        patterns = PatternLibrary()
        extractor = UniversalExtractor(None)
        result = {"success": True, "field_type": field_type, "extracted": None, "confidence": 0.0}

        if field_type == "patient_name":
            value, conf = extractor.extract_with_confidence(text, patterns.PATIENT_NAME)
            result.update({"extracted": value, "confidence": conf})
        elif field_type == "doctor_name":
            value, conf = extractor.extract_with_confidence(text, patterns.DOCTOR_NAME)
            result.update({"extracted": value, "confidence": conf})
        elif field_type == "medications":
            meds = extractor.extract_medications(text, [])
            from dataclasses import asdict

            result["extracted"] = [asdict(m) for m in meds]
        elif field_type == "date":
            value, conf = extractor.extract_with_confidence(text, patterns.DATE_PATTERNS)
            result.update({"extracted": value, "confidence": conf})
        else:
            return jsonify({"success": False, "error": f"Unknown field_type: {field_type}"}), 400

        return jsonify(result)

    except Exception as e:
        logging.error(f"Error in /extract: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/validate", methods=["POST"])
def validate():
    """Validate extracted prescription data."""
    try:
        data = request.get_json()
        extraction = data.get("extraction", {})
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": [],
        }

        patient = extraction.get("patient", {})
        if not patient.get("name"):
            validation_results["errors"].append("Missing patient name")
            validation_results["is_valid"] = False

        prescriber = extraction.get("prescriber", {})
        if not prescriber.get("name"):
            validation_results["errors"].append("Missing prescriber name")
            validation_results["is_valid"] = False

        medications = extraction.get("medications", [])
        if not medications:
            validation_results["errors"].append("No medications found")
            validation_results["is_valid"] = False

        metadata = extraction.get("metadata", {})
        if metadata.get("prescription_type") == PrescriptionType.CONTROLLED_SUBSTANCE.value:
            if not prescriber.get("dea"):
                validation_results["errors"].append(
                    "Controlled substance prescription missing DEA number"
                )
                validation_results["is_valid"] = False

        return jsonify(validation_results)

    except Exception as e:
        logging.error(f"Error in /validate: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/test", methods=["GET"])
def test():
    return jsonify(
        {
            "service": "Universal Prescription OCR System v3.0 (Regex + LLM)",
            "status": "running",
            "llm_enabled": USE_LLM_REFINEMENT,
            "capabilities": [
                "Multi-format OCR (PDF, images)",
                "Regex-based extraction",
                "Offline LLM refinement (Phi-3-mini GGUF)",
                "Comprehensive field validation",
                "Routing automation",
            ],
            "endpoints": {
                "/health": "GET - System health check",
                "/process": "POST - Upload & extract prescription",
                "/extract": "POST - Extract specific fields",
                "/validate": "POST - Validate extraction",
                "/test": "GET - Info",
            },
        }
    )


# -------- MAIN ENTRY POINT --------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    logging.info("🚀 Starting Universal Prescription OCR System v3.0 (Regex + LLM)")
    logging.info(f"🌐 Server: http://0.0.0.0:{port}")

    try:
        device = init_device()
        ocr_model = init_ocr()
        # ner_pipeline = load_biomedical_ner()  # optional, currently unused in extraction

        if ocr_model is None:
            logging.critical("❌ OCR failed to initialize. Exiting.")
            raise SystemExit(1)

        logging.info("✅ OCR initialized successfully.")
        app.run(host="0.0.0.0", port=port, debug=False)
    except KeyboardInterrupt:
        logging.info("\n👋 Shutting down gracefully...")
    except Exception as e:
        logging.critical(f"❌ Fatal error: {e}", exc_info=True)
        raise SystemExit(1)
