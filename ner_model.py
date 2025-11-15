# ner_model.py
import logging
from transformers import pipeline


def load_biomedical_ner():
    """
    Optional helper to load the biomedical NER model.

    Not currently wired into extraction (v3.0 is regex + LLM),
    but this gives you a ready-to-use NER pipeline for future use.
    """
    try:
        model_name = "d4data/biomedical-ner-all"
        logging.info("Loading biomedical NER model...")

        ner = pipeline(
            "ner",
            model=model_name,
            tokenizer=model_name,
            aggregation_strategy="simple",
            device=-1,  # CPU by default; set to 0 if you want GPU
        )
        logging.info("NER model loaded successfully.")
        return ner

    except Exception as e:
        logging.error(f"NER failed to load: {e}")
        return None
