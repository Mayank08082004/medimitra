# llm_refiner.py — FINAL A1 STRICT + FALLBACK SAFE VERSION

import logging
import re
from pathlib import Path

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

MODEL_PATH = "models/Phi-3-mini-4k-instruct-q4.gguf"


def init_llm():
    """Initialize the Phi-3-mini LLM cleanly with safe params."""
    if not Llama:
        logging.error("llama_cpp not installed — LLM disabled.")
        return None

    if not Path(MODEL_PATH).exists():
        logging.error(f"LLM model missing: {MODEL_PATH}")
        return None

    try:
        logging.info(f"Loading LLM: {MODEL_PATH}")
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=1200,
            n_threads=4,
            n_gpu_layers=0,  # Metal fails silently >0 on many Macs
            verbose=False,
        )
        logging.info("LLM loaded successfully.")
        return llm
    except Exception as e:
        logging.error(f"LLM load failed: {e}")
        return None


# ----------- STRICT PROMPT -----------
PROMPT = """
You are cleaning OCR text.

Rules:
- Correct only obvious OCR mistakes.
- DO NOT add new information.
- DO NOT invent medications.
- DO NOT add diagnoses, procedures, or interpretations.
- DO NOT summarize.
- If a word cannot be corrected, KEEP IT AS-IS.
- Preserve original meaning strictly.

Return ONLY the cleaned version of this text:

{text}

CLEANED:
"""


# ----------- SANITIZER (removes hallucinations if LLM misbehaves) -----------

FORBIDDEN_PATTERNS = [
    r"\bAspirin\b",
    r"\bAmlodipine\b",
    r"\bOxygen therapy\b",
    r"\bBronchoscopy\b",
    r"\bCT scan\b",
    r"\bonce daily\b",
    r"\bthree times\b",
    r"\bfour times\b",
    r"\bnot indicated\b",
]


def sanitize(text: str) -> str:
    cleaned = text

    # Remove hallucinated lines
    for pat in FORBIDDEN_PATTERNS:
        cleaned = re.sub(pat + r".*", "", cleaned, flags=re.IGNORECASE)

    # Remove bullet-point lists LLM might create
    cleaned = re.sub(r"^[\-\•].*", "", cleaned, flags=re.MULTILINE)

    # Collapse blank lines
    cleaned = "\n".join([ln for ln in cleaned.split("\n") if ln.strip()])

    return cleaned.strip()


# ----------- MAIN REFINE FUNCTION -----------

def refine_text_with_llm(text: str) -> str:
    """Strict OCR cleaning with hallucination-proof fallback."""
    llm = init_llm()
    if not llm:
        return text  # fallback to original

    try:
        logging.info("LLM refinement triggered.")

        output = llm(
            PROMPT.format(text=text),
            max_tokens=300,
            temperature=0.0,
            top_k=10,
            top_p=0.9,
            stop=["CLEANED:"],
        )

        raw = output["choices"][0]["text"].strip()

        # Fallback if LLM gave nothing
        if not raw or len(raw) < 3:
            logging.warning("LLM returned empty or unusable text — using fallback.")
            return text

        cleaned = sanitize(raw)

        # Fallback if sanitization removed everything
        if not cleaned or len(cleaned) < 3:
            logging.warning("Sanitizer removed all text — using OCR fallback.")
            return text

        return cleaned

    except Exception as e:
        logging.error(f"LLM refine error: {e}")
        return text  # safe fallback
