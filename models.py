"""
models.py

Defines all data structures (dataclasses) and Enums for the
Universal Prescription OCR System.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# -------- DATA MODELS --------
class PrescriptionType(Enum):
    STANDARD = "standard"
    CONTROLLED_SUBSTANCE = "controlled_substance"
    HOSPITAL_DISCHARGE = "hospital_discharge"
    E_PRESCRIPTION = "e_prescription"
    COMPOUND = "compound"
    UNKNOWN = "unknown"


@dataclass
class Medication:
    name: str
    strength: Optional[str] = None
    form: Optional[str] = None
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    route: Optional[str] = None
    quantity: Optional[str] = None
    refills: Optional[str] = None
    instructions: Optional[str] = None
    confidence: float = 0.0


@dataclass
class Patient:
    name: Optional[str] = None
    id: Optional[str] = None
    dob: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    allergies: List[str] = field(default_factory=list)
    insurance: Optional[str] = None


@dataclass
class Prescriber:
    name: Optional[str] = None
    id: Optional[str] = None
    license: Optional[str] = None
    specialty: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    signature_present: bool = False
    npi: Optional[str] = None
    dea: Optional[str] = None


@dataclass
class Pharmacy:
    name: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    license: Optional[str] = None


@dataclass
class PrescriptionMetadata:
    prescription_number: Optional[str] = None
    date_written: Optional[str] = None
    date_filled: Optional[str] = None
    expiration_date: Optional[str] = None
    prescription_type: PrescriptionType = PrescriptionType.UNKNOWN
    is_original: bool = True
    is_electronic: bool = False
    language: str = "en"


@dataclass
class ClinicalInfo:
    diagnosis: List[str] = field(default_factory=list)
    icd_codes: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    vital_signs: Dict[str, str] = field(default_factory=dict)
    lab_results: Dict[str, str] = field(default_factory=dict)
