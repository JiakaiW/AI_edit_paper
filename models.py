from pydantic import BaseModel
from typing import List

class GrammarCorrection(BaseModel):
    corrected_sentence: str
    confidence: float
    explanation: str

class QualityAssurance(BaseModel):
    is_valid: bool
    maintains_meaning: bool
    technical_accuracy: bool
    concerns: List[str]

class PreprocessingVerification(BaseModel):
    is_valid: bool 