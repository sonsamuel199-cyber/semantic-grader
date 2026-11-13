"""
Semantic Similarity Scoring with Calibration
--------------------------------------------
Implements the core grading algorithm with a built-in Calibration Curve.
"""

from functools import lru_cache
from typing import List, Dict

from sentence_transformers import SentenceTransformer, util

from .preprocessing import normalize_text

# Lightweight pre-trained SBERT model
MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Loads the SBERT model once and caches it for reuse."""
    return SentenceTransformer(MODEL_NAME)


def semantic_similarity_score(reference_answer: str, student_response: str) -> float:
    """
    Returns a calibrated similarity score from 0-100 between a reference answer 
    and a student response.
    """
    model = get_model()

    ref_clean = normalize_text(reference_answer)
    stu_clean = normalize_text(student_response)

    if not ref_clean or not stu_clean:
        return 0.0

    embeddings = model.encode([ref_clean, stu_clean], convert_to_tensor=True)
    cosine_score = util.cos_sim(embeddings[0], embeddings[1]).item()
    
    # Clamp between 0.0 and 1.0 for safety
    cosine_score = max(0.0, min(1.0, cosine_score))

    # ==========================================
    # CALIBRATION LOGIC (Grading Curve)
    # ==========================================
    # Formula: (Raw Score * Multiplier) + Offset
    # Halimbawa: 0.60 -> (0.60 * 1.3) + 0.05 = 0.83
    calibrated_score = (cosine_score * 1.3) + 0.05
    
    # Siguraduhing hindi lalampas sa 1.0 (100%)
    calibrated_score = min(1.0, calibrated_score)

    # Ibalik bilang percentage (0-100)
    return round(calibrated_score * 100, 2)


def rubric_breakdown(student_response: str, criteria: List[Dict]) -> List[Dict]:
    """
    Returns a per-criterion keyword-presence breakdown for instructor feedback ONLY.
    """
    clean_response = normalize_text(student_response)
    results = []

    for c in criteria:
        keywords = [
            normalize_text(k.strip()) for k in c["keywords"].split(",") if k.strip()
        ]
        matched = [k for k in keywords if k and k in clean_response]
        results.append(
            {
                "keywords": c["keywords"],
                "weight": c["weight"],
                "matched": matched,
                "hit": len(matched) > 0,
            }
        )

    return results
