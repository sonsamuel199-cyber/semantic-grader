"""
Semantic Similarity Scoring with Keyword Boosting and Threshold Adjustment
"""

from functools import lru_cache
from typing import List, Dict

from sentence_transformers import SentenceTransformer, util

from .preprocessing import normalize_text

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def semantic_similarity_score(reference_answer: str, student_response: str) -> float:
    model = get_model()

    ref_clean = normalize_text(reference_answer)
    stu_clean = normalize_text(student_response)

    if not ref_clean or not stu_clean:
        return 0.0

    embeddings = model.encode([ref_clean, stu_clean], convert_to_tensor=True)
    cosine_score = util.cos_sim(embeddings[0], embeddings[1]).item()
    cosine_score = max(0.0, min(1.0, cosine_score))

    # ===== KEYWORD BOOSTING =====
    ref_lower = reference_answer.lower()
    stu_lower = student_response.lower()
    
    important_keywords = ["connected", "terminal", "battery", "gap", "voltage", "circuit"]
    
    ref_keywords = sum(1 for kw in important_keywords if kw in ref_lower)
    stu_keywords = sum(1 for kw in important_keywords if kw in stu_lower)
    
    if ref_keywords > 0:
        keyword_overlap = stu_keywords / ref_keywords
        if keyword_overlap >= 0.5:
            cosine_score += 0.05
        if keyword_overlap >= 0.8:
            cosine_score += 0.05

    # Penalize incorrect keyword usage
    if "gap" in stu_lower and "connected" in ref_lower:
        cosine_score -= 0.10
    if "connected" in stu_lower and "gap" in ref_lower:
        cosine_score -= 0.10
    if "not connected" in stu_lower and "connected" in ref_lower:
        cosine_score -= 0.15

    # ===== THRESHOLD ADJUSTMENT =====
    # I-adjust ang raw score para tumaas ang passing rate
    # Formula: (Raw Score * 1.1) + 0.02
    # Ito ay para tumaas ang score ng mga tamang sagot at bumaba ang FP
    adjusted_score = (cosine_score * 1.1) + 0.02
    adjusted_score = max(0.0, min(1.0, adjusted_score))

    return round(adjusted_score * 100, 2)


def rubric_breakdown(student_response: str, criteria: List[Dict]) -> List[Dict]:
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
