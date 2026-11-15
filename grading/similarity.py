"""
Semantic Similarity Scoring with Keyword Boosting
---------------------------------------------------
Implements the core grading algorithm described in Chapter 3 (Methodology):
SBERT generates semantic vector representations, and Cosine Similarity
measures the degree of conceptual alignment.

A keyword boosting layer is added to penalize incorrect keyword usage
(e.g., using "gap" when the correct answer is "connected").
"""

from functools import lru_cache
from typing import List, Dict

from sentence_transformers import SentenceTransformer, util

from .preprocessing import normalize_text

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Loads the SBERT model once and caches it for reuse."""
    return SentenceTransformer(MODEL_NAME)


def semantic_similarity_score(reference_answer: str, student_response: str) -> float:
    """
    Returns a similarity score from 0-100 between a reference answer and a
    student response, with keyword boosting for domain-specific terms.
    """
    model = get_model()

    ref_clean = normalize_text(reference_answer)
    stu_clean = normalize_text(student_response)

    if not ref_clean or not stu_clean:
        return 0.0

    embeddings = model.encode([ref_clean, stu_clean], convert_to_tensor=True)
    cosine_score = util.cos_sim(embeddings[0], embeddings[1]).item()
    cosine_score = max(0.0, min(1.0, cosine_score))

    # ===== KEYWORD BOOSTING (Science/Electricity Domain) =====
    ref_lower = reference_answer.lower()
    stu_lower = student_response.lower()
    
    # Important keywords for Science (electricity)
    important_keywords = ["connected", "terminal", "battery", "gap", "voltage", "circuit"]
    
    ref_keywords = sum(1 for kw in important_keywords if kw in ref_lower)
    stu_keywords = sum(1 for kw in important_keywords if kw in stu_lower)
    
    # Boost if student has keywords that reference also has
    if ref_keywords > 0:
        keyword_overlap = stu_keywords / ref_keywords
        if keyword_overlap >= 0.5:
            cosine_score += 0.05  # +5% boost
        if keyword_overlap >= 0.8:
            cosine_score += 0.05  # +10% total

    # Penalize if student says "gap" but reference says "connected"
    if "gap" in stu_lower and "connected" in ref_lower:
        cosine_score -= 0.10  # -10% penalty
    if "connected" in stu_lower and "gap" in ref_lower:
        cosine_score -= 0.10  # -10% penalty

    # Penalize if student says "not connected" when reference says "connected"
    if "not connected" in stu_lower and "connected" in ref_lower:
        cosine_score -= 0.15  # -15% penalty

    cosine_score = max(0.0, min(1.0, cosine_score))
    return round(cosine_score * 100, 2)


def rubric_breakdown(student_response: str, criteria: List[Dict]) -> List[Dict]:
    """
    Returns a per-criterion keyword-presence breakdown for instructor
    feedback ONLY. This is never used to compute the final grade.
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
