"""
Semantic Similarity Scoring
----------------------------
Implements the core grading algorithm described in Chapter 3 (Methodology):
a pre-trained Sentence-BERT (SBERT) model generates semantic vector
representations of the reference answer and the student response, and
Cosine Similarity measures the degree of conceptual alignment between them.

This score is the ONLY thing that determines the final grade. The optional
rubric (see rubric_breakdown below) never affects it — per Chapter 3
Scope and Delimitations, the rubric is for instructor feedback only.
"""

from functools import lru_cache
from typing import List, Dict

from sentence_transformers import SentenceTransformer, util

from .preprocessing import normalize_text

# Lightweight pre-trained SBERT model — matches the thesis's emphasis on a
# "lightweight, accessible prototype" that doesn't require heavy compute.
MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Loads the SBERT model once and caches it for reuse."""
    return SentenceTransformer(MODEL_NAME)


def semantic_similarity_score(reference_answer: str, student_response: str) -> float:
    """
    Returns a similarity score from 0-100 between a reference answer and a
    student response:

        score = cosine_similarity( SBERT(normalize(reference)),
                                    SBERT(normalize(student)) ) * 100
    """
    model = get_model()

    ref_clean = normalize_text(reference_answer)
    stu_clean = normalize_text(student_response)

    if not ref_clean or not stu_clean:
        return 0.0

    embeddings = model.encode([ref_clean, stu_clean], convert_to_tensor=True)
    cosine_score = util.cos_sim(embeddings[0], embeddings[1]).item()
    cosine_score = max(0.0, min(1.0, cosine_score))  # clamp for safety

    return round(cosine_score * 100, 2)


def rubric_breakdown(student_response: str, criteria: List[Dict]) -> List[Dict]:
    """
    criteria: list of dicts like {"keywords": "sunlight, water, CO2", "weight": 40}

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
