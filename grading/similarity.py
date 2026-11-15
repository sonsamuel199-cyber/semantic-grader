"""
Semantic Similarity Scoring
----------------------------
Implements the core grading algorithm described in Chapter 3 (Methodology):
a pre-trained Sentence-BERT (SBERT) model generates semantic vector
representations of the reference answer and the student response, and
Cosine Similarity measures the degree of conceptual alignment between them.
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
    student response using pure SBERT + Cosine Similarity.
    """
    model = get_model()

    ref_clean = normalize_text(reference_answer)
    stu_clean = normalize_text(student_response)

    if not ref_clean or not stu_clean:
        return 0.0

    embeddings = model.encode([ref_clean, stu_clean], convert_to_tensor=True)
    cosine_score = util.cos_sim(embeddings[0], embeddings[1]).item()
    cosine_score = max(0.0, min(1.0, cosine_score))

    return round(cosine_score * 100, 2)


def rubric_breakdown(student_response: str, criteria: List[Dict]) -> List[Dict]:
    """Returns a per-criterion keyword-presence breakdown for instructor feedback ONLY."""
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
